"""No-LLM M8 process, failover and HTTP/SSE capacity verification.

This script starts a real Uvicorn API and two real Worker OS processes against
an explicitly configured PostgreSQL database.  Its graph is test-only and waits
on files, so it incurs no model/search/fetch cost.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import delete, func, select, text

from deepsearch_agent.service.db import make_engine, make_session_factory
from deepsearch_agent.service.models import Run, RunEvent, User
from deepsearch_agent.service.settings import get_service_config

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "m8-automation-password"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ProcessGroup:
    env: dict[str, str]
    log_dir: Path
    processes: list[subprocess.Popen] = field(default_factory=list)
    logs: list[Any] = field(default_factory=list)

    def start(self, name: str, *command: str) -> subprocess.Popen:
        log = (self.log_dir / f"{name}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.processes.append(process)
        self.logs.append(log)
        return process

    def stop(self, process: subprocess.Popen, *, hard: bool = False) -> None:
        if process.poll() is not None:
            return
        process.send_signal(signal.SIGKILL if hard else signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    def close(self) -> None:
        for process in reversed(self.processes):
            self.stop(process)
        for log in self.logs:
            log.close()


async def _eventually(operation, predicate, *, timeout: float, label: str):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = await operation()
            if predicate(last):
                return last
        except (httpx.TransportError, KeyError):
            pass
        await asyncio.sleep(0.1)
    raise AssertionError(f"timeout waiting for {label}; last={last!r}")


async def _wait_api(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=2) as client:
        await _eventually(
            lambda: client.get("/"),
            lambda response: response.status_code == 200,
            timeout=20,
            label="API readiness",
        )


async def _run_detail(client: httpx.AsyncClient, token: str, run_id: str) -> dict:
    response = await client.get(f"/api/runs/{run_id}", headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()


def _entered_pids(marker_dir: Path, run_id: str) -> set[int]:
    return {int(path.name.rsplit("-", 1)[1]) for path in marker_dir.glob(f"entered-{run_id}-*")}


async def _create_run(client: httpx.AsyncClient, token: str, query: str) -> str:
    response = await client.post(
        "/api/runs",
        json={"query": query},
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    return str(response.json()["run_id"])


async def _hold_sse_connections(
    client: httpx.AsyncClient, token: str, run_id: str, count: int
) -> AsyncExitStack:
    stack = AsyncExitStack()
    headers = {"Authorization": f"Bearer {token}"}
    for _ in range(count):
        response = await stack.enter_async_context(
            client.stream("GET", f"/api/runs/{run_id}/events", headers=headers)
        )
        if response.status_code != 200:
            await stack.aclose()
            raise AssertionError(f"SSE connection failed: {response.status_code}")
    return stack


async def _validate_database(database_url: str, run_id: str) -> None:
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    try:
        async with factory() as session:
            run = await session.get(Run, run_id)
            assert run is not None
            assert run.status == "completed"
            assert run.attempt == 2, f"expected takeover attempt=2, got {run.attempt}"
            done_count = await session.scalar(
                select(func.count())
                .select_from(RunEvent)
                .where(RunEvent.run_id == run_id, RunEvent.event_type == "run_done")
            )
            seqs = list(
                (
                    await session.scalars(
                        select(RunEvent.seq).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
                    )
                ).all()
            )
        assert done_count == 1, f"run_done count is {done_count}"
        assert seqs == list(range(1, len(seqs) + 1)), f"event sequence has gaps: {seqs}"
    finally:
        await engine.dispose()


async def _cleanup(database_url: str, email: str, run_ids: list[str]) -> None:
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    try:
        async with factory() as session:
            # LangGraph checkpoint tables intentionally have no FK to runs.
            bind = session.get_bind()
            if bind.dialect.name == "postgresql" and run_ids:
                for run_id in run_ids:
                    for table_name in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                        await session.execute(
                            text(f"DELETE FROM {table_name} WHERE thread_id = :run_id"),
                            {"run_id": run_id},
                        )
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
    finally:
        await engine.dispose()


async def verify(args: argparse.Namespace) -> dict[str, Any]:
    database_url = args.database_url or get_service_config().database_url
    if not database_url.startswith("postgresql"):
        raise ValueError("M8 process verification requires PostgreSQL")

    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"
    email = f"m8-{uuid4().hex}@test.invalid"
    run_ids: list[str] = []
    work_dir = Path(tempfile.mkdtemp(prefix="deepsearch-m8-"))
    marker_dir = work_dir / "markers"
    marker_dir.mkdir()
    env = {
        **os.environ,
        "DEEPSEARCH_ENV_FILE": str(work_dir / "no-env-file"),
        "SERVICE_DATABASE_URL": database_url,
        "SERVICE_JWT_SECRET": "m8-automation-secret-must-be-at-least-32-chars",
        "SERVICE_HOST": "127.0.0.1",
        "SERVICE_PORT": str(port),
        "SERVICE_API_EMBEDDED_WORKER": "false",
        "SERVICE_MAX_CONCURRENT_RUNS_PER_USER": "32",
        "SERVICE_MAX_GLOBAL_RUNNING_RUNS": "8",
        "SERVICE_MAX_GLOBAL_QUEUED_RUNS": "100",
        "SERVICE_WORKER_LEASE_SECONDS": str(args.lease_seconds),
        "SERVICE_WORKER_HEARTBEAT_SECONDS": "1",
        "SERVICE_WORKER_POLL_SECONDS": "0.1",
        "SERVICE_JSONL_EVENTS": "false",
        "SERVICE_LOG_DIR": str(work_dir / "service"),
        "OBSERVABILITY_LOG_DIR": str(work_dir / "logs"),
        "TOOL_CACHE_ENABLED": "false",
        "M8_MARKER_DIR": str(marker_dir),
        "APP_ENV": "development",
    }
    group = ProcessGroup(env=env, log_dir=work_dir)
    api = group.start("api-1", sys.executable, "server.py")
    workers = [
        group.start(f"worker-{index}", sys.executable, "scripts/m8_fake_worker.py")
        for index in (1, 2)
    ]
    started = time.monotonic()
    passed = False
    try:
        await _wait_api(base_url)
        limits = httpx.Limits(
            max_connections=max(120, args.sse_connections + 10),
            max_keepalive_connections=max(20, args.sse_connections),
        )
        async with httpx.AsyncClient(base_url=base_url, timeout=20, limits=limits) as client:
            response = await client.post(
                "/api/register", json={"email": email, "password": PASSWORD}
            )
            response.raise_for_status()
            token = response.json()["token"]

            failover_run = await _create_run(client, token, "M8 kill takeover")
            run_ids.append(failover_run)
            await _eventually(
                lambda: _run_detail(client, token, failover_run),
                lambda item: item["status"] == "running",
                timeout=20,
                label="first Worker claim",
            )
            pids = await _eventually(
                lambda: asyncio.sleep(0, result=_entered_pids(marker_dir, failover_run)),
                lambda item: len(item) == 1,
                timeout=10,
                label="single initial graph execution",
            )
            await asyncio.sleep(0.5)
            assert _entered_pids(marker_dir, failover_run) == pids

            sse_stack = await _hold_sse_connections(
                client, token, failover_run, args.sse_connections
            )
            headers = {"Authorization": f"Bearer {token}"}
            responses = await asyncio.gather(
                *[
                    client.get(
                        "/api/runs" if index % 2 == 0 else f"/api/runs/{failover_run}",
                        headers=headers,
                    )
                    for index in range(args.request_count)
                ]
            )
            assert all(item.status_code == 200 for item in responses)
            await sse_stack.aclose()

            # Rolling API replacement while the Worker-owned graph stays blocked.
            group.stop(api)
            api = group.start("api-2", sys.executable, "server.py")
            await _wait_api(base_url)
            rolled = await _eventually(
                lambda: _run_detail(client, token, failover_run),
                lambda item: item["status"] == "running",
                timeout=10,
                label="API rolling restart",
            )
            assert rolled["status"] == "running"

            owner_pid = next(iter(pids))
            owner = next(process for process in workers if process.pid == owner_pid)
            group.stop(owner, hard=True)
            replacement_pids = await _eventually(
                lambda: asyncio.sleep(0, result=_entered_pids(marker_dir, failover_run)),
                lambda item: len(item) == 2,
                timeout=args.lease_seconds + 12,
                label="replacement Worker takeover",
            )
            assert owner_pid in replacement_pids
            (marker_dir / f"release-{failover_run}").touch()
            completed = await _eventually(
                lambda: _run_detail(client, token, failover_run),
                lambda item: item["status"] == "completed",
                timeout=20,
                label="takeover completion",
            )
            assert completed["report_markdown"].startswith("# M8")
            await _validate_database(database_url, failover_run)

            # Restore two live Workers, then verify queue/capacity at 1/2/4/8.
            workers.append(
                group.start("worker-replacement", sys.executable, "scripts/m8_fake_worker.py")
            )
            concurrency_results: dict[int, float] = {}
            for level in args.concurrency_levels:
                wave_started = time.monotonic()
                wave = await asyncio.gather(
                    *[
                        _create_run(client, token, f"M8 concurrency {level}/{index}")
                        for index in range(level)
                    ]
                )
                run_ids.extend(wave)
                await asyncio.gather(
                    *[
                        _eventually(
                            lambda run_id=run_id: _run_detail(client, token, run_id),
                            lambda item: item["status"] == "running",
                            timeout=20,
                            label=f"concurrency claim {run_id}",
                        )
                        for run_id in wave
                    ]
                )
                await _eventually(
                    lambda: asyncio.sleep(
                        0,
                        result={run_id: _entered_pids(marker_dir, run_id) for run_id in wave},
                    ),
                    lambda items: all(len(pids) == 1 for pids in items.values()),
                    timeout=10,
                    label=f"single graph execution for concurrency wave {level}",
                )
                # A second claimant would normally enter within one poll interval.
                await asyncio.sleep(0.3)
                assert all(len(_entered_pids(marker_dir, run_id)) == 1 for run_id in wave)
                for run_id in wave:
                    (marker_dir / f"release-{run_id}").touch()
                await asyncio.gather(
                    *[
                        _eventually(
                            lambda run_id=run_id: _run_detail(client, token, run_id),
                            lambda item: item["status"] == "completed",
                            timeout=20,
                            label=f"concurrency completion {run_id}",
                        )
                        for run_id in wave
                    ]
                )
                concurrency_results[level] = round(time.monotonic() - wave_started, 3)

        result = {
            "status": "passed",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "sse_connections": args.sse_connections,
            "http_requests": args.request_count,
            "concurrency_seconds": concurrency_results,
            "failover_attempt": 2,
            "run_done_count": 1,
        }
        passed = True
        return result
    finally:
        group.close()
        try:
            await _cleanup(database_url, email, run_ids)
        finally:
            if args.keep_artifacts or not passed:
                print(f"artifacts={work_dir}")
            else:
                shutil.rmtree(work_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url", help="dedicated PostgreSQL URL; defaults to service config"
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--lease-seconds", type=int, default=10)
    parser.add_argument("--sse-connections", type=int, default=100)
    parser.add_argument("--request-count", type=int, default=100)
    parser.add_argument("--concurrency-levels", default="1,2,4,8")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()
    args.lease_seconds = max(10, args.lease_seconds)
    args.sse_connections = max(0, args.sse_connections)
    args.request_count = max(0, args.request_count)
    args.concurrency_levels = tuple(
        int(item) for item in args.concurrency_levels.split(",") if int(item) > 0
    )
    if any(level > 8 for level in args.concurrency_levels):
        parser.error("this harness caps global active runs at 8")
    return args


def main() -> None:
    result = asyncio.run(verify(_parse_args()))
    print(result)


if __name__ == "__main__":
    main()
