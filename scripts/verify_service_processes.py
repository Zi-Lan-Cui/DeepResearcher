"""无 LLM 的 API 与 Worker 进程、故障接管与 HTTP/SSE 容量验证。

本脚本对着显式配置的 PostgreSQL 启动真实 Uvicorn API 与两个真实
Worker OS 进程。图仅供测试、以文件同步,不产生模型/搜索/抓取费用。
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

from deepresearcher.service.bootstrap.loop import asyncio_loop_factory
from deepresearcher.service.settings import get_service_config
from deepresearcher.service.storage.engine import make_engine, make_session_factory
from deepresearcher.service.storage.tables import Run, RunEvent, User

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "service-automation-password"
_WINDOWS = sys.platform == "win32"
# 三个真实进程同时冷启动要等建库、建 checkpoint 表与连 broker，本机实测 30–90 秒。
_STARTUP_TIMEOUT_SECONDS = 180.0


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
    by_name: dict[str, subprocess.Popen] = field(default_factory=dict)

    def start(self, name: str, *command: str) -> subprocess.Popen:
        log = (self.log_dir / f"{name}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            # 子进程用名字标识自己：Windows 上 `Popen.pid` 是 venv 启动器的 pid，
            # 写标记的是它拉起的真实解释器，两者对不上，只能按名字回查进程。
            env={**self.env, "WORKER_INSTANCE": name},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            # 独立进程组是「只给这个子进程发停止信号」的前提：Windows 没有 SIGTERM，
            # 优雅停止靠 CTRL_BREAK_EVENT，而它只能投给一个进程组。
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if _WINDOWS else 0,
        )
        self.processes.append(process)
        self.logs.append(log)
        self.by_name[name] = process
        return process

    def stop(self, process: subprocess.Popen, *, hard: bool = False) -> None:
        if process.poll() is not None:
            return
        if hard:
            if _WINDOWS:
                # 启动器与真实解释器是两个进程，只杀启动器会把执行中的 worker 留成孤儿，
                # 它照样续租，接管永远不会发生。
                subprocess.run(  # noqa: S603
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    check=False,
                    capture_output=True,
                )
            else:
                process.kill()
        elif _WINDOWS:
            # Windows 没有 SIGTERM，terminate() 就是强杀；优雅停止只能投 CTRL_BREAK。
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGTERM)
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
            timeout=_STARTUP_TIMEOUT_SECONDS,
            label="API readiness",
        )


def _wait_log(
    path: Path,
    needle: str,
    *,
    process: subprocess.Popen | None = None,
    timeout: float = _STARTUP_TIMEOUT_SECONDS,
) -> None:
    """等子进程把自己的就绪行写进日志再返回。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise AssertionError(
                f"{path.name} exited with {process.returncode}:\n"
                + path.read_text(encoding="utf-8", errors="replace")
            )
        if path.exists() and needle in path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(0.1)
    tail = path.read_text(encoding="utf-8", errors="replace")[-2000:] if path.exists() else ""
    raise AssertionError(f"timeout waiting for {needle!r} in {path}:\n{tail}")


def _start_worker(group: ProcessGroup, log_dir: Path, name: str) -> None:
    """起一个测试 Worker，等它把 worker_started 写进日志再返回。"""
    process = group.start(name, sys.executable, "scripts/fake_service_worker.py")
    _wait_log(log_dir / f"{name}.log", "worker_started", process=process)


async def _run_detail(client: httpx.AsyncClient, token: str, run_id: str) -> dict:
    response = await client.get(f"/api/runs/{run_id}", headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()


def _entered_instances(marker_dir: Path, run_id: str) -> dict[str, set[int]]:
    """进过图的 worker 实例 → 写标记的真实解释器 pid。

    标记名形如 entered-<run_id>-<instance>-<pid>；实例名自带连字符，从右侧切。
    """
    prefix = f"entered-{run_id}-"
    instances: dict[str, set[int]] = {}
    for path in marker_dir.glob(f"{prefix}*"):
        instance, _, pid = path.name[len(prefix) :].rpartition("-")
        instances.setdefault(instance, set()).add(int(pid))
    return instances


def _entered_pids(marker_dir: Path, run_id: str) -> set[int]:
    return {pid for pids in _entered_instances(marker_dir, run_id).values() for pid in pids}


def _only_instance(instances: dict[str, set[int]]) -> str:
    (instance,) = instances
    return instance


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
            assert run.fence_token == 2, f"expected takeover fence_token=2, got {run.fence_token}"
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
            # LangGraph 的 checkpoint 表不对 runs 建外键，删 run 之前要自己清。
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
        raise ValueError("Service process verification requires PostgreSQL")

    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"
    email = f"service-{uuid4().hex}@test.invalid"
    run_ids: list[str] = []
    work_dir = Path(tempfile.mkdtemp(prefix="deepresearcher-service-"))
    marker_dir = work_dir / "markers"
    marker_dir.mkdir()
    env = {
        **os.environ,
        "DEEPRESEARCHER_ENV_FILE": str(work_dir / "no-env-file"),
        "SERVICE_DATABASE_URL": database_url,
        "SERVICE_JWT_SECRET": "service-automation-secret-must-be-at-least-32-chars",
        "SERVICE_HOST": "127.0.0.1",
        "SERVICE_PORT": str(port),
        "SERVICE_MAX_CONCURRENT_RUNS_PER_USER": "32",
        "SERVICE_WORKER_MAX_RUNNING_RUNS": "8",
        "SERVICE_MAX_GLOBAL_QUEUED_RUNS": "100",
        "SERVICE_WORKER_HEARTBEAT_SECONDS": "1",
        # 唯一租约：接管时延等于不可见时长，续期压到下限让租约确认能跑满一轮。
        "SERVICE_ROCKETMQ_INVISIBLE_SECONDS": str(args.lease_seconds),
        "SERVICE_ROCKETMQ_VISIBILITY_RENEW_SECONDS": "10",
        "SERVICE_JSONL_EVENTS": "false",
        "SERVICE_LOG_DIR": str(work_dir / "service"),
        "OBSERVABILITY_LOG_DIR": str(work_dir / "logs"),
        "TOOL_CACHE_ENABLED": "false",
        "WORKER_MARKER_DIR": str(marker_dir),
        "APP_ENV": "development",
    }
    group = ProcessGroup(env=env, log_dir=work_dir)
    api = group.start("api-1", sys.executable, "-m", "deepresearcher.service.bootstrap.main")
    for index in (1, 2):
        _start_worker(group, work_dir, f"worker-{index}")
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

            failover_run = await _create_run(client, token, "Service kill takeover")
            run_ids.append(failover_run)
            await _eventually(
                lambda: _run_detail(client, token, failover_run),
                lambda item: item["status"] == "running",
                timeout=20,
                label="first Worker claim",
            )
            pids = await _eventually(
                lambda: asyncio.sleep(0, result=_entered_instances(marker_dir, failover_run)),
                lambda item: sum(len(pids) for pids in item.values()) == 1,
                timeout=10,
                label="single initial graph execution",
            )
            await asyncio.sleep(0.5)
            assert _entered_instances(marker_dir, failover_run) == pids

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

            # 滚动替换 API,Worker 持有的图全程阻塞。
            group.stop(api)
            api = group.start(
                "api-2", sys.executable, "-m", "deepresearcher.service.bootstrap.main"
            )
            await _wait_api(base_url)
            rolled = await _eventually(
                lambda: _run_detail(client, token, failover_run),
                lambda item: item["status"] == "running",
                timeout=10,
                label="API rolling restart",
            )
            assert rolled["status"] == "running"

            owner_pid = next(iter(_entered_pids(marker_dir, failover_run)))
            group.stop(group.by_name[_only_instance(pids)], hard=True)
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
                timeout=args.lease_seconds + 10,
                label="takeover completion",
            )
            assert completed["report_markdown"].startswith("# Service")
            await _validate_database(database_url, failover_run)

            # 优雅 SIGTERM → interrupted → checkpoint 续跑。Worker 优雅停止时把运行中的
            # run 置为 interrupted 且不确认消息；这条消息在可见性到期后重新投递，
            # 新启动的 Worker 以 resume=True 接管。启动时不再做任何扫描。
            graceful_run = await _create_run(client, token, "Service graceful resume")
            run_ids.append(graceful_run)
            await _eventually(
                lambda: _run_detail(client, token, graceful_run),
                lambda item: item["status"] == "running",
                timeout=20,
                label="graceful run first claim",
            )
            graceful_pids = await _eventually(
                lambda: asyncio.sleep(0, result=_entered_instances(marker_dir, graceful_run)),
                lambda item: sum(len(pids) for pids in item.values()) == 1,
                timeout=10,
                label="graceful run single initial execution",
            )

            # SIGTERM（Windows 上是 CTRL_BREAK）而非强杀：Worker 排空并干净地释放该 run。
            group.stop(group.by_name[_only_instance(graceful_pids)], hard=False)
            interrupted = await _eventually(
                lambda: _run_detail(client, token, graceful_run),
                lambda item: item["status"] == "interrupted",
                timeout=10,
                label="graceful shutdown marks the run interrupted",
            )
            assert interrupted["terminal_reason"] == "server_shutdown", (
                f"expected server_shutdown, got {interrupted['terminal_reason']!r}"
            )

            # 新启动的 Worker 在消息重新可见后从 checkpoint 续跑该 run，
            # 以新 PID 重入图。
            _start_worker(group, work_dir, "worker-resume")
            resumed = await _eventually(
                lambda: _run_detail(client, token, graceful_run),
                lambda item: item["status"] == "running",
                timeout=args.lease_seconds + 15,
                label="redelivery resumes the interrupted run",
            )
            assert resumed["status"] == "running"
            await _eventually(
                lambda: asyncio.sleep(0, result=_entered_pids(marker_dir, graceful_run)),
                lambda item: len(item) == 2,
                timeout=15,
                label="resumed execution under a new Worker PID",
            )
            (marker_dir / f"release-{graceful_run}").touch()
            completed_graceful = await _eventually(
                lambda: _run_detail(client, token, graceful_run),
                lambda item: item["status"] == "completed",
                timeout=20,
                label="graceful resume completion",
            )
            assert completed_graceful["report_markdown"].startswith("# Service")
            await _validate_database(database_url, graceful_run)

            # 补回一个 Worker，再按 1/2/4/8 验证队列与容量。
            _start_worker(group, work_dir, "worker-replacement")
            concurrency_results: dict[int, float] = {}
            accept_results: dict[int, float] = {}
            for level in args.concurrency_levels:
                wave_started = time.monotonic()
                wave = await asyncio.gather(
                    *[
                        _create_run(client, token, f"Service concurrency {level}/{index}")
                        for index in range(level)
                    ]
                )
                # 受理耗时单独记：这一段慢说明瓶颈在事务消息写入或受理事务，
                # 后面几段慢才是领取与执行的问题。
                accept_results[level] = round(time.monotonic() - wave_started, 3)
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
                    lambda wave=wave: asyncio.sleep(
                        0,
                        result={run_id: _entered_pids(marker_dir, run_id) for run_id in wave},
                    ),
                    lambda items: all(len(pids) == 1 for pids in items.values()),
                    timeout=10,
                    label=f"single graph execution for concurrency wave {level}",
                )
                # 若发生双领，第二个执行会在一个轮询间隔内进入；睡过窗口后仍应是单执行。
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
            "accept_seconds": accept_results,
            "failover_fence_token": 2,
            "run_done_count": 1,
            "graceful_shutdown_recovered": True,
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
    parser.add_argument("--lease-seconds", type=int, default=30, help="接管窗口（秒）")
    parser.add_argument("--sse-connections", type=int, default=100)
    parser.add_argument("--request-count", type=int, default=100)
    parser.add_argument("--concurrency-levels", default="1,2,4,8")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()
    args.lease_seconds = max(30, args.lease_seconds)
    args.sse_connections = max(0, args.sse_connections)
    args.request_count = max(0, args.request_count)
    args.concurrency_levels = tuple(
        int(item) for item in args.concurrency_levels.split(",") if int(item) > 0
    )
    if any(level > 8 for level in args.concurrency_levels):
        parser.error("this harness caps global active runs at 8")
    return args


def main() -> None:
    result = asyncio.run(verify(_parse_args()), loop_factory=asyncio_loop_factory())
    print(result)


if __name__ == "__main__":
    main()
