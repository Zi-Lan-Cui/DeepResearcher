"""Runner：用户旅程走产品唯一入口（HTTP API），地面真相走数据库。

分工刻意为之：
- 提交/澄清回答/取消/详情 全部经 API —— 评测必须在与真实用户相同的路径上跑，
  旁路即失真；
- run_events 原始记录、runs.attempt 等取数走 DB 直读 —— SSE 只有投影帧，
  评测需要原始事件序列做行为断言，harness 是特权操作者视角，产品 API 保持
  用户最小面不被评测需求污染。

故障注入（kill/stop 接管类，F1/F2）需要 harness 掌控 worker 进程生命周期，
按 M8 验收脚本的模式在第二阶段接线；本文件对 fault=none/rerun_cache 与
duplicate_answer 给出完整闭环。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from deepsearch_agent.service.persistence.database import make_engine, make_session_factory
from deepsearch_agent.service.persistence.models import Run, RunEvent
from evals.deterministic import Artifact, process_metrics
from evals.schemas import EvalCase

TERMINAL = ("completed", "failed", "cancelled")

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
POLL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 4200.0


class RunHarness:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        email: str | None = None,
        password: str | None = None,
        database_url: str | None = None,
        out_dir: Path | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url or os.environ.get("EVAL_BASE_URL", DEFAULT_BASE_URL)
        self.email = email or os.environ.get("EVAL_EMAIL", "eval@local.dev")
        self.password = password or os.environ.get("EVAL_PASSWORD", "")
        if not self.password:
            raise RuntimeError("EVAL_PASSWORD 未设置（评测账号密码）。")
        dsn = database_url or os.environ.get("SERVICE_DATABASE_URL", "")
        if not dsn:
            raise RuntimeError("SERVICE_DATABASE_URL 未设置：行为断言需要 DB 特权读。")
        self._session_factory = make_session_factory(make_engine(dsn))
        self.out_dir = out_dir
        self.timeout_seconds = timeout_seconds
        self._token = ""

    async def close(self) -> None:
        engine = self._session_factory.kw["bind"]
        await engine.dispose()

    # ---- 用户旅程（API） ----

    async def _auth(self, client: httpx.AsyncClient) -> dict[str, str]:
        if not self._token:
            register = await client.post(
                f"{self.base_url}/api/register",
                json={"email": self.email, "password": self.password},
            )
            if register.status_code not in (201, 409):
                raise RuntimeError(f"注册异常: {register.status_code} {register.text[:200]}")
            login = await client.post(
                f"{self.base_url}/api/login",
                json={"email": self.email, "password": self.password},
            )
            login.raise_for_status()
            self._token = login.json()["token"]
        return {"authorization": f"Bearer {self._token}"}

    async def _await_state(
        self, client: httpx.AsyncClient, headers: dict[str, str], run_id: str
    ) -> dict[str, Any]:
        """轮询到终态或 awaiting_input（等待态交给 case 的澄清回答处理）。"""
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        while True:
            response = await client.get(f"{self.base_url}/api/runs/{run_id}", headers=headers)
            response.raise_for_status()
            detail = response.json()
            if detail.get("status") in TERMINAL or detail.get("status") == "awaiting_input":
                return detail
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(f"case 超时未收敛: {run_id} status={detail.get('status')}")
            await asyncio.sleep(POLL_SECONDS)

    async def _drive_once(self, case: EvalCase, client: httpx.AsyncClient) -> Artifact:
        headers = await self._auth(client)
        create = await client.post(
            f"{self.base_url}/api/runs", json={"query": case.prompt}, headers=headers
        )
        create.raise_for_status()
        run_id = create.json()["run_id"]
        control: dict[str, Any] = {}
        detail = await self._await_state(client, headers, run_id)

        while detail.get("status") == "awaiting_input":
            if not case.clarify_answer:
                raise RuntimeError(
                    f"{case.case_id} 进入 awaiting_input 但 case 没配澄清回答；先 cancel 清理。"
                )
            resume = await client.post(
                f"{self.base_url}/api/runs/{run_id}/resume",
                json={"answer": case.clarify_answer},
                headers=headers,
            )
            control["first_resume_status"] = resume.status_code
            if case.duplicate_answer:
                again = await client.post(
                    f"{self.base_url}/api/runs/{run_id}/resume",
                    json={"answer": case.clarify_answer},
                    headers=headers,
                )
                control["second_resume_status"] = again.status_code  # 期待 409
            detail = await self._await_state(client, headers, run_id)

        events, run_row = await self._read_db(run_id)
        return Artifact(
            case_id=case.case_id,
            attempt=1,
            run_id=run_id,
            detail=detail,
            events=events,
            run_row=run_row,
            control=control,
        )

    async def _read_db(self, run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
                )
            ).all()
            run = await session.get(Run, run_id)
            row = (
                {}
                if run is None
                else {
                    "attempt": run.attempt,
                    "status": run.status,
                    "lease_owner": run.lease_owner,
                    "terminal_reason": run.terminal_reason,
                }
            )
        return [r.record for r in rows], row

    # ---- case 编排 ----

    async def run_case(self, case: EvalCase, attempt: int) -> list[Artifact]:
        if case.fault in {"kill", "stop", "stop_during_awaiting"}:
            raise NotImplementedError(
                f"{case.case_id} 的故障注入（fault={case.fault}）需要 harness 掌控 worker "
                "进程，第二阶段按 scripts/verify_m8_processes.py 模式接线。"
            )
        async with httpx.AsyncClient(timeout=60.0) as client:
            artifacts = [await self._drive_once(case, client)]
            if case.fault == "rerun_cache":
                # 同题二跑：验证 L1/L2/L3 语义缓存兑现"重提交不重付费"。
                artifacts.append(await self._drive_once(case, client))
        for index, artifact in enumerate(artifacts, start=1):
            artifact.attempt = index + (attempt - 1) * (2 if case.fault == "rerun_cache" else 1)
        if self.out_dir is not None:
            self._persist(case, attempt, artifacts)
        return artifacts

    def _persist(self, case: EvalCase, attempt: int, artifacts: list[Artifact]) -> None:
        target = self.out_dir / case.case_id
        target.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "run_id": a.run_id,
                "attempt": a.attempt,
                "detail": a.detail,
                "run_row": a.run_row,
                "control": a.control,
                "events": a.events,
                "metrics": process_metrics(a),
            }
            for a in artifacts
        ]
        (target / f"round{attempt}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", "utf-8"
        )
