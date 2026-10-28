"""Runner：用户旅程走 HTTP API，门禁判定所需的事件原始记录走数据库直读。

分工刻意为之：
- 提交、澄清代答、取消、详情全部经 API:评测必须跑在与真实用户相同的路径上,
  绕过 API 就测不到用户可见行为;
- run_events 走 DB 直读:SSE 只有投影帧,门禁(seq 连续、run_done 帧数)需要
  原始事件序列;harness 用操作者视图,产品 API 不为评测需求扩展。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from deepresearcher.service.storage.engine import make_engine, make_session_factory
from deepresearcher.service.storage.tables import RunEvent
from evals.deterministic import Artifact, process_metrics
from evals.schemas import EvalCase

TERMINAL = ("completed", "failed", "cancelled")
# DRB 题属直接作答式,被 Clarifier 偶发追问时的默认代答:让系统自行界定范围、直接研究。
DEFAULT_CLARIFY_ANSWER = "请以我的原问题为准，按你认为最合理的范围直接开展研究，不必再追问。"

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
POLL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 4200.0
DEFAULT_QUEUE_WAIT_SECONDS = 900.0


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
        queue_wait_seconds: float = DEFAULT_QUEUE_WAIT_SECONDS,
    ) -> None:
        self.base_url = base_url or os.environ.get("EVAL_BASE_URL", DEFAULT_BASE_URL)
        self.email = email or os.environ.get("EVAL_EMAIL", "eval@local.dev")
        self.password = password or os.environ.get("EVAL_PASSWORD", "")
        if not self.password:
            raise RuntimeError("EVAL_PASSWORD 未设置（评测账号密码）。")
        dsn = database_url or os.environ.get("SERVICE_DATABASE_URL", "")
        if not dsn:
            raise RuntimeError("SERVICE_DATABASE_URL 未设置：门禁判定需要 DB 特权读。")
        self._session_factory = make_session_factory(make_engine(dsn))
        self.out_dir = out_dir
        self.timeout_seconds = timeout_seconds
        self.queue_wait_seconds = queue_wait_seconds
        self._token = ""

    async def close(self) -> None:
        engine = self._session_factory.kw["bind"]
        await engine.dispose()

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
        """轮询到终态或 awaiting_input 返回;等待态交由调用方做澄清回答。

        执行预算 timeout_seconds 从 run 离开 queued 起算;并发批次里排队属于
        容量而非(case,attempt)失败,排队另有 queue_wait_seconds 上限兜底。
        """
        now = asyncio.get_running_loop().time
        queue_deadline = now() + self.queue_wait_seconds
        execution_deadline: float | None = None
        while True:
            response = await client.get(f"{self.base_url}/api/runs/{run_id}", headers=headers)
            response.raise_for_status()
            detail = response.json()
            status = detail.get("status")
            if status in TERMINAL or status == "awaiting_input":
                return detail
            current = now()
            if status == "queued":
                if current > queue_deadline:
                    raise TimeoutError(f"case 排队超时未获领取: {run_id}")
            else:
                if execution_deadline is None:
                    execution_deadline = current + self.timeout_seconds
                if current > execution_deadline:
                    raise TimeoutError(f"case 执行超时未收敛: {run_id} status={status}")
            await asyncio.sleep(POLL_SECONDS)

    async def _drive_once(
        self, case: EvalCase, attempt: int, client: httpx.AsyncClient
    ) -> Artifact:
        headers = await self._auth(client)
        create = await client.post(
            f"{self.base_url}/api/runs", json={"query": case.prompt}, headers=headers
        )
        create.raise_for_status()
        run_id = create.json()["run_id"]
        detail = await self._await_state(client, headers, run_id)

        # Clarifier 偶发追问时代答一次继续研究;再次追问说明无法收敛,取消本 run
        # 并抛出,由 cmd_run 记为该题失败。
        auto_resumed = False
        while detail.get("status") == "awaiting_input":
            if auto_resumed:
                await client.post(f"{self.base_url}/api/runs/{run_id}/cancel", headers=headers)
                raise RuntimeError(f"{case.case_id} 连续多轮澄清，评测放弃该题")
            auto_resumed = True
            resume = await client.post(
                f"{self.base_url}/api/runs/{run_id}/resume",
                json={"answer": DEFAULT_CLARIFY_ANSWER},
                headers=headers,
            )
            resume.raise_for_status()
            detail = await self._await_state(client, headers, run_id)

        events = await self._read_db(run_id)
        return Artifact(
            case_id=case.case_id,
            attempt=attempt,
            run_id=run_id,
            detail=detail,
            events=events,
        )

    async def _read_db(self, run_id: str) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
                )
            ).all()
        return [r.record for r in rows]

    async def warmup_auth(self) -> None:
        """并发执行前预热评测账号 token,避免 gather 下重复注册登录。"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            await self._auth(client)

    def _load_round(self, case: EvalCase, attempt: int) -> list[Artifact] | None:
        """读取已落盘的 round 产物;不存在或解析失败返回 None 按缺跑处理。"""
        if self.out_dir is None:
            return None
        path = self.out_dir / case.case_id / f"round{attempt}.json"
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        artifacts: list[Artifact] = []
        for row in payload:
            # 只取白名单键:metrics 由 process_metrics 现算,历史产物中的多余键忽略;
            # case_id 按目录名归位。
            fields = {
                key: row[key] for key in ("attempt", "run_id", "detail", "events") if key in row
            }
            artifacts.append(Artifact(case_id=case.case_id, **fields))
        return artifacts

    async def run_case(
        self, case: EvalCase, attempt: int, *, resume: bool = False
    ) -> list[Artifact]:
        if resume:
            existing = self._load_round(case, attempt)
            if existing is not None:
                return existing
        async with httpx.AsyncClient(timeout=60.0) as client:
            artifacts = [await self._drive_once(case, attempt, client)]
        self._persist(case, attempt, artifacts)
        return artifacts

    async def capture(self, case: EvalCase, run_id: str, attempt: int) -> Artifact:
        """把一次已存在或恢复中的 durable run 收成产物,不重跑。

        clarify 这类跨进程流程常需人工或外部推动,评测可以附着到正在运行的
        run:capture 用 API 与 DB 现状补齐 artifact,免去为收集结果重跑一遍的
        费用。
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = await self._auth(client)
            detail = (
                await client.get(f"{self.base_url}/api/runs/{run_id}", headers=headers)
            ).json()
        events = await self._read_db(run_id)
        artifact = Artifact(
            case_id=case.case_id,
            attempt=attempt,
            run_id=run_id,
            detail=detail,
            events=events,
        )
        self._persist(case, attempt, [artifact])
        return artifact

    def _persist(self, case: EvalCase, attempt: int, artifacts: list[Artifact]) -> None:
        """out_dir 未配置时直接返回;判空集中在这里,调用点不再各自检查。"""
        if self.out_dir is None:
            return
        target = self.out_dir / case.case_id
        target.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "run_id": a.run_id,
                "attempt": a.attempt,
                "detail": a.detail,
                "events": a.events,
                "metrics": process_metrics(a),
            }
            for a in artifacts
        ]
        path = target / f"round{attempt}.json"
        tmp_path = path.with_name(path.name + ".tmp")
        # 原子替换:并发或中断都不留半截 JSON,--resume 的判定才可信赖。
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", "utf-8")
        tmp_path.replace(path)
