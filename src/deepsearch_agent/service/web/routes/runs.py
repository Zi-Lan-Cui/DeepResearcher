"""Run admission, query, cancellation, and clarification-resume routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from deepsearch_agent.service.persistence.models import Run, RunEvent, User
from deepsearch_agent.service.runs.service import QuotaExceededError
from deepsearch_agent.service.web.dependencies import app_state, current_user, owned_run
from deepsearch_agent.service.web.presenters import run_summary
from deepsearch_agent.service.web.schemas import CreateRunBody, ResumeRunBody

router = APIRouter(prefix="/api/runs")


@router.post("", status_code=202)
async def create_run(
    body: CreateRunBody,
    request: Request,
    user: User = Depends(current_user),
) -> dict[str, str]:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="研究问题不能为空。")
    try:
        run_id = await app_state(request).manager.start(user.id, query)
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from None
    return {"run_id": run_id}


@router.get("")
async def list_runs(
    request: Request,
    user: User = Depends(current_user),
) -> list[dict[str, Any]]:
    state = app_state(request)
    async with state.session_factory() as session:
        runs = (
            await session.scalars(
                select(Run)
                .where(Run.user_id == user.id)
                .order_by(Run.created_at.desc())
                .limit(50)
            )
        ).all()
    return [run_summary(run) for run in runs]


@router.get("/{run_id}")
async def get_run(
    request: Request,
    run: Run = Depends(owned_run),
) -> dict[str, Any]:
    clarification = None
    if run.status == "awaiting_input":
        state = app_state(request)
        async with state.session_factory() as session:
            row = await session.scalar(
                select(RunEvent)
                .where(
                    RunEvent.run_id == run.id,
                    RunEvent.event_type == "clarification_requested",
                )
                .order_by(RunEvent.seq.desc())
                .limit(1)
            )
        if row is not None:
            payload = row.record.get("payload", {})
            if isinstance(payload, dict):
                raw_options = payload.get("options")
                clarification = {
                    "question": str(payload.get("question") or "")[:500],
                    "options": (
                        [str(item)[:120] for item in raw_options[:3]]
                        if isinstance(raw_options, list)
                        else []
                    ),
                }
    return {
        **run_summary(run),
        "report_markdown": run.report_markdown,
        "citations": run.citations_json or [],
        "error_message": run.error_message,
        "clarification": clarification,
    }


@router.post("/{run_id}/cancel")
async def cancel_run(
    request: Request,
    run: Run = Depends(owned_run),
) -> dict[str, str]:
    try:
        cancelled = await app_state(request).manager.cancel(run.user_id, run.id)
    except LookupError:
        raise HTTPException(status_code=404, detail="运行不存在。") from None
    return {"id": cancelled.id, "status": cancelled.status}


@router.post("/{run_id}/resume")
async def resume_run(
    body: ResumeRunBody,
    request: Request,
    run: Run = Depends(owned_run),
) -> dict[str, str]:
    try:
        status = await app_state(request).manager.resume_with_input(
            run.user_id,
            run.id,
            body.answer,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="运行不存在。") from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        detail = (
            "恢复断点不存在，请重新发起。"
            if str(exc) == "checkpoint_missing"
            else "该运行当前不在等待输入。"
        )
        raise HTTPException(status_code=409, detail=detail) from None
    return {"id": run.id, "status": status}
