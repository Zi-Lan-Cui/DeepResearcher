"""评测 CLI：`python -m evals.cli <子命令>`。

子命令对应评测流水线各阶段：
  split      生成或重看 DRB 中文 dev/holdout 切分,固定种子
  cases      列出题集,校验并报告问题
  run        经 HTTP API 跑题,产物落 evals/results/
  score      确定性校验
  judge      RACE 对比 0-10 评分,每题每轮正序+换序各判一次
  report     记分表汇总,含 RACE 双序得分/确定性校验/pass^k
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from evals import aggregate, drb
from evals.deterministic import Artifact, score_artifact
from evals.judge import OpenAICompatJudge, judge_case_race
from evals.runner import RunHarness
from evals.schemas import (
    CriterionResult,
    EvalCase,
    RaceCriterionScore,
    load_race_results,
    load_results,
    merge_race_results,
    merge_results,
)

EVALS_DIR = Path(__file__).resolve().parent
# 每轮评测用 EVAL_RESULTS_DIR 指定独立目录,避免旧 round 产物混入新基线。
RESULTS_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", EVALS_DIR / "results"))
SPLIT_FILE = EVALS_DIR / "split_drb_zh.json"
PROFILES_DIR = EVALS_DIR / "profiles"


def collect_cases(args: argparse.Namespace) -> list[EvalCase]:
    profile = getattr(args, "profile", "")
    if profile:
        spec = json.loads((PROFILES_DIR / f"{profile}.json").read_text("utf-8"))
        external = {case.case_id: case for case in drb.build_cases(drb.drb_root(args.drb_root))}
        requested = spec["external_ids"]
        missing = [case_id for case_id in requested if case_id not in external]
        if missing:
            raise KeyError(f"评测 profile={profile} 缺题: {missing}")
        cases = [external[case_id] for case_id in requested]
    else:
        cases = drb.build_cases(
            drb.drb_root(args.drb_root),
            split_file=SPLIT_FILE,
            split_name=args.split,
        )
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c.case_id in wanted]
    return cases


def _artifacts_from_round(case_id: str, round_path: Path) -> list[Artifact]:
    rows = json.loads(round_path.read_text("utf-8"))
    return [
        Artifact(
            case_id=case_id,
            attempt=int(row["attempt"]),
            run_id=str(row["run_id"]),
            detail=row["detail"],
            events=row["events"],
        )
        for row in rows
    ]


def _iter_result_rounds(only_case: str | None):
    for case_dir in sorted(RESULTS_DIR.iterdir()):
        if not case_dir.is_dir() or (only_case and case_dir.name != only_case):
            continue
        for round_path in sorted(case_dir.glob("round*.json")):
            yield case_dir.name, round_path


async def cmd_run(args: argparse.Namespace) -> None:
    cases = collect_cases(args)
    if args.dry_run:
        for case in cases:
            print(f"{case.case_id}  repeat={case.repeat}")
        print(f"共 {len(cases)} 题")
        return
    harness = RunHarness(out_dir=RESULTS_DIR)
    failures: list[str] = []
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def _run_one(case: EvalCase, attempt: int, repeats: int) -> None:
        async with semaphore:
            print(f"running {case.case_id} attempt={attempt}/{repeats} ...", flush=True)
            try:
                await harness.run_case(case, attempt, resume=args.resume)
            except Exception as exc:  # noqa: BLE001 - 单题因超时、澄清环、网关抖动等失败时不拖垮整批
                failures.append(f"{case.case_id}#{attempt}: {type(exc).__name__}: {exc}")
                print(f"  FAILED {case.case_id} attempt={attempt}: {exc}", flush=True)

    try:
        if args.concurrency > 1:
            await harness.warmup_auth()
        await asyncio.gather(
            *[
                _run_one(case, attempt, repeats)
                for case in cases
                for repeats in [args.attempts or case.repeat]
                for attempt in range(max(1, args.from_attempt), repeats + 1)
            ]
        )
    finally:
        await harness.close()
    if failures:
        print(f"\n{len(failures)} 个 run 失败（其余照常评分）：")
        for line in failures:
            print("  " + line)


async def cmd_capture(args: argparse.Namespace) -> None:
    """把一次已存在或恢复中的 run 收成为评测产物,不重跑、不产生 LLM 费用。"""
    all_cases = {c.case_id: c for c in drb.build_cases(drb.drb_root(None))}
    case = all_cases.get(args.case_id)
    if case is None:
        sys.exit(f"未知 case_id: {args.case_id}")
    harness = RunHarness(out_dir=RESULTS_DIR)
    try:
        artifact = await harness.capture(case, args.run_id, args.attempt)
        print(
            f"captured {args.run_id} → {case.case_id} attempt={args.attempt} "
            f"status={artifact.detail.get('status')} events={len(artifact.events)}"
        )
    finally:
        await harness.close()


def cmd_score(args: argparse.Namespace) -> None:
    results: list[CriterionResult] = []
    evaluated: set[tuple[str, int]] = set()
    for case_id, round_path in _iter_result_rounds(args.case):
        for artifact in _artifacts_from_round(case_id, round_path):
            evaluated.add((artifact.case_id, artifact.attempt))
            results.extend(score_artifact(artifact))
    out = RESULTS_DIR / "deterministic.jsonl"
    merged = merge_results(out, results, evaluated)
    print(f"本次门禁判定 {len(results)} 条，累计 {len(merged)} 条 → {out}")


async def cmd_judge(args: argparse.Namespace) -> None:
    root = drb.drb_root(args.drb_root)
    cases = {c.case_id: c for c in drb.build_cases(root)}
    references = {int(c.case_id.rsplit("-", 1)[1]): article for c, article in _pairs(cases, root)}
    invoker = OpenAICompatJudge()
    forward_rows: list[RaceCriterionScore] = []
    swapped_rows: list[RaceCriterionScore] = []
    evaluated: set[tuple[str, int]] = set()
    for case_id, round_path in _iter_result_rounds(args.case):
        case = cases.get(case_id)
        if case is None or not case.criteria:
            continue
        for artifact in _artifacts_from_round(case_id, round_path):
            if args.attempt is not None and artifact.attempt != args.attempt:
                continue
            report = artifact.report
            if not report:
                continue  # 无报告可判,保留该轮既有判定
            reference = references.get(_drb_numeric(case_id))
            if not reference:
                sys.exit(f"RACE 口径需要 DRB 专家参考文章，缺失：{case_id}")
            evaluated.add((artifact.case_id, artifact.attempt))
            for swap_positions, rows in ((False, forward_rows), (True, swapped_rows)):
                rows.extend(
                    await judge_case_race(
                        case,
                        report=report,
                        attempt=artifact.attempt,
                        invoker=invoker,
                        reference=reference,
                        swap_positions=swap_positions,
                    )
                )
            print(f"judged[双序] {case_id} attempt={artifact.attempt}")
    forward_out = RESULTS_DIR / "judge_race.jsonl"
    swapped_out = RESULTS_DIR / "judge_race_swapped.jsonl"
    merged_forward = merge_race_results(forward_out, forward_rows, evaluated)
    merged_swapped = merge_race_results(swapped_out, swapped_rows, evaluated)
    (RESULTS_DIR / "judge_meta.json").write_text(
        json.dumps(invoker.identity, ensure_ascii=False) + "\n", "utf-8"
    )
    if invoker.self_judging:
        print("⚠ 本轮为自评（judge=被测同模型），分数仅供调试，勿对外引用")
    print(
        f"本次 RACE 判定 正序 {len(forward_rows)} 条 + 换序 {len(swapped_rows)} 条，"
        f"累计 正序 {len(merged_forward)} / 换序 {len(merged_swapped)} → {forward_out}、{swapped_out}"
    )


def _pairs(external: dict[str, EvalCase], root: Path):
    refs = drb.load_references(root)
    for case_id, case in external.items():
        yield case, refs.get(_drb_numeric(case_id), "")


def _drb_numeric(case_id: str) -> int:
    try:
        return int(case_id.rsplit("-", 1)[1])
    except ValueError:
        return -1


def cmd_report(args: argparse.Namespace) -> None:
    det_file = RESULTS_DIR / "deterministic.jsonl"
    forward_file = RESULTS_DIR / "judge_race.jsonl"
    swapped_file = RESULTS_DIR / "judge_race_swapped.jsonl"
    det_rows: list[CriterionResult] = load_results(det_file) if det_file.is_file() else []
    forward_rows = load_race_results(forward_file) if forward_file.is_file() else []
    swapped_rows = load_race_results(swapped_file) if swapped_file.is_file() else []
    if not det_rows and not forward_rows and not swapped_rows:
        sys.exit("results/ 下无任何判定")
    pairs = {(r.case_id, r.attempt) for r in det_rows}
    pairs |= {(r.case_id, r.attempt) for r in forward_rows}
    pairs |= {(r.case_id, r.attempt) for r in swapped_rows}
    try:
        cases = {c.case_id: c for c in drb.build_cases(drb.drb_root(None))}
    except FileNotFoundError:
        cases = {}
    scores = []
    for case_id, attempt in sorted(pairs):
        # DRB 不在场时只出门计数,质量分算不出就不合成。
        case = cases.get(case_id) or EvalCase(case_id=case_id, prompt="", criteria=())
        scores.append(aggregate.score_case(case, attempt, det_rows, forward_rows, swapped_rows))

    metrics = aggregate.k_metrics(scores)
    dual_means = [s.race_mean for s in scores if s.race_mean is not None]
    forward_ratios: list[float] = []
    swapped_ratios: list[float] = []
    missing_count = 0
    dimension_values: dict[str, list[float]] = {}
    for score in scores:
        for race, ratio_sink in (
            (score.race_forward, forward_ratios),
            (score.race_swapped, swapped_ratios),
        ):
            if race is None:
                continue
            missing_count += race.missing_count
            if race.overall is not None:
                ratio_sink.append(race.overall)
        if score.race_forward is not None:
            for dimension, ratio in score.race_forward.dimension_ratios.items():
                dimension_values.setdefault(dimension, []).append(ratio)
    summary = {
        "cases": len(metrics),
        "rounds": len(scores),
        "race": {
            "meaning": "正序与换序 target/(target+reference) 的均值,0.5=与专家参考打平",
            "rounds_scored": len(dual_means),
            "mean_overall": round(sum(dual_means) / len(dual_means), 4) if dual_means else None,
            "mean_forward": round(sum(forward_ratios) / len(forward_ratios), 4)
            if forward_ratios
            else None,
            "mean_swapped": round(sum(swapped_ratios) / len(swapped_ratios), 4)
            if swapped_ratios
            else None,
            "attempt1_overall": [
                s.race_mean for s in scores if s.attempt == 1 and s.race_mean is not None
            ],
            "dimension_mean": {
                dimension: round(sum(values) / len(values), 4)
                for dimension, values in sorted(dimension_values.items())
            },
            "missing_criterion_count": missing_count,
        },
        "pass": {
            "criteria": f"三门均通过 ∧ 双序均值 > {aggregate.RACE_PASS_RATIO}",
            "pass_at_k": sum(1 for m in metrics.values() if m["pass_any"]),
            "pass_pow_k": sum(1 for m in metrics.values() if m["pass_all"]),
        },
        "gate_failures": sorted({f for s in scores for f in s.gate_failures}),
        "efficiency": aggregate.summarize_metrics(_all_round_metrics()),
    }
    judge_meta_file = RESULTS_DIR / "judge_meta.json"
    if forward_file.is_file() or swapped_file.is_file():
        # 无 judge_meta 的旧轮次如实标 unknown,不假设与被测同模型。
        summary["judge"] = (
            json.loads(judge_meta_file.read_text("utf-8"))
            if judge_meta_file.is_file()
            else {"model": "unknown", "self_judging": None}
        )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps({"summary": summary, "per_case": metrics}, ensure_ascii=False, indent=1) + "\n",
        "utf-8",
    )


def _all_round_metrics() -> list[dict]:
    """从所有 round 产物里抽 process_metrics,供 summarize_metrics 汇总效率块。"""
    rows: list[dict] = []
    for _case_id, round_path in _iter_result_rounds(None):
        for row in json.loads(round_path.read_text("utf-8")):
            metrics = row.get("metrics")
            if metrics:
                rows.append(metrics)
    return rows


def cmd_cases(args: argparse.Namespace) -> None:
    cases = collect_cases(args)
    bad = 0
    for case in cases:
        problems = _validate(case)
        if problems:
            bad += 1
            print(f"[INVALID] {case.case_id}: {'; '.join(problems)}")
        print(f"{case.case_id}  criteria={len(case.criteria)}  repeat={case.repeat}")
    print(f"共 {len(cases)} 题，无效 {bad}")


def _validate(case: EvalCase) -> list[str]:
    from evals.schemas import validate_case

    return validate_case(case)


def cmd_split(args: argparse.Namespace) -> None:
    root = drb.drb_root(args.drb_root)
    split = drb.make_split(root, dev_size=args.dev, out_path=SPLIT_FILE)
    print(f"dev={len(split['dev'])} holdout={len(split['holdout'])} → {SPLIT_FILE}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evals")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", default="", choices=["", "dev", "test30"])
        p.add_argument("--ids", default="")
        p.add_argument("--split", default=None, choices=[None, "dev", "holdout"])
        p.add_argument("--drb-root", default=None)

    sp = sub.add_parser("split")
    sp.add_argument("--drb-root", default=None)
    sp.add_argument("--dev", type=int, default=20)
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser("cases")
    add_filters(sp)
    sp.set_defaults(func=cmd_cases)

    sp = sub.add_parser("run")
    add_filters(sp)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--attempts", type=int, default=0, help="覆盖每题次数（pass^k 用）")
    sp.add_argument("--from-attempt", type=int, default=1, help="从第几个 attempt 起跑（续跑用）")
    sp.add_argument(
        "--resume",
        action="store_true",
        help="round 产物已存在且可解析的直接复用,不重跑",
    )
    sp.add_argument("--concurrency", type=int, default=1, help="批内并发 (题,attempt) 任务数")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("capture")
    sp.add_argument("--case-id", required=True)
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--attempt", type=int, default=1)
    sp.set_defaults(func=cmd_capture)

    sp = sub.add_parser("score")
    sp.add_argument("--case", default=None)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("judge")
    sp.add_argument("--case", default=None)
    sp.add_argument("--drb-root", default=None)
    sp.add_argument("--attempt", type=int, default=None, help="只判指定 attempt(如 1 即 pass@1)")
    sp.set_defaults(func=cmd_judge)

    sp = sub.add_parser("report")
    sp.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.command in {"run", "judge", "capture"}:
        asyncio.run(args.func(args))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
