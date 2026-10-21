"""评测 CLI：`python -m evals.cli <子命令>`。

子命令对应评测流水线各阶段：
  split      生成或重看 DRB 中文 dev/holdout 切分,固定种子
  cases      列出两轨题集,校验并报告问题
  run        经 HTTP API 跑题,产物落 evals/results/
  score      确定性门 + 行为断言,零 LLM 费用
  judge      二元 rubric LLM judge,external 轨带专家参考对照
  annotate   导出人工标注 CSV,重新导入并算人机一致率
  report     记分表汇总,含 Quality/Grounding/Behavior/pass^k/unknown
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
from evals.judge import OpenAICompatJudge, judge_case, judge_case_race
from evals.runner import RunHarness
from evals.schemas import (
    CriterionResult,
    EvalCase,
    RaceCriterionScore,
    load_cases,
    load_results,
    load_race_results,
    merge_results,
    merge_race_results,
)

EVALS_DIR = Path(__file__).resolve().parent
# 每轮评测用 EVAL_RESULTS_DIR 指定独立目录,避免旧 round 产物混入新基线。
RESULTS_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", EVALS_DIR / "results"))
BEHAVIOR_FILE = EVALS_DIR / "cases" / "behavior.jsonl"
SPLIT_FILE = EVALS_DIR / "split_drb_zh.json"
PROFILES_DIR = EVALS_DIR / "profiles"


def collect_cases(args: argparse.Namespace) -> list[EvalCase]:
    profile = getattr(args, "profile", "")
    if profile:
        spec = json.loads((PROFILES_DIR / f"{profile}.json").read_text("utf-8"))
        behavior = {case.case_id: case for case in load_cases(BEHAVIOR_FILE)}
        external = {case.case_id: case for case in drb.build_cases(drb.drb_root(args.drb_root))}
        requested = [*spec["external_ids"], *spec["behavior_ids"]]
        available = {**external, **behavior}
        missing = [case_id for case_id in requested if case_id not in available]
        if missing:
            raise KeyError(f"评测 profile={profile} 缺题: {missing}")
        cases = [available[case_id] for case_id in requested]
    else:
        cases = []
        want_behavior = args.track in {"all", "behavior"}
        want_external = args.track in {"all", "external"}
        if want_behavior:
            cases.extend(load_cases(BEHAVIOR_FILE))
        if want_external:
            cases.extend(
                drb.build_cases(
                    drb.drb_root(args.drb_root),
                    split_file=SPLIT_FILE,
                    split_name=args.split,
                )
            )
    if args.group:
        cases = [c for c in cases if c.group == args.group]
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c.case_id in wanted]
    if not args.with_faults:
        skipped = [c.case_id for c in cases if c.phase > 1]
        if skipped:
            print(f"[skip] phase2 故障注入题（待 worker 进程编排接线）: {skipped}", file=sys.stderr)
        cases = [c for c in cases if c.phase == 1]
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
            run_row=row.get("run_row") or {},
            control=row.get("control") or {},
            evidences=row.get("evidences") or [],
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
            print(f"{case.case_id}  track={case.track} repeat={case.repeat} fault={case.fault}")
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
    all_cases = {c.case_id: c for c in load_cases(BEHAVIOR_FILE)}
    try:
        for c in drb.build_cases(drb.drb_root(None)):
            all_cases.setdefault(c.case_id, c)
    except FileNotFoundError:
        pass  # external 轨 case 需要 DRB 在场,capture behavior 题时缺场无妨
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
    behavior = {c.case_id: c for c in load_cases(BEHAVIOR_FILE)}
    for case_id, round_path in _iter_result_rounds(args.case):
        artifacts = _artifacts_from_round(case_id, round_path)
        case = behavior.get(case_id)
        for artifact in artifacts:
            evaluated.add((artifact.case_id, artifact.attempt))
            if case is None:
                case = EvalCase(case_id=case_id, track="external", prompt="", criteria=())
            results.extend(score_artifact(artifact, case))
    out = RESULTS_DIR / "deterministic.jsonl"
    merged = merge_results(out, results, evaluated)
    print(f"本次确定性判定 {len(results)} 条，累计 {len(merged)} 条 → {out}")


async def cmd_judge(args: argparse.Namespace) -> None:
    root = drb.drb_root(args.drb_root)
    external = {c.case_id: c for c in drb.build_cases(root)}
    behavior = {c.case_id: c for c in load_cases(BEHAVIOR_FILE)}
    references = {
        int(c.case_id.rsplit("-", 1)[1]): article for c, article in _pairs(external, root)
    }
    invoker = OpenAICompatJudge()
    results: list[CriterionResult] = []
    race_rows: list[RaceCriterionScore] = []
    evaluated: set[tuple[str, int]] = set()
    race_mode = args.mode == "race"
    for case_id, round_path in _iter_result_rounds(args.case):
        case = external.get(case_id) or behavior.get(case_id)
        if case is None or not case.judge_criteria():
            continue
        if race_mode and case.track != "external":
            continue  # RACE 口径只适用于带官方 rubric 与参考文章的 external 轨
        for artifact in _artifacts_from_round(case_id, round_path):
            if args.attempt is not None and artifact.attempt != args.attempt:
                continue
            report = artifact.report
            if not report:
                if not race_mode:
                    evaluated.add((artifact.case_id, artifact.attempt))
                continue
            evaluated.add((artifact.case_id, artifact.attempt))
            reference = references.get(_drb_numeric(case_id))
            if race_mode:
                if not reference:
                    sys.exit(f"race 口径需要 DRB 专家参考文章，缺失：{case_id}")
                race_rows.extend(
                    await judge_case_race(
                        case,
                        report=report,
                        attempt=artifact.attempt,
                        invoker=invoker,
                        reference=reference,
                        swap_positions=args.swap,
                    )
                )
            else:
                results.extend(
                    await judge_case(
                        case,
                        report=report,
                        attempt=artifact.attempt,
                        invoker=invoker,
                        reference=reference,
                    )
                )
            print(f"judged[{args.mode}] {case_id} attempt={artifact.attempt}")
    if race_mode:
        out = RESULTS_DIR / ("judge_race_swapped.jsonl" if args.swap else "judge_race.jsonl")
        merged_rows: list[RaceCriterionScore] | list[CriterionResult] = merge_race_results(
            out, race_rows, evaluated
        )
    else:
        out = RESULTS_DIR / "judge.jsonl"
        merged_rows = merge_results(out, results, evaluated)
    (RESULTS_DIR / "judge_meta.json").write_text(
        json.dumps(invoker.identity, ensure_ascii=False) + "\n", "utf-8"
    )
    if invoker.self_judging:
        print("⚠ 本轮为自评（judge=被测同模型），分数仅供调试，勿对外引用")
    judged_count = len(race_rows) if race_mode else len(results)
    print(f"本次 judge 判定 {judged_count} 条，累计 {len(merged_rows)} 条 → {out}")


def _pairs(external: dict[str, EvalCase], root: Path):
    refs = drb.load_references(root)
    for case_id, case in external.items():
        yield case, refs.get(_drb_numeric(case_id), "")


def _drb_numeric(case_id: str) -> int:
    try:
        return int(case_id.rsplit("-", 1)[1])
    except ValueError:
        return -1


def cmd_annotate(args: argparse.Namespace) -> None:
    judge_file = RESULTS_DIR / "judge.jsonl"
    if args.import_csv:
        rows = aggregate.import_annotation_csv(Path(args.import_csv).read_text("utf-8"))
        import csv as _csv

        out = Path(args.import_csv).with_suffix(".graded.csv")
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(handle, fieldnames=aggregate.ANNOTATION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"graded → {out}")
        print(json.dumps(aggregate.agreement_report(rows), ensure_ascii=False, indent=1))
        return
    if not judge_file.is_file():
        sys.exit("先跑 judge 生成结果")
    behavior = {c.case_id: c for c in load_cases(BEHAVIOR_FILE)}
    external: dict[str, EvalCase] = {}
    try:
        external = {c.case_id: c for c in drb.build_cases(drb.drb_root(None))}
    except FileNotFoundError:
        pass
    results = load_results(judge_file)
    csv_text = aggregate.export_annotation_csv(results, {**external, **behavior})
    out = RESULTS_DIR / "annotate.csv"
    out.write_text(csv_text, "utf-8")
    print(f"标注表 → {out}（填 human_verdict 列后 --import-csv 回导）")


def cmd_report(args: argparse.Namespace) -> None:
    judge_file = RESULTS_DIR / "judge.jsonl"
    det_file = RESULTS_DIR / "deterministic.jsonl"
    results: list[CriterionResult] = []
    if det_file.is_file():
        results.extend(load_results(det_file))
    if judge_file.is_file():
        results.extend(load_results(judge_file))
    if not results:
        sys.exit("results/ 下无任何判定")
    pairs = {(r.case_id, r.attempt) for r in results}
    behavior = {c.case_id: c for c in load_cases(BEHAVIOR_FILE)}
    try:
        external = {c.case_id: c for c in drb.build_cases(drb.drb_root(None))}
    except FileNotFoundError:
        external = {}
    cases = {**external, **behavior}
    scores = []
    for cid, attempt in sorted(pairs):
        case = cases.get(cid)
        if case is None:  # external 轨题需 DRB 在场才有判分定义,缺场时只出门计数不合成质量分
            case = EvalCase(case_id=cid, track="external", prompt="", criteria=())
        scores.append(aggregate.score_case(case, attempt, results))
    metrics = aggregate.k_metrics(scores)
    tracks = aggregate.track_summary(scores, metrics)
    summary = {
        "cases": len(metrics),
        "rounds": len(scores),
        **tracks,
        "unknown_rate": round(
            sum(s.unknown_count for s in scores) / max(1, sum(s.judge_count for s in scores)), 4
        ),
        "gate_failures": sorted({f for s in scores for f in s.gate_failures}),
        "efficiency": aggregate.summarize_metrics(_all_round_metrics()),
    }
    judge_meta_file = RESULTS_DIR / "judge_meta.json"
    if judge_file.is_file():
        # 无 judge_meta 的旧轮次如实标 unknown,不假设与被测同模型。
        summary["judge"] = (
            json.loads(judge_meta_file.read_text("utf-8"))
            if judge_meta_file.is_file()
            else {"model": "unknown", "self_judging": None}
        )
    race_file = RESULTS_DIR / "judge_race.jsonl"
    if race_file.is_file():
        race_rows = load_race_results(race_file)
        race_scores = [
            aggregate.score_case_race(external[cid], attempt, race_rows)
            for cid, attempt in sorted({(r.case_id, r.attempt) for r in race_rows})
            if cid in external
        ]
        valid_overall = [item.overall for item in race_scores if item.overall is not None]
        dimension_values: dict[str, list[float]] = {}
        for item in race_scores:
            for dimension, ratio in item.dimension_ratios.items():
                dimension_values.setdefault(dimension, []).append(ratio)
        summary["race_quality"] = {
            "meaning": "target/(target+reference) 归一,0.5=与专家参考打平",
            "rounds": len(race_scores),
            "mean_overall": round(sum(valid_overall) / len(valid_overall), 4)
            if valid_overall
            else None,
            "attempt1_overall": [
                round(item.overall, 4)
                for item in race_scores
                if item.attempt == 1 and item.overall is not None
            ],
            "dimension_mean": {
                dimension: round(sum(values) / len(values), 4)
                for dimension, values in sorted(dimension_values.items())
            },
            "missing_criterion_count": sum(item.missing_count for item in race_scores),
        }
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
        print(
            f"{case.case_id}  [{case.track}/{case.group or '-'}]  criteria={len(case.criteria)}  repeat={case.repeat}"
        )
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
        p.add_argument("--track", default="all", choices=["all", "external", "behavior"])
        p.add_argument("--group", default="")
        p.add_argument("--ids", default="")
        p.add_argument("--split", default=None, choices=[None, "dev", "holdout"])
        p.add_argument("--drb-root", default=None)
        p.add_argument("--with-faults", action="store_true")

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

    for name, func in (("score", cmd_score), ("annotate", cmd_annotate)):
        sp = sub.add_parser(name)
        sp.add_argument("--case", default=None)
        if name == "annotate":
            sp.add_argument("--import-csv", default="")
        sp.set_defaults(func=func)

    sp = sub.add_parser("judge")
    sp.add_argument("--case", default=None)
    sp.add_argument("--drb-root", default=None)
    sp.add_argument(
        "--mode",
        choices=("binary", "race"),
        default="binary",
        help="binary=既有 yes/no 曲线;race=官方 0-10 双文对比口径,写 judge_race.jsonl",
    )
    sp.add_argument("--attempt", type=int, default=None, help="只判指定 attempt(如 1 即 pass@1)")
    sp.add_argument(
        "--swap",
        action="store_true",
        help="race 口径下把参考文章放文章一(位置偏置对照),写 judge_race_swapped.jsonl",
    )
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
