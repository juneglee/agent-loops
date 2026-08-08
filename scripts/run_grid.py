from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loops.bench.bfcl.adapter import dataset_revision, load_cases
from agent_loops.bench.bfcl.runner import run_case
from agent_loops.bench.core.config import full_config
from agent_loops.bench.core.local_llm import LocalLLM
from agent_loops.bench.core.runner import pass_at_k, summarize
from agent_loops.bench.prompts import PROMPT_VERSION
from agent_loops.loops import (
    adapt,
    codeact,
    fixed_pipeline,
    plan_and_act,
    plan_and_execute,
    plan_and_solve,
    react,
    rewoo,
    single_call,
)

LOOPS = {
    m.NAME: m
    for m in (
        single_call,
        react,
        rewoo,
        plan_and_solve,
        plan_and_execute,
        plan_and_act,
        adapt,
        codeact,
        fixed_pipeline,
    )
}

LOOP_KWARGS = {
    "react": {"max_steps": 12},
    "plan_and_act": {"max_steps": 12},
    "adapt": {"max_depth": 3},
    "codeact": {"max_steps": 8},
}


def _stratified(cases: list[dict], k: int, seed: int) -> list[dict]:
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    groups: dict[tuple, list] = defaultdict(list)
    for c in cases:
        groups[tuple(sorted(c["involved_classes"]))].append(c)

    picked: list[dict] = []
    for key in sorted(groups):
        pool = sorted(groups[key], key=lambda c: c["id"])
        rng.shuffle(pool)
        share = max(1, round(k * len(groups[key]) / len(cases)))
        picked.extend(pool[:share])
    picked.sort(key=lambda c: c["id"])
    return picked[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="multi_turn_base")
    ap.add_argument("--loops", nargs="*", default=list(LOOPS))
    ap.add_argument("--limit", type=int, default=0, help="0 means all cases")
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="stratified sample preserving backend-combination ratios; 0 disables",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gemma-4-E4B-it-qat-q4_0")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--fs-only", action="store_true", default=True)
    ap.add_argument("--all-classes", dest="fs_only", action="store_false")
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeats per case k (pass^k)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature; pass^k is meaningful only with temp>0",
    )
    ap.add_argument(
        "--decode-seed",
        type=int,
        default=0,
        help="decoding seed (--seed is for stratified sampling); per-run seed = decode_seed + run_index",
    )
    ap.add_argument(
        "--code-timeout",
        type=float,
        default=5.0,
        help="codeact code execution timeout in seconds",
    )
    a = ap.parse_args()
    cases = load_cases(
        a.category, only_classes={"GorillaFileSystem"} if a.fs_only else None
    )
    if a.sample and a.sample < len(cases):
        cases = _stratified(cases, a.sample, a.seed)
    if a.limit:
        cases = cases[: a.limit]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = a.tag or f"{a.model}_{a.category}{'_fs' if a.fs_only else ''}"
    out_path = Path("results") / f"grid_{tag}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"model {a.model} | {len(cases)} cases | {len(a.loops)} loops "
        f",  prompt {PROMPT_VERSION}\n"
    )

    all_results: dict[str, list] = {}
    for loop_name in a.loops:
        module = LOOPS[loop_name]
        rows = []
        for i, case in enumerate(cases, 1):
            for rep in range(a.repeats):

                def factory(tools, _rep=rep):
                    return LocalLLM(
                        tools,
                        base_url=a.base_url,
                        model=a.model,
                        temperature=a.temperature,
                        seed=a.decode_seed + _rep,
                    )

                r = run_case(
                    case,
                    module,
                    factory,
                    a.category,
                    loop_kwargs=LOOP_KWARGS.get(loop_name, {}),
                    code_timeout=a.code_timeout,
                )
                r.run_index = rep
                rows.append(r)
                print(
                    f"  {loop_name:16s} [{i}/{len(cases)}]"
                    f"{f' r{rep + 1}' if a.repeats > 1 else ''} {r.case_id:22s} "
                    f"{'O' if r.valid else 'X'} LLM={r.n_llm_calls:2d} "
                    f"tools={r.n_tool_calls:2d} {r.seconds:.0f}s",
                    flush=True,
                )
        all_results[loop_name] = rows
        s = summarize(rows)
        print(
            f"  → {loop_name}: accuracy {s['accuracy']:.1%} | "
            f"parse {s['parse_rate']:.1%} | LLM {s['mean_llm_calls']:.1f} calls\n",
            flush=True,
        )

        out_path.write_text(
            json.dumps(
                {
                    "when": stamp,
                    "model": a.model,
                    "category": a.category,
                    "fs_only": a.fs_only,
                    "prompt_version": PROMPT_VERSION,
                    "dataset": dataset_revision(),
                    "n_cases": len(cases),
                    "repeats": a.repeats,
                    "config": full_config(
                        model=a.model,
                        base_url=a.base_url,
                        started_at=stamp,
                        temperature=a.temperature,
                        seed=a.decode_seed,
                    ),
                    "summaries": {
                        k: {**summarize(v), **pass_at_k(v)}
                        for k, v in all_results.items()
                    },
                    "cases": {k: [vars(r) for r in v] for k, v in all_results.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("=" * 78)
    hdr = f"{'loop':<17}{'acc':>8}{'parse':>8}{'quiet':>8}{'LLM':>8}{'tools':>7}{'sec':>7}"
    print(hdr)
    print("-" * 78)
    for name, rows in all_results.items():
        s = summarize(rows)
        print(
            f"{name:<17}{s['accuracy']:>7.1%}{s['parse_rate']:>8.1%}"
            f"{s['quiet_failure_rate']:>8.1%}{s['mean_llm_calls']:>8.1f}"
            f"{s['mean_tool_calls']:>7.1f}{s['mean_seconds']:>7.0f}"
        )
    print("=" * 78)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
