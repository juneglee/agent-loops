from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loops.bench.bfcl.adapter import dataset_revision
from agent_loops.bench.bfcl.single import load_single_cases, single_tools
from agent_loops.bench.bfcl.single_runner import run_single_case, summarize_single
from agent_loops.bench.core.config import full_config
from agent_loops.bench.core.local_llm import LocalLLM
from agent_loops.bench.core.runner import pass_at_k
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
    "react": {"max_steps": 4},
    "plan_and_execute": {"max_rounds": 3},
    "plan_and_act": {"max_steps": 4},
    "adapt": {"max_depth": 2},
    "codeact": {"max_steps": 3},
}


def measure_case(
    case,
    module,
    loop_kwargs,
    *,
    base_url,
    model,
    category,
    repeats,
    temperature=0.0,
    seed=0,
    _runner=None,
):
    runner = _runner or run_single_case
    rows = []
    for rep in range(repeats):
        llm = LocalLLM(
            single_tools(case),
            base_url=base_url,
            model=model,
            temperature=temperature,
            seed=seed + rep,
        )
        r = runner(case, llm, category, loop_module=module, loop_kwargs=loop_kwargs)
        r.run_index = rep
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="simple")
    ap.add_argument("--loops", nargs="*", default=["direct", *LOOPS])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeats per case k (pass^k)",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="decoding seed; per-run seed = seed + run_index",
    )
    ap.add_argument("--model", default="gemma-4-E4B-it-qat-q4_0")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--bare", action="store_true", help="no prompt (true baseline)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cases = load_single_cases(a.category)
    if a.limit:
        cases = cases[: a.limit]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = a.tag or f"{a.model}_{a.category}"
    tag += "_bare" if a.bare else f"_prompt{PROMPT_VERSION}"
    out_path = Path("results") / f"single_{tag}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"model {a.model} | {a.category} {len(cases)} cases | {len(a.loops)} loops "
        f",  prompt {'none (bare)' if a.bare else PROMPT_VERSION}\n"
    )

    all_results: dict[str, list] = {}
    for loop_name in a.loops:
        module = None if loop_name == "direct" else LOOPS[loop_name]
        rows = []
        for i, case in enumerate(cases, 1):
            rows.extend(
                measure_case(
                    case,
                    module,
                    LOOP_KWARGS.get(loop_name, {}),
                    base_url=a.base_url,
                    model=a.model,
                    category=a.category,
                    repeats=a.repeats,
                    temperature=a.temperature,
                    seed=a.seed,
                )
            )
            if i % 25 == 0 or i == len(cases):
                s = summarize_single(rows)
                print(
                    f"  {loop_name:16s} [{i}/{len(cases)}] "
                    f"accuracy {s['accuracy']:.1%} call_rate {s['call_rate']:.1%} "
                    f"parse {s['parse_rate']:.1%}",
                    flush=True,
                )
        all_results[loop_name] = rows
        s = summarize_single(rows)
        print(
            f"  → {loop_name}: accuracy {s['accuracy']:.1%} | call_rate {s['call_rate']:.1%} "
            f",  parse {s['parse_rate']:.1%} | {s['mean_seconds']:.1f}s/case\n",
            flush=True,
        )

        out_path.write_text(
            json.dumps(
                {
                    "when": stamp,
                    "model": a.model,
                    "category": a.category,
                    "bare": a.bare,
                    "prompt_version": None if a.bare else PROMPT_VERSION,
                    "dataset": dataset_revision(),
                    "repeats": a.repeats,
                    "config": full_config(
                        model=a.model,
                        base_url=a.base_url,
                        started_at=stamp,
                        temperature=a.temperature,
                        seed=a.seed,
                    ),
                    "n_cases": len(cases),
                    "summaries": {
                        k: {**summarize_single(v), **pass_at_k(v)}
                        for k, v in all_results.items()
                    },
                    "cases": {k: [vars(r) for r in v] for k, v in all_results.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("=" * 74)
    print(
        f"{'loop':<17}{'accuracy':>9}{'call':>9}{'parse':>9}{'quiet':>9}{'LLM':>8}{'sec':>7}"
    )
    print("-" * 74)
    for name, rows in all_results.items():
        s = summarize_single(rows)
        print(
            f"{name:<17}{s['accuracy']:>8.1%}{s['call_rate']:>9.1%}{s['parse_rate']:>9.1%}"
            f"{s['quiet_failure_rate']:>9.1%}{s['mean_llm_calls']:>8.1f}{s['mean_seconds']:>7.1f}"
        )
    print("=" * 74)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
