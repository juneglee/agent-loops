from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loops.bench.bfcl.adapter import dataset_revision
from agent_loops.bench.bfcl.partition import partition
from agent_loops.bench.bfcl.track import BfclTrack
from agent_loops.bench.core.config import full_config
from agent_loops.bench.core.local_llm import LocalLLM
from agent_loops.bench.core.registry import (
    LAYERS,
    LOOPS,
    Stack,
    kwargs_for,
    with_layers,
)
from agent_loops.bench.core.runner import Budgets, Runner, pass_at_k, summarize
from agent_loops.bench.prompts import INSTRUCTION_VARIANTS, apply_variant

CATEGORY = "multi_turn_base"


def category_of(case_id: str) -> str:
    return case_id.rsplit("_", 1)[0]


def make_factory(a, rep):
    def factory(tools):
        return LocalLLM(
            [*tools, *getattr(a, "extra_tools", [])],
            base_url=a.base_url,
            model=a.model,
            temperature=a.temperature,
            seed=a.seed + rep,
        )

    return factory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells",
        nargs="*",
        default=["single_turn_multi_step", "multi_turn_single_step"],
    )
    ap.add_argument("--loops", nargs="*", default=list(LOOPS))
    ap.add_argument("--limit", type=int, default=30)
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
        "--seed",
        type=int,
        default=0,
        help="decoding seed; per-run seed = seed + run_index",
    )
    ap.add_argument(
        "--instruction-variant",
        default=None,
        choices=sorted(INSTRUCTION_VARIANTS),
        help="instruction A/B variant; recorded in the result prompt_version",
    )
    ap.add_argument(
        "--code-timeout",
        type=float,
        default=5.0,
        help="codeact code execution timeout in seconds",
    )
    ap.add_argument(
        "--layers",
        nargs="*",
        default=[],
        choices=sorted(LAYERS),
        help="harness layers applied to every selected loop (e.g. --layers todo → react+todo)",
    )
    ap.add_argument("--model", default="gemma-4-E4B-it-qat-q4_0")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--categories", nargs="*", default=[CATEGORY])
    ap.add_argument(
        "--all-classes",
        action="store_true",
        help="include all classes (default: GorillaFileSystem file management only)",
    )
    a = ap.parse_args()
    a.prompt_version = apply_variant(a.instruction_variant)
    a.registry, a.extra_tools = with_layers({n: LOOPS[n] for n in a.loops}, a.layers)
    a.loops = list(a.registry)

    from agent_loops.bench.bfcl.partition import FILE_MANAGEMENT

    only = None if a.all_classes else FILE_MANAGEMENT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for category in a.categories:
        cells = partition(category, only_classes=only)
        scope = "all classes" if a.all_classes else "file management"
        print(f"\n{'#' * 62}\n# {category} | {scope}\n{'#' * 62}", flush=True)
        run_category(a, category, cells, stamp)
    return 0


def run_category(a, category: str, cells: dict[str, list], stamp: str) -> None:
    from agent_loops.bench.bfcl.partition import FILE_MANAGEMENT

    track = BfclTrack(category, only_classes=None if a.all_classes else FILE_MANAGEMENT)
    for cell in a.cells:
        cases = cells.get(cell, [])
        if a.limit:
            cases = cases[: a.limit]
        if not cases:
            print(f"\n### {category} / {cell} — 0 cases, skipped\n", flush=True)
            continue

        n_turns = 1 if cell.startswith("single_turn") else 99
        out = Path("results") / f"{category}_{cell}_{a.model}_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"\n### {category} / {cell} — {len(cases)} cases | {len(a.loops)} loops "
            f",  prompt {a.prompt_version}\n",
            flush=True,
        )

        results: dict[str, list] = {}
        for name in a.loops:
            rows = []
            for i, case in enumerate(cases, 1):
                for rep in range(a.repeats):
                    stack = Stack(
                        name=name,
                        run=a.registry[name].run,
                        kwargs=dict(kwargs_for(name)),
                        layers=tuple(a.layers),
                    )
                    res = Runner(
                        track,
                        make_factory(a, rep),
                        Budgets(code_timeout=a.code_timeout),
                    ).run_case(case, stack, n_turns=n_turns)
                    res.run_index = rep
                    rows.append(res)
                if i % 10 == 0 or i == len(cases):
                    s = summarize(rows)
                    print(
                        f"  {name:16s} [{i}/{len(cases)}] accuracy {s['accuracy']:.1%}",
                        flush=True,
                    )
            results[name] = rows
            s = summarize(rows)
            print(
                f"  → {name}: accuracy {s['accuracy']:.1%} | parse {s['parse_rate']:.1%} "
                f",  LLM {s['mean_llm_calls']:.1f} calls | {s['mean_seconds']:.0f}s\n",
                flush=True,
            )
            out.write_text(
                json.dumps(
                    {
                        "when": stamp,
                        "category": category,
                        "cell": cell,
                        "model": a.model,
                        "all_classes": a.all_classes,
                        "prompt_version": a.prompt_version,
                        "dataset": dataset_revision(),
                        "n_cases": len(cases),
                        "config": full_config(
                            model=a.model,
                            base_url=a.base_url,
                            started_at=stamp,
                            temperature=a.temperature,
                            seed=a.seed,
                        ),
                        "repeats": a.repeats,
                        "layers": a.layers,
                        "summaries": {
                            k: {**summarize(v), **pass_at_k(v)}
                            for k, v in results.items()
                        },
                        "cases": {k: [vars(r) for r in v] for k, v in results.items()},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        print("=" * 62)
        print(
            f"{category} / {cell}\n"
            f"{'loop':<17}{'accuracy':>9}{'parse':>9}{'LLM':>8}{'sec':>7}"
        )
        print("-" * 62)
        for name, rows in results.items():
            s = summarize(rows)
            print(
                f"{name:<17}{s['accuracy']:>8.1%}{s['parse_rate']:>9.1%}"
                f"{s['mean_llm_calls']:>8.1f}{s['mean_seconds']:>7.0f}"
            )
        print("=" * 62, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
