from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from agent_loops.bench.tasks.format import CELLS
from agent_loops.bench.tasks.track import TaskTrack
from agent_loops.tools import TOOLS_VERSION


def _trace_sink(path: Path):
    def sink(turn_index: int, task: str, trace: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for i, step in enumerate(trace.steps):
                response = step.llm_response or {}
                fh.write(
                    json.dumps(
                        {
                            "turn": turn_index,
                            "task": task if i == 0 else None,
                            "step": i,
                            "text": str(response.get("text") or "")
                            if step.llm_response
                            else None,
                            "tool_calls": response.get("tool_calls")
                            if step.llm_response
                            else None,
                            "tool_name": step.tool_name,
                            "arguments": step.tool_arguments or None,
                            "observation": (
                                None
                                if step.observation is None
                                else json.dumps(
                                    step.observation, ensure_ascii=False, default=str
                                )[:300]
                            ),
                            "stage": step.stage,
                            "terminated_by": trace.terminated_by
                            if i == len(trace.steps) - 1
                            else None,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

    return sink


def run_dataset(
    tasks_path: Path,
    cells: list[str],
    loops: list[str],
    layers: list[str],
    limit: int,
    repeats: int,
    temperature: float,
    seed: int,
    instruction_variant: str | None,
    code_timeout: float,
    bash_timeout: float,
    model: str,
    base_url: str,
    out_dir: Path = Path("results/tasks"),
    llm_factory: Any = None,
) -> list[Path]:
    track = TaskTrack(Path(tasks_path))
    prompt_version = apply_variant(instruction_variant)
    registry, extra_tools = with_layers({n: LOOPS[n] for n in loops}, layers)
    cases_all = track.all_cases()
    set_name = track.base.name
    revision = track.revision()
    budgets = Budgets(code_timeout=code_timeout, bash_timeout=bash_timeout)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for case in cases_all:
        by_cell.setdefault(case["cell"], []).append(case)

    def local_factory(rep: int):
        def factory(tools: list[dict[str, Any]]) -> LocalLLM:
            return LocalLLM(
                [*tools, *extra_tools],
                base_url=base_url,
                model=model,
                temperature=temperature,
                seed=seed + rep,
            )

        return factory

    written: list[Path] = []
    for cell in cells:
        cases = by_cell.get(cell, [])[:limit] if limit else by_cell.get(cell, [])
        if not cases:
            print(f"\n### {set_name} / {cell}: 0 cases, skipped\n", flush=True)
            continue
        n_turns = 1 if cell.startswith("single_turn") else None
        out = out_dir / f"tasks_{set_name}_{cell}_{model}_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"\n### {set_name} / {cell}: {len(cases)} cases, {len(registry)} stacks, prompt {prompt_version} "
            f",  tools {TOOLS_VERSION} | {revision}\n",
            flush=True,
        )
        results: dict[str, list] = {}
        for name in registry:
            rows = []
            for i, case in enumerate(cases, 1):
                for rep in range(repeats):
                    factory = (
                        llm_factory(case, name)
                        if llm_factory is not None
                        else local_factory(rep)
                    )
                    sink = _trace_sink(
                        out_dir
                        / "traces"
                        / stamp
                        / cell
                        / name
                        / f"{case['id']}_r{rep}.jsonl"
                    )
                    stack = Stack(
                        name=name,
                        run=registry[name].run,
                        kwargs=dict(kwargs_for(name)),
                        layers=tuple(layers),
                    )
                    res = Runner(track, factory, budgets, trace_sink=sink).run_case(
                        case, stack, n_turns=n_turns
                    )
                    res.run_index = rep
                    rows.append(res)
                    print(
                        f"  {name:20s} [{i}/{len(cases)}] {case['id']} "
                        f"{'valid' if res.valid else 'x'} | {res.n_llm_calls} calls | {res.seconds:.0f}s "
                        f",  {res.tps:.0f} tps | {','.join(res.terminated_by)}"
                        f"{'' if res.valid else ' | ' + str(res.error)[:80]}",
                        flush=True,
                    )
            results[name] = rows
            s = summarize(rows)
            print(
                f"  {name}: accuracy {s['accuracy']:.1%} | LLM calls {s['mean_llm_calls']:.1f} | "
                f"{s['mean_seconds']:.0f}s | {s['mean_tps']:.0f} tps\n",
                flush=True,
            )
            out.write_text(
                json.dumps(
                    {
                        "when": stamp,
                        "category": f"tasks:{set_name}",
                        "cell": cell,
                        "model": model,
                        "all_classes": False,
                        "prompt_version": prompt_version,
                        "tools_version": TOOLS_VERSION,
                        "dataset": revision,
                        "n_cases": len(cases),
                        "config": full_config(
                            model=model,
                            base_url=base_url,
                            started_at=stamp,
                            temperature=temperature,
                            seed=seed,
                        ),
                        "repeats": repeats,
                        "layers": layers,
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
        written.append(out)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks/generic_v1/tasks.json")
    ap.add_argument("--cells", nargs="*", default=list(CELLS))
    ap.add_argument("--loops", nargs="*", default=list(LOOPS))
    ap.add_argument("--layers", nargs="*", default=[], choices=sorted(LAYERS))
    ap.add_argument("--limit", type=int, default=0, help="cases per cell (0 = all)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--instruction-variant", default=None, choices=sorted(INSTRUCTION_VARIANTS)
    )
    ap.add_argument("--code-timeout", type=float, default=5.0)
    ap.add_argument("--bash-timeout", type=float, default=10.0)
    ap.add_argument("--model", default="gemma-4-E4B-it-qat-q4_0")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--out-dir", default="results/tasks")
    a = ap.parse_args()
    run_dataset(
        Path(a.tasks),
        a.cells,
        a.loops,
        a.layers,
        a.limit,
        a.repeats,
        a.temperature,
        a.seed,
        a.instruction_variant,
        a.code_timeout,
        a.bash_timeout,
        a.model,
        a.base_url,
        Path(a.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
