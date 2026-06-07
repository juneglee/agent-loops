from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loops.bench.bfcl.adapter import load_cases, tools_for_case
from agent_loops.bench.bfcl.env import BFCLEnv
from agent_loops.bench.core.local_llm import LocalLLM
from agent_loops.bench.core.registry import LOOPS, kwargs_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", default="react", choices=sorted(LOOPS))
    ap.add_argument("--case", default="multi_turn_base_1")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="gemma-4-E4B-it-qat-q4_0")
    a = ap.parse_args()

    case = next(c for c in load_cases("multi_turn_base") if c["id"] == a.case)
    env = BFCLEnv(case)
    llm = LocalLLM(tools_for_case(case), base_url=a.base_url, model=a.model)
    task = case["question"][0][0]["content"]
    trace = LOOPS[a.loop].run(task=task, env=env, llm=llm, **kwargs_for(a.loop))

    for i, step in enumerate(trace.steps, 1):
        if step.tool_name:
            print(
                f"{i}. {step.tool_name}({json.dumps(step.tool_arguments, ensure_ascii=False)})"
            )
        else:
            print(f"{i}. text: {(step.llm_response or {}).get('text', '')[:160]}")
    print(
        f"\nterminated_by={trace.terminated_by} llm_calls={trace.n_llm_calls} tool_calls={trace.n_tool_calls}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
