from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_loops.loops import (
    adapt,
    codeact,
    dfsdt,
    fixed_pipeline,
    llm_compiler,
    plan_and_act,
    plan_and_solve,
    react,
    reflexion,
    rewoo,
    single_call,
)


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls_made = 0

    def __call__(self, messages, **_):
        self.calls_made += 1
        if not self._responses:
            raise RuntimeError("script exhausted")
        return dict(self._responses.pop(0), parse_ok=True)


class RecordingEnv:
    def __init__(self, tools):
        self._tools = dict(tools)
        self.executed = []

    def execute(self, name, arguments):
        self.executed.append((name, dict(arguments)))
        fn = self._tools.get(name)
        if fn is None:
            return {"ok": False, "error": f"unknown tool: {name}", "output": ""}
        try:
            return {"ok": True, "output": fn(**arguments)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "output": ""}


TASK = "Collect the file listings of folders a, b and c"


def _bad(**_kw):
    raise RuntimeError("no such path")


TOOLS = {"ls": lambda path: f"{path}/file.txt", "bad": _bad}


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


SCENARIOS = {
    "A1 single_call": (
        single_call.run,
        {},
        [_tc("ls", path="a")],
    ),
    "A2 react": (
        react.run,
        {"max_steps": 10},
        [
            _tc("ls", path="a"),
            _tc("ls", path="b"),
            _tc("ls", path="c"),
            _txt("Final: done"),
        ],
    ),
    "A3 rewoo": (
        rewoo.run,
        {},
        [_txt("#E1 = ls[path=a]\n#E2 = ls[path=b]\n#E3 = ls[path=c]"), _txt("summary")],
    ),
    "A4 plan_and_solve": (
        plan_and_solve.run,
        {},
        [_txt("1. ls[path=a]\n2. ls[path=b]\n3. ls[path=c]")],
    ),
    "A5 plan_and_act": (
        plan_and_act.run,
        {"max_steps": 10},
        [
            _txt("1. look at folder a\n2. look at folder b\n3. look at folder c"),
            _tc("ls", path="a"),
            _txt("1. look at folder b\n2. look at folder c"),
            _tc("ls", path="b"),
            _txt("1. look at folder c"),
            _tc("ls", path="c"),
            _txt("1. report the result"),
            _txt("Final: done"),
        ],
    ),
    "A6 adapt": (
        adapt.run,
        {"max_depth": 3},
        [_tc("ls", path="a"), _txt("Task completed")],
    ),
    "A7 codeact": (
        codeact.run,
        {"max_steps": 5},
        [_tc("execute_code", code="for p in 'abc': ls(p)"), _txt("Final: done")],
    ),
    "A8 fixed_pipeline": (
        fixed_pipeline.run,
        {},
        [_tc("ls", path="a"), _tc("ls", path="b"), _tc("ls", path="c")],
    ),
    "A11 reflexion": (
        reflexion.run,
        {"max_trials": 2, "max_steps": 5},
        [
            _txt("Task failed: cannot do it"),
            _txt("I should have gone through the folders one by one"),
            _tc("ls", path="a"),
            _tc("ls", path="b"),
            _tc("ls", path="c"),
            _txt("Task completed"),
        ],
    ),
    "A12 dfsdt": (
        dfsdt.run,
        {"breadth": 2, "max_calls": 8},
        [
            _tc("bad", path="a"),
            _txt("Give up: that path does not exist"),
            _tc("ls", path="a"),
            _tc("ls", path="b"),
            _tc("ls", path="c"),
            _txt("Final: done"),
        ],
    ),
    "A13 llm_compiler": (
        llm_compiler.run,
        {"max_replans": 1},
        [
            _txt("#E1 = ls[path=a]\n#E2 = ls[path=b]\n#E3 = ls[path=c]"),
            _txt("Final: done"),
        ],
    ),
}


class _InnerToolCounter:
    def __init__(self, inner_calls: tuple[str, ...] = ("a", "b", "c")) -> None:
        self._inner = inner_calls
        self.made: list[str] = []

    def __call__(self, code: str) -> str:
        self.made.extend(self._inner)
        return "\n".join(self.made)

    def __len__(self) -> int:
        return len(self.made)


def main() -> int:
    print(f"task: {TASK}\n")
    hdr = f"{'loop':<20}{'llm calls':>10}{'tool runs':>10}{'steps':>6}  {'terminated_by':<14}{'parse':>6}"
    print(hdr)
    print("-" * len(hdr))

    for name, (fn, kwargs, script) in SCENARIOS.items():
        tools = dict(TOOLS)
        counter = _InnerToolCounter()
        if "codeact" in name:
            tools["execute_code"] = counter
        env = RecordingEnv(tools)
        llm = ScriptedLLM(script)
        trace = fn(task=TASK, env=env, llm=llm, **kwargs)

        n_tools = len(counter) if "codeact" in name else len(env.executed)
        print(
            f"{name:<20}{llm.calls_made:>9}{n_tools:>10}{len(trace.steps):>6}  "
            f"{trace.terminated_by:<12}{'OK' if trace.parse_ok else 'FAIL':>5}"
        )

    print("\nhow to read")
    print(
        "  llm calls dominate on-device cost: every call is a full local forward pass"
    )
    print(
        "  rewoo: three tools but two calls, the structural basis of its token-efficiency claim"
    )
    print(
        "  plan_and_act: replans after every action, so calls grow with the number of actions"
    )
    print(
        "  codeact: one piece of code runs three tools, compressing multi-step into one inference"
    )
    print("  reflexion: one extra reflection call after a failed trial")
    print(
        "  dfsdt: branches to a sibling where the model gave up; without a give-up it equals react"
    )
    print(
        "  llm_compiler: flat calls like rewoo, but the plan is a DAG and a joiner decides the end"
    )
    print(
        "  single_call cannot finish this task; adapt runs until the executor declares completion"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
