import pytest

from agent_loops.bench.bfcl.adapter import load_cases, tools_for_case
from agent_loops.bench.bfcl.env import BFCLEnv
from agent_loops.bench.core.codeact_setup import prepare

pytestmark = pytest.mark.integration


def _registry_names() -> list[str]:
    import scripts.run_cells as rc

    layered, _ = rc.with_layers(dict(rc.LOOPS), ["todo"])
    return [*rc.LOOPS, *layered]


def test_every_registry_entry_gets_the_right_tool_set():
    case = load_cases("multi_turn_base", only_classes={"GorillaFileSystem"})[0]
    base_names = sorted(t["function"]["name"] for t in tools_for_case(case))
    failures = []

    for name in _registry_names():
        env = BFCLEnv(case)
        tools, extra = prepare(env, tools_for_case(case), name)
        got = sorted(t["function"]["name"] for t in tools)

        if "codeact" in name.split("+"):
            ok = got == ["execute_code"] and extra is not None and env._code_enabled
        else:
            ok = got == base_names and extra is None and not env._code_enabled
        if not ok:
            failures.append((name, got[:3], bool(extra)))

    assert failures == [], f"{len(failures)} tool sets mismatch: {failures}"
