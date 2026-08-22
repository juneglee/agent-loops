from types import SimpleNamespace

from agent_loops.harness.todo import TODO_TOOL_SCHEMA
from agent_loops.loops.base import Trace


def _fake_loop(name):
    def run(task, env, llm, history=None, **kwargs):
        return Trace(task=task, loop=name)

    return SimpleNamespace(NAME=name, run=run)


def test_with_layers_wraps_every_loop_and_collects_tool_schemas():
    from agent_loops.bench.core.registry import with_layers

    registry, extra = with_layers(
        {"react": _fake_loop("react"), "single_call": _fake_loop("single_call")},
        ["todo"],
    )

    assert sorted(registry) == ["react+todo", "single_call+todo"]
    assert [s["function"]["name"] for s in extra] == ["update_todo"]


def test_with_layers_is_a_noop_without_layers():
    from agent_loops.bench.core.registry import with_layers

    loops = {"react": _fake_loop("react")}
    registry, extra = with_layers(loops, [])

    assert registry is loops and extra == []


def test_make_factory_appends_layer_tool_schemas(monkeypatch):
    import scripts.run_cells as rc

    made: list[dict] = []
    monkeypatch.setattr(
        rc,
        "LocalLLM",
        lambda tools, **kw: made.append({"tools": tools, **kw}) or object(),
    )

    a = SimpleNamespace(
        base_url="u", model="m", temperature=0.0, seed=0, extra_tools=[TODO_TOOL_SCHEMA]
    )
    case_tools = [{"type": "function", "function": {"name": "ls", "parameters": {}}}]
    rc.make_factory(a, 0)(case_tools)

    names = [t["function"]["name"] for t in made[0]["tools"]]
    assert names == ["ls", "update_todo"]


def test_loop_kwargs_survive_layering():
    from agent_loops.bench.core.registry import LOOP_KWARGS, kwargs_for

    assert kwargs_for("react+todo") == LOOP_KWARGS["react"]
    assert kwargs_for("react") == LOOP_KWARGS["react"]
    assert kwargs_for("planner+react") == {}
