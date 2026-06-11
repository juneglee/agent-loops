import pytest

from agent_loops.bench.tasks.env import WorkspaceEnv
from agent_loops.tools import TOOLS_VERSION, implementations, schemas
from agent_loops.tools.fs import ToolError
from agent_loops.tools.guard import Blocked, Guard
from agent_loops.tools.toolset import Toolset


def test_toolset_bundles_schemas_names_and_calls(tmp_path):
    (tmp_path / "a.txt").write_text("hi\n")
    ts = Toolset(tmp_path)
    assert ts.names() == [s["function"]["name"] for s in schemas()]
    assert ts.schemas() == schemas() and ts.version == TOOLS_VERSION
    assert ts.call("read_file", {"path": "a.txt"}) == "hi\n"
    assert ts.call("bash", {"command": "ls"}).strip() == "a.txt"


def test_toolset_guards_before_calling(tmp_path):
    ts = Toolset(tmp_path)
    with pytest.raises(ToolError) as exc:
        ts.call("read_file", {"path": "../x"})
    assert "outside the workspace" in str(exc.value)
    with pytest.raises(ToolError):
        ts.call("bash", {"command": "curl http://x"})
    with pytest.raises(ToolError):
        ts.call("teleport", {})


def test_guard_object_is_configurable(tmp_path):
    strict = Guard(allowed_programs=frozenset({"ls"}))
    assert strict.check("bash", {"command": "ls"}, tmp_path) is None
    with pytest.raises(Blocked):
        strict.check("bash", {"command": "mv a b"}, tmp_path)
    ts = Toolset(tmp_path, guard=strict)
    with pytest.raises(ToolError):
        ts.call("bash", {"command": "mkdir d"})


def test_workspace_env_accepts_a_toolset(tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "a.txt").write_text("x")
    env = WorkspaceEnv(
        fx,
        toolset_factory=lambda root: Toolset(
            root, guard=Guard(allowed_programs=frozenset({"ls"}))
        ),
    )
    assert env.execute("bash", {"command": "ls"})["ok"] is True
    blocked = env.execute("bash", {"command": "mkdir d"})
    assert blocked["ok"] is False and "allow list" in blocked["error"]
    env.close()


def test_legacy_function_api_still_works(tmp_path):
    impl = implementations(tmp_path)
    assert set(impl) == set(Toolset(tmp_path).names())
