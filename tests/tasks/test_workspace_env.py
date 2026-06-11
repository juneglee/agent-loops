from pathlib import Path

from agent_loops.bench.tasks.env import WorkspaceEnv


def _fixture(tmp_path: Path) -> Path:
    fx = tmp_path / "fx"
    (fx / "docs").mkdir(parents=True)
    (fx / "docs" / "a.md").write_text("A")
    return fx


def test_env_copies_fixture_and_never_touches_the_original(tmp_path):
    fx = _fixture(tmp_path)
    env = WorkspaceEnv(fx)
    env.execute("write_file", {"path": "docs/a.md", "content": "changed"})
    assert (fx / "docs" / "a.md").read_text() == "A"
    assert (env.root / "docs" / "a.md").read_text() == "changed"
    env.close()
    assert not env.root.exists()


def test_execute_records_calls_and_returns_observation_shape(tmp_path):
    env = WorkspaceEnv(_fixture(tmp_path))
    obs = env.execute("list_dir", {"path": "docs"})
    assert (
        set(obs) >= {"ok", "error", "output"}
        and obs["ok"] is True
        and "a.md" in obs["output"]
    )
    assert env.calls == [{"name": "list_dir", "arguments": {"path": "docs"}}]
    bad = env.execute("read_file", {"path": "../x"})
    assert bad["ok"] is False and bad["error"] and bad["output"] == ""
    assert len(env.calls) == 2


def test_unknown_tool_is_an_error_observation(tmp_path):
    obs = WorkspaceEnv(_fixture(tmp_path)).execute("teleport", {})
    assert obs["ok"] is False and "teleport" in obs["error"]


def test_tool_names_follow_the_schema_order(tmp_path):
    assert WorkspaceEnv(_fixture(tmp_path)).tool_names() == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
        "bash",
    ]


def test_snapshot_restore_and_reset(tmp_path):
    fx = _fixture(tmp_path)
    env = WorkspaceEnv(fx)
    snap = env.snapshot()
    env.execute("write_file", {"path": "new.txt", "content": "x"})
    env.restore(snap)
    assert not (env.root / "new.txt").exists()
    env.execute("write_file", {"path": "new2.txt", "content": "x"})
    env.reset()
    assert env.state() == WorkspaceEnv(fx).state()
    assert env.calls and env.calls[-1]["name"] == "write_file"


def test_state_is_a_content_hash_map_with_directories(tmp_path):
    st = WorkspaceEnv(_fixture(tmp_path)).state()
    assert st["docs"] == "<dir>" and len(st["docs/a.md"]) == 64


def test_state_normalises_trailing_whitespace_of_text_files(tmp_path):
    fx = _fixture(tmp_path)
    a = WorkspaceEnv(fx)
    b = WorkspaceEnv(fx)
    a.execute("write_file", {"path": "t.txt", "content": "one \ntwo\n\n"})
    b.execute("write_file", {"path": "t.txt", "content": "one\ntwo"})
    assert a.state()["t.txt"] == b.state()["t.txt"]


def test_execute_code_calls_tools_and_records_inner_calls(tmp_path):
    env = WorkspaceEnv(_fixture(tmp_path))
    env.enable_code_execution()
    obs = env.execute("execute_code", {"code": "print(read_file(path='docs/a.md'))"})
    assert obs["ok"] and "A" in obs["output"]
    assert {"name": "read_file", "arguments": {"path": "docs/a.md"}} in env.calls
    assert obs["calls"] == [{"name": "read_file", "arguments": {"path": "docs/a.md"}}]


def test_execute_code_reports_traceback_and_respects_timeout(tmp_path):
    env = WorkspaceEnv(_fixture(tmp_path), code_timeout=1.0)
    env.enable_code_execution()
    bad = env.execute("execute_code", {"code": "read_file(path='nope')"})
    assert bad["ok"] is False and "nope" in bad["error"]
    slow = env.execute("execute_code", {"code": "while True: pass"})
    assert slow["ok"] is False and "timeout" in slow["error"].lower()
