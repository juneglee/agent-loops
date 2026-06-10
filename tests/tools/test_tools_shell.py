import pytest

from agent_loops.tools import implementations
from agent_loops.tools.fs import ToolError


def test_bash_runs_in_workspace_root_and_returns_stdout(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    assert implementations(tmp_path)["bash"](command="ls").strip() == "a.txt"


def test_bash_nonzero_exit_is_a_tool_error_with_stderr(tmp_path):
    with pytest.raises(ToolError) as exc:
        implementations(tmp_path)["bash"](command="ls no_such_dir")
    assert "no_such_dir" in str(exc.value)


def test_bash_timeout_kills_the_process(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ToolError) as exc:
        implementations(tmp_path, bash_timeout=0.5)["bash"](command="tail -f a.txt")
    assert "timeout" in str(exc.value).lower()


def test_bash_is_guarded_before_execution(tmp_path):
    with pytest.raises(ToolError) as exc:
        implementations(tmp_path)["bash"](command="curl http://example.com")
    assert "curl" in str(exc.value)


def test_bash_cannot_escape_root_via_cd(tmp_path):
    with pytest.raises(ToolError):
        implementations(tmp_path)["bash"](command="cd .. && ls")


def test_bash_file_operations_change_the_workspace(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    t = implementations(tmp_path)
    t["bash"](command="mkdir docs && mv a.txt docs/ && cp docs/a.txt docs/b.txt")
    assert sorted(p.name for p in (tmp_path / "docs").iterdir()) == ["a.txt", "b.txt"]
