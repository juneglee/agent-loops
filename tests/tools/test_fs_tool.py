from pathlib import Path

import pytest

from agent_loops.tools import TOOL_VERSIONS, Toolset, implementations, schemas
from agent_loops.tools.fs import ToolError

FIXTURE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "report.pdf"
NAMES = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]


def _ws(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("hello\nworld\nhello again\n")
    (tmp_path / "docs" / "b.txt").write_text("plain text\n")
    (tmp_path / "notes.txt").write_text("todo\n")
    return Toolset(tmp_path, version="fs_tool")


def test_version_is_registered_with_the_six_tools():
    assert "fs_tool" in TOOL_VERSIONS
    assert [s["function"]["name"] for s in schemas("fs_tool")] == NAMES
    assert sorted(implementations(Path("."), version="fs_tool")) == sorted(NAMES)


def test_schema_argument_names_follow_cli_agent_conventions():
    params = {
        s["function"]["name"]: list(s["function"]["parameters"]["properties"])
        for s in schemas("fs_tool")
    }
    assert params["Read"] == ["file_path", "offset", "limit"]
    assert params["Write"] == ["file_path", "content"]
    assert params["Edit"] == ["file_path", "old_string", "new_string", "replace_all"]
    assert params["Bash"] == ["command", "timeout"]
    assert params["Glob"] == ["pattern", "path"]
    assert params["Grep"] == ["pattern", "path", "glob", "output_mode"]


def test_read_returns_numbered_lines_and_pages_with_offset_and_limit(tmp_path):
    t = _ws(tmp_path)
    whole = t.call("Read", {"file_path": "docs/a.md"})
    assert whole.splitlines()[0].endswith("\thello") and whole.splitlines()[
        0
    ].strip().startswith("1")
    part = t.call("Read", {"file_path": "docs/a.md", "offset": 2, "limit": 1})
    assert part.splitlines()[0].strip().startswith("2") and "world" in part
    assert "more lines" in part and "offset=3" in part


def test_read_reports_missing_files_and_directories(tmp_path):
    t = _ws(tmp_path)
    with pytest.raises(ToolError, match="does not exist"):
        t.call("Read", {"file_path": "nope.txt"})
    with pytest.raises(ToolError, match="directory"):
        t.call("Read", {"file_path": "docs"})


def test_read_extracts_pdf_text(tmp_path):
    t = _ws(tmp_path)
    (tmp_path / "r.pdf").write_bytes(FIXTURE_PDF.read_bytes())
    assert "Quarterly report Q1" in t.call("Read", {"file_path": "r.pdf"})


def test_write_creates_parents_and_overwrites(tmp_path):
    t = _ws(tmp_path)
    t.call("Write", {"file_path": "out/new.txt", "content": "x\n"})
    assert (tmp_path / "out" / "new.txt").read_text() == "x\n"
    t.call("Write", {"file_path": "out/new.txt", "content": "y\n"})
    assert (tmp_path / "out" / "new.txt").read_text() == "y\n"


def test_edit_requires_a_prior_read(tmp_path):
    t = _ws(tmp_path)
    with pytest.raises(ToolError, match="Read it first"):
        t.call(
            "Edit",
            {"file_path": "notes.txt", "old_string": "todo", "new_string": "done"},
        )


def test_edit_replaces_a_unique_string_and_rejects_ambiguous_ones(tmp_path):
    t = _ws(tmp_path)
    t.call("Read", {"file_path": "docs/a.md"})
    with pytest.raises(ToolError, match="Found 2 matches"):
        t.call(
            "Edit",
            {"file_path": "docs/a.md", "old_string": "hello", "new_string": "bye"},
        )
    with pytest.raises(ToolError, match="not found"):
        t.call(
            "Edit",
            {"file_path": "docs/a.md", "old_string": "absent", "new_string": "x"},
        )
    t.call(
        "Edit",
        {
            "file_path": "docs/a.md",
            "old_string": "hello again",
            "new_string": "bye again",
        },
    )
    assert (tmp_path / "docs" / "a.md").read_text() == "hello\nworld\nbye again\n"
    t.call(
        "Edit",
        {
            "file_path": "docs/a.md",
            "old_string": "e",
            "new_string": "E",
            "replace_all": True,
        },
    )
    assert "hEllo" in (tmp_path / "docs" / "a.md").read_text()


def test_bash_runs_inside_the_workspace_and_reports_exit_codes(tmp_path):
    t = _ws(tmp_path)
    assert "a.md" in t.call("Bash", {"command": "ls docs"})
    t.call("Bash", {"command": "mkdir -p archive && mv docs/b.txt archive/"})
    assert (tmp_path / "archive" / "b.txt").exists()
    with pytest.raises(ToolError, match="Exit code"):
        t.call("Bash", {"command": "ls missing_dir"})
    with pytest.raises(ToolError, match="blocked"):
        t.call("Bash", {"command": "curl http://example.com"})


def test_bash_honours_the_timeout_argument(tmp_path):
    t = _ws(tmp_path)
    with pytest.raises(ToolError, match="timed out"):
        t.call("Bash", {"command": "tail -f notes.txt", "timeout": 0.2})


def test_glob_matches_recursively_and_reports_no_files(tmp_path):
    t = _ws(tmp_path)
    found = t.call("Glob", {"pattern": "**/*.md"})
    assert found.splitlines() == ["docs/a.md"]
    assert t.call("Glob", {"pattern": "*.txt", "path": "docs"}) == "docs/b.txt"
    assert t.call("Glob", {"pattern": "*.py"}) == "No files found"


def test_grep_output_modes(tmp_path):
    t = _ws(tmp_path)
    content = t.call("Grep", {"pattern": "hello"})
    assert content.splitlines() == ["docs/a.md:1:hello", "docs/a.md:3:hello again"]
    assert (
        t.call("Grep", {"pattern": "hello", "output_mode": "files_with_matches"})
        == "docs/a.md"
    )
    assert t.call("Grep", {"pattern": "hello", "output_mode": "count"}) == "docs/a.md:2"
    assert t.call("Grep", {"pattern": "hello", "glob": "*.txt"}) == "No matches found"
    with pytest.raises(ToolError, match="Invalid regex"):
        t.call("Grep", {"pattern": "("})


def test_paths_outside_the_workspace_are_rejected(tmp_path):
    t = _ws(tmp_path)
    for name, args in [
        ("Read", {"file_path": "../secret"}),
        ("Write", {"file_path": "/etc/x", "content": ""}),
        ("Glob", {"pattern": "*", "path": ".."}),
    ]:
        with pytest.raises(ToolError):
            t.call(name, args)


def test_workspace_env_and_replay_use_the_selected_version(tmp_path):
    from agent_loops.bench.tasks.format import load_tasks
    from agent_loops.bench.tasks.runner import run_task_case
    from agent_loops.loops import react
    from tests.conftest import ScriptedLLM

    base = Path(__file__).resolve().parents[1] / "fixtures" / "samples"
    cases = {c["id"]: c for c in load_tasks(base / "tasks_fs_tool.json")}
    for cid in ("c001", "c002", "c003"):
        case = cases[cid]
        script = []
        for turn in case["gt_calls"]:
            script += [{"tool_calls": [c]} for c in turn]
            answer = " ".join(case.get("expect", {}).get("answer_contains", []))
            script.append({"tool_calls": None, "text": f"Final: {answer or 'done'}"})
        result = run_task_case(
            case,
            react,
            lambda tools, script=script: ScriptedLLM(script),
            base,
            loop_kwargs={"max_steps": 10},
            tools_version="fs_tool",
        )
        assert result.valid is True, f"{cid}: {result.error}"
