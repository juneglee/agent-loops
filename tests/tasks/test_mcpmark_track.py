from pathlib import Path

from agent_loops.bench.tasks.format import dataset_revision, load_tasks
from agent_loops.bench.tasks.runner import run_task_case
from agent_loops.bench.tasks.track import TaskTrack
from agent_loops.loops import react
from tests.conftest import ScriptedLLM

MINI = Path(__file__).resolve().parents[1] / "fixtures" / "mcpmark_mini"


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _final(text="Final: done"):
    return {"tool_calls": None, "text": text}


def test_mcpmark_directory_loads_as_cases_with_verify_scripts():
    cases = load_tasks(MINI)

    assert [c["id"] for c in cases] == ["easy/demo/make_note"]
    case = cases[0]
    assert case["fixture"] == "fixtures/demo"
    assert case["turns"][0].startswith("Create a file named")
    assert case["expect"]["verify"] == "easy/demo/make_note/verify.py"
    assert "difficulty:L1" in case["tags"] and case["gt_calls"] == [[]]
    assert dataset_revision(MINI).startswith("tasks:mcpmark_mini@")


def test_track_accepts_the_directory_and_runs_the_verifier():
    track = TaskTrack(MINI, tools_version="fs_tool")
    assert track.base == MINI and len(track.all_cases()) == 1


def test_verifier_passes_when_the_agent_creates_the_file():
    case = load_tasks(MINI)[0]
    llm = ScriptedLLM([_tc("Write", file_path="note.txt", content="done\n"), _final()])

    result = run_task_case(
        case, react, lambda tools: llm, MINI, tools_version="fs_tool"
    )

    assert result.valid is True, result.error


def test_verifier_fails_when_the_agent_does_nothing():
    case = load_tasks(MINI)[0]
    llm = ScriptedLLM([_final()])

    result = run_task_case(
        case, react, lambda tools: llm, MINI, tools_version="fs_tool"
    )

    assert result.valid is False
    assert result.error.startswith("verify: exit 1")


def test_verifier_runs_from_a_relative_dataset_path(monkeypatch, tmp_path):
    import os
    import shutil

    target = tmp_path / "mini"
    shutil.copytree(MINI, target)
    monkeypatch.chdir(tmp_path)
    case = load_tasks(Path("mini"))[0]
    llm = ScriptedLLM([_tc("Write", file_path="note.txt", content="done\n"), _final()])

    result = run_task_case(
        case, react, lambda tools: llm, Path("mini"), tools_version="fs_tool"
    )

    assert result.valid is True, result.error
    assert os.getcwd() == str(tmp_path)


def test_run_dataset_can_override_the_step_budget(tmp_path):
    from scripts.run_tasks import _budget

    assert _budget({"max_steps": 10}, 30) == {"max_steps": 30}
    assert _budget({"max_rounds": 5}, 30) == {"max_rounds": 30}
    assert _budget({"max_depth": 3, "max_calls": 30}, 40) == {
        "max_depth": 3,
        "max_calls": 40,
    }
    assert _budget({"max_replans": 1}, 30) == {"max_replans": 1}
    assert _budget({"max_steps": 10}, None) == {"max_steps": 10}
