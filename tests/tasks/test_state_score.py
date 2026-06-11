from pathlib import Path

from agent_loops.bench.tasks.env import WorkspaceEnv
from agent_loops.bench.tasks.format import load_tasks
from agent_loops.bench.tasks.score import compare, expected_state, score

BASE = Path(__file__).resolve().parents[1] / "fixtures" / "samples"


def _case(cid):
    return next(c for c in load_tasks(BASE / "tasks.json") if c["id"] == cid)


def _replay(case):
    env = WorkspaceEnv(BASE / case["fixture"])
    for turn in case["gt_calls"]:
        for call in turn:
            env.execute(call["name"], call["arguments"])
    return env


def test_expected_state_is_the_fixture_after_replaying_gt_calls():
    case = _case("s002")
    st = expected_state(BASE / case["fixture"], case["gt_calls"], ignore=[])
    assert "docs/notes.txt" in st and "notes.txt" not in st and st["archive"] == "<dir>"


def test_compare_lists_differences_and_is_empty_on_match():
    a = {"x": "h1", "d": "<dir>"}
    assert compare(a, dict(a)) == []
    assert compare(a, {"x": "h2"}) == ["content: x", "missing: d"]
    assert compare(a, {"x": "h1", "d": "<dir>", "z": "h9"}) == ["extra: z"]


def test_score_valid_when_replay_reproduces_expected_state():
    case = _case("s002")
    assert score(case, _replay(case), BASE, answer="") == (True, None)


def test_score_invalid_reports_the_first_differences():
    case = _case("s002")
    valid, error = score(case, WorkspaceEnv(BASE / case["fixture"]), BASE, answer="")
    assert (
        valid is False and error.startswith("state_mismatch:") and "notes.txt" in error
    )


def test_query_case_needs_the_answer_tokens():
    case = _case("s001")
    env = _replay(case)
    assert score(
        case, env, BASE, answer="Final: there are a.md, b.txt, sample.pdf"
    ) == (True, None)
    valid, error = score(case, env, BASE, answer="Final: there are some files")
    assert valid is False and error.startswith("answer_missing:") and "a.md" in error


def test_query_case_fails_if_state_was_changed():
    case = _case("s001")
    env = _replay(case)
    env.execute("write_file", {"path": "junk.txt", "content": "x"})
    valid, error = score(case, env, BASE, answer="a.md b.txt sample.pdf")
    assert valid is False and "extra: junk.txt" in error


def test_ignore_globs_are_excluded_from_comparison():
    case = dict(_case("s002"))
    case["expect"] = {"ignore": ["**/.DS_Store", "*.tmp"]}
    env = _replay(case)
    (env.root / "x.tmp").write_text("t")
    (env.root / "docs" / ".DS_Store").write_bytes(b"\x00")
    assert score(case, env, BASE, answer="") == (True, None)
