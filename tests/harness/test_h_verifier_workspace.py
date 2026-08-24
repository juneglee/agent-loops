from pathlib import Path

from agent_loops.bench.tasks.env import WorkspaceEnv
from agent_loops.harness.checks.workspace import WORKSPACE_CHECKS
from agent_loops.harness.verifier import verifier


def _wrapped(tmp_path: Path):
    fx = tmp_path / "fx"
    (fx / "docs").mkdir(parents=True)
    (fx / "docs" / "a.md").write_text("hello\n")
    env = WorkspaceEnv(fx)
    return env, verifier(checks=WORKSPACE_CHECKS)().wrap_env(env)


def test_normal_sequence_has_no_false_positives(tmp_path):
    env, v = _wrapped(tmp_path)
    steps = [
        ("write_file", {"path": "n.txt", "content": "x"}),
        ("edit_file", {"path": "docs/a.md", "old": "hello", "new": "bye"}),
        ("bash", {"command": "mkdir out && mv n.txt out/ && cp docs/a.md out/b.md"}),
        ("list_dir", {"path": "out"}),
        ("read_file", {"path": "out/b.md"}),
        ("grep", {"pattern": "bye", "path": "docs"}),
        ("bash", {"command": "rm out/n.txt"}),
    ]
    for name, args in steps:
        obs = v.execute(name, args)
        assert obs["ok"] is True, (name, obs)
    assert len(env.calls) == len(steps)


def test_silent_failure_is_turned_into_a_postcondition_error(tmp_path):
    env, v = _wrapped(tmp_path)
    obs = v.execute("write_file", {"path": "n.txt", "content": "x"})
    (env.root / "n.txt").write_text("corrupted")
    assert obs["ok"] is True
    obs2 = v.execute("edit_file", {"path": "n.txt", "old": "corrupted", "new": "fixed"})
    assert obs2["ok"] is True and (env.root / "n.txt").read_text() == "fixed"


def test_bash_move_that_did_not_happen_is_flagged(tmp_path, monkeypatch):
    env, _ = _wrapped(tmp_path)
    inner = env.execute

    def lying(name, args=None):
        obs = inner(name, args)
        if name == "bash" and "mv" in (args or {}).get("command", ""):
            (env.root / "docs" / "a.md").write_text("back")
        return obs

    monkeypatch.setattr(env, "execute", lying)
    obs = (
        verifier(checks=WORKSPACE_CHECKS)()
        .wrap_env(env)
        .execute("bash", {"command": "mv docs/a.md docs/moved.md"})
    )
    assert obs["ok"] is False and "postcondition" in obs["error"]
