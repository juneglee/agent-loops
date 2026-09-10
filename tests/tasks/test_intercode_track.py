from pathlib import Path

from agent_loops.bench.tasks.format import load_tasks
from agent_loops.bench.tasks.intercode import build_fixtures, is_intercode_dir
from agent_loops.bench.tasks.score import expected_state

DATA = Path(__file__).resolve().parents[2] / "data" / "tasks" / "intercode_bash"


def test_intercode_directory_is_recognised_and_loads_single_step_cases():
    assert is_intercode_dir(DATA)
    cases = load_tasks(DATA)

    assert len(cases) == 26
    assert all(c["cell"] == "single_turn_single_step" for c in cases)
    assert all(c["gt_calls"][0][0]["name"] == "Bash" for c in cases)
    assert all("/testbed" not in c["turns"][0] for c in cases)


def test_fixtures_are_built_from_the_setup_scripts(tmp_path):
    built = build_fixtures(DATA, tmp_path)

    assert (tmp_path / "fs_1" / "testbed" / "dir1").is_dir()
    assert (tmp_path / "fs_2" / "system" / "folder1").is_dir()
    assert (tmp_path / "fs_3" / "workspace").is_dir()
    assert not any(p.name == "index.html" for p in tmp_path.iterdir())
    assert set(built) == {"fs_1", "fs_2", "fs_3"}


def test_every_gold_command_produces_a_state_change():
    for case in load_tasks(DATA):
        state = expected_state(
            DATA / case["fixture"], case["gt_calls"], [], tools_version="fs_tool"
        )
        untouched = expected_state(DATA / case["fixture"], [[]], [], "fs_tool")
        assert state != untouched, case["id"]
