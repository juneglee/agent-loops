import time

import pytest

from agent_loops.bench.bfcl.adapter import load_cases
from agent_loops.bench.bfcl.env import BFCLEnv

pytestmark = pytest.mark.integration


def _fs_env(**kw):
    case = next(
        c
        for c in load_cases("multi_turn_base")
        if c["involved_classes"] == ["GorillaFileSystem"]
    )
    env = BFCLEnv(case, **kw)
    env.enable_code_execution()
    return env


def test_infinite_loop_is_killed_by_timeout_and_env_stays_usable():
    env = _fs_env(code_timeout=1.0)
    t0 = time.time()
    obs = env.execute("execute_code", {"code": "while True:\n    pass"})
    assert time.time() - t0 < 5
    assert obs["ok"] is False and "timeout" in obs["error"].lower()
    assert env.execute("ls", {})["ok"] is True


def test_state_changes_made_by_code_propagate_back_to_the_parent_env():
    env = _fs_env(code_timeout=5.0)
    obs = env.execute(
        "execute_code", {"code": "mkdir(dir_name='from_child')\nprint(ls())"}
    )
    assert obs["ok"] is True, obs
    assert [c["name"] for c in obs["calls"]] == ["mkdir", "ls"]
    listing = env.execute("ls", {})["output"]
    assert "from_child" in str(listing)
    assert [c["name"] for c in env.calls] == ["mkdir", "ls", "ls"]


def test_exception_inside_code_becomes_observation_with_traceback():
    env = _fs_env(code_timeout=5.0)
    obs = env.execute("execute_code", {"code": "x = 1 / 0"})
    assert obs["ok"] is False and "ZeroDivisionError" in obs["error"]


def test_import_failure_gives_an_actionable_hint_without_harness_internals():
    env = _fs_env(code_timeout=5.0)
    obs = env.execute("execute_code", {"code": "import os\nprint(os.getcwd())"})
    assert obs["ok"] is False
    assert (
        "bfcl_env.py" not in obs["error"] and "_run_code_in_child" not in obs["error"]
    )
    assert "import" in obs["error"].lower()
    assert "ls" in obs["error"] and "cd" in obs["error"]
