import pytest

from agent_loops.bench.bfcl.env import BFCLEnv

pytestmark = pytest.mark.integration

CASE = {
    "id": "env_test_0",
    "involved_classes": ["GorillaFileSystem"],
    "initial_config": {
        "GorillaFileSystem": {
            "root": {
                "workspace": {
                    "type": "directory",
                    "contents": {"a.txt": {"type": "file", "content": "hello"}},
                }
            }
        }
    },
    "question": [],
}


def test_executes_official_backend_method():
    env = BFCLEnv(CASE)

    result = env.execute("ls", {})

    assert result["ok"] is True
    assert "a.txt" in str(result["output"])


def test_state_persists_across_calls():
    env = BFCLEnv(CASE)

    env.execute("mkdir", {"dir_name": "temp"})
    result = env.execute("cd", {"folder": "temp"})

    assert result["ok"] is True
    assert "temp" in str(env.execute("pwd", {})["output"])


def test_unknown_tool_is_error_not_exception():
    env = BFCLEnv(CASE)

    result = env.execute("no_such_tool", {})

    assert result["ok"] is False
    assert result["error"]


def test_bad_arguments_are_error_not_exception():
    env = BFCLEnv(CASE)

    result = env.execute("cd", {"folder": "missing_folder"})

    assert result["ok"] is False


def test_records_calls_for_scoring():
    env = BFCLEnv(CASE)

    env.execute("mkdir", {"dir_name": "t"})
    env.execute("ls", {})

    assert env.calls == [
        {"name": "mkdir", "arguments": {"dir_name": "t"}},
        {"name": "ls", "arguments": {}},
    ]


def test_two_envs_do_not_share_state():
    a = BFCLEnv(CASE)
    b = BFCLEnv(CASE)

    a.execute("mkdir", {"dir_name": "only_in_a"})

    assert "only_in_a" not in str(b.execute("ls", {})["output"])
