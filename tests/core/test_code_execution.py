import pytest

from agent_loops.bench.bfcl.env import BFCLEnv
from agent_loops.bench.bfcl.partition import partition

pytestmark = pytest.mark.integration


@pytest.fixture
def env():
    case = partition()["single_turn_multi_step"][0]
    e = BFCLEnv(case)
    e.enable_code_execution()
    return e


class TestCodeToolAvailability:
    def test_execute_code_appears_in_the_tool_list(self, env):
        assert "execute_code" in env.tool_names

    def test_file_tools_are_still_available(self, env):
        assert {"ls", "pwd", "cat"} <= set(env.tool_names)

    def test_disabled_by_default(self):
        plain = BFCLEnv(partition()["single_turn_multi_step"][0])

        assert "execute_code" not in plain.tool_names


class TestCodeExecution:
    def test_code_can_call_a_file_tool(self, env):
        out = env.execute("execute_code", {"code": "print(pwd())"})

        assert out["ok"] is True, out.get("error")
        assert out["output"].strip()

    def test_tool_calls_made_inside_code_are_recorded(self, env):
        out = env.execute("execute_code", {"code": "pwd()\nls()"})

        assert [c["name"] for c in out["calls"]] == ["pwd", "ls"]

    def test_recorded_calls_carry_their_arguments(self, env):
        out = env.execute("execute_code", {"code": "ls(a=True)"})

        assert out["calls"] == [{"name": "ls", "arguments": {"a": True}}]

    def test_positional_arguments_are_recorded(self, env):
        out = env.execute("execute_code", {"code": "cd('workspace')"})

        assert out["calls"][0]["name"] == "cd"
        assert "workspace" in str(out["calls"][0]["arguments"])

    def test_multiple_tools_compose_in_one_call(self, env):
        out = env.execute("execute_code", {"code": "pwd()\nls()\nls(a=True)"})

        assert len(out["calls"]) == 3
        assert out["ok"] is True

    def test_env_calls_includes_the_inner_calls(self, env):
        env.execute("execute_code", {"code": "pwd()"})

        assert [c["name"] for c in env.calls if c["name"] != "execute_code"] == ["pwd"]


class TestFailuresBecomeObservations:
    def test_syntax_error_is_an_observation_not_a_crash(self, env):
        out = env.execute("execute_code", {"code": "def (:"})

        assert out["ok"] is False
        assert "SyntaxError" in out["error"]

    def test_runtime_error_returns_the_traceback(self, env):
        out = env.execute("execute_code", {"code": "undefined_tool()"})

        assert out["ok"] is False
        assert "undefined_tool" in out["error"]

    def test_partial_progress_before_an_error_is_still_recorded(self, env):
        out = env.execute("execute_code", {"code": "pwd()\nboom()"})

        assert out["ok"] is False
        assert [c["name"] for c in out["calls"]] == ["pwd"]


class TestSandbox:
    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "__import__('os').system('echo hi')",
            "open('/etc/passwd')",
            "eval('1+1')",
            "exec('x=1')",
        ],
    )
    def test_dangerous_constructs_are_blocked(self, env, code):
        out = env.execute("execute_code", {"code": code})

        assert out["ok"] is False, f"not blocked: {code}"

    @pytest.mark.parametrize(
        "code",
        [
            "x = [1, 2, 3]\nprint(len(x))",
            "for p in ['a', 'b']:\n    print(p)",
            "print(str(1) + 'x')",
            "print(sorted([3, 1, 2]))",
        ],
    )
    def test_ordinary_python_still_works(self, env, code):
        out = env.execute("execute_code", {"code": code})

        assert out["ok"] is True, out.get("error")


class TestCallAccounting:
    def test_execute_code_is_not_counted_as_a_tool_call(self, env):
        env.execute("execute_code", {"code": "pwd()\nls()"})

        assert [c["name"] for c in env.calls] == ["pwd", "ls"]

    def test_code_actions_are_counted_separately(self, env):
        env.execute("execute_code", {"code": "pwd()"})
        env.execute("execute_code", {"code": "ls()"})

        assert len(env.code_actions) == 2
        assert len(env.calls) == 2

    def test_code_that_calls_nothing_records_no_tool_call(self, env):
        env.execute("execute_code", {"code": "x = 1"})

        assert env.calls == []
        assert len(env.code_actions) == 1

    def test_direct_tool_calls_still_counted(self, env):
        env.execute("pwd", {})

        assert [c["name"] for c in env.calls] == ["pwd"]
