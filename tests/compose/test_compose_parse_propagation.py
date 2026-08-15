from agent_loops.compose import adaptive, hierarchical, routed
from agent_loops.loops import single_call
from tests.conftest import RecordingEnv, ScriptedLLM


def _plan(*items):
    return {"tool_calls": None, "text": "\n".join(f"- {t}" for t in items)}


def _tc(tool, **args):
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _txt(text):
    return {"tool_calls": None, "text": text}


def test_hierarchical_keeps_worker_parse_failure_visible_after_recovery():
    run = hierarchical(single_call)
    llm = ScriptedLLM(
        [
            _plan("check folder a", "check folder b"),
            _txt("Hmm... I am not sure which tool to use."),
            _tc("ls", path="b"),
            _txt("Final: done"),
        ]
    )

    trace = run(task="t", env=RecordingEnv({"ls": lambda path: "x"}), llm=llm)

    assert trace.terminated_by == "success"
    assert trace.parse_ok is False


def test_adaptive_keeps_direct_attempt_parse_failure_visible_after_recovery():
    run = adaptive(single_call)
    llm = ScriptedLLM(
        [
            _txt("I am not sure."),
            _plan("check folder a"),
            _tc("ls", path="a"),
        ]
    )

    trace = run(task="t", env=RecordingEnv({"ls": lambda path: "x"}), llm=llm)

    assert trace.terminated_by == "success"
    assert trace.parse_ok is False


def test_routed_complex_path_propagates_inner_worker_parse_failure():
    run = routed(single_call)
    llm = ScriptedLLM(
        [
            _txt("complex"),
            _plan("check folder a", "check folder b"),
            _txt("I am not sure which tool to use."),
            _tc("ls", path="b"),
            _txt("Final: done"),
        ]
    )

    trace = run(task="t", env=RecordingEnv({"ls": lambda path: "x"}), llm=llm)

    assert trace.terminated_by == "success"
    assert trace.parse_ok is False


def test_clean_success_still_reports_parse_ok_true():
    run = hierarchical(single_call)
    llm = ScriptedLLM(
        [
            _plan("check folder a"),
            _tc("ls", path="a"),
            _txt("Final: done"),
        ]
    )

    trace = run(task="t", env=RecordingEnv({"ls": lambda path: "x"}), llm=llm)

    assert trace.terminated_by == "success"
    assert trace.parse_ok is True
