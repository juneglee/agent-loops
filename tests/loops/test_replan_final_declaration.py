from agent_loops.loops.plan_and_act import run as run_pa
from agent_loops.loops.plan_and_execute import run as run_pe


def _txt(text):
    return {"tool_calls": None, "text": text}


def test_first_round_final_declaration_is_no_action_not_parse_fail_pe(
    scripted_llm, recording_env
):
    trace = run_pe(
        task="t",
        env=recording_env({}),
        llm=scripted_llm([_txt("Final: nothing to do for this request")]),
        max_rounds=3,
    )
    assert trace.terminated_by == "no_action"


def test_first_round_final_declaration_from_the_planner_is_parse_fail_pa(
    scripted_llm, recording_env
):
    trace = run_pa(
        task="t",
        env=recording_env({}),
        llm=scripted_llm([_txt("Task completed")]),
        max_steps=3,
    )
    assert trace.terminated_by == "parse_fail"


def test_executor_final_declaration_is_success_pa(scripted_llm, recording_env):
    trace = run_pa(
        task="t",
        env=recording_env({}),
        llm=scripted_llm([_txt("1. wrap up"), _txt("Final: nothing to do")]),
        max_steps=3,
    )
    assert trace.terminated_by == "success"


def test_first_round_prose_without_marker_is_parse_fail_pe(scripted_llm, recording_env):
    trace = run_pe(
        task="t",
        env=recording_env({}),
        llm=scripted_llm([_txt("Hmm, let me look at the files first.")]),
        max_rounds=3,
    )
    assert trace.terminated_by == "parse_fail"
