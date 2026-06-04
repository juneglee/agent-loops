from agent_loops.bench.core.history import turn_messages
from agent_loops.loops.base import Step, Trace


def test_turn_messages_reconstructs_request_calls_observations_and_final_answer():
    trace = Trace(task="show the file", loop="react")
    resp1 = {
        "tool_calls": [{"name": "cat", "arguments": {"file_name": "log.txt"}}],
        "text": "",
    }
    obs1 = {"ok": True, "error": None, "output": "line1\nline2"}
    trace.steps.append(
        Step(
            llm_response=resp1,
            tool_name="cat",
            tool_arguments={"file_name": "log.txt"},
            observation=obs1,
        )
    )
    trace.steps.append(
        Step(llm_response={"tool_calls": None, "text": "Task completed: two lines"})
    )

    messages = turn_messages("show the file", trace)

    assert messages[0] == {"role": "user", "content": "show the file"}
    assert messages[1]["role"] == "assistant" and "cat" in str(messages[1]["content"])
    assert messages[2]["role"] == "tool" and "line1" in str(messages[2]["content"])
    assert messages[-1]["role"] == "assistant" and "two lines" in str(
        messages[-1]["content"]
    )
    assert "raw" not in str(messages)


def test_turn_messages_groups_parallel_calls_under_one_assistant_message():
    trace = Trace(task="t", loop="react")
    resp = {
        "tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
    }
    trace.steps.append(
        Step(
            llm_response=resp,
            tool_name="a",
            tool_arguments={},
            observation={"ok": True},
        )
    )
    trace.steps.append(
        Step(
            llm_response=None,
            tool_name="b",
            tool_arguments={},
            observation={"ok": True},
        )
    )

    messages = turn_messages("t", trace)

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]
