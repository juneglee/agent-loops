from agent_loops.bench.core.llm import parse_response


def test_detects_length_truncation():
    raw = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}

    out = parse_response(raw)

    assert out["truncated"] is True


def test_normal_stop_is_not_truncated():
    raw = {"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}

    assert parse_response(raw)["truncated"] is False


def test_tool_call_finish_is_not_truncated():
    raw = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [{"function": {"name": "ls", "arguments": "{}"}}],
                },
            }
        ]
    }

    out = parse_response(raw)

    assert out["truncated"] is False
    assert out["tool_calls"] == [{"name": "ls", "arguments": {}}]


def test_missing_finish_reason_is_not_truncated():
    raw = {"choices": [{"message": {"content": "x"}}]}

    assert parse_response(raw)["truncated"] is False


def test_empty_response_is_flagged_even_when_parse_succeeds():
    raw = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}

    out = parse_response(raw)

    assert out["parse_ok"] is True
    assert out["truncated"] is True
