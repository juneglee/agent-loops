import pytest

from agent_loops.bench.core.llm import RUNTIMES, build_payload, parse_response


def test_parses_tool_call_from_openai_response():
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "ls", "arguments": '{"path": "docs"}'}}
                    ],
                }
            }
        ]
    }

    out = parse_response(raw)

    assert out["tool_calls"] == [{"name": "ls", "arguments": {"path": "docs"}}]
    assert out["parse_ok"] is True


def test_parses_plain_text_response_as_no_tool_call():
    raw = {"choices": [{"message": {"content": "the file is a.txt"}}]}

    out = parse_response(raw)

    assert out["tool_calls"] is None
    assert out["text"] == "the file is a.txt"
    assert out["parse_ok"] is True


def test_malformed_arguments_json_is_parse_failure_not_exception():
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "ls", "arguments": '{"path": "docs"'}}
                    ],
                }
            }
        ]
    }

    out = parse_response(raw)

    assert out["parse_ok"] is False
    assert out["tool_calls"] is None


def test_detects_quiet_failure_when_text_mentions_tool_but_no_call():
    raw = {
        "choices": [
            {"message": {"content": '{"name": "ls", "arguments": {"path": "."}}'}}
        ]
    }

    out = parse_response(raw)

    assert out["tool_calls"] is None
    assert out["quiet_failure"] is True


def test_payload_forces_temperature_zero_by_default():
    payload = build_payload(runtime="llamacpp", model="m", messages=[], tools=[])

    assert payload["temperature"] == 0


def test_llamacpp_payload_carries_grammar():
    payload = build_payload(
        runtime="llamacpp", model="m", messages=[], tools=[], grammar='root ::= "x"'
    )

    assert payload["grammar"] == 'root ::= "x"'


def test_ollama_rejects_grammar_instead_of_dropping_it_silently():
    with pytest.raises(ValueError, match="grammar"):
        build_payload(
            runtime="ollama", model="m", messages=[], tools=[], grammar='root ::= "x"'
        )


def test_runtime_registry_declares_grammar_support():
    assert RUNTIMES["llamacpp"].supports_grammar is True
    assert RUNTIMES["ollama"].supports_grammar is False
    assert RUNTIMES["llamacpp"].default_base_url.endswith("/v1")
    assert RUNTIMES["ollama"].default_base_url.endswith("/v1")


def test_non_dict_arguments_are_a_parse_failure():
    from agent_loops.bench.core.llm import parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "ls", "arguments": "[1, 2]"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    out = parse_response(raw)
    assert out["parse_ok"] is False and out["tool_calls"] is None


def test_missing_tool_name_is_a_parse_failure():
    from agent_loops.bench.core.llm import parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    out = parse_response(raw)
    assert out["parse_ok"] is False and out["tool_calls"] is None


def test_build_payload_carries_seed_through_extra():
    from agent_loops.bench.core.llm import build_payload

    payload = build_payload(
        runtime="llamacpp", model="m", messages=[], tools=None, seed=42
    )
    assert payload["seed"] == 42
