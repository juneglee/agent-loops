import json

from agent_loops.bench.core.local_llm import _serialize

RAW = {
    "id": "chatcmpl-x",
    "system_fingerprint": "fp_abc",
    "usage": {"prompt_tokens": 1234},
    "timings": {"predicted_ms": 12.3},
    "choices": [{"message": {"content": "hi"}}],
}


def test_serialize_drops_raw_server_payload_from_assistant_history():
    messages = [
        {"role": "user", "content": "request"},
        {
            "role": "assistant",
            "content": {
                "tool_calls": [{"name": "ls", "arguments": {"path": "."}}],
                "text": "",
                "parse_ok": True,
                "quiet_failure": False,
                "truncated": False,
                "raw": RAW,
            },
        },
    ]

    out = _serialize(messages)

    joined = str(out)
    assert (
        "fp_abc" not in joined
        and "chatcmpl-x" not in joined
        and "timings" not in joined
    )
    assert "ls" in joined and "path" in joined


def test_serialize_keeps_plain_string_content_untouched():
    out = _serialize([{"role": "user", "content": "as is"}])
    assert out[0]["content"] == "as is"


def test_assistant_tool_calls_become_native_openai_tool_calls():
    messages = [
        {"role": "user", "content": "request"},
        {
            "role": "assistant",
            "content": {
                "tool_calls": [{"name": "cd", "arguments": {"folder": "communal"}}],
                "text": "",
            },
        },
        {
            "role": "tool",
            "content": {"ok": True, "error": None, "output": {"cwd": "communal"}},
        },
    ]

    out = _serialize(messages)

    a = out[1]
    assert a["role"] == "assistant"
    assert a["tool_calls"][0]["type"] == "function"
    assert a["tool_calls"][0]["function"]["name"] == "cd"
    assert json.loads(a["tool_calls"][0]["function"]["arguments"]) == {
        "folder": "communal"
    }
    t = out[2]
    assert t["role"] == "tool"
    assert t["tool_call_id"] == a["tool_calls"][0]["id"]
    assert json.loads(t["content"])["output"] == {"cwd": "communal"}


def test_parallel_calls_get_matching_tool_call_ids_in_order():
    messages = [
        {
            "role": "assistant",
            "content": {
                "tool_calls": [
                    {"name": "a", "arguments": {}},
                    {"name": "b", "arguments": {}},
                ],
                "text": "",
            },
        },
        {"role": "tool", "content": {"ok": True, "output": "A"}},
        {"role": "tool", "content": {"ok": True, "output": "B"}},
    ]

    out = _serialize(messages)

    ids = [c["id"] for c in out[0]["tool_calls"]]
    assert len(set(ids)) == 2
    assert [m["tool_call_id"] for m in out[1:]] == ids


def test_tool_content_without_native_call_falls_back_to_flattened_observation():
    messages = [
        {"role": "assistant", "content": {"plan": "1. ls[path=a]"}},
        {"role": "tool", "content": {"step": {"tool": "ls"}, "result": {"ok": True}}},
    ]

    out = _serialize(messages)

    assert out[0]["role"] == "assistant" and "tool_calls" not in out[0]
    assert out[1]["role"] == "user" and out[1]["content"].startswith("[observation]")


def test_serialize_drops_raw_payload_nested_inside_dict_content():
    transcript = [
        {
            "role": "assistant",
            "content": {
                "tool_calls": [{"name": "ls", "arguments": {"path": "."}}],
                "text": "",
                "parse_ok": True,
                "raw": RAW,
            },
        },
        {"role": "tool", "content": {"ok": True, "output": "a.txt"}},
        {
            "role": "assistant",
            "content": {"tool_calls": None, "text": "I cannot do this", "raw": RAW},
        },
    ]
    messages = [
        {"role": "user", "content": "request"},
        {"role": "tool", "content": {"failed_trial": transcript}},
    ]

    out = _serialize(messages)

    joined = json.dumps(out, ensure_ascii=False)
    assert "fp_abc" not in joined and "chatcmpl-x" not in joined
    assert "I cannot do this" in joined and "a.txt" in joined


def test_consecutive_assistant_thoughts_are_merged_into_one_message():
    messages = [
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": {"text": "first thought"}},
        {"role": "assistant", "content": {"text": "second thought"}},
    ]

    out = _serialize(messages)

    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[1]["content"] == "first thought\nsecond thought"


def test_thought_followed_by_a_tool_call_is_merged_with_the_call():
    messages = [
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": {"text": "let me look"}},
        {
            "role": "assistant",
            "content": {
                "text": "",
                "tool_calls": [{"name": "ls", "arguments": {"path": "a"}}],
            },
        },
        {"role": "tool", "content": {"ok": True, "output": "x"}},
    ]

    out = _serialize(messages)

    assert [m["role"] for m in out] == ["user", "assistant", "tool"]
    assert out[1]["content"] == "let me look" and len(out[1]["tool_calls"]) == 1
    assert out[2]["tool_call_id"] == out[1]["tool_calls"][0]["id"]


def test_assistant_after_a_tool_result_is_not_merged():
    messages = [
        {"role": "user", "content": "t"},
        {
            "role": "assistant",
            "content": {"text": "", "tool_calls": [{"name": "ls", "arguments": {}}]},
        },
        {"role": "tool", "content": {"ok": True, "output": "x"}},
        {"role": "assistant", "content": {"text": "done"}},
    ]

    out = _serialize(messages)

    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]
