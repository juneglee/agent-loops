from agent_loops.bench.core.freetext import (
    parse_freetext_calls,
    render_tools_for_prompt,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "creates a directory",
            "parameters": {
                "type": "object",
                "properties": {"dir_name": {"type": "string"}},
                "required": ["dir_name"],
            },
        },
    }
]


def test_renders_tools_as_text_for_the_prompt():
    text = render_tools_for_prompt(TOOLS)

    assert "mkdir" in text
    assert "dir_name" in text
    assert "creates a directory" in text


def test_parses_fenced_json_block():
    out = parse_freetext_calls(
        '```json\n{"name": "mkdir", "arguments": {"dir_name": "t"}}\n```'
    )

    assert out == [{"name": "mkdir", "arguments": {"dir_name": "t"}}]


def test_parses_bare_json_object():
    out = parse_freetext_calls('{"name": "mkdir", "arguments": {"dir_name": "t"}}')

    assert out == [{"name": "mkdir", "arguments": {"dir_name": "t"}}]


def test_parses_json_embedded_in_prose():
    text = 'I will create the directory.\n{"name": "mkdir", "arguments": {"dir_name": "t"}}\nDone.'

    assert parse_freetext_calls(text) == [
        {"name": "mkdir", "arguments": {"dir_name": "t"}}
    ]


def test_returns_none_for_plain_prose():
    assert parse_freetext_calls("Just an explanation, no call.") is None


def test_returns_none_for_malformed_json():
    assert (
        parse_freetext_calls('{"name": "mkdir", "arguments": {"dir_name": "t"') is None
    )


def test_accepts_alternative_key_names():
    out = parse_freetext_calls('{"tool": "mkdir", "parameters": {"dir_name": "t"}}')

    assert out == [{"name": "mkdir", "arguments": {"dir_name": "t"}}]


def test_parses_multiple_calls_in_a_list():
    out = parse_freetext_calls(
        '[{"name": "mkdir", "arguments": {"dir_name": "a"}}, '
        '{"name": "ls", "arguments": {}}]'
    )

    assert len(out) == 2
    assert out[1]["name"] == "ls"
