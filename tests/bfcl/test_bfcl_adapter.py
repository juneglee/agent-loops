import pytest

from agent_loops.bench.bfcl.adapter import (
    calls_to_strings,
    load_cases,
    tools_for_case,
    turns_of,
)

CASE = {
    "id": "multi_turn_base_0",
    "question": [
        [{"role": "user", "content": "first request"}],
        [{"role": "user", "content": "second request"}],
    ],
    "initial_config": {"GorillaFileSystem": {"root": {}}},
    "involved_classes": ["GorillaFileSystem"],
    "path": [],
}


def test_turns_of_returns_one_task_per_user_turn():
    turns = turns_of(CASE)

    assert turns == ["first request", "second request"]


def test_tools_for_case_only_includes_involved_backends():
    tools = tools_for_case(CASE)

    names = {t["function"]["name"] for t in tools}
    assert "ls" in names and "mkdir" in names
    assert "post_tweet" not in names


def test_tools_are_openai_shaped():
    for t in tools_for_case(CASE):
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "parameters" in t["function"]


def test_calls_to_strings_renders_python_call_syntax():
    calls = [{"name": "mkdir", "arguments": {"dir_name": "temp"}}]

    assert calls_to_strings(calls) == ["mkdir(dir_name='temp')"]


def test_calls_to_strings_quotes_strings_and_leaves_other_types():
    calls = [
        {"name": "ls", "arguments": {"a": True}},
        {"name": "tail", "arguments": {"file_name": "a.txt", "lines": 3}},
    ]

    out = calls_to_strings(calls)

    assert out == ["ls(a=True)", "tail(file_name='a.txt', lines=3)"]


def test_calls_to_strings_escapes_quotes_in_values():
    calls = [{"name": "echo", "arguments": {"content": "it's here"}}]

    rendered = calls_to_strings(calls)[0]

    assert rendered.startswith("echo(content=")
    assert "it's here" in rendered.replace('\\"', '"').replace("\\'", "'")


def test_calls_to_strings_handles_no_arguments():
    assert calls_to_strings([{"name": "pwd", "arguments": {}}]) == ["pwd()"]


@pytest.mark.integration
def test_load_cases_reads_official_file():
    cases = load_cases("multi_turn_base")

    assert len(cases) == 200
    assert cases[0]["id"].startswith("multi_turn_base")
    assert "involved_classes" in cases[0]


@pytest.mark.integration
def test_load_cases_can_filter_to_file_system_only():
    cases = load_cases("multi_turn_base", only_classes={"GorillaFileSystem"})

    assert all(c["involved_classes"] == ["GorillaFileSystem"] for c in cases)
    assert len(cases) == 13
