import pytest

from agent_loops.bench.bfcl.single import (
    load_single_cases,
    single_tools,
    single_turn_task,
)

CASE = {
    "id": "simple_0",
    "question": [
        [{"role": "user", "content": "area of a triangle with base 10 and height 5"}]
    ],
    "function": [
        {
            "name": "calculate_triangle_area",
            "description": "triangle area",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer"},
                    "height": {"type": "integer"},
                },
                "required": ["base", "height"],
            },
        }
    ],
}


def test_task_is_the_single_user_message():
    assert single_turn_task(CASE) == "area of a triangle with base 10 and height 5"


def test_tools_come_from_the_case_itself():
    tools = single_tools(CASE)

    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "calculate_triangle_area"


def test_bfcl_dict_type_is_converted_to_openai_object():
    tools = single_tools(CASE)

    assert tools[0]["function"]["parameters"]["type"] == "object"


@pytest.mark.integration
def test_load_single_cases_reads_official_file():
    cases = load_single_cases("simple")

    assert len(cases) == 400
    assert cases[0]["id"].startswith("simple")
