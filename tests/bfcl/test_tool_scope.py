import pytest

from agent_loops.bench.bfcl.adapter import tools_for_case
from agent_loops.bench.bfcl.partition import partition

pytestmark = pytest.mark.integration

FILE_TOOLS = {
    "cat",
    "cd",
    "cp",
    "diff",
    "du",
    "echo",
    "find",
    "grep",
    "ls",
    "mkdir",
    "mv",
    "pwd",
    "rm",
    "rmdir",
    "sort",
    "tail",
    "touch",
    "wc",
}


def test_partition_defaults_to_file_management_only():
    cells = partition("multi_turn_base")

    for name, cases in cells.items():
        for case in cases:
            assert case["involved_classes"] == ["GorillaFileSystem"], (
                f"{name} case {case['id']} mixes in another backend: "
                f"{case['involved_classes']}"
            )


def test_every_case_gets_exactly_the_file_tools():
    cells = partition("multi_turn_base")

    for cases in cells.values():
        for case in cases:
            names = {t["function"]["name"] for t in tools_for_case(case)}
            assert names == FILE_TOOLS, f"{case['id']}: {sorted(names ^ FILE_TOOLS)}"


def test_tool_count_is_constant_across_cases():
    cells = partition("multi_turn_base")
    counts = {len(tools_for_case(case)) for cases in cells.values() for case in cases}

    assert counts == {18}, f"tool count is not uniform: {sorted(counts)}"
