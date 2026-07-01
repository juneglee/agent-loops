import pytest

from agent_loops.bench.core.freetext import render_tool_catalog, render_tools_for_prompt
from agent_loops.loops.plan_and_solve import parse_plan
from agent_loops.loops.rewoo import parse_plan_line

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "lists the directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "a": {"type": "boolean"}},
                "required": ["path"],
            },
        },
    }
]


class TestToolDocsDoNotDictateFormat:
    def test_catalog_lists_tools(self):
        text = render_tool_catalog(TOOLS)

        assert "ls" in text
        assert "path" in text
        assert "lists the directory" in text

    def test_catalog_does_not_prescribe_an_output_format(self):
        text = render_tool_catalog(TOOLS)

        assert "JSON" not in text
        assert '{"name"' not in text

    def test_freetext_prompt_still_prescribes_json(self):
        text = render_tools_for_prompt(TOOLS)

        assert "JSON" in text
        assert "ls" in text


class TestPlanSyntaxTolerance:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("1. ls[path=a]", ("ls", {"path": "a"})),
            ("1. ls(path=a)", ("ls", {"path": "a"})),
            ("2) ls()", ("ls", {})),
            ("1. ls[]", ("ls", {})),
            ("1. ls(path='a', a=True)", ("ls", {"path": "a", "a": True})),
        ],
    )
    def test_plan_and_solve_accepts_both_bracket_styles(self, line, expected):
        assert parse_plan(line) == [expected]

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("#E1 = ls[path=a]", ("#E1", "ls", {"path": "a"})),
            ("#E1 = ls(path=a)", ("#E1", "ls", {"path": "a"})),
            ("#E2 = ls()", ("#E2", "ls", {})),
        ],
    )
    def test_rewoo_accepts_both_bracket_styles(self, line, expected):
        assert parse_plan_line(line) == [expected]

    def test_nested_brackets_still_work_with_parens(self):
        assert parse_plan('1. mv(paths=["a", "b"], dest=out)') == [
            ("mv", {"paths": ["a", "b"], "dest": "out"})
        ]

    def test_multiline_plan_with_mixed_styles(self):
        plan = "1. ls[path=a]\n2. pwd()\n3. mv(src=x, dest=y)"

        assert parse_plan(plan) == [
            ("ls", {"path": "a"}),
            ("pwd", {}),
            ("mv", {"src": "x", "dest": "y"}),
        ]
