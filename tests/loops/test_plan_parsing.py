import pytest

from agent_loops.loops.planargs import parse_args, parse_value


class TestArgumentValueTypes:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("10", 10),
            ("3.5", 3.5),
            ('"units"', "units"),
            ("'units'", "units"),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("none", None),
            ("null", None),
            ("[1, 2]", [1, 2]),
            ("bare_word", "bare_word"),
            ("#E1", "#E1"),
        ],
    )
    def test_parse_value_preserves_type(self, raw, expected):
        assert parse_value(raw) == expected
        assert type(parse_value(raw)) is type(expected)


class TestArgumentSplitting:
    def test_commas_inside_brackets_do_not_split(self):
        assert parse_args("paths=[a, b], flag=true") == {
            "paths": ["a", "b"],
            "flag": True,
        }

    def test_commas_inside_quotes_do_not_split(self):
        assert parse_args('msg="hi, there", n=2') == {"msg": "hi, there", "n": 2}

    def test_escaped_quote_does_not_end_the_string(self):
        assert parse_args(r'msg="say \", hello", count=2') == {
            "msg": 'say ", hello',
            "count": 2,
        }

    def test_backslash_before_quote_inside_a_word(self):
        assert parse_args(r'path="a\\b", n=1') == {"path": r"a\b", "n": 1}
