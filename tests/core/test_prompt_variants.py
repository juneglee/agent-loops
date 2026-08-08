import pytest

from agent_loops.bench.prompts import PROMPT_VERSION, apply_variant, build_messages


def _instruction(loop: str) -> str:
    return build_messages(loop, "task")[-1]["content"]


@pytest.fixture(autouse=True)
def _reset_variant():
    yield
    apply_variant(None)


def test_default_instructions_contain_the_empty_plan_line():
    assert "empty plan" in _instruction("plan_and_execute")
    assert "empty plan" not in _instruction("plan_and_act")


def test_no_empty_plan_variant_drops_the_line_only_for_the_two_loops():
    version = apply_variant("no-empty-plan")

    assert version == f"{PROMPT_VERSION}+no-empty-plan"
    assert "empty plan" not in _instruction("plan_and_execute")
    assert "independent" in _instruction("plan_and_solve")
    assert "one step at a time" in _instruction("react")


def test_final_declaration_variant_replaces_instead_of_removing():
    apply_variant("final-declaration")
    text = _instruction("plan_and_execute")
    assert "empty plan" not in text
    assert "Final:" in text


def test_reset_restores_the_canonical_instructions():
    apply_variant("no-empty-plan")
    assert apply_variant(None) == PROMPT_VERSION
    assert "empty plan" in _instruction("plan_and_execute")


def test_unknown_variant_name_raises_instead_of_silently_measuring_v2():
    with pytest.raises(KeyError):
        apply_variant("no-such-variant")
