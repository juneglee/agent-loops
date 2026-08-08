from agent_loops.bench import prompts

_DOMAIN_WORDS = ["file", "folder", "directory", "workspace"]


def _leaks(text: str) -> list[str]:
    lowered = text.lower()
    return [w for w in _DOMAIN_WORDS if w in lowered]


def test_system_prompt_is_domain_neutral():
    leaked = _leaks(prompts.SYSTEM)

    assert leaked == [], f"domain-specific wording in the system prompt: {leaked}"


def test_loop_instructions_are_domain_neutral():
    leaked = {name: _leaks(text) for name, text in prompts.LOOP_INSTRUCTIONS.items()}
    offenders = {k: v for k, v in leaked.items() if v}

    assert offenders == {}, f"domain-specific wording in loop instructions: {offenders}"


def test_system_prompt_still_tells_the_model_to_act():
    assert "tool" in prompts.SYSTEM


def test_builtin_tool_schemas_are_not_used_for_benchmark():
    assert hasattr(prompts, "DEMO_TOOLS"), (
        "the name must reveal that these are demo-only"
    )
    assert not hasattr(prompts, "tool_schemas"), (
        "a name that can be confused with the benchmark must go"
    )
