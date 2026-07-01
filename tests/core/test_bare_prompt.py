from agent_loops.bench import prompts


def test_bare_sends_only_the_task():
    messages = prompts.build_messages("react", task="move the file", bare=True)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "move the file"


def test_bare_has_no_system_message():
    for loop in prompts.LOOP_INSTRUCTIONS:
        messages = prompts.build_messages(loop, task="task", bare=True)
        assert all(m["role"] != "system" for m in messages)


def test_bare_is_identical_across_loops():
    seen = {
        str(prompts.build_messages(loop, task="same task", bare=True))
        for loop in prompts.LOOP_INSTRUCTIONS
    }

    assert len(seen) == 1


def test_bare_still_carries_history():
    history = [{"role": "user", "content": "[observation] a.txt"}]

    messages = prompts.build_messages("react", task="task", history=history, bare=True)

    assert messages[-1] == history[0]


def test_default_is_not_bare():
    messages = prompts.build_messages("react", task="task")

    assert messages[0]["role"] == "system"
