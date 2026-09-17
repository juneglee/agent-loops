from agent_loops.bench.tasks.rules import actions_of, score_rules
from agent_loops.loops.base import Step, Trace


def _trace(calls, final_text="Final: done"):
    t = Trace(task="t", loop="x")
    for name, args in calls:
        t.steps.append(
            Step(
                llm_response={"tool_calls": [{"name": name, "arguments": args}]},
                tool_name=name,
                tool_arguments=args,
                observation={"ok": True, "error": None, "output": ""},
            )
        )
    if final_text is not None:
        t.steps.append(Step(llm_response={"tool_calls": None, "text": final_text}))
    return t


def test_actions_are_derived_from_shell_commands_and_tools():
    t = _trace(
        [
            ("Glob", {"pattern": "*.jpg"}),
            ("Bash", {"command": "mkdir -p photos"}),
            ("Bash", {"command": 'mv "IMG_1.jpg" photos/'}),
            ("Bash", {"command": 'mv "a.txt" "b.txt"'}),
            ("Bash", {"command": "cp data.csv backup/"}),
            ("Bash", {"command": 'zip -r "backup.zip" data'}),
            ("Write", {"file_path": "summary.md", "content": "x"}),
            ("Read", {"file_path": "summary.md"}),
        ]
    )
    assert actions_of([t]) == [
        "list",
        "mkdir",
        "move",
        "rename",
        "copy",
        "zip",
        "write",
        "read",
    ]


def test_state_change_required_fails_when_nothing_changed():
    t = _trace([("Glob", {"pattern": "*"})])
    ok, why = score_rules({"state_change_required": True}, {"a": "1"}, {"a": "1"}, [t])
    assert ok is False and "recon" in why


def test_refusal_passes_only_with_text_and_no_change():
    t = _trace([], final_text="Final: this system cannot send email")
    assert score_rules({"refusal": True}, {"a": "1"}, {"a": "1"}, [t])[0] is True
    changed = _trace([("Bash", {"command": "mkdir x"})])
    assert (
        score_rules({"refusal": True}, {"a": "1"}, {"a": "1", "x": "<dir>"}, [changed])[
            0
        ]
        is False
    )


def test_must_include_any_is_and_of_or_groups():
    t = _trace(
        [
            ("Bash", {"command": "mkdir photos"}),
            ("Bash", {"command": "mv a.jpg photos/"}),
        ]
    )
    s = ({"a.jpg": "h"}, {"photos": "<dir>", "photos/a.jpg": "h"})
    assert score_rules({"must_include_any": [["mkdir"], ["move", "copy"]]}, *s, [t])[0]
    assert not score_rules({"must_include_any": [["zip"]]}, *s, [t])[0]


def test_answer_kind_means_text_only_without_change():
    t = _trace([("Glob", {"pattern": "*"})], final_text="Final: where should it go?")
    assert score_rules(
        {"must_include_any": [["move", "answer"]]}, {"a": "1"}, {"a": "1"}, [t]
    )[0]


def test_end_forbid_order_args_and_action_budget():
    t = _trace(
        [
            ("Bash", {"command": "mkdir archive"}),
            ("Bash", {"command": 'mv "meeting notes final (rev).md" archive/'}),
        ]
    )
    s = (
        {"meeting notes final (rev).md": "h"},
        {"archive": "<dir>", "archive/meeting notes final (rev).md": "h"},
    )
    assert score_rules({"must_end_with_any": ["move"]}, *s, [t])[0]
    assert not score_rules({"must_end_with_any": ["zip"]}, *s, [t])[0]
    assert not score_rules({"forbid": ["mkdir"]}, *s, [t])[0]
    assert score_rules({"require_order": ["mkdir", "move"]}, *s, [t])[0]
    assert not score_rules({"require_order": ["move", "mkdir"]}, *s, [t])[0]
    assert score_rules(
        {"require_arg_contains": ["meeting notes final (rev).md"]}, *s, [t]
    )[0]
    assert not score_rules({"max_actions": 1}, *s, [t])[0]
    assert not score_rules({"min_actions": 3}, *s, [t])[0]


def test_answer_contains_checks_the_final_text():
    t = _trace([("Read", {"file_path": "settings.txt"})], final_text="port is 8080")
    initial = {"settings.txt": "port=8080\n"}
    ok, _ = score_rules({"answer_contains": ["8080"]}, initial, initial, [t])
    assert ok
    ok, reason = score_rules({"answer_contains": ["9090"]}, initial, initial, [t])
    assert not ok and "9090" in reason
