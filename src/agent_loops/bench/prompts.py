from __future__ import annotations

from typing import Any

PROMPT_VERSION = "v1"

SYSTEM = (
    "You are an assistant that carries out the user's request with the given tools. "
    "Pick the tool that does the job and call it. Do not just describe what to do."
)

LOOP_INSTRUCTIONS: dict[str, str] = {
    "single_call": ("Call the tool that carries out the request **once**."),
    "react": (
        "Work one step at a time. Call one tool, look at its result, then decide the "
        "next step. You may write a short thought before calling a tool "
        "(a thought alone keeps the loop going).\n"
        "When the request is complete, do not call a tool; answer with a line that starts with `Final:`."
    ),
    "rewoo": (
        "First write the **whole plan**. Do not execute any tool yet.\n"
        "Write each step as two lines: a `Plan:` line saying what this step does, then the tool call:\n"
        "  Plan: what this step does\n"
        "  #E1 = tool_name[arg=value, arg2=value2]\n"
        "Refer to an earlier result as `#E1`. Example:\n"
        "  Plan: first do A\n"
        "  #E1 = toolA[arg=value]\n"
        "  Plan: use the result of A to do B\n"
        "  #E2 = toolB[arg=#E1]"
    ),
    "plan_and_solve": (
        "First write the whole plan as a numbered list, then it is executed as written.\n"
        "Line format: `1. tool_name[arg=value]`\n"
        "Do not refer to earlier results; every step is independent."
    ),
}

INSTRUCTION_VARIANTS: dict[str, dict[str, str]] = {
    "no-empty-plan": {
        "plan_and_execute": (
            "Write the plan as a numbered list. Format: `1. tool_name[arg=value]`\n"
            "After **every step of the plan has been executed** you will see the results and plan again."
        ),
    },
    "final-declaration": {
        "plan_and_execute": (
            "Write the plan as a numbered list. Format: `1. tool_name[arg=value]`\n"
            "After **every step of the plan has been executed** you will see the results and plan again.\n"
            "If nothing is left to do, answer with a line that starts with `Final:`."
        ),
    },
}

_active_overrides: dict[str, str] = {}


def apply_variant(name: str | None) -> str:
    overrides = {} if name is None else INSTRUCTION_VARIANTS[name]
    _active_overrides.clear()
    _active_overrides.update(overrides)
    return PROMPT_VERSION if name is None else f"{PROMPT_VERSION}+{name}"


DEMO_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List the direct contents of one directory. Does not descend into subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "path of the directory to list",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Return the contents of one file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "path of the file to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": (
                "Run a shell command for everything the tools above cannot do: move, copy or rename "
                "(mv/cp/for loops), zip/unzip (zip -r / unzip -d), create directories (mkdir -p), "
                "date or size conditions (find -mtime / du). Quote names that contain spaces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "shell command to run"}
                },
                "required": ["command"],
            },
        },
    },
]


def demo_tool_schemas() -> list[dict[str, Any]]:
    return [dict(t, function=dict(t["function"])) for t in DEMO_TOOLS]


def build_messages(
    loop: str,
    task: str,
    history: list[dict[str, Any]] | None = None,
    bare: bool = False,
    prior: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if bare:
        messages: list[dict[str, Any]] = [
            *(prior or []),
            {"role": "user", "content": task},
        ]
        if history:
            messages.extend(history)
        return messages

    instruction = _active_overrides.get(loop, LOOP_INSTRUCTIONS[loop])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        *(prior or []),
        {"role": "user", "content": f"{instruction}\n\nRequest: {task}"},
    ]
    if history:
        messages.extend(history)
    return messages
