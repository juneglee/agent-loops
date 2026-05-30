from __future__ import annotations

from typing import Any

CODE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute Python code. The file-management functions below can be called directly "
            "inside the code, and several can be combined in one call. stdout is returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    },
}


def code_tools(_case_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [CODE_TOOL_SCHEMA]


def prepare(
    env: Any, case_tools: list[dict[str, Any]], loop_name: str
) -> tuple[list[dict[str, Any]], str | None]:
    if "codeact" not in loop_name.split("+"):
        return case_tools, None

    env.enable_code_execution()
    return code_tools(case_tools), code_signatures(case_tools)


def code_signatures(case_tools: list[dict[str, Any]]) -> str:
    lines = [
        "Functions callable inside the code (no imports | virtual environment | nothing exists besides these functions):"
    ]
    for tool in case_tools:
        fn = tool.get("function", tool)
        params = (fn.get("parameters") or {}).get("properties", {}) or {}
        required = set((fn.get("parameters") or {}).get("required", []) or [])
        args = ", ".join(
            f"{k}: {v.get('type', 'string')}"
            if k in required
            else f"{k}: {v.get('type', 'string')} = None"
            for k, v in params.items()
        )
        lines.append(f"  {fn.get('name')}({args})")
    return "\n".join(lines)
