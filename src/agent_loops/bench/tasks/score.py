from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from agent_loops.bench.tasks.env import WorkspaceEnv
from agent_loops.bench.tasks.format import fixture_dir


def _ignored(path: str, patterns: list[str]) -> bool:
    name = path.rsplit("/", 1)[-1]
    for pattern in patterns:
        stripped = pattern.removeprefix("**/")
        if (
            fnmatch.fnmatch(path, pattern)
            or fnmatch.fnmatch(name, stripped)
            or fnmatch.fnmatch(path, stripped)
        ):
            return True
    return False


def _filtered(state: dict[str, str], ignore: list[str]) -> dict[str, str]:
    return {k: v for k, v in state.items() if not _ignored(k, ignore)}


def expected_state(
    fixture: Path | str, gt_calls: list[list[dict[str, Any]]], ignore: list[str]
) -> dict[str, str]:
    env = WorkspaceEnv(fixture)
    try:
        for turn in gt_calls:
            for call in turn:
                obs = env.execute(call["name"], call["arguments"])
                if not obs.get("ok"):
                    raise ValueError(
                        f"ground-truth call fails (data defect): {call} -> {obs.get('error')}"
                    )
        return _filtered(env.state(), ignore)
    finally:
        env.close()


def compare(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    diffs: list[str] = []
    for key, value in expected.items():
        if key not in actual:
            diffs.append(f"missing: {key}")
        elif actual[key] != value:
            diffs.append(f"content: {key}")
    diffs.extend(f"extra: {key}" for key in sorted(set(actual) - set(expected)))
    return diffs


def score(
    case: dict[str, Any], env: WorkspaceEnv, base: Path | str, answer: str
) -> tuple[bool, str | None]:
    expect = case.get("expect", {}) or {}
    ignore = list(expect.get("ignore", []))
    expected = expected_state(fixture_dir(case, base), case["gt_calls"], ignore)
    diffs = compare(expected, _filtered(env.state(), ignore))
    if diffs:
        more = f" (+{len(diffs) - 3})" if len(diffs) > 3 else ""
        return False, "state_mismatch: " + "; ".join(diffs[:3]) + more
    missing = [
        token
        for token in expect.get("answer_contains", [])
        if token not in (answer or "")
    ]
    if missing:
        return False, "answer_missing: " + ", ".join(missing)
    return True, None


def final_answer(trace: Any) -> str:
    parts = []
    for step in trace.steps:
        if step.tool_name is not None or step.llm_response is None:
            continue
        text = str(step.llm_response.get("text") or "")
        if text.strip():
            parts.append(text)
    return "\n".join(parts)
