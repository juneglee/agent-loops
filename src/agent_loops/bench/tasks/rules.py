from __future__ import annotations

import posixpath
import re
import shlex
from typing import Any

STATE_KINDS = frozenset(
    {"mkdir", "move", "rename", "copy", "zip", "unzip", "write", "delete"}
)
_SPLIT = re.compile(r"\n|;|&&|\|\|")
_ARCHIVE_CREATE = re.compile(r"^tar\s+(?:-?\w*c\w*|--create)")
_ARCHIVE_EXTRACT = re.compile(r"^tar\s+(?:-?\w*x\w*|--extract)")


def _segments(command: str) -> list[str]:
    loop = re.search(r"\bdo\b(.*?)\bdone\b", command, re.DOTALL)
    body = loop.group(1) if loop else command
    return [s.strip() for s in _SPLIT.split(body) if s.strip()]


def _args(segment: str) -> list[str]:
    try:
        words = shlex.split(segment)
    except ValueError:
        words = segment.split()
    return [w for w in words[1:] if not w.startswith("-")]


def _mv_kind(args: list[str]) -> str:
    if len(args) < 2:
        return "move"
    src, dst = args[0], args[-1]
    if dst.endswith("/") or posixpath.dirname(src) != posixpath.dirname(dst):
        return "move"
    return "rename"


def _shell_kinds(command: str) -> list[str]:
    kinds: list[str] = []
    for seg in _segments(command):
        head = seg.split(None, 1)[0] if seg else ""
        if head in ("mkdir",):
            kinds.append("mkdir")
        elif head == "mv":
            kinds.append(_mv_kind(_args(seg)))
        elif head == "cp":
            kinds.append("copy")
        elif head == "rm" or head == "rmdir":
            kinds.append("delete")
        elif head == "zip" or _ARCHIVE_CREATE.match(seg):
            kinds.append("zip")
        elif head == "unzip":
            kinds.append("list" if re.search(r"\s-l\b", seg) else "unzip")
        elif _ARCHIVE_EXTRACT.match(seg) or head in ("gunzip", "gzip"):
            kinds.append("unzip")
        elif head in ("cat", "head", "tail", "less", "more", "wc"):
            kinds.append("read")
        elif head in ("ls", "find", "tree", "du", "stat"):
            kinds.append("list")
        elif head in (
            "echo",
            "printf",
            "touch",
            "tee",
            "sed",
            "awk",
            "sort",
            "split",
        ) and (
            ">" in seg
            or head in ("touch", "tee", "split")
            or re.search(r"\bsed\s+-i", seg)
        ):
            kinds.append("write")
        elif head:
            kinds.append(head)
    return kinds


def actions_of(traces: list[Any]) -> list[str]:
    kinds: list[str] = []
    for trace in traces:
        for step in trace.steps:
            if not step.tool_name:
                continue
            if step.observation is not None and step.observation.get("ok") is False:
                continue
            args = step.tool_arguments or {}
            if step.tool_name == "Bash":
                kinds.extend(_shell_kinds(str(args.get("command", ""))))
            elif step.tool_name in ("Write", "Edit"):
                kinds.append("write")
            elif step.tool_name == "Read":
                kinds.append("read")
            elif step.tool_name in ("Glob", "Grep"):
                kinds.append("list")
            elif step.tool_name == "execute_code":
                kinds.append("code")
    return kinds


def _final_text(traces: list[Any]) -> str:
    for trace in reversed(traces):
        for step in reversed(trace.steps):
            if step.tool_name is None and step.llm_response:
                text = str(step.llm_response.get("text") or "").strip()
                if text:
                    return text
    return ""


def _argument_text(traces: list[Any]) -> str:
    parts = []
    for trace in traces:
        for step in trace.steps:
            if step.tool_name:
                parts.append(str(step.tool_arguments or {}))
    return "\n".join(parts)


def score_rules(
    rules: dict[str, Any],
    initial: dict[str, str],
    final: dict[str, str],
    traces: list[Any],
) -> tuple[bool, str | None]:
    kinds = actions_of(traces)
    changed = initial != final
    state_kinds = [k for k in kinds if k in STATE_KINDS]
    answered = bool(_final_text(traces)) and not changed
    present = set(kinds) | ({"answer"} if answered else set())
    fails: list[str] = []

    if rules.get("refusal"):
        if changed or not _final_text(traces):
            fails.append("refusal expected: workspace changed or no answer")
    if rules.get("state_change_required") and not changed:
        fails.append("recon only: workspace unchanged")
    for group in rules.get("must_include_any", []):
        if not any(k in present for k in group):
            fails.append(f"none of {group} happened")
    ends = rules.get("must_end_with_any")
    if ends:
        last = state_kinds[-1] if state_kinds else None
        if last not in ends:
            fails.append(f"last action {last} not in {ends}")
    for k in rules.get("forbid", []):
        if k in present:
            fails.append(f"forbidden action {k}")
    order = rules.get("require_order")
    if order:
        pos = 0
        for k in order:
            try:
                pos = kinds.index(k, pos) + 1
            except ValueError:
                fails.append(f"order {order} not followed")
                break
    text = _argument_text(traces)
    for needle in rules.get("require_arg_contains", []):
        if needle not in text:
            fails.append(f"argument lacks {needle!r}")
    if "max_actions" in rules and len(state_kinds) > rules["max_actions"]:
        fails.append(f"{len(state_kinds)} actions over max {rules['max_actions']}")
    if "min_actions" in rules and len(state_kinds) < rules["min_actions"]:
        fails.append(f"{len(state_kinds)} actions under min {rules['min_actions']}")
    if fails:
        return False, "rules: " + "; ".join(fails)
    return True, None
