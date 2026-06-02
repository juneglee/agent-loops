from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_VERSION = "BFCL_v4"


def package_data_dir() -> Path:
    import bfcl_eval

    return Path(bfcl_eval.__file__).resolve().parent / "data"


def dataset_revision() -> str:
    from importlib.metadata import version

    return f"bfcl-eval {version('bfcl-eval')} ({DATASET_VERSION})"


_FUNC_DOC = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_cases(
    category: str,
    only_classes: set[str] | None = None,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    root = data_dir or package_data_dir()
    cases = _load_jsonl(root / f"{DATASET_VERSION}_{category}.json")
    if only_classes is not None:
        cases = [c for c in cases if set(c["involved_classes"]) == only_classes]
    return cases


def load_ground_truth(
    category: str, data_dir: Path | None = None
) -> dict[str, list[list[str]]]:
    root = data_dir or package_data_dir()
    rows = _load_jsonl(root / "possible_answer" / f"{DATASET_VERSION}_{category}.json")
    return {r["id"]: r["ground_truth"] for r in rows}


def turns_of(case: dict[str, Any]) -> list[str]:
    turns = []
    for turn in case["question"]:
        texts = [m.get("content", "") for m in turn if m.get("role") == "user"]
        turns.append("\n".join(t for t in texts if t))
    return turns


def tools_for_case(
    case: dict[str, Any], data_dir: Path | None = None
) -> list[dict[str, Any]]:
    root = data_dir or package_data_dir()
    tools: list[dict[str, Any]] = []
    for class_name in case["involved_classes"]:
        filename = _FUNC_DOC.get(class_name)
        if filename is None:
            continue
        for fn in _load_jsonl(root / "multi_turn_func_doc" / filename):
            tools.append({"type": "function", "function": fn})
    return tools


def _render_value(value: Any) -> str:
    return repr(value)


def calls_to_strings(calls: list[dict[str, Any]]) -> list[str]:
    out = []
    for call in calls:
        args = call.get("arguments") or {}
        rendered = ", ".join(f"{k}={_render_value(v)}" for k, v in args.items())
        out.append(f"{call['name']}({rendered})")
    return out


def trace_to_decoded(traces: list[Any]) -> list[list[list[str]]]:
    decoded: list[list[list[str]]] = []
    for trace in traces:
        steps: list[list[str]] = []
        for step in trace.steps:
            if step.tool_name is None:
                continue
            steps.append(
                calls_to_strings(
                    [{"name": step.tool_name, "arguments": step.tool_arguments}]
                )
            )
        decoded.append(steps)
    return decoded
