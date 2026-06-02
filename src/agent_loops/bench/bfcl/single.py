from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loops.bench.bfcl.adapter import DATASET_VERSION, package_data_dir

_V4_FILE = {"simple": "simple_python"}

CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_single_cases(
    category: str, data_dir: Path | None = None
) -> list[dict[str, Any]]:
    root = data_dir or package_data_dir()
    return _load_jsonl(
        root / f"{DATASET_VERSION}_{_V4_FILE.get(category, category)}.json"
    )


def load_single_answers(
    category: str, data_dir: Path | None = None
) -> dict[str, list[dict[str, Any]]]:
    root = data_dir or package_data_dir()
    rows = _load_jsonl(
        root
        / "possible_answer"
        / f"{DATASET_VERSION}_{_V4_FILE.get(category, category)}.json"
    )
    return {r["id"]: r["ground_truth"] for r in rows}


def single_turn_task(case: dict[str, Any]) -> str:
    turns = case["question"]
    first = turns[0] if turns else []
    texts = [m.get("content", "") for m in first if m.get("role") == "user"]
    return "\n".join(t for t in texts if t)


def _to_openai_schema(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    out = dict(params)
    if out.get("type") == "dict":
        out["type"] = "object"
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {
            k: (
                _to_openai_schema(v)
                if isinstance(v, dict) and v.get("type") == "dict"
                else v
            )
            for k, v in props.items()
        }
    return out


def single_tools(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": fn["name"].replace(".", "_"),
                "description": fn.get("description", ""),
                "parameters": _to_openai_schema(fn.get("parameters", {})),
            },
        }
        for fn in case.get("function", [])
    ]
