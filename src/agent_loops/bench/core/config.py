from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    loop: str
    dataset: str = "bfcl"
    category: str = ""

    model_id: str = ""
    model_file: str = ""
    quantization: str = "q4_0"

    runtime: str = "llamacpp"
    runtime_version: str = ""
    base_url: str = ""

    temperature: float = 0.0
    seed: int | None = 0
    context_length: int = 8192
    grammar: str | None = None
    action_format: str = "json_text"
    jinja: bool = False

    max_steps: int = 10
    loop_kwargs: dict[str, Any] = field(default_factory=dict)

    dataset_revision: str = ""
    scorer_commit: str = ""
    code_commit: str = ""
    machine: str = ""
    started_at: str = ""

    @property
    def grammar_hash(self) -> str | None:
        if self.grammar is None:
            return None
        return hashlib.sha256(self.grammar.encode("utf-8")).hexdigest()[:12]

    @property
    def constrained(self) -> bool:
        return self.grammar is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("grammar", None)
        d["grammar_hash"] = self.grammar_hash
        d["constrained"] = self.constrained
        return d

    def key(self) -> str:
        parts = [
            self.loop,
            self.model_id or "?",
            self.runtime,
            "grammar" if self.constrained else "free",
        ]
        if self.category:
            parts.append(self.category)
        return "/".join(parts)

    def __str__(self) -> str:
        return self.key()


def dumps(config: RunConfig, **extra: Any) -> str:
    return json.dumps({**config.to_dict(), **extra}, ensure_ascii=False, indent=2)


def run_env_info(base_url: str | None = None) -> dict[str, Any]:
    import platform
    import sys as _sys

    runtime_version = ""
    if base_url:
        try:
            import requests

            root = base_url.rsplit("/v1", 1)[0]
            props = requests.get(f"{root}/props", timeout=3).json()
            runtime_version = str(props.get("build_info") or props.get("version") or "")
        except Exception:  # noqa: BLE001
            runtime_version = ""
    try:
        from importlib.metadata import version

        scorer = f"bfcl-eval {version('bfcl-eval')}"
    except Exception:  # noqa: BLE001
        scorer = ""
    return {
        "runtime_version": runtime_version,
        "scorer": scorer,
        "code_commit": _code_identity(),
        "machine": f"{platform.node()} ({platform.machine()})",
        "python": _sys.version.split()[0],
    }


def _code_identity() -> str:
    import hashlib
    import subprocess
    from pathlib import Path

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        if out:
            return out
    except OSError:
        pass
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for sub in ("loops", "bench", "compose", "scripts"):
        base = root / sub
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            digest.update(f.name.encode())
            digest.update(f.read_bytes())
    return "src-" + digest.hexdigest()[:12]


def full_config(
    *,
    model: str,
    base_url: str | None = None,
    started_at: str = "",
    grammar: str | None = None,
    temperature: float = 0.0,
    seed: int | None = 0,
) -> dict[str, Any]:
    env = run_env_info(base_url)
    try:
        from agent_loops.bench.bfcl.adapter import dataset_revision

        dataset = dataset_revision()
    except Exception:  # noqa: BLE001
        dataset = ""
    cfg = RunConfig(
        loop="*",
        model_id=model,
        runtime="llamacpp",
        runtime_version=env["runtime_version"],
        base_url=base_url or "",
        temperature=temperature,
        seed=seed,
        grammar=grammar,
        action_format="native_tools+text",
        jinja=None,
        dataset_revision=dataset,
        scorer_commit=env["scorer"],
        code_commit=env["code_commit"],
        machine=env["machine"],
        started_at=started_at,
    ).to_dict()
    cfg["python"] = env["python"]
    return cfg
