from __future__ import annotations

from pathlib import Path


class PathEscape(Exception):
    pass


def resolve(root: Path | str, path: str) -> Path:
    root = Path(root).resolve()
    raw = Path(str(path)).expanduser()
    candidate = (raw if raw.is_absolute() else root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise PathEscape(f"path outside the workspace: {path}")
    return candidate
