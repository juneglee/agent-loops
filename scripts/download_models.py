from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import hf_hub_download

from agent_loops.bench.core.models import MODELS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for key in a.keys:
        spec = MODELS[key]
        dest = MODELS_DIR / key
        if (dest / spec.filename).exists():
            print(f"[skip] {key} already present", flush=True)
            continue
        print(f"[download] {key} <- {spec.repo}/{spec.filename}", flush=True)
        try:
            hf_hub_download(
                repo_id=spec.repo, filename=spec.filename, local_dir=str(dest)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        size = (dest / spec.filename).stat().st_size
        print(f"[ok] {key}: {size / 1e9:.2f} GB", flush=True)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
