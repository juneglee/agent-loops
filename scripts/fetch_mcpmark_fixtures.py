from __future__ import annotations

import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "tasks" / "mcpmark_fs"


def main() -> int:
    urls: dict[str, str] = {}
    for meta in ROOT.glob("*/*/*/meta.json"):
        data = json.loads(meta.read_text(encoding="utf-8"))
        url = data.get("meta_data", {}).get("stateUrl")
        if url:
            urls[data["category_id"]] = url
    out = ROOT / "fixtures"
    out.mkdir(exist_ok=True)
    for category, url in sorted(urls.items()):
        target = out / category
        if target.is_dir():
            print(f"{category}: present")
            continue
        print(f"{category}: downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        tmp = out / f"_{category}"
        shutil.rmtree(tmp, ignore_errors=True)
        archive.extractall(tmp)
        inner = next(p for p in tmp.iterdir() if p.is_dir() and p.name != "__MACOSX")
        shutil.move(str(inner), str(target))
        shutil.rmtree(tmp, ignore_errors=True)
        for junk in target.rglob(".DS_Store"):
            junk.unlink()
    print(f"fixtures ready under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
