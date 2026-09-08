from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MCPMARK_ARCHIVES = {
    "desktop": "fd9efa7f596e94c3ab12bb574f6190c208a094d9f2d66a3f2c1f236e1eba75bd",
    "desktop_template": "a25a080440ba46aa88f8fe0dfae01d3bb47ba3c76d533046b184e09d7c334082",
    "file_context": "d133c03da0ea824a41dbae4ae9af38cf3c4086a0145174aad0ea5666249bbcd8",
    "file_property": "99d5449cef45bfcda6e5260f6ef4cd356bdbae59818d37ffa840054cdee19ec4",
    "folder_structure": "231f9d42040216bb64d4148154b89040f00ac830271dc8f338ac2df719424137",
    "legal_document": "225e469124d2159a2ba5b77b2d39805ac2cce8099870a7482fb37a630dd74c09",
    "papers": "927da33d642186fbf23eb551bb52171cc0b8902813c238de1a41fb258e7bf28d",
    "student_database": "b070246c99332f817a65fd4ad50b6a60db3ad635718915afae7a12d463358633",
    "threestudio": "87a664c1eca18b8052081943d70e0547561aff35ef60af18aafb9a39db4e5063",
    "votenet": "0a3e60344f5021ceff03b220465df61a5c6a6fe8630c10eee809ec905be7d275",
}
_JUNK = ("__MACOSX/", ".DS_Store")


def _wanted(name: str) -> bool:
    return not name.endswith("/") and not any(part in name for part in _JUNK)


def _strip_top(names: list[str]) -> str:
    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1 and all("/" in n for n in names):
        return tops.pop() + "/"
    return ""


def extract_workspace(archive: Path | str, target: Path | str) -> int:
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if _wanted(n)]
        prefix = _strip_top(names)
        for name in names:
            rel = name[len(prefix) :]
            dest = (target / rel).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError(f"archive entry escapes the workspace: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    return len(names)


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "agent-loops"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def archive_urls(tasks_root: Path | str) -> dict[str, str]:
    urls: dict[str, str] = {}
    for meta in sorted(Path(tasks_root).glob("*/*/*/meta.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        url = data.get("meta_data", {}).get("stateUrl")
        if url:
            urls[data["category_id"]] = url
    return urls


def fetch_mcpmark(
    tasks_root: Path | str,
    categories: list[str] | None = None,
    verify: bool = True,
) -> dict[str, int]:
    tasks_root = Path(tasks_root)
    urls = archive_urls(tasks_root)
    if not urls:
        raise ValueError(f"no MCPMark tasks with a stateUrl under {tasks_root}")
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for category in categories or sorted(urls):
            archive = Path(tmp) / f"{category}.zip"
            _download(urls[category], archive)
            if verify:
                expected = MCPMARK_ARCHIVES.get(category)
                actual = sha256_of(archive)
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"{category}.zip changed upstream: sha256 {actual} != {expected}"
                    )
            counts[category] = extract_workspace(
                archive, tasks_root / "fixtures" / category
            )
    return counts
