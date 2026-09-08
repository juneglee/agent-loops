import json
import zipfile

import pytest

from agent_loops.bench.tasks.fetch import (
    MCPMARK_ARCHIVES,
    extract_workspace,
    fetch_mcpmark,
    sha256_of,
)


def _archive(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def test_extract_strips_the_top_folder_and_mac_junk(tmp_path):
    archive = _archive(
        tmp_path / "desktop.zip",
        {
            "desktop/a.txt": "a",
            "desktop/sub/b.txt": "b",
            "desktop/.DS_Store": "x",
            "__MACOSX/desktop/._a.txt": "x",
        },
    )
    n = extract_workspace(archive, tmp_path / "out")
    assert n == 2
    assert sorted(p.name for p in (tmp_path / "out").rglob("*") if p.is_file()) == [
        "a.txt",
        "b.txt",
    ]
    assert (tmp_path / "out" / "sub" / "b.txt").read_text() == "b"


def test_extract_keeps_a_flat_archive_flat(tmp_path):
    archive = _archive(tmp_path / "flat.zip", {"a.txt": "a", "b/c.txt": "c"})
    extract_workspace(archive, tmp_path / "out")
    assert (tmp_path / "out" / "a.txt").is_file()
    assert (tmp_path / "out" / "b" / "c.txt").is_file()


def test_extract_rejects_entries_escaping_the_workspace(tmp_path):
    archive = _archive(
        tmp_path / "bad.zip", {"x/../../escape.txt": "x", "x/ok.txt": "o"}
    )
    with pytest.raises(ValueError):
        extract_workspace(archive, tmp_path / "out")


def _tasks_root(tmp_path, category, url):
    task = tmp_path / "tasks" / "easy" / category / "t1"
    task.mkdir(parents=True)
    (task / "meta.json").write_text(
        json.dumps({"category_id": category, "meta_data": {"stateUrl": url}})
    )
    return tmp_path / "tasks"


def test_fetch_checks_the_archive_digest(tmp_path, monkeypatch):
    archive = _archive(tmp_path / "src.zip", {"papers/p.html": "<html/>"})
    root = _tasks_root(tmp_path, "papers", "https://example.test/papers.zip")

    def fake_download(url, dest):
        assert url.endswith("/papers.zip")
        dest.write_bytes(archive.read_bytes())

    monkeypatch.setattr("agent_loops.bench.tasks.fetch._download", fake_download)
    with pytest.raises(ValueError):
        fetch_mcpmark(root, ["papers"], verify=True)
    monkeypatch.setitem(MCPMARK_ARCHIVES, "papers", sha256_of(archive))
    counts = fetch_mcpmark(root, ["papers"], verify=True)
    assert counts == {"papers": 1}
    assert (root / "fixtures" / "papers" / "p.html").read_text() == "<html/>"
