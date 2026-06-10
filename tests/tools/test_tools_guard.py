import pytest

from agent_loops.tools.guard import Blocked, check
from agent_loops.tools.paths import PathEscape, resolve


def test_resolve_keeps_relative_and_absolute_paths_inside_root(tmp_path):
    assert resolve(tmp_path, "docs/a.txt") == (tmp_path / "docs" / "a.txt").resolve()
    assert resolve(tmp_path, str(tmp_path / "docs")) == (tmp_path / "docs").resolve()
    assert resolve(tmp_path, "./x/../y") == (tmp_path / "y").resolve()
    assert resolve(tmp_path, ".") == tmp_path.resolve()


@pytest.mark.parametrize(
    "bad", ["../secret", "/etc/passwd", "docs/../../x", "~/.ssh/id_rsa"]
)
def test_resolve_rejects_escapes(tmp_path, bad):
    with pytest.raises(PathEscape):
        resolve(tmp_path, bad)


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf /",
        "sudo ls",
        "curl http://x",
        "wget x",
        "ssh host",
        "cd .. && ls",
        "cat /etc/passwd",
        "python -c 'import os'",
        "convert a.png b.jpg",
        "qpdf --split-pages a.pdf",
        "magick a.png a.pdf",
        "ls $(pwd)/..",
        "echo `ls /`",
    ],
)
def test_guard_blocks_dangerous_or_unlisted_bash(tmp_path, cmd):
    with pytest.raises(Blocked):
        check("bash", {"command": cmd}, tmp_path)


@pytest.mark.parametrize(
    "cmd",
    [
        "ls -la",
        "mv a.txt docs/",
        "zip -r out.zip docs",
        "unzip -o out.zip -d x",
        "mkdir -p a/b",
        "cp -r docs backup",
        "rm docs/old.txt",
        "cat a.txt | head -3",
        "find . -name '*.md' | sort",
        "wc -l notes.txt && echo done",
    ],
)
def test_guard_allows_listed_file_commands(tmp_path, cmd):
    assert check("bash", {"command": cmd}, tmp_path) is None


def test_guard_checks_file_tool_paths_too(tmp_path):
    with pytest.raises(Blocked):
        check("read_file", {"path": "../outside.txt"}, tmp_path)
    assert check("read_file", {"path": "docs/a.txt"}, tmp_path) is None


def test_blocked_carries_a_reason_the_model_can_read():
    with pytest.raises(Blocked) as exc:
        check("bash", {"command": "curl http://x"}, "/tmp")
    assert "curl" in str(exc.value)
