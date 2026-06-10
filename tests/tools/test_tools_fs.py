from pathlib import Path

import pytest

from agent_loops.tools import TOOLS_VERSION, implementations, schemas
from agent_loops.tools.fs import ToolError

FIXTURE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "report.pdf"


def _tools(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("hello\nworld\nTODO x\n")
    (tmp_path / "docs" / "b.txt").write_text("b")
    return implementations(tmp_path)


def test_schemas_are_openai_shaped_and_versioned():
    names = [s["function"]["name"] for s in schemas()]
    assert names == ["read_file", "write_file", "edit_file", "list_dir", "grep", "bash"]
    for s in schemas():
        assert s["type"] == "function"
        assert s["function"]["description"].strip()
        assert s["function"]["parameters"]["type"] == "object"
    assert TOOLS_VERSION


def test_implementations_cover_every_schema(tmp_path):
    assert set(implementations(tmp_path)) == {s["function"]["name"] for s in schemas()}


def test_read_write_edit_roundtrip(tmp_path):
    t = _tools(tmp_path)
    t["write_file"](path="docs/c.md", content="one\ntwo\n")
    assert t["read_file"](path="docs/c.md") == "one\ntwo\n"
    t["edit_file"](path="docs/c.md", old="two", new="2")
    assert (tmp_path / "docs" / "c.md").read_text() == "one\n2\n"


def test_edit_requires_unique_match_unless_replace_all(tmp_path):
    t = _tools(tmp_path)
    (tmp_path / "x.txt").write_text("a a")
    with pytest.raises(ToolError):
        t["edit_file"](path="x.txt", old="a", new="b")
    with pytest.raises(ToolError):
        t["edit_file"](path="x.txt", old="zzz", new="b")
    t["edit_file"](path="x.txt", old="a", new="b", replace_all=True)
    assert (tmp_path / "x.txt").read_text() == "b b"


def test_list_dir_plain_glob_and_recursive(tmp_path):
    t = _tools(tmp_path)
    plain = t["list_dir"](path="docs")
    assert "a.md" in plain and "b.txt" in plain
    assert (
        t["list_dir"](path=".", pattern="*.md", recursive=True).strip() == "docs/a.md"
    )
    assert "docs/" in t["list_dir"](path=".")


def test_grep_reports_file_line_and_text(tmp_path):
    t = _tools(tmp_path)
    assert t["grep"](pattern="TODO", path="docs") == "docs/a.md:3:TODO x"
    assert t["grep"](pattern="nothing-here", path="docs") == ""


def test_missing_file_is_a_tool_error(tmp_path):
    t = _tools(tmp_path)
    with pytest.raises(ToolError):
        t["read_file"](path="docs/nope.md")


def test_write_creates_parent_directories(tmp_path):
    t = _tools(tmp_path)
    t["write_file"](path="new/deep/f.txt", content="x")
    assert (tmp_path / "new" / "deep" / "f.txt").read_text() == "x"


def test_read_file_offset_and_limit_are_line_based(tmp_path):
    t = _tools(tmp_path)
    assert t["read_file"](path="docs/a.md", offset=2, limit=1) == "world\n"


def test_read_file_extracts_pdf_text_with_page_range(tmp_path):
    (tmp_path / "s.pdf").write_bytes(FIXTURE_PDF.read_bytes())
    t = implementations(tmp_path)
    whole = t["read_file"](path="s.pdf")
    first = t["read_file"](path="s.pdf", pages="1")
    assert "Quarterly report Q1" in whole and "Second page totals" in whole
    assert "Quarterly report Q1" in first and "Second page totals" not in first


def test_read_file_refuses_binary_that_is_not_pdf(tmp_path):
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    with pytest.raises(ToolError):
        implementations(tmp_path)["read_file"](path="img.png")


def test_path_escape_is_a_tool_error(tmp_path):
    with pytest.raises(ToolError):
        implementations(tmp_path)["read_file"](path="../x")
