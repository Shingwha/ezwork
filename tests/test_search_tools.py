"""Glob/Grep tool tests."""

from __future__ import annotations

from pathlib import Path

from ezwork.tools import GlobTool, GrepTool


# ---- glob ----


def test_glob_finds_files_by_pattern(tmp_path: Path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("y", encoding="utf-8")
    (tmp_path / "c.txt").write_text("z", encoding="utf-8")
    out = GlobTool().run({"pattern": "*.py", "path": str(tmp_path)})
    assert "a.py" in out and "b.py" in out
    assert "c.txt" not in out
    assert "2" in out.split("\n")[0]  # header count


def test_glob_recursive(tmp_path: Path):
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "deep.py").write_text("x", encoding="utf-8")
    out = GlobTool().run({"pattern": "**/*.py", "path": str(tmp_path)})
    assert "deep.py" in out


def test_glob_ignores_common_dirs(tmp_path: Path):
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    (ignored / "junk.py").write_text("x", encoding="utf-8")
    (tmp_path / "real.py").write_text("y", encoding="utf-8")
    out = GlobTool().run({"pattern": "**/*.py", "path": str(tmp_path)})
    assert "real.py" in out
    assert "junk.py" not in out


def test_glob_no_matches(tmp_path: Path):
    out = GlobTool().run({"pattern": "*.rs", "path": str(tmp_path)})
    assert out.startswith("No files matching")


def test_glob_missing_dir_raises(tmp_path: Path):
    from ezwork.core import ToolError

    try:
        GlobTool().run({"pattern": "*.py", "path": str(tmp_path / "nope")})
        assert False, "should raise"
    except ToolError as e:
        assert e.code == "path_not_found"


# ---- grep ----


def _make_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(
        "import os\n\ndef hello():\n    return 'hi'\n", encoding="utf-8"
    )
    (tmp_path / "b.py").write_text(
        "def world():\n    return 'hello world'\n", encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text("# hello\nplain text\n", encoding="utf-8")
    return tmp_path


def test_grep_content_mode(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run({"pattern": "hello", "path": str(tmp_path), "type": "py"})
    assert "a.py:3" in out and "b.py:2" in out
    assert "match(es)" in out


def test_grep_files_mode(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run({"pattern": "hello", "path": str(tmp_path), "output_mode": "files"})
    assert "a.py" in out and "b.py" in out and "notes.md" in out
    assert "Found" in out


def test_grep_count_mode(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run({"pattern": "hello", "path": str(tmp_path), "output_mode": "count"})
    lines = out.splitlines()
    assert any(l.endswith(":1") for l in lines)  # a.py has 1 'hello'
    assert any(l.endswith(":1") for l in lines)  # b.py has 1 'hello'


def test_grep_type_filter(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run({"pattern": "hello", "path": str(tmp_path), "type": "py"})
    assert "notes.md" not in out


def test_grep_ignore_case(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run(
        {"pattern": "HELLO", "path": str(tmp_path), "type": "py", "ignore_case": True}
    )
    assert "a.py" in out


def test_grep_context_lines(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run({"pattern": "hello", "path": str(tmp_path), "context": 1})
    # context separator '-' lines exist; match separator ':' lines exist
    assert "a.py-2" in out and "a.py:3" in out


def test_grep_single_file(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run({"pattern": "world", "path": str(tmp_path / "b.py")})
    assert "b.py:2" in out


def test_grep_no_matches(tmp_path: Path):
    _make_tree(tmp_path)
    out = GrepTool().run({"pattern": "zzz", "path": str(tmp_path)})
    assert out.startswith("No matches")


def test_grep_ignores_non_text(tmp_path: Path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01hello\x02")
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    out = GrepTool().run({"pattern": "hello", "path": str(tmp_path)})
    assert "blob.bin" not in out
    assert "a.txt" in out
