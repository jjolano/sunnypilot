from __future__ import annotations

from scripts.ci.check_touch_points import missing_touch_points


def _always_exists(path: str) -> bool:
    return True


def _never_exists(path: str) -> bool:
    return False


def test_missing_upstream_touch_point_fails() -> None:
    changed = [("M", "selfdrive/controls/lib/foo.py", None)]
    touched: set[str] = set()
    assert missing_touch_points(
        changed,
        touched,
        _always_exists,
        "docs/touch-points.md",
    ) == ["selfdrive/controls/lib/foo.py"]


def test_listed_touch_point_passes() -> None:
    changed = [("M", "selfdrive/controls/lib/foo.py", None)]
    touched = {"selfdrive/controls/lib/foo.py"}
    assert missing_touch_points(
        changed,
        touched,
        _always_exists,
        "docs/touch-points.md",
    ) == []


def test_new_upstream_absent_file_ignored() -> None:
    changed = [("A", "some/new/file.py", None)]
    touched: set[str] = set()
    assert missing_touch_points(
        changed,
        touched,
        _never_exists,
        "docs/touch-points.md",
    ) == []


def test_touch_points_md_ignored() -> None:
    changed = [("M", "docs/touch-points.md", None)]
    touched: set[str] = set()
    assert missing_touch_points(
        changed,
        touched,
        _always_exists,
        "docs/touch-points.md",
    ) == []


def test_rename_checks_both_paths() -> None:
    changed = [("R100", "old/path.py", "new/path.py")]
    touched: set[str] = set()
    missing = missing_touch_points(
        changed,
        touched,
        lambda p: p == "old/path.py",
        "docs/touch-points.md",
    )
    assert missing == ["old/path.py"]
