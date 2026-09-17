"""Tests for the read-only session-state guard (tools/session_check.py).

These lock the drift-detection logic so a future session cannot make the guard
silently pass while the tree, branch or commit disagree with
experiments/session_state.json.

Run:
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_session_check.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.session_check import (
    REQUIRED_KEYS,
    ROOT,
    collect_actual,
    compare_state,
    parse_porcelain,
    sync_state,
)

STATE_PATH = ROOT / "experiments" / "session_state.json"


def _consistent_state(actual: dict) -> dict:
    state = {key: None for key in REQUIRED_KEYS}
    state.update(
        {
            "version": "session-state-v1",
            "updated_utc": "2026-09-17T00:00:00+00:00",
            "branch": actual["branch"],
            "checkpoint_sha": actual["checkpoint_sha"],
            "commit_sha": actual["head"],
            "working_tree": "clean" if not actual["dirty_files"] else "dirty",
            "dirty_files": list(actual["dirty_files"]),
            "current_phase": "test",
            "last_successful_command": "test",
            "next_command": "test",
            "blockers": [],
        }
    )
    return state


# --------------------------------------------------------------------------
# parse_porcelain
# --------------------------------------------------------------------------

def test_parse_porcelain_modified_only() -> None:
    parsed = parse_porcelain(" M tokenizer/artifacts/bpe32k/tokenizer.json\n")
    assert parsed["modified"] == ["tokenizer/artifacts/bpe32k/tokenizer.json"]
    assert parsed["staged"] == []
    assert parsed["untracked"] == []


def test_parse_porcelain_keeps_path_of_first_line_intact() -> None:
    """Regression: a global strip() used to eat the first char of the path."""
    raw = " M data/heldout/arz_test_heldout.quality.json\n?? tools/\n"
    parsed = parse_porcelain(raw)
    assert parsed["modified"] == ["data/heldout/arz_test_heldout.quality.json"]
    assert parsed["untracked"] == ["tools/"]


def test_parse_porcelain_staged_and_untracked() -> None:
    raw = "M  experiments/session_state.json\n?? tools/\n"
    parsed = parse_porcelain(raw)
    assert parsed["staged"] == ["experiments/session_state.json"]
    assert parsed["untracked"] == ["tools/"]
    assert parsed["modified"] == []


def test_parse_porcelain_ignores_blank_lines() -> None:
    assert parse_porcelain("\n\n") == {"modified": [], "staged": [], "untracked": []}


# --------------------------------------------------------------------------
# compare_state
# --------------------------------------------------------------------------

def test_compare_state_consistent_has_no_problems() -> None:
    actual = collect_actual()
    assert actual["branch"], "git branch must be detectable in this repo"
    assert compare_state(_consistent_state(actual), actual) == []


def test_compare_state_detects_checkpoint_drift() -> None:
    actual = collect_actual()
    state = _consistent_state(actual)
    state["checkpoint_sha"] = "0" * 40
    problems = compare_state(state, actual)
    assert any("checkpoint drift" in problem for problem in problems), problems


def test_compare_state_falls_back_to_commit_sha_when_checkpoint_missing() -> None:
    actual = collect_actual()
    state = _consistent_state(actual)
    state.pop("checkpoint_sha")
    state["commit_sha"] = "0" * 40
    problems = compare_state(state, actual)
    assert any("checkpoint drift" in problem for problem in problems), problems


def test_compare_state_detects_branch_drift() -> None:
    actual = collect_actual()
    state = _consistent_state(actual)
    state["branch"] = "not-a-real-branch"
    problems = compare_state(state, actual)
    assert any("branch drift" in problem for problem in problems), problems


def test_compare_state_detects_dirty_file_drift() -> None:
    actual = dict(collect_actual())
    actual["dirty_files"] = ["some/other/file.py"]
    state = _consistent_state(actual)
    state["dirty_files"] = []
    problems = compare_state(state, actual)
    assert any("dirty_files drift" in problem for problem in problems), problems


def test_compare_state_rejects_clean_claim_on_dirty_tree() -> None:
    actual = dict(collect_actual())
    actual["dirty_files"] = ["tokenizer/artifacts/bpe32k/tokenizer.json"]
    state = _consistent_state(actual)
    state["working_tree"] = "clean"
    problems = compare_state(state, actual)
    assert any("working_tree drift" in problem for problem in problems), problems


def test_compare_state_flags_missing_required_key() -> None:
    actual = collect_actual()
    state = _consistent_state(actual)
    state.pop("blockers")
    problems = compare_state(state, actual)
    assert any("missing required key: blockers" in problem for problem in problems), problems


# --------------------------------------------------------------------------
# sync_state
# --------------------------------------------------------------------------

def test_sync_state_refreshes_mechanical_fields_only() -> None:
    actual = dict(collect_actual())
    actual["checkpoint_sha"] = "a" * 40
    actual["head"] = "b" * 40
    actual["dirty_files"] = ["some/file.py"]
    actual["untracked_files"] = ["new/"]
    state = {
        "version": "session-state-v1",
        "updated_utc": "old",
        "branch": "wrong",
        "checkpoint_sha": "0" * 40,
        "current_phase": "phase-x",
        "last_successful_command": "cmd",
        "next_command": "next",
        "blockers": ["b1"],
        "notes": "keep me",
    }
    updated = sync_state(state, actual, "2026-09-17T00:00:00+00:00")
    assert updated["checkpoint_sha"] == "a" * 40
    assert updated["commit_sha"] == "b" * 40
    assert updated["branch"] == actual["branch"]
    assert updated["working_tree"] == "dirty"
    assert updated["dirty_files"] == ["some/file.py"]
    assert updated["untracked"] == ["new/"]
    # narrative fields must survive untouched
    assert updated["current_phase"] == "phase-x"
    assert updated["blockers"] == ["b1"]
    assert updated["notes"] == "keep me"


def test_sync_state_marks_clean_when_no_dirty_files() -> None:
    actual = dict(collect_actual())
    actual["dirty_files"] = []
    actual["untracked_files"] = []
    updated = sync_state({"version": "session-state-v1"}, actual, "now")
    assert updated["working_tree"] == "clean"
    assert updated["dirty_files"] == []


# --------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------

def test_cli_json_report_shape() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "session_check.py"), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    report = json.loads(result.stdout)
    for key in (
        "declared_checkpoint_sha",
        "actual_checkpoint_sha",
        "actual_head",
        "branch",
        "origin_main",
        "head_matches_origin_main",
        "dirty_files",
        "problems",
        "status",
    ):
        assert key in report, f"missing key in report: {key}"
    assert report["status"] in ("PASS", "FAIL")
    assert isinstance(report["dirty_files"], list)


def test_shipped_session_state_is_valid_json_with_required_keys() -> None:
    if not STATE_PATH.exists():
        pytest.skip("session_state.json not present")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in state]
    assert missing == [], f"session_state.json missing keys: {missing}"
    assert isinstance(state["blockers"], list)