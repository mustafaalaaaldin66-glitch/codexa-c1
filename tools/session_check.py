"""Read-only session-state guard for Codexa C1.

Compares ``experiments/session_state.json`` with the *actual* repository state
so a new session cannot silently drift or claim a clean tree while it is dirty.

Checks (read-only, never writes, never trains):
1. session_state.json parses and has the required keys
2. declared branch == current git branch
3. declared commit_sha == ``git rev-parse HEAD``
4. declared dirty_files == real modified tracked files
5. ``working_tree`` flag agrees with reality (clean vs dirty)
6. HEAD vs ``origin/main`` sync status is reported

Exit code 0 = consistent, 1 = drift detected.

Run:
    .\\.venv\\Scripts\\python.exe tools\\session_check.py
    .\\.venv\\Scripts\\python.exe tools\\session_check.py --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "experiments" / "session_state.json"

REQUIRED_KEYS = (
    "version",
    "updated_utc",
    "branch",
    "checkpoint_sha",
    "commit_sha",
    "working_tree",
    "dirty_files",
    "current_phase",
    "last_successful_command",
    "next_command",
    "blockers",
)

STATE_RELATIVE = "experiments/session_state.json"
PATHSPEC_EXCLUDE = f":(exclude){STATE_RELATIVE}"
# MARKER-G2


def git_raw(*args: str) -> str:
    """Run a read-only git command and return *unmodified* stdout.

    Porcelain output is position-sensitive: a leading space is part of the
    status columns, so it must never be stripped before parsing.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def git(*args: str) -> str:
    """Run a read-only git command and return stripped stdout (scalar queries)."""
    return git_raw(*args).strip()


def git_available() -> bool:
    return bool(git("rev-parse", "--is-inside-work-tree"))
def parse_porcelain(raw: str) -> dict[str, list[str]]:
    """Split ``git status --porcelain`` into modified / staged / untracked."""
    modified: list[str] = []
    staged: list[str] = []
    untracked: list[str] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        index_status, worktree_status = line[0], line[1]
        path = line[3:].strip().strip('"')
        if index_status == "?" and worktree_status == "?":
            untracked.append(path)
            continue
        if index_status not in (" ", "?"):
            staged.append(path)
        if worktree_status == "M":
            modified.append(path)
    return {
        "modified": sorted(modified),
        "staged": sorted(staged),
        "untracked": sorted(untracked),
    }


def compare_state(state: dict, actual: dict) -> list[str]:
    """Return a list of human-readable drift problems (empty == consistent)."""
    problems: list[str] = []

    for key in REQUIRED_KEYS:
        if key not in state:
            problems.append(f"session_state.json missing required key: {key}")

    declared_branch = state.get("branch")
    if declared_branch and declared_branch != actual.get("branch"):
        problems.append(
            f"branch drift: state={declared_branch} actual={actual.get('branch')}"
        )

    declared_checkpoint = state.get("checkpoint_sha") or state.get("commit_sha")
    actual_checkpoint = actual.get("checkpoint_sha")
    if declared_checkpoint and actual_checkpoint and declared_checkpoint != actual_checkpoint:
        problems.append(
            f"checkpoint drift: state={declared_checkpoint[:12]} "
            f"actual={actual_checkpoint[:12]}"
        )

    declared_dirty = sorted(state.get("dirty_files") or [])
    real_dirty = sorted(actual.get("dirty_files") or [])
    if declared_dirty != real_dirty:
        problems.append(f"dirty_files drift: state={declared_dirty} actual={real_dirty}")

    declared_tree = state.get("working_tree")
    real_tree = "clean" if not real_dirty else "dirty"
    if declared_tree and declared_tree != real_tree:
        problems.append(f"working_tree drift: state={declared_tree} actual={real_tree}")

    return problems
def checkpoint_sha() -> str:
    """Last commit that changed code/data, excluding the session-state file."""
    return git("log", "-1", "--format=%H", "--", ".", PATHSPEC_EXCLUDE)


def collect_actual() -> dict:
    """Read-only snapshot of the real repository state."""
    parsed = parse_porcelain(git_raw("status", "--porcelain"))
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "checkpoint_sha": checkpoint_sha(),
        "origin_main": git("rev-parse", "origin/main"),
        "dirty_files": parsed["modified"],
        "staged_files": parsed["staged"],
        "untracked_files": parsed["untracked"],
    }
def sync_state(state: dict, actual: dict, timestamp: str) -> dict:
    """Return a copy of ``state`` with the mechanical fields refreshed.

    Only machine-derived fields are touched; narrative fields
    (current_phase, commands, blockers, notes) are left to the human/agent.
    """
    updated = dict(state)
    updated["version"] = state.get("version", "session-state-v1")
    updated["updated_utc"] = timestamp
    updated["branch"] = actual["branch"]
    updated["checkpoint_sha"] = actual["checkpoint_sha"]
    updated["commit_sha"] = actual["head"]
    real_dirty = sorted(actual["dirty_files"])
    updated["working_tree"] = "clean" if not real_dirty else "dirty"
    updated["dirty_files"] = real_dirty
    updated["untracked"] = sorted(actual["untracked_files"])
    return updated
# MARKER-G4
def main() -> int:
    parser = argparse.ArgumentParser(description="Session state guard (read-only by default).")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument(
        "--write",
        action="store_true",
        help="refresh the machine-derived fields in experiments/session_state.json",
    )
    args = parser.parse_args()

    if not STATE_PATH.exists():
        print(f"FAIL: session state not found: {STATE_PATH}")
        return 1

    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL: session state is not valid JSON: {exc}")
        return 1

    if not git_available():
        print("FAIL: not inside a git work tree")
        return 1

    actual = collect_actual()

    if args.write:
        from datetime import datetime, timezone

        updated = sync_state(
            state, actual, datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        STATE_PATH.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        state = updated
        print(f"synced: {STATE_RELATIVE}")
    problems = compare_state(state, actual)
    in_sync = bool(actual["head"]) and actual["head"] == actual["origin_main"]

    report = {
        "state_path": str(STATE_PATH.relative_to(ROOT)),
        "declared_checkpoint_sha": state.get("checkpoint_sha") or state.get("commit_sha"),
        "actual_checkpoint_sha": actual["checkpoint_sha"],
        "declared_commit_sha": state.get("commit_sha"),
        "actual_head": actual["head"],
        "branch": actual["branch"],
        "origin_main": actual["origin_main"],
        "head_matches_origin_main": in_sync,
        "dirty_files": actual["dirty_files"],
        "staged_files": actual["staged_files"],
        "untracked_files": actual["untracked_files"],
        "current_phase": state.get("current_phase"),
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=" * 62)
        print(" CODEXA C1 SESSION STATE CHECK")
        print("=" * 62)
        print(f" branch          : {report['branch']}")
        print(f" checkpoint (sha) : {str(report['declared_checkpoint_sha'])[:12]} "
              f"(actual {report['actual_checkpoint_sha'][:12]})")
        print(f" HEAD            : {report['actual_head'][:12]}")
        print(f" origin/main     : {report['origin_main'][:12]}")
        print(f" HEAD==origin    : {in_sync}")
        print(f" phase           : {report['current_phase']}")
        print(f" dirty (tracked) : {report['dirty_files']}")
        print(f" staged          : {report['staged_files']}")
        print(f" untracked       : {report['untracked_files']}")
        if problems:
            print("-" * 62)
            for problem in problems:
                print(f"  [FAIL] {problem}")
        print("-" * 62)
        print(f"RESULT: {report['status']}")

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())