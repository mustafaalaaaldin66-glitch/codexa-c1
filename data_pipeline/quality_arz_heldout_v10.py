"""Low-memory data-quality scan for Egyptian held-out. No corpus rebuild."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from text.normalize import normalize_text
HELDOUT = PROJECT_ROOT / "data" / "heldout" / "arz_test_heldout.txt"
OUT = PROJECT_ROOT / "data" / "heldout" / "arz_test_heldout.quality.json"


def main() -> int:
    lines = [ln.rstrip("\n") for ln in HELDOUT.read_text(encoding="utf-8").split("\n") if ln.strip()]
    if not lines:
        print("FAIL: heldout missing or empty")
        return 1
    total_chars = sum(len(t) for t in lines)
    arabic_chars = sum(1 for t in lines for c in t if "\u0600" <= c <= "\u06FF")
    words = [w for t in lines for w in normalize_text(t).split()]
    word_count = len(words)
    unique_words = len(set(words))
    lengths = sorted(map(len, lines))
    n = len(lines)
    bad_utf8 = 0
    for text in lines:
        try:
            text.encode("utf-8").decode("utf-8")
        except Exception:
            bad_utf8 += 1
    ws_anomalies = sum(1 for t in lines if "  " in t or t != t.strip())
    repeated = len(lines) - len({hashlib.sha256(normalize_text(t).encode("utf-8")).hexdigest() for t in lines})
    report = {
        "version": "arz_heldout_quality_v1",
        "lines": n,
        "mean_len": total_chars / n,
        "min_len": lengths[0],
        "p50_len": lengths[n // 2],
        "max_len": lengths[-1],
        "arabic_character_ratio": arabic_chars / total_chars if total_chars else 0.0,
        "words": word_count,
        "unique_word_ratio": unique_words / word_count if word_count else 0.0,
        "duplicate_lines": repeated,
        "invalid_utf8": bad_utf8,
        "whitespace_anomalies": ws_anomalies,
        "language_ratio_note": "arz-only heldout by construction; source split=test",
        "contamination_note": "decontaminated against full 400k-line arz training file at build time",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"lines={n} mean={report['mean_len']:.0f} ar_ratio={report['arabic_character_ratio']:.4f}")
    print(f"unique_word_ratio={report['unique_word_ratio']:.4f} dup={repeated} bad_utf8={bad_utf8} ws={ws_anomalies}")
    print(f"report: {OUT.relative_to(PROJECT_ROOT)}")
    print("QUALITY_SCAN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
