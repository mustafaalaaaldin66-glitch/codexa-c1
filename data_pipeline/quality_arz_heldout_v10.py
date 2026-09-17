"""Low-memory data-quality scan for the Egyptian held-out set.

Produces a small, reproducible JSON report: length distribution, duplicate and
repeated-line counts, invalid UTF-8, whitespace anomalies, Arabic character
ratio, unique-word ratio, plus source balance and contamination counts read
from the existing manifests. Never rebuilds the corpus and never loads a large
file into memory beyond the held-out text itself.

Run:
    .\\.venv\\Scripts\\python.exe data_pipeline\\quality_arz_heldout_v10.py
"""
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
HELDOUT_MANIFEST = PROJECT_ROOT / "data" / "heldout" / "arz_test_heldout.manifest.json"
TRAIN_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "tokenizer_train_v10.json"

QUALITY_VERSION = "arz_heldout_quality_v2"
ARABIC_RANGE = ("\u0600", "\u06FF")


def is_arabic(char: str) -> bool:
    return ARABIC_RANGE[0] <= char <= ARABIC_RANGE[1]


def scan_lines(lines: list[str]) -> dict:
    """Pure text-quality metrics for normalized held-out lines."""
    texts = [text for text in (normalize_text(line) for line in lines) if text]
    if not texts:
        return {"lines": 0}

    lengths = sorted(len(text) for text in texts)
    count = len(texts)
    total_chars = sum(lengths)
    arabic_chars = sum(1 for text in texts for char in text if is_arabic(char))
    words = [word for text in texts for word in text.split()]
    word_count = len(words)

    hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts}
    repeated = count - len(hashes)

    bad_utf8 = 0
    for text in texts:
        try:
            text.encode("utf-8").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            bad_utf8 += 1

    whitespace_anomalies = sum(
        1 for text in texts if "  " in text or text != text.strip() or "\t" in text
    )

    return {
        "lines": count,
        "mean_len": total_chars / count,
        "min_len": lengths[0],
        "p50_len": lengths[count // 2],
        "p90_len": lengths[int(count * 0.9)],
        "max_len": lengths[-1],
        "chars": total_chars,
        "arabic_character_ratio": arabic_chars / total_chars if total_chars else 0.0,
        "words": word_count,
        "unique_words": len(set(words)),
        "unique_word_ratio": len(set(words)) / word_count if word_count else 0.0,
        "duplicate_lines": repeated,
        "duplicate_rate": repeated / count,
        "invalid_utf8": bad_utf8,
        "whitespace_anomalies": whitespace_anomalies,
    }
def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def source_balance(manifest: dict | None) -> dict:
    """Byte share per language from the v10 training manifest (tiny file)."""
    if not manifest:
        return {"available": False}
    counts = manifest.get("counts", {})
    total = sum(int(entry.get("bytes", 0)) for entry in counts.values())
    share = {
        language: (int(entry.get("bytes", 0)) / total if total else 0.0)
        for language, entry in counts.items()
    }
    msa = share.get("ar", 0.0)
    egyptian = share.get("arz", 0.0)
    return {
        "available": True,
        "bytes_by_language": {k: int(v.get("bytes", 0)) for k, v in counts.items()},
        "share_by_language": share,
        "msa_share": msa,
        "egyptian_share": egyptian,
        "msa_to_egyptian_ratio": (msa / egyptian) if egyptian else 0.0,
    }


def contamination(manifest: dict | None) -> dict:
    """Contamination counters recorded by the held-out builder."""
    if not manifest:
        return {"available": False}
    stats = manifest.get("stats", {})
    return {
        "available": True,
        "train_overlap_rejected": int(stats.get("train_overlap", 0)),
        "eval_overlap_rejected": int(stats.get("eval_overlap", 0)),
        "train_hashes_checked": int(manifest.get("train_hashes_checked", 0)),
        "dup_rejected": int(stats.get("dup", 0)),
        "contamination_count": int(stats.get("train_overlap", 0))
        + int(stats.get("eval_overlap", 0)),
        "heldout_lines_final": int(stats.get("accepted", 0)),
    }
def main() -> int:
    if not HELDOUT.exists():
        print(f"FAIL: held-out file not found: {HELDOUT}")
        return 1

    lines = [ln for ln in HELDOUT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    metrics = scan_lines(lines)
    if not metrics.get("lines"):
        print("FAIL: held-out file has no usable lines")
        return 1

    report = {
        "version": QUALITY_VERSION,
        "heldout": {
            "path": str(HELDOUT.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(HELDOUT.read_bytes()).hexdigest(),
        },
        "text": metrics,
        "source_balance": source_balance(read_json(TRAIN_MANIFEST)),
        "contamination": contamination(read_json(HELDOUT_MANIFEST)),
        "language_ratio_note": "held-out is arz-only by construction (FineWeb2 arz_Arab, split=test)",
        "status": "PASS"
        if metrics["duplicate_lines"] == 0 and metrics["invalid_utf8"] == 0
        else "FAIL",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f" lines            : {metrics['lines']}")
    print(f" mean/p50/max len : {metrics['mean_len']:.0f} / {metrics['p50_len']} / {metrics['max_len']}")
    print(f" arabic char ratio: {metrics['arabic_character_ratio']:.4f}")
    print(f" unique word ratio: {metrics['unique_word_ratio']:.4f}")
    print(f" duplicates       : {metrics['duplicate_lines']} (rate {metrics['duplicate_rate']:.4f})")
    print(f" invalid utf8     : {metrics['invalid_utf8']}")
    print(f" whitespace anom. : {metrics['whitespace_anomalies']}")
    balance = report["source_balance"]
    if balance.get("available"):
        print(f" msa/egyptian     : {balance['msa_share']:.3f} / {balance['egyptian_share']:.3f} "
              f"(ratio {balance['msa_to_egyptian_ratio']:.2f})")
    contam = report["contamination"]
    if contam.get("available"):
        print(f" contamination    : {contam['contamination_count']} rejected "
              f"of {contam['train_hashes_checked']} training hashes checked")
    print(f" report           : {OUT.relative_to(PROJECT_ROOT)}")
    print(f"RESULT: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
