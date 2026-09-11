"""Build a small, language-proportional sample of the v10 corpus.

The v10 corpus is a plain concatenation of data/tokenizer_train_v10/ar.txt
followed by arz.txt. A naive prefix sample would therefore contain *only*
Modern Standard Arabic and miss the Egyptian dialect entirely. This tool
takes a byte budget from each language file in proportion to its share of
the full corpus, producing a representative mixed sample.

The sample is written under runs/ (git-ignored) and is meant only for
tokenizer-training probes. It never modifies the real corpus files.

Run:
    .\\.venv\\Scripts\\python.exe tokenizer\\build_v10_sample.py --mb 40
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V10 = PROJECT_ROOT / "data" / "tokenizer_train_v10"
DEFAULT_OUT = PROJECT_ROOT / "runs" / "v10_sample.txt"

LANGUAGES = ("ar", "arz")


def take_head(path: Path, byte_budget: int, out) -> tuple[int, int]:
    """Copy whole lines from path until byte_budget is reached."""
    written = 0
    lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            size = len(line.encode("utf-8"))
            if written + size > byte_budget:
                break
            out.write(line)
            written += size
            lines += 1
    return written, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=int, default=40)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sizes = {}
    for language in LANGUAGES:
        path = V10 / f"{language}.txt"
        if not path.exists():
            print(f"FAIL: missing {path}")
            return 1
        sizes[language] = path.stat().st_size

    grand = sum(sizes.values())
    total_budget = args.mb * 1024 * 1024

    args.out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    print(f"Building proportional sample -> {args.out}")
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        stats = {}
        for language in LANGUAGES:
            budget = int(total_budget * sizes[language] / grand)
            written, lines = take_head(V10 / f"{language}.txt", budget, out)
            stats[language] = (written, lines)
            print(f"  {language}: budget={budget:,} written={written:,} lines={lines:,}")

    raw = args.out.read_bytes()
    digest.update(raw)
    print(f"total bytes: {len(raw):,}")
    print(f"sha256     : {digest.hexdigest()}")
    print("SAMPLE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
