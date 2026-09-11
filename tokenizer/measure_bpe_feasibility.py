"""Measure BPE-32K training feasibility on this machine (read-only probe).

Trains the *same* BPE configuration as tokenizer/train_bpe_v10.py on a
bounded byte sample of the v10 corpus, in memory only. It never writes a
tokenizer artifact and never modifies existing artifacts.

Reports: sample size, wall-clock time, MB/s, peak RSS, achieved vocab size,
and an extrapolation to the full corpus.

Run:
    .\\.venv\\Scripts\\python.exe tokenizer\\measure_bpe_feasibility.py --sample-mb 64
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

CORPUS = PROJECT_ROOT / "data" / "tokenizer" / "corpus.txt"

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>", "<mask>",
    "<ar>", "<en>", "<code>", "<math>",
    "<user>", "<assistant>", "<system>",
]


class PeakRssMonitor:
    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                self.peak = max(self.peak, rss)
            except psutil.Error:
                pass
            self._stop.wait(0.2)

    def __enter__(self) -> "PeakRssMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def build_config(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    return tokenizer, trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS,
                        help="file to train on (default: full v10 corpus)")
    parser.add_argument("--vocab", type=int, default=32768)
    args = parser.parse_args()

    corpus_path: Path = args.corpus
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path
    if not corpus_path.exists():
        print(f"FAIL: corpus missing: {corpus_path}")
        return 1

    try:  # keep the machine responsive on this 2-core box
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (psutil.Error, AttributeError):
        pass

    total_bytes = corpus_path.stat().st_size

    print("=" * 60)
    print(" BPE FEASIBILITY PROBE (production code path)")
    print("=" * 60)
    print(f" corpus          : {corpus_path} ({total_bytes:,} bytes)")
    print(f" vocab target    : {args.vocab:,}")
    free = psutil.virtual_memory().available
    print(f" RAM available   : {free:,} bytes ({free / 1024**2:,.0f} MB)")

    tokenizer, trainer = build_config(args.vocab)

    print("\n training via tokenizer.train([file])...", flush=True)
    with PeakRssMonitor() as monitor:
        start = time.perf_counter()
        tokenizer.train([str(corpus_path)], trainer)
        elapsed = time.perf_counter() - start

    mb = total_bytes / 1024**2
    mbps = mb / elapsed if elapsed else 0.0
    vocab_now = tokenizer.get_vocab_size()
    peak_mb = monitor.peak / 1024**2

    print("-" * 60)
    print(f" elapsed         : {elapsed:,.1f} s")
    print(f" throughput      : {mbps:,.2f} MB/s")
    print(f" peak RSS        : {peak_mb:,.0f} MB")
    print(f" achieved vocab  : {vocab_now:,} / {args.vocab:,}")
    if total_bytes == 3_989_915_552:
        print(" (this WAS the full v10 corpus)")
    else:
        extrapolated = (3_989_915_552 / 1024**2) / mbps if mbps else float("inf")
        print(f" naive linear extrapolation to full corpus: {extrapolated / 60:,.1f} min")
    print("=" * 60)
    print("FEASIBILITY_PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

