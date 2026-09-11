"""Measure C1-66 CPU step time and throughput for capacity planning.

Runs a few real optimiser steps on CPU and reports:
  seconds/step, tokens/second, peak RSS, and extrapolations to 1,000 steps
  and to a 1B-token budget. No checkpoint is written.

Run:
    .\\.venv\\Scripts\\python.exe bench_c166.py --steps 3 --seq 128 --batch 1
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import torch

from config import C1_CONFIG
from model import C1Config, C1Micro


class PeakRss:
    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self.peak = 0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            self._stop.wait(0.2)

    def stop(self) -> None:
        self._stop.set()


def build(seq: int) -> tuple[C1Config, C1Micro]:
    cfg = C1Config(
        vocab_size=C1_CONFIG.vocab_size,
        hidden_size=C1_CONFIG.hidden_size,
        num_layers=C1_CONFIG.num_layers,
        num_q_heads=C1_CONFIG.num_q_heads,
        num_kv_heads=C1_CONFIG.num_kv_heads,
        head_dim=C1_CONFIG.head_dim,
        intermediate_size=C1_CONFIG.intermediate_size,
        max_seq_len=max(seq, C1_CONFIG.max_seq_len),
        rope_theta=C1_CONFIG.rope_theta,
    )
    return cfg, C1Micro(cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(2)

    cfg, model = build(args.seq)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq))
    labels = torch.randint(0, cfg.vocab_size, (args.batch, args.seq))

    def one_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        output = model(x, labels=labels)
        loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss)

    print("warmup step...", flush=True)
    one_step()

    monitor = PeakRss()
    started = time.perf_counter()
    losses = [one_step() for _ in range(args.steps)]
    elapsed = time.perf_counter() - started
    monitor.stop()

    sec_per_step = elapsed / args.steps
    tokens_per_step = args.batch * args.seq
    tokens_per_sec = tokens_per_step / sec_per_step
    peak_mb = monitor.peak / 1024**2

    print("=" * 60)
    print(" C1-66 CPU STEP BENCHMARK")
    print("=" * 60)
    print(f" seq x batch        : {args.seq} x {args.batch} ({tokens_per_step} tok/step)")
    print(f" measured steps     : {args.steps}")
    print(f" loss (first/last)  : {losses[0]:.3f} / {losses[-1]:.3f}")
    print(f" seconds / step     : {sec_per_step:.3f} s")
    print(f" tokens / second    : {tokens_per_sec:,.2f}")
    print(f" peak RSS           : {peak_mb:,.0f} MB")
    print(f" 1,000 steps ~      : {sec_per_step * 1000 / 60:,.1f} min")
    print(f" 1B tokens ~        : {1e9 / tokens_per_sec / 3600:,.1f} h")
    print("=" * 60)
    print("BENCH_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
