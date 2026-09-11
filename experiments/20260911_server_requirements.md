# 2026-09-11 — Server gate: what needs a server, with measured numbers

## Measured on the local machine
Hardware: Intel Core2 Duo E8400 (2 cores @ 3.0 GHz), 4,058 MB RAM
(~1.1–1.9 GB free), ~62 GB free on D:, torch 2.6.0+cpu, tokenizers 0.23.1.
Disk (sequential): **read 92.1 MB/s, write 73.7 MB/s**.

### BPE-32K tokenizer training (production path `tokenizer.train([file])`)
| corpus     | bytes      | time     | MB/s | peak RSS |
|------------|-----------:|---------:|-----:|---------:|
| 42 MB mix  | 41.9 M     | 75.8 s   | 0.53 | 415 MB   |
| 120 MB mix | 125.8 M    | 248.7 s  | 0.48 | 730 MB   |
| full 4 GB  | 3,989.9 M  | ~131 min | 0.48 | grows with corpus; unsafe on 4 GB |

BPE training is CPU-bound (0.48 MB/s « 92 MB/s disk). Full-corpus training is
possible locally only if you accept ~2.2 h and elevated memory pressure.

### C1-66 training step (seq 128, batch 1, AdamW, fwd+bwd+step)
- **11.238 s / step**, **11.39 tokens/s**, peak RSS **1,354 MB**
- 1,000 steps ≈ **187.3 min (3.1 h)**
- 1B tokens ≈ **24,389 h (2.8 years)** on this CPU

## Verdict
- CPU local is **sufficient** for: corpus validation, tokenizer training on a
  representative sample, model unit/smoke tests, capacity benchmarking.
- CPU local is **not** sufficient for: real pretraining, training on the full
  4 GB corpus at scale, or running many experiments.

Start a server **only** when we begin real pretraining / need GPU / need many
runs / need a much larger corpus or token cache.

## Required server specification (when we go)
- GPU: 1× NVIDIA A100 40 GB (or RTX 4090 24 GB / A6000 48 GB)
- CPU: 8+ cores, RAM 32–64 GB (for tokenization + dataloaders)
- Disk: 256 GB+ NVMe SSD
- OS: Ubuntu 22.04, CUDA 12.x, PyTorch cu121 build
- Network: enough to pull the 4 GB corpus + raw data once

## Rough time / cost estimate
- A100 40 GB on a 66 M-param model: ~50k–150k tokens/s
  → 1 B tokens ≈ **1.9–5.6 h**; 5 B tokens ≈ **9–28 h**
- On-demand A100 ≈ $1.5–3/h → ~$5–20 per 1 B-token run
- Cheaper: RTX 4090 spot ≈ $0.3–0.6/h → ~$2–8 per run
- Tokenizer on the server CPU (more cores) ≈ minutes, not hours.

## Pre-server checklist
| item | status |
|------|--------|
| model.py (correct init, step-0 loss ≈ 10.4) | done |
| config.py (C1-66 frozen) | done |
| tokenizer validated (round-trip 100%, unk 0%) | done |
| corpus manifest (+ validator) | done |
| token cache format | `data/tokenizer_train_v10/tokens.uint32.bin`, raw uint32 LE — must be rebuilt from the v10 corpus on the server |
| train.py (+ checkpoint resume) | present |
| validation scripts | `validate_v10_corpus.py`, `evaluate_v10.py`, `smoke_test_c166.py`, `bench_c166.py` |
| requirements lock | regenerated, UTF-8 (`requirements-lock.txt`) |
| README | `README.md` |
| clean git commit | see session log |
| cloud run instructions | see below |

## Cloud run outline (to be executed later, NOT now)
1. Provision the GPU instance; install CUDA + `pip install -r requirements-lock.txt`.
2. Sync code via git; upload `data/tokenizer/corpus.txt` + `data/raw/` once.
3. `python data_pipeline/validate_v10_corpus.py` on the server.
4. Rebuild the token cache: `python train.py` (first run builds `tokens.uint32.bin`).
5. Train with checkpointing: `python train.py` (resumable).
6. Pull back only checkpoints/artifacts, never raw data.
