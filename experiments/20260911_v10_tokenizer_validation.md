# 2026-09-11 — v10 tokenizer training feasibility, validation, and server gate

## 1. Baseline state (verified)
- HEAD at the time of the probe: 316bf5e.
- C1-66 = 65,690,496 parameters (calculator == C1Micro == MATCH, tied embeddings True).
- pytest: 17 passed.
- v10 corpus + manifest verified byte-for-byte (see section 2).
- Machine: Intel Core2 Duo E8400 (2 cores), 4,058 MB RAM total, ~1.1–1.6 GB free, ~62 GB disk free.

## 2. Corpus validation (data_pipeline/validate_v10_corpus.py)
Command: `.\.venv\Scripts\python.exe data_pipeline\validate_v10_corpus.py`
Result: **PASS** (exit 0)
- manifest JSON valid, required keys present
- corpus.txt = 3,989,915,552 bytes (sha256 matches manifest)
- ar.txt  = 2,806,077,399 bytes, 61,825 lines, sha256 match
- arz.txt = 1,183,838,153 bytes, 400,000 lines, sha256 match
- no leftover lock/tmp builder files
Not rebuilt, parquet not re-read.

## 3. BPE training feasibility (tokenizer/measure_bpe_feasibility.py)
Production code path `tokenizer.train([file])`, vocab 32,768, min_frequency 2.

| corpus    | bytes      | time     | MB/s | peak RSS | vocab |
|-----------|-----------:|---------:|-----:|---------:|------:|
| 8 MB head | 8.4 M      | 11.8 s   | 0.68 | 563 MB   | 32,768 |
| 32 MB head| 33.6 M     | 59.0 s   | 0.54 | 1,792 MB*| 32,768 |
| 42 MB mix | 41.9 M     | 75.8 s   | 0.53 | 415 MB   | 32,768 |
| 120 MB mix| 125.8 M    | 248.7 s  | 0.48 | 730 MB   | 32,768 |

*The 32 MB row used `train_from_iterator` with one giant document and is NOT
representative; the file path is the production path and peaks far lower.

**Full 4 GB corpus on this box:** extrapolated ≈ **131 min (~2.2 h)** at
0.48 MB/s, and peak RSS grows with corpus size (415 MB @ 42 MB → 730 MB @
120 MB). On a 4 GB machine with <1.6 GB free this is not safe.
=> Full-corpus tokenizer training is **deferred to a server**.

## 4. Decision — train on a proportional 120 MB sample locally
`tokenizer/build_v10_sample.py --mb 120` builds a language-proportional sample
(88.5 MB ar + 37.3 MB arz) so Egyptian dialect is represented. A naive prefix
would have been 100% MSA because corpus.txt = ar.txt then arz.txt.
Sample: 125,825,428 bytes, sha256 a9f22e360bb891f35ddb8f8e009947fd7e4390bf0d06262b0a8324f119c1ec96.

## 5. v10 tokenizer training (tokenizer/train_bpe_v10.py --corpus runs/v10_sample_120.txt)
- artifact: tokenizer/artifacts/bpe32k_v10/tokenizer.json (new; legacy bpe32k untouched)
- meta:     tokenizer/artifacts/bpe32k_v10/meta.json (vocab, corpus sha256, normalizer, specials)
- vocab: 32,768; **all 12 special tokens registered as added tokens** (legacy only had 4)
- normalization idempotency check: 6 / 200 sampled MSA lines differed — all due to
  double spaces (U+0020 U+0020) that `normalize_text` collapses. The MSA file
  (ar.txt, built 2026-09-09) predates the current normalizer revision. This is a
  data-vintage issue, cosmetic for tokenization, and the corpus is NOT rebuilt.

## 6. Tokenizer sample-probe evaluation (tokenizer/evaluate_v10.py)
Result: **PASS** (exit 0). Report: data/tokenizer_eval/report_v10.json
- round-trip failures (v10): **0** on every dataset (curated + real holdout)
- unknown rate (v10): **0.0000%** everywhere
- tokens/word vs legacy bpe32k:
  - holdout/ar_msa  : 1.4255 -> **1.3589** (better)
  - holdout/arz_egy: 1.5592 -> **1.4105** (better)
  - curated/english: 3.6744 -> 2.6279 (better)
- holdout = TAIL slice of ar.txt / arz.txt (training used the HEAD) => disjoint.

The sample-probe criteria passed: round-trip 100%, unk ~0, no regression on the
reported holdouts, vocab 32768, evaluation saved, tests pass (17 passed). This
does **not** promote the sample artifact to the production tokenizer; production
acceptance requires retraining on the full corpus on an external server and
rerunning this evaluation.

## 7. Model smoke test (smoke_test_c166.py)
Instantiate C1-66 -> forward loss finite -> backward grads > 0 -> checkpoint
save/load exact -> tied embeddings. No long training started on CPU.

## 8. CPU training benchmark (bench_c166.py)
One low-priority step at sequence length 64 completed without a crash:

- 4.606 seconds/step
- 13.90 tokens/second
- peak RSS: 1,332 MB
- 1,000 steps: approximately 76.8 minutes
- 1B tokens: approximately 19,990 hours

This confirms that the local CPU is suitable for smoke tests only, not real
pretraining.

## 9. Server gate
A server is required only for: full-corpus tokenizer training (2.2 h, RAM),
real pretraining, GPU work, or many experiments. The server command and exact
requirements must be documented before the external run is started.
