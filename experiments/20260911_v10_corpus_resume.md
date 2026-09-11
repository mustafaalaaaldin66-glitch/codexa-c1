# CODEXA C1 — v10 Corpus Resume

Date: 2026-09-11
Authority: `CODEXA_C1_MASTER_PLAN_v10`
Track A target: `C1-66 = 65,690,496` parameters (frozen)

## State Found

- `data/tokenizer_train_v10/ar.txt` existed (~2.65GB) from the previous session.
- `arz.txt` and `data/manifests/tokenizer_train_v10.json` were missing.
- `train.py` referenced the old v9.1 corpus/tokenizer (39.33M run, forbidden to resume).

## Work Completed This Session

1. **`data_pipeline/build_arz_v10_resume.py`** — new resume script:
   - Builds `arz.txt` only (target 400,000 accepted lines), preserving existing `ar.txt`.
   - Same pipeline rules: `normalize_text`, SHA-256 dedup, tokenizer_eval exclusion, 40..8000 chars, `en=0`.
   - Writes the combined `data/tokenizer/corpus.txt` and `data/manifests/tokenizer_train_v10.json`.
   - Must be run with the venv interpreter: `.venv\Scripts\python.exe` (system python lacks duckdb).
2. **Corpus build completed** — `arz.txt` contains 400,000 accepted samples; the local MSA sample produced 61,825 accepted samples.
3. **C1-66 parameter verification** (`check_model_c166.py`):
   - `parameter_calculator` total: 65,690,496 ✓
   - `C1Micro` with `C1_CONFIG` fields: 65,690,496 ✓ (MATCH)
   - Tied embeddings confirmed (shared tensor).
4. **`train.py` rewritten** for the v10 pipeline:
   - Tokenizer: `tokenizer/artifacts/bpe32k_v10/tokenizer.json`
   - Corpus: `data/tokenizer_train_v10/corpus_combined.txt`
   - Probe run (Nano-13 scale: 8L/384h, 6Q/3KV heads) — C1-66 is NOT trained directly on this CPU.
   - 20-step calibration, checkpoint/resume in `runs/c1_66_first/`.
5. **`tokenizer/train_bpe_v10.py`** — new v10 BPE-32K trainer, writes to a separate artifact dir (`bpe32k_v10`) so the old T0 tokenizer is untouched.
6. **`model.py` left unchanged** (a trial C1_CONFIG aliasing was reverted to keep the module self-contained).

## Completed Verification

- `data/manifests/tokenizer_train_v10.json` was created successfully.
- Combined corpus size: 3,989,915,552 bytes.
- Combined corpus SHA-256: `60a8fa6583fd215c59ef01ae5163f32cc889f53b301bf5172ac96b154b79d93b`.
- C1-66 parameter check passed: 65,690,496 parameters with tied embeddings.
- Test suite passed: 17 tests.
- Large local corpora and raw data remain outside GitHub by design.
- Kaggle is not required for the current local validation and probe work.

## Pending / Next

- Decide whether the 61,825-sample MSA corpus is sufficient for the tokenizer probe or obtain a larger approved MSA source before production training.
- Train the v10 tokenizer: `.venv\Scripts\python.exe tokenizer\train_bpe_v10.py`
- Evaluate it: `.venv\Scripts\python.exe tokenizer\evaluate.py` (after pointing it at the v10 artifact or copying).
- Build the token cache and run the 20-step probe: `.venv\Scripts\python.exe train.py`
- Acceptance for step-0 loss ≈ 10.4 (ln 32768 ≈ 10.397) before any longer run.

## Rules Honored

- 39.33M run not resumed; checkpoints untouched.
- Old tokenizer artifacts not rebuilt.
- English target remains 0.