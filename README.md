# Codexa C1

Arabic-first decoder-only language model project. Delivery target **C1-66**
(65,690,496 parameters). All heavyweight work runs locally on CPU until a
server is justified by measured numbers (see `experiments/`).

## Environment

- Windows, Python 3.11 (`.venv`)
- PyTorch CPU, `tokenizers==0.23.1`
- Always use the project interpreter: `.\.venv\Scripts\python.exe`

Verify:

```powershell
python --version
python -c "import torch, tokenizers; print(torch.__version__, tokenizers.__version__)"
```

## Data (local only, never pushed)

- `data/tokenizer/corpus.txt` — v10 tokenizer corpus (3,989,915,552 bytes)
- `data/tokenizer_train_v10/ar.txt` (61,825 lines), `arz.txt` (400,000 lines)
- `data/manifests/tokenizer_train_v10.json` — manifest (sizes + sha256)
- `tokenizer/artifacts/bpe32k_v10/` — sample-probe tokenizer + meta; production
	retraining on the full corpus is deferred to an external server

Validate the corpus against its manifest (read-only):

```powershell
.\.venv\Scripts\python.exe data_pipeline\validate_v10_corpus.py
```

## Tokenizer v10

Train (writes a NEW artifact; the legacy `bpe32k` is never touched):

```powershell
# full corpus (deferred to an external server; RAM-heavy on this box):
.\.venv\Scripts\python.exe tokenizer\train_bpe_v10.py

# proportional sample (recommended locally, ~4 min):
.\.venv\Scripts\python.exe tokenizer\build_v10_sample.py --mb 120
.\.venv\Scripts\python.exe tokenizer\train_bpe_v10.py --corpus runs\v10_sample_120.txt
```

Evaluate v10 against the legacy tokenizer (round-trip, unk rate, tokens/word):

```powershell
.\.venv\Scripts\python.exe tokenizer\evaluate_v10.py
```

Inspect any tokenizer artifact:

```powershell
.\.venv\Scripts\python.exe tokenizer\inspect_tokenizer.py
```

Measure BPE feasibility before training a large corpus:

```powershell
.\.venv\Scripts\python.exe tokenizer\measure_bpe_feasibility.py --sample-mb 16
```

## Model (C1-66)

```powershell
.\.venv\Scripts\python.exe check_model_c166.py     # param count + tied embeddings
.\.venv\Scripts\python.exe smoke_test_c166.py      # forward/backward/checkpoint
.\.venv\Scripts\python.exe bench_c166.py           # step time + tokens/s
.\.venv\Scripts\python.exe -m pytest -q            # unit tests
```

## Notes

- Do not rebuild the v10 corpus or delete any checkpoint/artifact.
- Raw data and the 4 GB corpus stay local and are git-ignored.
- `runs/` holds local logs and sample files and is git-ignored.
- Real pretraining and full-corpus tokenizer training require a GPU/RAM server;
	the CPU benchmark is recorded under `experiments/`.
