# CODEXA C1 — SESSION RESUME
# V10 FREEZE
Date: 2026-09-08
Authority: CODEXA_C1_MASTER_PLAN_v10
Delivery model: C1-66 (65,690,496) — NOT 604M.
604M/MobileLLM-600 reference-only, forbidden as training target.
Probe models: Nano-13 and Proxy-21.
Do not resume the 39.33M run or any run with step-0 loss != ~10.4.
E8400 is not a trainer.
Next: production tokenizer corpus, then model.py.

# Last updated: 2026-09-06

## ENVIRONMENT
OS: Windows
Python: 3.11.9 64-bit
Virtual environment: .venv
CPU: Intel Core 2 Duo E8400 @ 3.00GHz
CPU cores/logical processors: 2/2
Device: CPU
PyTorch: 2.6.0+cpu
CUDA: False

## HOW TO RESUME
# From the project directory:
.\.venv\Scripts\Activate.ps1

# Verify:
python --version
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

## PROJECT STATE
Project: Codexa C1
Current language scope: Arabic only
Final goal: Arabic + English
Do NOT introduce English work until the Arabic pipeline is verified.

## COMPLETED
- GitHub repository connected and push verified.
- Arabic normalization pipeline established.
- FineWeb-2 Arabic source approved for current pipeline.
- DuckDB selected for FineWeb-2 ingestion on this CPU.
- PyArrow 14.0.2 installed because newer PyArrow caused illegal-instruction failure on this CPU.
- datasets 2.19.0 installed for compatibility.
- Arabic corpus pipeline tested.
- 20K corpus test completed: 20,208 accepted samples.
- Source-row boundary checkpoint/resume design established.
- BPE-32K tokenizer trained successfully.
- Vocabulary size: 32,768.
- Token cache built successfully.

## TOKENIZER
Tokenizer:
tokenizer/artifacts/bpe32k/tokenizer.json

Vocabulary:
32,768

## TOKEN CACHE
Cache:
data/tokenizer_train_v9_1/tokens.uint32.bin

Token count:
946,225

Training split:
898,913 tokens

Validation split:
47,312 tokens

## FIRST REAL TRAINING
Model parameters:
39,330,304

Sequence length:
128

Batch size:
1

Gradient accumulation:
8

Original configured steps:
500

Device:
cpu

First recorded result:
step=0000
loss=478.1236
val_loss=473.4499
tokens/s=47.2

Checkpoint:
BEST checkpoint was successfully saved.

## STOP CONDITION
Training was manually interrupted with Ctrl+C during backward().
This was NOT a model compilation/configuration failure.

## IMPORTANT
Do not restart the 500-step run blindly.
At ~47.2 tokens/s, the original 500-step run would take roughly hours on this CPU.

## NEXT ACTION
Reduce STEPS from 500 to 20.

PowerShell command:
(Get-Content .\train.py) -replace 'STEPS = 500', 'STEPS = 20' | Set-Content .\train.py

Verify:
Select-String -Path .\train.py -Pattern 'STEPS ='

Then run:
python .\train.py

## NEXT OBJECTIVE
20-step calibration:
- confirm loss decreases
- measure real tokens/sec
- confirm checkpoint saving
- confirm training loop stability
- decide whether to optimize the training configuration before a longer run

Do not delete existing checkpoints.
Do not rebuild the tokenizer.
Do not rebuild the token cache unless a verification proves it is corrupted.

## ARCHITECTURE TARGETS
REVOKED as delivery.
Final C1 target:
~604M parameters
40 Transformer layers
hidden size 1152
18 Q heads
6 KV heads
head_dim 64
GQA
SwiGLU intermediate 3072
RMSNorm
RoPE
tied embeddings
vocab 32768
context target 4096

C1-Micro target:
~346M parameters

Current first-real-training model:
~39.33M parameters
This is a training/verification model, not the final C1 model.

## PRINCIPLE
Do not chase parameter count.
Measure:
- quality
- loss
- validation loss
- tokens/sec
- step time
- RAM
- checkpoint/resume
- generation
- FP32/BF16/FP16/INT8/INT4 where hardware permits
- ONNX Runtime where useful

## GIT
Repository:
https://github.com/mustafaalaaaldin66-glitch/codexa-c1.git

Last previously confirmed commit:
aab0667

Before pushing, always inspect:
git status
git diff

Then:
git add .
git commit -m "docs: save Codexa C1 training session state"
git push

## RESUME RULE
When returning to this project:
1. Open PowerShell.
2. cd to D:\codexa-c1 if not already there.
3. Activate .venv:
   .\.venv\Scripts\Activate.ps1
4. Read this file.
5. Check git status.
6. Continue from NEXT ACTION.
7. Never repeat completed setup without a reason.

## END OF SESSION
The project is paused after the first real training attempt.
No need to restart previous completed stages.

## SESSION 2026-09-11 — v10 tokenizer validated + model init fixed + server gate

Baseline: HEAD == origin/main == 316bf5e. Machine: E8400 2 cores, 4 GB RAM.

DONE
- Corpus v10 validated byte-for-byte (validate_v10_corpus.py -> PASS):
  corpus 3,989,915,552 B; ar 61,825 lines; arz 400,000 lines; sha256 all match.
- BPE feasibility measured (production path): 0.48-0.53 MB/s, peak RSS 415 MB
  @42 MB and 730 MB @120 MB. Full 4 GB would take ~131 min and more RAM.
- v10 tokenizer trained on a proportional 120 MB sample -> bpe32k_v10:
  vocab 32,768, all 12 special tokens registered. Legacy bpe32k untouched.
- v10 tokenizer accepted (evaluate_v10.py -> PASS): 0 round-trip failures,
  0.0000% unknown, tokens/word improves on real holdout
  (ar_msa 1.4255->1.3589, arz_egy 1.5592->1.4105).
- BUG FIXED in model.py: default nn.Embedding init N(0,1) made step-0 loss
  ~267. Added LLaMA-style init (std 0.02 + depth scaling). Step-0 loss is now
  ~10.46 (~ln 32768 = 10.397). This resolves the old "loss != ~10.4" warning.
- C1-66 smoke test OK; pytest 17 passed; check_model MATCH 65,690,496.
- C1-66 CPU step benchmark: 11.238 s/step, 11.39 tok/s, peak RSS 1,354 MB.

DECISION
- Do NOT train the full 4 GB corpus or C1-66 on this CPU. Server needed only
  for real pretraining / GPU / many runs. See
  experiments/20260911_server_requirements.md.

KNOWN CAVEAT
- ~3% of ar.txt lines contain double spaces (older normalizer vintage);
  cosmetic for tokenization, corpus NOT rebuilt.

NEXT
- Commit the validated tokenizer. Keep everything else local.
