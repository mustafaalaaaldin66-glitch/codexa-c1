# CODEXA C1 — SESSION RESUME
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
