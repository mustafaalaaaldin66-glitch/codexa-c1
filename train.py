from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import C1Config, C1Micro
from config import C1_CONFIG


ROOT = Path.cwd()

TOKENIZER_FILE = (
    ROOT / "tokenizer" / "artifacts" / "bpe32k_v10" / "tokenizer.json"
)

CORPUS_FILE = (
    ROOT / "data" / "tokenizer" / "corpus.txt"
)

CACHE_FILE = (
    ROOT / "data" / "tokenizer_train_v10" / "tokens.uint32.bin"
)

RUN_DIR = ROOT / "runs" / "c1_66_first"
RUN_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = RUN_DIR / "checkpoint.pt"
META_FILE = RUN_DIR / "run_meta.json"


# ============================================================
# C1-66 probe run configuration (Nano-13 scale first)
# ============================================================

SEED = 42

SEQ_LEN = 128
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

TRAIN_STEPS = 20
LOG_EVERY = 5
SAVE_EVERY = 10
VAL_EVERY = 10

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Track A probe: a small verification model using the same C1
# building blocks. The frozen C1-66 delivery config is available
# via C1_CONFIG but is not trained on this CPU directly.
PROBE_VOCAB = 32_768
PROBE_LAYERS = 8
PROBE_HIDDEN = 384
PROBE_Q_HEADS = 6
PROBE_KV_HEADS = 3
PROBE_HEAD_DIM = 64
PROBE_FFN = 1_152


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_token_cache(tokenizer: Tokenizer) -> np.ndarray:
    if CACHE_FILE.exists():
        print(f"Using token cache: {CACHE_FILE}")
        return np.fromfile(CACHE_FILE, dtype=np.uint32)

    print("Building token cache...")

    ids: list[int] = []

    with CORPUS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            encoded = tokenizer.encode(line)
            ids.extend(encoded.ids)

    if not ids:
        raise RuntimeError("Token cache is empty.")

    array = np.asarray(ids, dtype=np.uint32)
    array.tofile(CACHE_FILE)

    print(f"Token count: {len(array):,}")
    print(f"Cache: {CACHE_FILE}")

    return array


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:

    max_start = len(tokens) - seq_len - 1

    starts = np.random.randint(
        0,
        max_start + 1,
        size=batch_size,
    )

    x = np.stack(
        [tokens[s : s + seq_len] for s in starts]
    )

    y = np.stack(
        [tokens[s + 1 : s + seq_len + 1] for s in starts]
    )

    return (
        torch.from_numpy(x.astype(np.int64)).to(device),
        torch.from_numpy(y.astype(np.int64)).to(device),
    )


def evaluate(
    model: C1Micro,
    tokens: np.ndarray,
    device: torch.device,
    batches: int = 10,
) -> float:

    model.eval()

    losses = []

    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(
                tokens,
                BATCH_SIZE,
                SEQ_LEN,
                device,
            )

            output = model(x, labels=y)
            losses.append(float(output["loss"]))

    model.train()

    return sum(losses) / len(losses)


def save_checkpoint(
    model: C1Micro,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
) -> None:

    payload = {
        "step": step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(CHECKPOINT_FILE)


def main() -> None:
    set_seed(SEED)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    device = torch.device("cpu")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    vocab_size = tokenizer.get_vocab_size()

    tokens = build_token_cache(tokenizer)

    split = int(len(tokens) * 0.95)

    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    if len(train_tokens) <= SEQ_LEN:
        raise RuntimeError("Training corpus is too small.")

    # Probe config (C1 building blocks, small scale).
    config = C1Config(
        vocab_size=min(vocab_size, PROBE_VOCAB),
        hidden_size=PROBE_HIDDEN,
        num_layers=PROBE_LAYERS,
        num_q_heads=PROBE_Q_HEADS,
        num_kv_heads=PROBE_KV_HEADS,
        head_dim=PROBE_HEAD_DIM,
        intermediate_size=PROBE_FFN,
        max_seq_len=SEQ_LEN,
        rope_theta=C1_CONFIG.rope_theta,
    )

    model = C1Micro(config).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 70)
    print("CODEXA C1 — v10 PROBE TRAINING RUN (Nano-13 scale)")
    print("=" * 70)
    print(f"Corpus          : {CORPUS_FILE}")
    print(f"Tokenizer       : {TOKENIZER_FILE}")
    print(f"Vocab size      : {vocab_size:,}")
    print(f"Probe parameters: {params:,}")
    print(f"C1-66 target    : {C1_CONFIG.num_layers}L/{C1_CONFIG.hidden_size}h "
          f"(65,690,496) — NOT trained here")
    print(f"Train tokens    : {len(train_tokens):,}")
    print(f"Val tokens      : {len(val_tokens):,}")
    print(f"Sequence length : {SEQ_LEN}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Grad accumulation: {GRAD_ACCUM_STEPS}")
    print(f"Steps           : {TRAIN_STEPS}")
    print(f"Device          : {device}")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    start_step = 0
    best_val = math.inf

    if CHECKPOINT_FILE.exists():
        print("Resuming checkpoint...")

        checkpoint = torch.load(
            CHECKPOINT_FILE,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_step = int(
            checkpoint["step"]
        ) + 1

        best_val = float(
            checkpoint["best_val"]
        )

        print(f"Resume step: {start_step}")

    META_FILE.write_text(
        json.dumps(
            {
                "tokenizer": str(TOKENIZER_FILE.relative_to(ROOT)),
                "vocab_size": vocab_size,
                "probe_parameters": params,
                "delivery_target": "C1-66 = 65,690,496",
                "train_tokens": int(len(train_tokens)),
                "val_tokens": int(len(val_tokens)),
                "seq_len": SEQ_LEN,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "steps": TRAIN_STEPS,
                "learning_rate": LEARNING_RATE,
                "probe_hidden_size": PROBE_HIDDEN,
                "probe_num_layers": PROBE_LAYERS,
                "probe_num_q_heads": PROBE_Q_HEADS,
                "probe_num_kv_heads": PROBE_KV_HEADS,
                "probe_intermediate_size": PROBE_FFN,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    model.train()

    started = time.perf_counter()

    for step in range(start_step, TRAIN_STEPS):

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):

            x, y = get_batch(
                train_tokens,
                BATCH_SIZE,
                SEQ_LEN,
                device,
            )

            output = model(
                x,
                labels=y,
            )

            loss = output["loss"]
            assert loss is not None

            accumulated_loss += float(loss)

            (loss / GRAD_ACCUM_STEPS).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        if step % LOG_EVERY == 0:
            elapsed = time.perf_counter() - started
            avg_loss = (
                accumulated_loss
                / GRAD_ACCUM_STEPS
            )

            tokens_seen = (
                (step - start_step + 1)
                * BATCH_SIZE
                * SEQ_LEN
                * GRAD_ACCUM_STEPS
            )

            tok_sec = (
                tokens_seen / elapsed
                if elapsed
                else 0.0
            )

            print(
                f"step={step:04d} "
                f"loss={avg_loss:.4f} "
                f"tokens/s={tok_sec:.1f}"
            )

        if step % VAL_EVERY == 0:
            val_loss = evaluate(
                model,
                val_tokens,
                device,
            )

            print(
                f"[VAL] step={step:04d} "
                f"loss={val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    best_val,
                )
                print("Saved BEST checkpoint.")

        elif step % SAVE_EVERY == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                best_val,
            )

    save_checkpoint(
        model,
        optimizer,
        TRAIN_STEPS - 1,
        best_val,
    )

    print("=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Checkpoint          : {CHECKPOINT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()