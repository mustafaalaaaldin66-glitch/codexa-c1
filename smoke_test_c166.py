"""Short CPU smoke test for the C1-66 model. No long training.

Verifies, in order:
  1. C1Micro built from the frozen C1_CONFIG has exactly 65,690,496 params
  2. tied embeddings (lm_head shares embed_tokens' tensor)
  3. a forward pass produces a finite loss
  4. gradients flow backwards (sum |grad| > 0)
  5. checkpoint save/load round-trips every parameter exactly

Run:
    .\\.venv\\Scripts\\python.exe smoke_test_c166.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import C1_CONFIG
from model import C1Config, C1Micro

EXPECTED_PARAMS = 65_690_496
SEQ_LEN = 64
BATCH = 1


def build_config() -> C1Config:
    return C1Config(
        vocab_size=C1_CONFIG.vocab_size,
        hidden_size=C1_CONFIG.hidden_size,
        num_layers=C1_CONFIG.num_layers,
        num_q_heads=C1_CONFIG.num_q_heads,
        num_kv_heads=C1_CONFIG.num_kv_heads,
        head_dim=C1_CONFIG.head_dim,
        intermediate_size=C1_CONFIG.intermediate_size,
        max_seq_len=C1_CONFIG.max_seq_len,
        rope_theta=C1_CONFIG.rope_theta,
    )


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(2)

    cfg = build_config()
    model = C1Micro(cfg)
    model.train()

    params = sum(p.numel() for p in model.parameters())
    print(f"[1] parameters           : {params:,}")
    assert params == EXPECTED_PARAMS, f"param mismatch: {params:,}"

    tied = model.lm_head.weight is model.embed_tokens.weight
    print(f"[2] tied embeddings      : {tied}")
    assert tied, "embeddings are not tied"

    x = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN))
    labels = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN))
    out = model(x, labels=labels)
    loss = out["loss"]
    print(f"[3] forward loss         : {float(loss):.4f}")
    assert loss is not None and torch.isfinite(loss), "loss is not finite"

    loss.backward()
    grad_sum = sum(
        float(p.grad.norm()) for p in model.parameters() if p.grad is not None
    )
    print(f"[4] backward sum|grad|   : {grad_sum:.4f}")
    assert grad_sum > 0.0, "no gradients flowed"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.pt"
        torch.save({"model": model.state_dict(), "step": 0}, path)

        model2 = C1Micro(cfg)
        checkpoint = torch.load(path, map_location="cpu")
        model2.load_state_dict(checkpoint["model"])

        mismatches = 0
        for (name, p1), (_, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
            if not torch.equal(p1, p2):
                mismatches += 1
                print(f"    mismatch: {name}")
        print(f"[5] checkpoint reload    : {'exact' if mismatches == 0 else 'MISMATCH'}")
        assert mismatches == 0, "checkpoint reload changed weights"

    print("SMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
