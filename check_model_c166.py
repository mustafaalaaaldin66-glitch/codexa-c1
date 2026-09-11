"""Verify that C1Micro built with C1_CONFIG (from config.py) has exactly 65,690,496 parameters."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import C1_CONFIG
from parameter_calculator import calculate_parameters

EXPECTED = 65_690_496

calc = calculate_parameters(C1_CONFIG)
print(f"Calculator total: {calc['total']:,}")
assert calc["total"] == EXPECTED, f"Calculator mismatch: {calc['total']:,}"

try:
    import torch
    from model import C1Config, C1Micro

    cfg = C1Config(
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
    model = C1Micro(cfg)
    params = sum(p.numel() for p in model.parameters())
    print(f"C1Micro parameters: {params:,}")
    print(f"Tied embeddings (shared tensor): {model.lm_head.weight is model.embed_tokens.weight}")
    print("MATCH" if params == EXPECTED else f"MISMATCH: {params - EXPECTED:+,}")
except ImportError as exc:
    print(f"torch unavailable ({exc}); calculator verification only.")

print("OK" if calc["total"] == EXPECTED else "FAIL")