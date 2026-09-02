from config import C1_CONFIG, ModelConfig


def calculate_parameters(config: ModelConfig) -> dict[str, int]:
    """
    Independent parameter calculator for the Codexa decoder-only model.

    Assumptions:
    - bias-free Q/K/V/O projections
    - GQA
    - SwiGLU
    - two RMSNorms per Transformer block
    - tied token embedding + LM head
    - final RMSNorm
    - no learned positional embeddings
    """

    config.validate()

    d = config.hidden_size
    q = config.num_q_heads
    kv = config.num_kv_heads
    hd = config.head_dim
    ffn = config.intermediate_size
    vocab = config.vocab_size
    layers = config.num_layers

    # -------------------------
    # Attention
    # -------------------------
    q_proj = d * d
    k_proj = d * (kv * hd)
    v_proj = d * (kv * hd)
    o_proj = d * d

    attention = q_proj + k_proj + v_proj + o_proj

    # -------------------------
    # SwiGLU
    #
    # gate + up + down
    # -------------------------
    mlp = 3 * d * ffn

    # -------------------------
    # RMSNorm
    # -------------------------
    norms = 2 * d

    # Optional QK-Norm
    qk_norm = 0

    if config.use_qk_norm:
        qk_norm = d + (kv * hd)

    # -------------------------
    # Transformer block
    # -------------------------
    block = attention + mlp + norms + qk_norm

    # -------------------------
    # All Transformer blocks
    # -------------------------
    transformer = layers * block

    # -------------------------
    # Token embedding
    # tied output head => counted once
    # -------------------------
    embedding = vocab * d

    # -------------------------
    # Final RMSNorm
    # -------------------------
    final_norm = d

    # -------------------------
    # Optional untied LM head
    # -------------------------
    output_head = 0

    if not config.tie_embeddings:
        output_head = vocab * d

    total = transformer + embedding + final_norm + output_head

    return {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": o_proj,
        "attention_per_layer": attention,
        "mlp_per_layer": mlp,
        "norms_per_layer": norms,
        "qk_norm_per_layer": qk_norm,
        "block_per_layer": block,
        "transformer_total": transformer,
        "embedding": embedding,
        "final_norm": final_norm,
        "output_head": output_head,
        "total": total,
    }


def print_report(config: ModelConfig) -> None:
    result = calculate_parameters(config)

    print("=== CODEXA PARAMETER REPORT ===")
    print(f"Layers:            {config.num_layers}")
    print(f"Hidden size:       {config.hidden_size}")
    print(f"Q heads:           {config.num_q_heads}")
    print(f"KV heads:          {config.num_kv_heads}")
    print(f"Head dim:          {config.head_dim}")
    print(f"FFN size:          {config.intermediate_size}")
    print(f"Vocab size:        {config.vocab_size}")
    print(f"Tied embeddings:   {config.tie_embeddings}")
    print()
    print(f"Attention/layer:   {result['attention_per_layer']:,}")
    print(f"MLP/layer:         {result['mlp_per_layer']:,}")
    print(f"Norm/layer:        {result['norms_per_layer']:,}")
    print(f"Block/layer:       {result['block_per_layer']:,}")
    print(f"Transformer total: {result['transformer_total']:,}")
    print(f"Embedding:         {result['embedding']:,}")
    print(f"Final norm:        {result['final_norm']:,}")
    print()
    print(f"TOTAL PARAMETERS:  {result['total']:,}")
    print(f"TOTAL (millions):  {result['total'] / 1_000_000:.6f}M")


if __name__ == "__main__":
    print_report(C1_CONFIG)