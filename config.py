from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # =========================
    # Vocabulary / sequence
    # =========================
    vocab_size: int = 32_768
    max_seq_len: int = 2_048

    # =========================
    # Transformer
    # =========================
    num_layers: int = 30
    hidden_size: int = 384

    # =========================
    # Attention / GQA
    # =========================
    num_q_heads: int = 6
    num_kv_heads: int = 3
    head_dim: int = 64

    # =========================
    # SwiGLU
    # =========================
    intermediate_size: int = 1_152

    # =========================
    # Normalization / position
    # =========================
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5

    # =========================
    # Architecture switches
    # =========================
    use_qk_norm: bool = False
    tie_embeddings: bool = True
    use_bias: bool = False

    def validate(self) -> None:
        """
        Validate all architectural invariants.
        Raises ValueError if the configuration is invalid.
        """

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")

        if self.num_q_heads <= 0:
            raise ValueError("num_q_heads must be > 0")

        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be > 0")

        if self.hidden_size % self.num_q_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_q_heads"
            )

        expected_head_dim = self.hidden_size // self.num_q_heads

        if expected_head_dim != self.head_dim:
            raise ValueError(
                f"head_dim mismatch: expected {expected_head_dim}, "
                f"got {self.head_dim}"
            )

        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads "
                "for GQA"
            )

        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be > 0")

        if self.intermediate_size % 128 != 0:
            raise ValueError(
                "intermediate_size should be a multiple of 128"
            )

        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be > 0")

        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")


# Canonical Track A starting configuration from v9.
C1_CONFIG = ModelConfig()

# Validate immediately when imported.
C1_CONFIG.validate()