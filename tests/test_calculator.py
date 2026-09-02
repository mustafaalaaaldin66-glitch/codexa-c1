from config import C1_CONFIG
from parameter_calculator import calculate_parameters


def test_c1_parameter_count():
    result = calculate_parameters(C1_CONFIG)

    expected = 65_690_496

    assert result["total"] == expected


def test_gqa_is_valid():
    assert C1_CONFIG.num_q_heads % C1_CONFIG.num_kv_heads == 0


def test_head_dim_is_valid():
    assert (
        C1_CONFIG.hidden_size // C1_CONFIG.num_q_heads
        == C1_CONFIG.head_dim
    )