"""Tests for RoPE position encodings in the FlexiViT encoder."""

import pytest
import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.encodings import (
    PositionEncoding,
    apply_2d_axial_rope,
    timestamps_to_days,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import (
    Encoder,
    EncoderConfig,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
)

# head_dim must be divisible by 4 for the 2D/3D RoPE variants, and the axial 3D
# split additionally needs (head_dim - d_t) divisible by 4. head_dim = 32 // 2 = 16
# (d_t = 4, remaining = 12) satisfies every mode.
ROPE_MODES = [
    PositionEncoding.AXIAL_2D_ROPE,
    PositionEncoding.MIXED_2D_ROPE,
    PositionEncoding.AXIAL_3D_ROPE,
    PositionEncoding.MIXED_3D_ROPE,
]


def _make_encoder(position_encoding: str) -> Encoder:
    torch.manual_seed(0)
    return Encoder(
        embedding_size=32,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        depth=2,
        drop_path=0.0,
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        max_sequence_length=12,
        position_encoding=position_encoding,
    )


def _make_sample() -> MaskedOlmoEarthSample:
    B, H, W, T, num_s2_bands = 1, 8, 8, 3, 12
    sentinel2_l2a = torch.randn((B, H, W, T, num_s2_bands))
    sentinel2_l2a_mask = torch.zeros((B, H, W, T, num_s2_bands), dtype=torch.long)
    days = torch.randint(1, 28, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        timestamps=timestamps,
    )


@pytest.mark.parametrize("position_encoding", ROPE_MODES)
def test_encoder_forward_rope_modes(position_encoding: str) -> None:
    """Each RoPE mode runs end-to-end and produces finite outputs."""
    encoder = _make_encoder(position_encoding)
    encoder.eval()
    sample = _make_sample()
    with torch.no_grad():
        output = encoder(sample, patch_size=4)
    tokens = output["tokens_and_masks"].sentinel2_l2a
    assert torch.isfinite(tokens).all()


def test_absolute_default_is_unchanged() -> None:
    """The default encoder uses absolute position encoding (no RoPE)."""
    encoder = _make_encoder(PositionEncoding.ABSOLUTE)
    assert encoder.position_encoding == PositionEncoding.ABSOLUTE
    for blk in encoder.blocks:
        assert blk.attn.rope_mixed_freqs is None


def test_mixed_rope_registers_learnable_freqs() -> None:
    """RoPE-Mixed modes register a learnable frequency parameter per block."""
    for mode, expected_lead in [
        (PositionEncoding.MIXED_2D_ROPE, 2),
        (PositionEncoding.MIXED_3D_ROPE, 3),
    ]:
        encoder = _make_encoder(mode)
        for blk in encoder.blocks:
            freqs = blk.attn.rope_mixed_freqs
            assert isinstance(freqs, torch.nn.Parameter)
            assert freqs.requires_grad
            assert freqs.shape[0] == expected_lead


def test_config_resolves_deprecated_spatial_pos_encoding() -> None:
    """The deprecated spatial_pos_encoding alias maps to position_encoding."""
    with pytest.warns(DeprecationWarning):
        config = EncoderConfig(
            supported_modality_names=["sentinel2_l2a", "latlon"],
            embedding_size=16,
            num_heads=2,
            spatial_pos_encoding="rope",
        )
    assert config.position_encoding == "rope"
    assert config.spatial_pos_encoding is None
    config.validate()


def test_timestamps_to_days_monotonic() -> None:
    """Later timestamps map to larger day counts."""
    # (day, month, year)
    early = torch.tensor([[1, 0, 2020]])
    late = torch.tensor([[1, 6, 2020]])
    assert timestamps_to_days(late).item() > timestamps_to_days(early).item()


def test_apply_2d_axial_rope_preserves_shape_and_norm() -> None:
    """Axial RoPE is a rotation, so it preserves per-token norms."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 8)  # (B, H, N, D)
    positions = torch.randint(0, 4, (1, 5, 2)).float()
    out = apply_2d_axial_rope(x, positions)
    assert out.shape == x.shape
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))
