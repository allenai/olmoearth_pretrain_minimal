"""Regression tests for low-precision (fp16/bf16) forward passes.

Prior to the fix in `flexi_vit.py::_apply_encodings_per_modality`, the
`modality_embed` accumulator was unconditionally allocated as fp32 (via
`torch.zeros(modality_tokens.shape, device=device)` with no `dtype=` kwarg).
PyTorch's type promotion then upcast the encoded tokens back to fp32 even if
the model and input had been converted to fp16/bf16, which broke the very
first `LayerNorm` inside the transformer blocks with:

    RuntimeError: expected scalar type Float but found Half

These tests run the encoder forward pass at fp16 and bf16 on CPU and assert
that:
  1. no dtype mismatch is raised, and
  2. the encoder output dtype matches the input dtype (i.e. it was not
     silently upcast back to fp32 by an internal allocation).

We construct the model directly (no Hugging Face download) and request a
patch size equal to the model's base patch size, which avoids unrelated CPU
bicubic-interpolation limits in `F.interpolate`.
"""

import pytest
import torch

from olmoearth_pretrain_minimal import OlmoEarthPretrain_v1
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
)

PATCH_SIZE = 8


def _make_sample(dtype: torch.dtype) -> MaskedOlmoEarthSample:
    B, H, W, T, num_s2_bands = 1, 16, 16, 3, 12
    sentinel2_l2a = torch.randn((B, H, W, T, num_s2_bands), dtype=dtype)
    sentinel2_l2a_mask = torch.zeros((B, H, W, T, num_s2_bands), dtype=torch.long)

    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)

    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nano_encoder_forward_low_precision(dtype: torch.dtype) -> None:
    """Encoder forward pass must not raise a dtype mismatch in fp16 / bf16."""
    model = OlmoEarthPretrain_v1(
        model_size="nano",
        supported_modality_names=["sentinel2_l2a"],
        max_patch_size=PATCH_SIZE,
        max_sequence_length=3,
    )
    model = model.to(dtype).eval()

    sample = _make_sample(dtype)

    with torch.inference_mode():
        out = model.encoder(
            sample, patch_size=PATCH_SIZE, input_res=10, fast_pass=True
        )

    # Encoder output dtype should match the model/input dtype, not have been
    # silently upcast by an internal fp32 allocation.
    tokens = out["tokens_and_masks"].sentinel2_l2a
    assert tokens.dtype == dtype, (
        f"Expected encoder output dtype {dtype}, got {tokens.dtype}"
    )


