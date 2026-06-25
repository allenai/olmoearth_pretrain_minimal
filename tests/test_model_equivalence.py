"""Tests for model loading functionality."""

import olmoearth_pretrain.model_loader as op_loader
import pytest
import torch
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample as opMaskedOlmoEarthSample,
)

import olmoearth_pretrain_minimal as opm
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample as opm_MaskedOlmoEarthSample,
)


@pytest.mark.slow
def test_nano_model_with_weights_equivalence() -> None:
    """Test loading nano model without weights."""
    opm_model = opm.load_model_from_id(opm.ModelID.OLMOEARTH_V1_NANO, load_weights=True)
    assert opm_model is not None
    opm_param_count = sum(p.numel() for p in opm_model.parameters())
    assert opm_param_count > 0
    assert opm_model.encoder.patch_embeddings.band_dropout_rate == 0

    # also test the forward functionality works
    B, H, W, T, num_s2_bands = 1, 16, 16, 3, 12
    sentinel2_l2a = torch.randn((B, H, W, T, num_s2_bands))
    sentinel2_l2a_mask = torch.zeros((B, H, W, T, num_s2_bands), dtype=torch.long)
    patch_size = 4

    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

    masked_sample_dict = {
        "sentinel2_l2a": sentinel2_l2a,
        "sentinel2_l2a_mask": sentinel2_l2a_mask,
        "timestamps": timestamps,
    }
    opm_sample = opm_MaskedOlmoEarthSample(**masked_sample_dict)
    opm_model.eval()
    opm_output = opm_model(opm_sample, patch_size=patch_size)

    # now lets do the same for an olmoearth_pretrain model
    op_model = op_loader.load_model_from_id(
        op_loader.ModelID.OLMOEARTH_V1_NANO, load_weights=True
    )
    assert op_model is not None
    op_param_count = sum(p.numel() for p in op_model.parameters())
    assert op_param_count == opm_param_count

    op_sample = opMaskedOlmoEarthSample(**masked_sample_dict)
    op_model.eval()
    op_output = op_model(op_sample, patch_size=patch_size)
    assert (op_output[0].sentinel2_l2a == opm_output[0].sentinel2_l2a).all()
