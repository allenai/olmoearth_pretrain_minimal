"""Tests for model loading functionality."""

import pytest
from pathlib import Path

from olmoearth_pretrain_minimal import ModelID, load_model_from_id, OlmoEarthPretrain_v1, load_model_from_path
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
import torch

ARTIFACTS = Path(__file__).parent / "artifacts"
assert ARTIFACTS.exists()


def test_load_nano_model_no_weights():
    """Test loading nano model without weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0

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
    sample = MaskedOlmoEarthSample(**masked_sample_dict)
    _ = model(sample, patch_size=patch_size)

def test_load_tiny_model_no_weights():
    """Test loading tiny model without weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_TINY, load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_load_base_model_no_weights():
    """Test loading base model without weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_load_large_model_no_weights():
    """Test loading large model without weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_LARGE, load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_nano_model_with_weights():
    """Test loading nano model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_tiny_model_with_weights():
    """Test loading tiny model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_TINY, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_base_model_with_weights():
    """Test loading base model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_large_model_with_weights():
    """Test loading large model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_LARGE, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_direct_initialization():
    """Test direct model initialization with custom modalities."""
    model = OlmoEarthPretrain_v1(
        model_size="nano",
        supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat"],
    )
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_direct_initialization_all_sizes():
    """Test direct model initialization for all model sizes."""
    for model_size in ["nano", "tiny", "base", "large"]:
        model = OlmoEarthPretrain_v1(model_size=model_size)
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_invalid_model_size():
    """Test that invalid model size raises an error."""
    with pytest.raises(ValueError, match="Invalid model_size"):
        OlmoEarthPretrain_v1(model_size="invalid")

def test_load_v1_1_config():
    """Test loading nano model from config."""
    model = load_model_from_path(model_path=ARTIFACTS / "v1_1_nano", load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0

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
    sample = MaskedOlmoEarthSample(**masked_sample_dict)
    _ = model(sample, patch_size=patch_size)
    assert model.encoder.patch_embeddings.band_dropout_rate == 0.2
