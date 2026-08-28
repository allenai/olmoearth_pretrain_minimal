"""Tests for model loading functionality."""

import json
from pathlib import Path

import pytest
import torch

from olmoearth_pretrain_minimal import (
    ModelID,
    OlmoEarthPretrain_v1,
    load_model_from_id,
    load_model_from_path,
)
from olmoearth_pretrain_minimal.model_loader import ENCODER_INPUT_MODALITY_NAMES
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.olmoearth_pretrain_v1 import (
    V1_1_SUPPORTED_SIZES,
    V1_2_SUPPORTED_SIZES,
    V1_SUPPORTED_SIZES,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
)

ARTIFACTS = Path(__file__).parent / "artifacts"
assert ARTIFACTS.exists()


def test_load_nano_model_no_weights() -> None:
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


def test_load_tiny_model_no_weights() -> None:
    """Test loading tiny model without weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_TINY, load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_load_base_model_no_weights() -> None:
    """Test loading base model without weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_load_large_model_no_weights() -> None:
    """Test loading large model without weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_LARGE, load_weights=False)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_nano_model_with_weights() -> None:
    """Test loading nano model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_tiny_model_with_weights() -> None:
    """Test loading tiny model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_TINY, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_base_model_with_weights() -> None:
    """Test loading base model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_large_model_with_weights() -> None:
    """Test loading large model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_LARGE, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


@pytest.mark.slow
def test_load_v1_1_nano_model_with_weights() -> None:
    """Test loading nano model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_1_NANO, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0
    assert model.encoder.band_dropout_rate == 0.2
    assert model.encoder.patch_embed_hidden_sizes == [12]


@pytest.mark.slow
def test_load_v1_1_tiny_model_with_weights() -> None:
    """Test loading tiny model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_1_TINY, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0
    assert model.encoder.band_dropout_rate == 0.2
    assert model.encoder.patch_embed_hidden_sizes == [64]


@pytest.mark.slow
def test_load_v1_1_base_model_with_weights() -> None:
    """Test loading base model with pre-trained weights."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_1_BASE, load_weights=True)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0
    assert model.encoder.band_dropout_rate == 0.2
    assert model.encoder.patch_embed_hidden_sizes == [64]


def test_direct_initialization() -> None:
    """Test direct model initialization with custom modalities."""
    model = OlmoEarthPretrain_v1(
        model_size="nano",
        supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat"],
    )
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_direct_initialization_all_sizes() -> None:
    """Test direct model initialization for all model sizes."""
    for model_size in ["nano", "tiny", "base", "large"]:
        model = OlmoEarthPretrain_v1(model_size=model_size, model_version="v1")  # type: ignore[arg-type]
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        assert model.encoder.patch_embeddings.band_dropout_rate == 0


def test_direct_initialization_v1_1_all_sizes() -> None:
    """Test direct model initialization of v1.1 models for all supported sizes."""
    expected_hidden = {"nano": [12], "tiny": [64], "base": [64]}
    for model_size in ["nano", "tiny", "base"]:
        model = OlmoEarthPretrain_v1(model_size=model_size, model_version="v1.1")  # type: ignore[arg-type]
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        # v1.1 keeps band dropout inactive at construction (enable_band_dropout)
        assert model.encoder.patch_embeddings.band_dropout_rate == 0.0
        assert model.encoder.band_dropout_rate == 0.2
        assert model.encoder.patch_embed_hidden_sizes == expected_hidden[model_size]


def test_direct_initialization_v1_2_all_sizes() -> None:
    """Test direct model initialization of v1.1 models for all supported sizes."""
    expected_hidden = {"nano": [12], "tiny": [64], "base": [64]}
    for model_size in ["nano", "tiny", "base"]:
        model = OlmoEarthPretrain_v1(model_size=model_size, model_version="v1.2")  # type: ignore[arg-type]
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        # v1.1 keeps band dropout inactive at construction (enable_band_dropout)
        assert model.encoder.patch_embeddings.band_dropout_rate == 0.0
        assert model.encoder.band_dropout_rate == 0.2
        assert model.encoder.patch_embed_hidden_sizes == expected_hidden[model_size]
        assert model.encoder.position_encoding == "rope_3d_mixed"
        assert model.encoder.rope_base == 10_000.0
        assert model.encoder.rope_temporal_coordinate_scale == 1.0 / 30.0


def test_v1_rejects_small() -> None:
    """Test that v1 does not support the small model size."""
    with pytest.raises(ValueError, match="not available for v1"):
        OlmoEarthPretrain_v1(model_size="small", model_version="v1")


def test_v1_1_rejects_large() -> None:
    """Test that v1.1 does not support the large model size."""
    with pytest.raises(ValueError, match="not available for v1.1"):
        OlmoEarthPretrain_v1(model_size="large", model_version="v1.1")


def test_v1_1_rejects_small() -> None:
    """Test that v1.1 does not support the small model size."""
    with pytest.raises(ValueError, match="not available for v1.1"):
        OlmoEarthPretrain_v1(model_size="small", model_version="v1.1")


def test_v1_2_rejects_large() -> None:
    """Test that v1.2 does not support the large model size."""
    with pytest.raises(ValueError, match="not available for v1.2"):
        OlmoEarthPretrain_v1(model_size="large", model_version="v1.2")


def test_invalid_model_size() -> None:
    """Test that invalid model size raises an error."""
    with pytest.raises(ValueError, match="Invalid model_size"):
        OlmoEarthPretrain_v1(model_size="invalid")  # type: ignore[arg-type]


def test_load_v1_1_config() -> None:
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

    # band dropout will be off by default. To enable it, call
    # model.encoder.enable_band_dropout()
    assert model.encoder.patch_embeddings.band_dropout_rate == 0.0


def test_load_v1_2_config() -> None:
    """Test loading nano model from config."""
    model = load_model_from_path(model_path=ARTIFACTS / "v1_2_nano", load_weights=False)
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

    # band dropout will be off by default. To enable it, call
    # model.encoder.enable_band_dropout()
    assert model.encoder.patch_embeddings.band_dropout_rate == 0.0
    assert model.encoder.position_encoding == "rope_3d_mixed"
    assert model.encoder.rope_base == 10_000.0
    assert model.encoder.rope_temporal_coordinate_scale == 1.0 / 30.0


EXPECTED_ENCODER_INPUTS = ["sentinel2_l2a", "sentinel1", "landsat"]


def test_all_model_ids_have_registered_encoder_input_modalities() -> None:
    """Every ModelID must explicitly declare its encoder input modalities.

    If this fails for a newly added model, add an entry to
    ENCODER_INPUT_MODALITY_NAMES listing the modalities its encoder was trained on.
    """
    for model_id in ModelID:
        assert model_id in ENCODER_INPUT_MODALITY_NAMES


def test_load_from_id_restricts_encoder_inputs() -> None:
    """Loading a pretrained model restricts the encoder to trained modalities."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
    assert model.encoder.encoder_input_modality_names == EXPECTED_ENCODER_INPUTS
    assert model.target_encoder.encoder_input_modality_names == EXPECTED_ENCODER_INPUTS


def test_load_from_path_restricts_encoder_inputs() -> None:
    """Loading from a path derives the encoder input modalities from the config."""
    model = load_model_from_path(model_path=ARTIFACTS / "v1_2_nano", load_weights=False)
    assert model.encoder.encoder_input_modality_names == EXPECTED_ENCODER_INPUTS


def test_load_from_path_without_decode_only_modalities_raises(tmp_path: Path) -> None:
    """A config that doesn't record decode-only modalities must fail loudly."""
    with (ARTIFACTS / "v1_2_nano" / "config.json").open() as f:
        config_dict = json.load(f)
    for section in ("train_module", "data_loader"):
        del config_dict[section]["masking_config"]["strategy_config"][
            "only_decode_modalities"
        ]
    with (tmp_path / "config.json").open("w") as f:
        json.dump(config_dict, f)

    with pytest.raises(ValueError, match="Explicitly mark"):
        load_model_from_path(model_path=tmp_path, load_weights=False)


def test_encoder_drops_untrained_modalities() -> None:
    """Decode-only modalities are dropped from the encoder input."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
    model.eval()

    B, H, W, T, num_s2_bands = 1, 16, 16, 3, 12
    patch_size = 4
    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)

    sentinel2_l2a = torch.randn((B, H, W, T, num_s2_bands))
    sentinel2_l2a_mask = torch.zeros((B, H, W, T, num_s2_bands), dtype=torch.long)
    s2_only_sample = MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
    )
    # worldcover was decode-only during pretraining, so the encoder should
    # ignore it even though the model config supports it
    with_worldcover_sample = MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        worldcover=torch.randn((B, H, W, 1)),
        worldcover_mask=torch.zeros((B, H, W, 1), dtype=torch.long),
    )

    with torch.no_grad():
        s2_only_output = model.encoder(s2_only_sample, patch_size=patch_size)
        with_worldcover_output = model.encoder(
            with_worldcover_sample, patch_size=patch_size
        )

    assert with_worldcover_output["tokens_and_masks"].worldcover is None
    torch.testing.assert_close(
        with_worldcover_output["tokens_and_masks"].sentinel2_l2a,
        s2_only_output["tokens_and_masks"].sentinel2_l2a,
    )


@pytest.mark.parametrize(
    ("model_version", "model_size"),
    [
        (version, size)
        for version, sizes in (
            ("v1", V1_SUPPORTED_SIZES),
            ("v1.1", V1_1_SUPPORTED_SIZES),
            ("v1.2", V1_2_SUPPORTED_SIZES),
        )
        for size in sizes
    ],
)
def test_every_supported_size_builds(model_version: str, model_size: str) -> None:
    """Every size a version declares support for must actually build.

    v1.2 declared support for "small" before PATCH_EMBED_HIDDEN_SIZES had an entry for
    it, so building it raised KeyError: 'small'.
    """
    model = OlmoEarthPretrain_v1(model_size=model_size, model_version=model_version)
    assert sum(p.numel() for p in model.parameters()) > 0
