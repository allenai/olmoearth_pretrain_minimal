"""Unit tests for normalize_sample."""

import numpy as np
import torch

from olmoearth_pretrain_minimal import Normalizer, normalize_sample
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
    load_computed_config,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
)

B, H, W, T = 1, 8, 8, 2


def _modality_tensor(name: str) -> torch.Tensor:
    """Random raw-ish (B, H, W, T, C) tensor for a modality, bands last."""
    num_bands = len(Modality.get(name).band_order)
    return torch.randn(B, H, W, T, num_bands) * 500 + 1500


def _timestamps() -> torch.Tensor:
    days = torch.randint(0, 28, (B, T, 1))
    months = torch.randint(0, 12, (B, T, 1))
    years = torch.randint(2018, 2021, (B, T, 1))
    return torch.cat([days, months, years], dim=-1)


def test_matches_numpy_normalizer() -> None:
    """normalize_sample matches the reference numpy Normalizer for Sentinel-2."""
    s2 = _modality_tensor("sentinel2_l2a")
    mask = torch.zeros_like(s2, dtype=torch.long)
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=_timestamps()
    )

    out = normalize_sample(sample, std_multiplier=2.0)
    ref = Normalizer(std_multiplier=2.0).normalize(Modality.SENTINEL2_L2A, s2.numpy())

    assert out.sentinel2_l2a.shape == s2.shape
    assert np.abs(out.sentinel2_l2a.numpy() - ref).max() < 1e-4
    # Original tensor untouched (a new sample is returned).
    assert torch.equal(sample.sentinel2_l2a, s2)


def test_leaves_mask_and_timestamps_untouched() -> None:
    """Only modality data is normalized; masks and timestamps pass through."""
    s2 = _modality_tensor("sentinel2_l2a")
    mask = torch.zeros_like(s2, dtype=torch.long)
    ts = _timestamps()
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=ts
    )

    out = normalize_sample(sample)

    assert torch.equal(out.sentinel2_l2a_mask, mask)
    assert torch.equal(out.timestamps, ts)
    assert not torch.allclose(out.sentinel2_l2a, s2)


def test_normalizes_multiple_modalities_and_skips_statless() -> None:
    """Modalities with stats are normalized; those without are left as-is."""
    assert "sentinel1" in load_computed_config()
    assert "worldcereal" not in load_computed_config()  # no stats -> skipped

    s2 = _modality_tensor("sentinel2_l2a")
    s1 = _modality_tensor("sentinel1")
    wc = _modality_tensor("worldcereal")
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=torch.zeros_like(s2, dtype=torch.long),
        sentinel1=s1,
        sentinel1_mask=torch.zeros_like(s1, dtype=torch.long),
        worldcereal=wc,
        worldcereal_mask=torch.zeros_like(wc, dtype=torch.long),
        timestamps=_timestamps(),
    )

    out = normalize_sample(sample)

    assert not torch.allclose(out.sentinel2_l2a, s2)  # normalized
    assert not torch.allclose(out.sentinel1, s1)  # normalized
    assert torch.equal(out.worldcereal, wc)  # no stats -> unchanged


def test_std_multiplier_widens_range() -> None:
    """A larger std_multiplier maps values closer to the center (smaller magnitude)."""
    s2 = _modality_tensor("sentinel2_l2a")
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=torch.zeros_like(s2, dtype=torch.long),
        timestamps=_timestamps(),
    )

    narrow = normalize_sample(sample, std_multiplier=1.0).sentinel2_l2a
    wide = normalize_sample(sample, std_multiplier=4.0).sentinel2_l2a

    # Wider range -> normalized values are pulled toward 0.5, i.e. smaller deviation.
    assert (wide - 0.5).abs().mean() < (narrow - 0.5).abs().mean()


def test_no_normalizable_modalities_returns_same_object() -> None:
    """A sample with only statless modalities is returned unchanged (identity)."""
    wc = _modality_tensor("worldcereal")
    sample = MaskedOlmoEarthSample(
        worldcereal=wc,
        worldcereal_mask=torch.zeros_like(wc, dtype=torch.long),
        timestamps=_timestamps(),
    )
    assert normalize_sample(sample) is sample
