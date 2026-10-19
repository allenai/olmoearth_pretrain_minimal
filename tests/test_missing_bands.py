"""Tests for ``data/missing_bands.py`` and the encoder's handling of missing bands."""

import warnings
from typing import Any

import numpy as np
import pytest
import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.missing_bands import (
    handle_missing_bands,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import Normalizer
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import Encoder
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
    MISSING_VALUE,
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
)

S2 = Modality.SENTINEL2_L2A
S1 = Modality.SENTINEL1
LANDSAT = Modality.LANDSAT
B, H, W, T = 1, 8, 8, 3
MISSING = MaskValue.MISSING.value
VISIBLE = MaskValue.ONLINE_ENCODER.value


def _encoder(v1: bool) -> Encoder:
    """v1-style encoder (no band dropout) or v1.1+-style (one S2/Landsat band set)."""
    kwargs: dict[str, Any] = {}
    if not v1:
        kwargs = {
            "tokenization_config": TokenizationConfig(
                overrides={
                    m.name: ModalityTokenization(band_groups=[m.band_order])
                    for m in (S2, LANDSAT)
                }
            ),
            "band_dropout_rate": 0.2,
            "random_band_dropout": True,
            "band_dropout_modalities": [S2.name, LANDSAT.name],
        }
    return Encoder(
        embedding_size=16,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        depth=2,
        drop_path=0.0,
        supported_modalities=[S2, S1, LANDSAT, Modality.LATLON],
        max_sequence_length=12,
        **kwargs,
    ).eval()


def _timestamps(batch: int = B) -> torch.Tensor:
    timestamps = torch.zeros(batch, T, 3, dtype=torch.long)
    timestamps[:, :, 1] = torch.arange(T)
    return timestamps


def _sample(
    modality: ModalitySpec = S2, mask: torch.Tensor | None = None, batch: int = B
) -> MaskedOlmoEarthSample:
    data = torch.rand(batch, H, W, T, modality.num_bands) + 0.5
    return MaskedOlmoEarthSample(
        timestamps=_timestamps(batch),
        **{modality.name: data, f"{modality.name}_mask": mask},
    )


def _with_nan(
    sample: MaskedOlmoEarthSample,
    modality: ModalitySpec,
    bands: list[str],
    t: int | slice = slice(None),
) -> MaskedOlmoEarthSample:
    data = getattr(sample, modality.name).clone()
    for band in bands:
        data[..., t, modality.band_order.index(band)] = float("nan")
    return sample._replace(**{modality.name: data})


def _band(
    sample: MaskedOlmoEarthSample, modality: ModalitySpec, band: str
) -> torch.Tensor:
    return getattr(sample, modality.name)[..., modality.band_order.index(band)]


def _data(sample: MaskedOlmoEarthSample) -> torch.Tensor:
    assert sample.sentinel2_l2a is not None
    return sample.sentinel2_l2a


def _mask(sample: MaskedOlmoEarthSample) -> torch.Tensor:
    assert sample.sentinel2_l2a_mask is not None
    return sample.sentinel2_l2a_mask


# v1.1 and later: missing bands are zeroed, fully missing band sets are masked.


def test_v1_2_zero_fills_partial_band_set() -> None:
    sample = _with_nan(_sample(), S2, ["B01", "B09"])
    with pytest.warns(UserWarning, match="set to 0"):
        out = handle_missing_bands(sample, _encoder(v1=False))
    assert (_band(out, S2, "B01") == 0).all() and (_band(out, S2, "B09") == 0).all()
    assert torch.equal(_band(out, S2, "B02"), _band(sample, S2, "B02"))
    assert _mask(out).shape == (B, H, W, T, 1)
    assert (_mask(out) == VISIBLE).all()


def test_v1_2_zero_fills_single_timestep() -> None:
    sample = _with_nan(_sample(), S2, ["B01"], t=1)
    with pytest.warns(UserWarning):
        out = handle_missing_bands(sample, _encoder(v1=False))
    assert (_band(out, S2, "B01")[..., 1] == 0).all()
    assert (_mask(out) == VISIBLE).all()


# v1: partially missing band sets are treated as fully missing.


def test_v1_masks_partial_band_set() -> None:
    sample = _with_nan(_sample(), S2, ["B05"])
    with pytest.warns(UserWarning, match="treated as fully missing"):
        out = handle_missing_bands(sample, _encoder(v1=True))
    band_sets = _encoder(v1=True).tokenization_config.get_bandset_indices(S2.name)
    index = next(
        i
        for i, band_set in enumerate(band_sets)
        if S2.band_order.index("B05") in band_set
    )
    for i, band_set in enumerate(band_sets):
        expected = MISSING if i == index else VISIBLE
        assert (_mask(out)[..., i] == expected).all()
        if i == index:
            assert (_data(out)[..., band_set] == 0).all()
        else:
            assert torch.equal(_data(out)[..., band_set], _data(sample)[..., band_set])


def test_v1_whole_band_set_missing_does_not_warn() -> None:
    sample = _with_nan(_sample(), S2, ["B01", "B09"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = handle_missing_bands(sample, _encoder(v1=True))
    assert (_mask(out)[..., 2] == MISSING).all()
    assert (_mask(out)[..., :2] == VISIBLE).all()


# Both versions.


@pytest.mark.parametrize("v1", [True, False], ids=["v1", "v1.2"])
def test_missing_timestep_is_masked(v1: bool) -> None:
    sample = _with_nan(_sample(), S2, S2.band_order, t=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = handle_missing_bands(sample, _encoder(v1))
    assert (_data(out)[..., 1, :] == 0).all()
    assert (_mask(out)[..., 1, :] == MISSING).all()
    assert (_mask(out)[..., [0, 2], :] == VISIBLE).all()


@pytest.mark.parametrize("v1", [True, False], ids=["v1", "v1.2"])
def test_per_band_mask_marks_bands_missing(v1: bool) -> None:
    mask = torch.zeros(B, H, W, T, S2.num_bands, dtype=torch.long)
    mask[..., S2.band_order.index("B01")] = MISSING
    sample = _sample(mask=mask)
    with pytest.warns(UserWarning):
        out = handle_missing_bands(sample, _encoder(v1))
    assert _mask(out).shape == (B, H, W, T, 3 if v1 else 1)
    by_nan = _with_nan(sample._replace(sentinel2_l2a_mask=None), S2, ["B01"])
    with pytest.warns(UserWarning):
        expected = handle_missing_bands(by_nan, _encoder(v1))
    assert torch.equal(_data(out), _data(expected))
    assert torch.equal(_mask(out), _mask(expected))


@pytest.mark.parametrize(
    "mask",
    [None, torch.zeros(B, H, W, T, dtype=torch.long)],
    ids=["none", "no-band-dim"],
)
def test_mask_is_made_per_band_set(mask: torch.Tensor | None) -> None:
    out = handle_missing_bands(_sample(mask=mask), _encoder(v1=True))
    assert _mask(out).shape == (B, H, W, T, 3)


def test_unchanged_when_nothing_is_missing() -> None:
    sample = _sample(mask=torch.zeros(B, H, W, T, 3, dtype=torch.long))
    assert handle_missing_bands(sample, _encoder(v1=True)) is sample


def test_bad_mask_shape() -> None:
    sample = _sample(mask=torch.zeros(B, H, W, T, 5, dtype=torch.long))
    with pytest.raises(ValueError, match="one entry per band set"):
        handle_missing_bands(sample, _encoder(v1=True))


def test_wrong_number_of_bands() -> None:
    sample = _sample()
    sample = sample._replace(sentinel2_l2a=_data(sample)[..., :10])
    with pytest.raises(ValueError, match="has 10 bands"):
        handle_missing_bands(sample, _encoder(v1=True))


def test_input_not_modified() -> None:
    sample = _with_nan(_sample(), S2, ["B01"])
    nan = _data(sample).isnan().clone()
    with pytest.warns(UserWarning):
        handle_missing_bands(sample, _encoder(v1=True))
    assert torch.equal(_data(sample).isnan(), nan)


# Encoder forward


def _encode(
    encoder: Encoder, sample: MaskedOlmoEarthSample, fast_pass: bool = False
) -> torch.Tensor:
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = encoder(sample, patch_size=4, input_res=10, fast_pass=fast_pass)
    return output["tokens_and_masks"].sentinel2_l2a


@pytest.mark.parametrize("v1", [True, False], ids=["v1", "v1.2"])
def test_forward_with_nan_is_finite(v1: bool) -> None:
    sample = _with_nan(_sample(), S2, ["B01", "B09"])
    sample = _with_nan(sample, S2, S2.band_order, t=1)
    assert torch.isfinite(_encode(_encoder(v1), sample)).all()


@pytest.mark.parametrize("v1", [True, False], ids=["v1", "v1.2"])
@pytest.mark.parametrize("fast_pass", [False, True], ids=["slow", "fast"])
def test_output_does_not_depend_on_rest_of_batch(v1: bool, fast_pass: bool) -> None:
    encoder = _encoder(v1)
    data = _data(_sample(batch=2))
    data[0, ..., 1, :] = float("nan")  # only the first sample is missing a timestep
    batch = MaskedOlmoEarthSample(timestamps=_timestamps(2), sentinel2_l2a=data)
    alone = MaskedOlmoEarthSample(timestamps=_timestamps(1), sentinel2_l2a=data[:1])
    expected = _encode(encoder, alone)[..., [0, 2], :, :]
    actual = _encode(encoder, batch, fast_pass)[:1, ..., [0, 2], :, :]
    assert torch.allclose(actual, expected, atol=1e-5)


def test_fast_pass_respects_missing_tokens() -> None:
    encoder = _encoder(v1=False)
    sample = _with_nan(_sample(), S2, S2.band_order, t=1)
    fast = _encode(encoder, sample, fast_pass=True)[..., [0, 2], :, :]
    slow = _encode(encoder, sample, fast_pass=False)[..., [0, 2], :, :]
    assert torch.allclose(fast, slow, atol=1e-5)


# Normalizer


def _raw_s2() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(100, 5000, (B, H, W, T, S2.num_bands)).astype(np.float32)


def _set(
    raw: np.ndarray,
    modality: ModalitySpec,
    bands: list[str],
    value: Any,
    t: int | slice = slice(None),
) -> np.ndarray:
    raw = raw.copy()
    for band in bands:
        raw[..., t, modality.band_order.index(band)] = value
    return raw


def _copy(
    raw: np.ndarray, pairs: dict[str, str], t: int | slice = slice(None)
) -> np.ndarray:
    raw = raw.copy()
    order = S2.band_order
    for target, source in pairs.items():
        raw[..., t, order.index(target)] = raw[..., t, order.index(source)]
    return raw


def test_normalizer_keeps_missing_values_missing() -> None:
    data = np.random.rand(1, 4, 4, 2, S2.num_bands).astype(np.float32) * 1000
    data[..., 0] = MISSING_VALUE
    data[..., 1] = np.nan
    normalized = Normalizer(std_multiplier=2.0).normalize(S2, data)
    assert np.isnan(normalized[..., :2]).all()
    assert np.isfinite(normalized[..., 2:]).all()


def test_normalizer_wrong_number_of_bands() -> None:
    with pytest.raises(ValueError, match="has 10 bands"):
        Normalizer(std_multiplier=2.0).normalize(S2, np.zeros((1, 10)))


@pytest.mark.parametrize("value", [np.nan, MISSING_VALUE], ids=["nan", "missing-value"])
def test_fill_copies_bands_like_olmoearth_pretrain_evals(value: float) -> None:
    raw = _raw_s2()
    normalizer = Normalizer(std_multiplier=2.0)
    expected = normalizer.normalize(S2, _copy(raw, {"B01": "B02", "B09": "B8A"}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = normalizer.normalize(
            S2, _set(raw, S2, ["B01", "B09"], value), fill_missing_bands=True
        )
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_fill_copies_from_the_same_timestep_only() -> None:
    raw = _raw_s2()
    normalizer = Normalizer(std_multiplier=2.0)
    expected = normalizer.normalize(S2, _copy(raw, {"B01": "B02"}, t=1))
    out = normalizer.normalize(
        S2, _set(raw, S2, ["B01"], np.nan, t=1), fill_missing_bands=True
    )
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_fill_copies_the_nearest_present_band() -> None:
    # B05 (705 nm) is nearest B06 (740 nm); with B06 also missing, B04 (665 nm).
    raw = _raw_s2()
    normalizer = Normalizer(std_multiplier=2.0)
    expected = normalizer.normalize(S2, _copy(raw, {"B05": "B04", "B06": "B07"}))
    out = normalizer.normalize(
        S2, _set(raw, S2, ["B05", "B06"], np.nan), fill_missing_bands=True
    )
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_fill_warns_when_many_bands_copied() -> None:
    rgbn = {"B02", "B03", "B04", "B08"}
    raw = _set(_raw_s2(), S2, [b for b in S2.band_order if b not in rgbn], np.nan)
    with pytest.warns(UserWarning, match="Up to 8 of the 12"):
        out = Normalizer(std_multiplier=2.0).normalize(S2, raw, fill_missing_bands=True)
    assert np.isfinite(out).all()


def test_fill_does_not_copy_thermal_from_reflective_bands() -> None:
    raw = np.random.default_rng(0).uniform(100, 5000, (B, H, W, T, LANDSAT.num_bands))
    raw = _set(raw, LANDSAT, ["B10", "B11"], np.nan)
    out = Normalizer(std_multiplier=2.0).normalize(
        LANDSAT, raw, fill_missing_bands=True
    )
    for band in ["B10", "B11"]:
        assert np.isnan(out[..., LANDSAT.band_order.index(band)]).all()


def test_fill_leaves_modalities_without_spectral_groups_missing() -> None:
    raw = np.random.default_rng(0).uniform(0, 1, (B, H, W, T, S1.num_bands))
    raw = _set(raw, S1, ["vh"], np.nan)
    out = Normalizer(std_multiplier=2.0).normalize(S1, raw, fill_missing_bands=True)
    assert np.isnan(out[..., S1.band_order.index("vh")]).all()


def test_fill_whole_timestep_stays_missing() -> None:
    raw = _set(_raw_s2(), S2, S2.band_order, np.nan, t=1)
    out = Normalizer(std_multiplier=2.0).normalize(S2, raw, fill_missing_bands=True)
    assert np.isnan(out[..., 1, :]).all()
    assert np.isfinite(out[..., [0, 2], :]).all()
