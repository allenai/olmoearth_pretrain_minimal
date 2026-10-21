"""Tests for how ``Normalizer`` fills missing bands for each model type."""

import warnings
from typing import Any

import numpy as np
import pytest
import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
    Normalizer,
    NormalizerV1,
    NormalizerV1_1,
    NormalizerV1_2,
    NormalizerV1_3,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import Encoder
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


def _raw(modality: ModalitySpec = S2) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (B, H, W, T, modality.num_bands)
    return rng.uniform(100, 5000, shape).astype(np.float32)


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


def _band(data: np.ndarray, modality: ModalitySpec, band: str) -> np.ndarray:
    return data[..., modality.band_order.index(band)]


# Model type


def test_model_type_subclasses() -> None:
    assert NormalizerV1().model_type == "v1"
    assert NormalizerV1_1().model_type == "v1.1"
    assert isinstance(NormalizerV1_2(), NormalizerV1_1)
    assert isinstance(NormalizerV1_3(), NormalizerV1_1)
    assert Normalizer().model_type is None


@pytest.mark.parametrize("model_type", ["v1.1", "v1.2", "v1.3"])
def test_later_model_types_zero_missing_bands(model_type: str) -> None:
    raw = _set(_raw(), S2, ["B01"], np.nan)
    out = Normalizer(model_type=model_type).normalize(S2, raw)
    np.testing.assert_array_equal(out, NormalizerV1_1().normalize(S2, raw))


def test_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        Normalizer(model_type="v2")


def test_no_model_type_leaves_missing_values() -> None:
    out = Normalizer().normalize(S2, _set(_raw(), S2, ["B01"], np.nan))
    assert np.isnan(_band(out, S2, "B01")).all()


@pytest.mark.parametrize("normalizer", [Normalizer(), NormalizerV1(), NormalizerV1_1()])
def test_same_as_plain_normalizer_when_nothing_missing(normalizer: Normalizer) -> None:
    raw = _raw()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = normalizer.normalize(S2, raw)
    np.testing.assert_array_equal(out, Normalizer().normalize(S2, raw))


@pytest.mark.parametrize("normalizer", [Normalizer(), NormalizerV1(), NormalizerV1_1()])
def test_wrong_number_of_bands(normalizer: Normalizer) -> None:
    with pytest.raises(ValueError, match="has 10 bands"):
        normalizer.normalize(S2, np.zeros((1, 10)))


# v1.1 and later: missing bands are set to 0 after normalization.


@pytest.mark.parametrize("value", [np.nan, MISSING_VALUE], ids=["nan", "missing-value"])
def test_v1_1_zeroes_missing_bands(value: float) -> None:
    raw = _raw()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = NormalizerV1_1().normalize(S2, _set(raw, S2, ["B01", "B09"], value))
    expected = Normalizer().normalize(S2, raw)
    assert (_band(out, S2, "B01") == 0).all() and (_band(out, S2, "B09") == 0).all()
    np.testing.assert_array_equal(_band(out, S2, "B02"), _band(expected, S2, "B02"))


def test_v1_1_warns_when_many_bands_missing() -> None:
    rgbn = {"B02", "B03", "B04", "B08"}
    raw = _set(_raw(), S2, [b for b in S2.band_order if b not in rgbn], np.nan)
    with pytest.warns(UserWarning, match="Up to 8 of the 12"):
        out = NormalizerV1_1().normalize(S2, raw)
    assert np.isfinite(out).all()


# v1: missing bands are copied from the nearest band in wavelength.


@pytest.mark.parametrize("value", [np.nan, MISSING_VALUE], ids=["nan", "missing-value"])
def test_v1_copies_bands_like_olmoearth_pretrain_evals(value: float) -> None:
    raw = _raw()
    expected = Normalizer().normalize(S2, _copy(raw, {"B01": "B02", "B09": "B8A"}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = NormalizerV1().normalize(S2, _set(raw, S2, ["B01", "B09"], value))
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_v1_copies_partial_pixels_next_to_whole_missing_timestep() -> None:
    raw = _raw()
    expected = Normalizer().normalize(S2, _copy(raw, {"B01": "B02"}, t=0))
    raw = _set(_set(raw, S2, ["B01"], np.nan, t=0), S2, S2.band_order, np.nan, t=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = NormalizerV1().normalize(S2, raw)
    np.testing.assert_allclose(out[..., [0, 2], :], expected[..., [0, 2], :], rtol=1e-6)
    assert (out[..., 1, :] == 0).all()


def test_v1_copies_from_the_same_timestep_only() -> None:
    raw = _raw()
    expected = Normalizer().normalize(S2, _copy(raw, {"B01": "B02"}, t=1))
    out = NormalizerV1().normalize(S2, _set(raw, S2, ["B01"], np.nan, t=1))
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_v1_copies_the_nearest_present_band() -> None:
    # B05 (705 nm) is nearest B06 (740 nm); with B06 also missing, B04 (665 nm).
    raw = _raw()
    expected = Normalizer().normalize(S2, _copy(raw, {"B05": "B04", "B06": "B07"}))
    out = NormalizerV1().normalize(S2, _set(raw, S2, ["B05", "B06"], np.nan))
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_v1_warns_when_many_bands_copied() -> None:
    rgbn = {"B02", "B03", "B04", "B08"}
    raw = _set(_raw(), S2, [b for b in S2.band_order if b not in rgbn], np.nan)
    with pytest.warns(UserWarning, match="Up to 8 of the 12"):
        out = NormalizerV1().normalize(S2, raw)
    assert np.isfinite(out).all()


def test_v1_does_not_copy_thermal_from_reflective_bands() -> None:
    raw = _set(_raw(LANDSAT), LANDSAT, ["B10", "B11"], np.nan)
    with pytest.warns(UserWarning, match="set to 0"):
        out = NormalizerV1().normalize(LANDSAT, raw)
    for band in ["B10", "B11"]:
        assert (_band(out, LANDSAT, band) == 0).all()


def test_v1_zeroes_modalities_without_spectral_groups() -> None:
    raw = _set(_raw(S1), S1, ["vh"], np.nan)
    with pytest.warns(UserWarning, match="set to 0"):
        out = NormalizerV1().normalize(S1, raw)
    assert (_band(out, S1, "vh") == 0).all()


@pytest.mark.parametrize("normalizer", [NormalizerV1(), NormalizerV1_1()])
def test_whole_timestep_missing_is_zeroed(normalizer: Normalizer) -> None:
    raw = _set(_raw(), S2, S2.band_order, np.nan, t=1)
    with pytest.warns(UserWarning, match="Mask band sets"):
        out = normalizer.normalize(S2, raw)
    assert (out[..., 1, :] == 0).all()
    assert np.isfinite(out).all()


def test_input_not_modified() -> None:
    raw = _set(_raw(), S2, ["B01"], np.nan)
    NormalizerV1().normalize(S2, raw)
    assert np.isnan(_band(raw, S2, "B01")).all()


# Encoder forward on normalized data with a masked missing timestep.


def test_forward_with_masked_missing_timestep_is_finite() -> None:
    encoder = Encoder(
        embedding_size=16,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        depth=2,
        drop_path=0.0,
        supported_modalities=[S2],
        max_sequence_length=12,
    ).eval()
    raw = _set(_set(_raw(), S2, ["B01"], np.nan), S2, S2.band_order, np.nan, t=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = NormalizerV1().normalize(S2, raw)
    num_band_sets = len(encoder.tokenization_config.get_bandset_indices(S2.name))
    mask = torch.full(
        (B, H, W, T, num_band_sets), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
    )
    mask[..., 1, :] = MaskValue.MISSING.value
    timestamps = torch.zeros(B, T, 3, dtype=torch.long)
    timestamps[:, :, 1] = torch.arange(T)
    sample = MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=torch.from_numpy(data).float(),
        sentinel2_l2a_mask=mask,
    )
    with torch.no_grad():
        output = encoder(sample, patch_size=4, input_res=10)
    assert torch.isfinite(output["tokens_and_masks"].sentinel2_l2a).all()
