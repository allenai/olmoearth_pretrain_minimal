"""Tests for ``data/missing_bands.py`` and the encoder's handling of missing bands."""

import warnings
from typing import Any

import numpy as np
import pytest
import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.missing_bands import (
    fill_missing_bands,
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = handle_missing_bands(sample, _encoder(v1=False))
    assert (_band(out, S2, "B01") == 0).all() and (_band(out, S2, "B09") == 0).all()
    assert torch.equal(_band(out, S2, "B02"), _band(sample, S2, "B02"))
    assert _mask(out).shape == (B, H, W, T, 1)
    assert (_mask(out) == VISIBLE).all()


def test_v1_2_warns_beyond_band_dropout_rate() -> None:
    sample = _with_nan(_sample(), S2, ["B05", "B06", "B07"])  # 3 / 12 > 0.2
    with pytest.warns(UserWarning, match="Up to 3 of the 12"):
        handle_missing_bands(sample, _encoder(v1=False))


def test_v1_2_warns_for_modality_without_band_dropout() -> None:
    sample = _with_nan(_sample(S1), S1, ["vh"])
    with pytest.warns(UserWarning, match="band dropout rate 0.0"):
        out = handle_missing_bands(sample, _encoder(v1=False))
    assert (_band(out, S1, "vh") == 0).all()


def test_v1_2_zero_fills_single_timestep() -> None:
    sample = _with_nan(_sample(), S2, ["B01"], t=1)
    out = handle_missing_bands(sample, _encoder(v1=False))
    assert (_band(out, S2, "B01")[..., 1] == 0).all()


# v1: each missing band is copied from the nearest present band in wavelength.


def _raw_s2(batch: int = B) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(100, 5000, (batch, H, W, T, S2.num_bands)).astype(np.float32)


def _normalized(raw: np.ndarray) -> MaskedOlmoEarthSample:
    data = Normalizer(std_multiplier=2.0).normalize(S2, raw)
    return MaskedOlmoEarthSample(
        timestamps=_timestamps(raw.shape[0]),
        sentinel2_l2a=torch.as_tensor(data, dtype=torch.float32),
    )


def test_v1_copies_bands_like_olmoearth_pretrain_evals() -> None:
    # olmoearth_pretrain's PASTIS eval fills B01 from B02 and B09 from B8A in raw
    # units, then normalizes with the pretraining stats.
    raw = _raw_s2()
    imputed = raw.copy()
    order = S2.band_order
    imputed[..., order.index("B01")] = raw[..., order.index("B02")]
    imputed[..., order.index("B09")] = raw[..., order.index("B8A")]
    expected = _normalized(imputed)

    sample = _with_nan(_normalized(raw), S2, ["B01", "B09"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = handle_missing_bands(sample, _encoder(v1=True))
    # B01 and B09 form a whole band set in v1; it is filled, not masked.
    assert torch.allclose(_data(out), _data(expected), atol=1e-5)
    assert (_mask(out) == VISIBLE).all()


def test_v1_copies_from_the_same_timestep_only() -> None:
    raw = _raw_s2()
    imputed = raw.copy()
    order = S2.band_order
    imputed[..., 1, order.index("B01")] = raw[..., 1, order.index("B02")]
    sample = _with_nan(_normalized(raw), S2, ["B01"], t=1)
    out = handle_missing_bands(sample, _encoder(v1=True))
    assert torch.allclose(_data(out), _data(_normalized(imputed)), atol=1e-5)


def test_v1_copies_the_nearest_present_band() -> None:
    # B05 (705 nm) is nearest B06 (740 nm); with B06 also missing, B04 (665 nm).
    raw = _raw_s2()
    order = S2.band_order
    sample = _with_nan(_normalized(raw), S2, ["B05", "B06"])
    out = handle_missing_bands(sample, _encoder(v1=True))
    imputed = raw.copy()
    imputed[..., order.index("B05")] = raw[..., order.index("B04")]
    imputed[..., order.index("B06")] = raw[..., order.index("B07")]
    assert torch.allclose(_data(out), _data(_normalized(imputed)), atol=1e-5)


@pytest.mark.parametrize("v1", [True, False], ids=["v1", "v1.2"])
def test_rgb_nir_only_warns(v1: bool) -> None:
    rgbn = {"B02", "B03", "B04", "B08"}
    sample = _with_nan(_sample(), S2, [b for b in S2.band_order if b not in rgbn])
    with pytest.warns(UserWarning, match="Up to 8 of the 12"):
        out = handle_missing_bands(sample, _encoder(v1))
    assert torch.isfinite(_data(out)).all()


def test_v1_warns_when_many_bands_copied() -> None:
    sample = _with_nan(_sample(), S2, ["B05", "B06", "B07"], t=1)
    with pytest.warns(UserWarning, match="Up to 3 of the 12 sentinel2_l2a bands were"):
        handle_missing_bands(sample, _encoder(v1=True))


def test_v1_does_not_copy_thermal_from_reflective_bands() -> None:
    sample = _with_nan(_sample(LANDSAT), LANDSAT, ["B10", "B11"])
    with pytest.warns(UserWarning, match="could not be copied"):
        out = handle_missing_bands(sample, _encoder(v1=True))
    assert (_band(out, LANDSAT, "B10") == 0).all()
    assert (_band(out, LANDSAT, "B11") == 0).all()


def test_v1_warns_when_nothing_to_copy_from() -> None:
    sample = _with_nan(_sample(S1), S1, ["vh"])
    with pytest.warns(UserWarning, match="could not be copied"):
        out = handle_missing_bands(sample, _encoder(v1=True))
    assert (_band(out, S1, "vh") == 0).all()


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
    out = handle_missing_bands(sample, _encoder(v1))
    assert _mask(out).shape == (B, H, W, T, 3 if v1 else 1)
    assert (_mask(out) == VISIBLE).all()
    by_nan = _with_nan(sample._replace(sentinel2_l2a_mask=None), S2, ["B01"])
    assert torch.equal(_data(out), _data(handle_missing_bands(by_nan, _encoder(v1))))


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
    handle_missing_bands(sample, _encoder(v1=True))
    assert torch.equal(_data(sample).isnan(), nan)


# fill_missing_bands


@pytest.mark.parametrize("v1", [True, False], ids=["v1", "v1.2"])
def test_fill_missing_bands_matches_nan(v1: bool) -> None:
    encoder = _encoder(v1)
    sample = _sample()
    by_name = fill_missing_bands(sample, {S2.name: ["B01", "B09"]}, encoder)
    by_nan = handle_missing_bands(_with_nan(sample, S2, ["B01", "B09"]), encoder)
    assert torch.equal(_data(by_name), _data(by_nan))
    assert torch.equal(_mask(by_name), _mask(by_nan))


def test_fill_missing_bands_accepts_model_with_encoder() -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _encoder(v1=False)

    out = fill_missing_bands(_sample(), {S2.name: ["B01"]}, Wrapper())
    assert (_band(out, S2, "B01") == 0).all()


def test_fill_missing_bands_errors() -> None:
    encoder = _encoder(v1=True)
    with pytest.raises(ValueError, match="Unknown band"):
        fill_missing_bands(_sample(), {S2.name: ["B13"]}, encoder)
    with pytest.raises(ValueError, match="not present in the sample"):
        fill_missing_bands(_sample(), {LANDSAT.name: ["B1"]}, encoder)
    with pytest.raises(TypeError, match="Expected an Encoder"):
        fill_missing_bands(_sample(), {S2.name: ["B01"]}, torch.nn.Linear(1, 1))


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
