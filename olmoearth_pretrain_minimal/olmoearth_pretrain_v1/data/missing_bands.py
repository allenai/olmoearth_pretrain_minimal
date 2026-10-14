"""Handle inputs whose modalities are missing some bands.

Missing values are marked with NaN in the data (``Normalizer`` keeps NaN and turns
``MISSING_VALUE`` into NaN), with ``MaskValue.MISSING`` in a per-band mask, or by
passing band names to :func:`fill_missing_bands`. The encoder calls
:func:`handle_missing_bands` on every input, so NaN and per-band masks are handled
without any extra step.

The encoder embeds each band set of a modality as one token, so missing data can be
excluded from attention only a whole band set at a time. How the rest is handled
depends on how the model was trained:

* Models trained with band dropout (v1.1 onward) saw dropped bands as 0 in normalized
  space. Missing bands are set to 0 and band sets with every band missing are masked.
* Models trained without band dropout (v1) never saw zeroed bands. Each missing band
  is copied from the present band nearest to it in wavelength, as olmoearth_pretrain's
  evals do.

In both cases a pixel and timestep with no band at all is masked. A warning is
raised when more than ``MAX_COPIED_FRACTION`` of a modality's bands are copied at a
pixel and timestep, or when more of a band set is set to 0 than the model was trained
with.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import (
    load_computed_config,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
)

if TYPE_CHECKING:
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import Encoder

# Warn when more than this fraction of a modality's bands are copied from other bands
# at a pixel and timestep. Matches the band dropout rate of v1.1 and v1.2.
MAX_COPIED_FRACTION = 0.2

# All released models were pretrained on data normalized with this multiplier.
TRAINING_STD_MULTIPLIER = 2.0

SPECTRAL_GROUPS: dict[str, list[dict[str, float]]] = {
    Modality.SENTINEL2_L2A.name: [
        {
            "B01": 443,
            "B02": 490,
            "B03": 560,
            "B04": 665,
            "B05": 705,
            "B06": 740,
            "B07": 783,
            "B08": 842,
            "B8A": 865,
            "B09": 945,
            "B11": 1610,
            "B12": 2190,
        }
    ],
    Modality.LANDSAT.name: [
        {
            "B1": 443,
            "B2": 482,
            "B3": 561,
            "B8": 590,
            "B4": 655,
            "B5": 865,
            "B9": 1373,
            "B6": 1609,
            "B7": 2201,
        },
        {"B10": 10900, "B11": 12000},
    ],
}


def _get_encoder(model: nn.Module) -> Encoder:
    """Return the encoder of a model, or the model itself if it is an encoder."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import Encoder

    if isinstance(model, Encoder):
        return model
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, Encoder):
        return encoder
    raise TypeError(
        f"Expected an Encoder or a model with an Encoder at `.encoder`, got "
        f"{type(model).__name__}."
    )


def _split_mask(
    data: torch.Tensor,
    mask: torch.Tensor | None,
    band_sets: list[list[int]],
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return a mask with one entry per band set, and the bands it marks missing.

    Accepts no mask (everything visible to the encoder), a mask without the band
    dimension, a mask with one entry per band, or one with one entry per band set.
    """
    num_band_sets = len(band_sets)
    shape = (*data.shape[:-1], num_band_sets)
    if mask is None:
        mask = torch.full(
            shape, MaskValue.ONLINE_ENCODER.value, dtype=torch.long, device=data.device
        )
        return mask, None
    if tuple(mask.shape) == shape:
        return mask, None
    if tuple(mask.shape) == tuple(data.shape[:-1]):
        return mask.unsqueeze(-1).expand(shape), None
    if tuple(mask.shape) == tuple(data.shape):
        # One entry per band. A band set takes the lowest value of its present bands,
        # so it is visible to the encoder if any of them is.
        band_missing = mask == MaskValue.MISSING.value
        sets = []
        for band_set in band_sets:
            values = mask[..., band_set].masked_fill(
                band_missing[..., band_set], MaskValue.MISSING.value
            )
            sets.append(values.amin(dim=-1))
        return torch.stack(sets, dim=-1), band_missing
    raise ValueError(
        f"{modality}_mask has shape {tuple(mask.shape)}; expected {shape} (one entry "
        f"per band set), {tuple(data.shape)} (one per band) or "
        f"{tuple(data.shape[:-1])} for data of shape {tuple(data.shape)}."
    )


@functools.cache
def _band_ranges(modality: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Lower bound and width of each band's normalization range, in band order."""
    stats = load_computed_config()[modality]
    bands = Modality.get(modality).band_order
    k = TRAINING_STD_MULTIPLIER
    low = tuple(stats[band]["mean"] - k * stats[band]["std"] for band in bands)
    width = tuple(2 * k * stats[band]["std"] for band in bands)
    return low, width


def _copy_nearest_bands(
    data: torch.Tensor, missing: torch.Tensor, modality: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill each missing band with the present band nearest to it in wavelength.

    This is how olmoearth_pretrain's evals fill bands a dataset lacks (B01 from B02,
    B09 from B8A, B10 from B11): the source band's raw value is copied and then
    normalized with the missing band's own statistics. Ties go to the shorter
    wavelength. Bands are only copied within a group of ``SPECTRAL_GROUPS``.
    """
    groups = SPECTRAL_GROUPS.get(modality)
    if not groups:
        return data, missing
    band_order = Modality.get(modality).band_order
    low_list, width_list = _band_ranges(modality)
    low = torch.tensor(low_list, dtype=data.dtype, device=data.device)
    width = torch.tensor(width_list, dtype=data.dtype, device=data.device)
    raw = data * width + low
    filled, still_missing = data.clone(), missing.clone()
    for group in groups:
        for target, wavelength in group.items():
            sources = sorted(
                (band for band in group if band != target),
                key=lambda band: (abs(group[band] - wavelength), group[band]),
            )
            t = band_order.index(target)
            value = torch.zeros_like(data[..., t])
            found = torch.zeros_like(missing[..., t])
            # Nearest last, so it overwrites any farther band that is present.
            for source in reversed(sources):
                s = band_order.index(source)
                present = ~missing[..., s]
                value = torch.where(present, raw[..., s], value)
                found = found | present
            copy = missing[..., t] & found
            filled[..., t] = torch.where(
                copy, (value - low[t]) / width[t], data[..., t]
            )
            still_missing[..., t] = missing[..., t] & ~found
    return filled, still_missing


def handle_missing_bands(
    sample: MaskedOlmoEarthSample,
    encoder: Encoder,
    missing_bands: Mapping[str, Sequence[str]] | None = None,
) -> MaskedOlmoEarthSample:
    """Fill missing bands and mask what cannot be filled; see the module docstring.

    Also checks that each modality has the expected number of bands and a mask of a
    supported shape, and converts masks to one entry per band set. Samples with no
    missing bands and masks already per band set are returned unchanged.

    Args:
        sample: Sample whose data has already been normalized.
        encoder: The encoder the sample is for.
        missing_bands: Optional modality name to names of bands missing everywhere,
            in addition to those marked with NaN or a per-band mask.

    Returns:
        A new sample; the input sample is not modified.

    Raises:
        ValueError: If a modality has the wrong number of bands, a band name is not
            part of its modality, a modality in ``missing_bands`` is not in the
            sample, or a mask has an unexpected shape.
    """
    missing_bands = missing_bands or {}
    for modality in missing_bands:
        if missing_bands[modality] and getattr(sample, modality, None) is None:
            raise ValueError(f"Modality {modality!r} is not present in the sample.")

    tokenization = encoder.tokenization_config
    copy_bands = encoder.band_dropout_rate == 0
    updates: dict[str, torch.Tensor] = {}
    for modality in sample.modalities:
        if modality not in encoder.patch_embeddings.supported_modality_names:
            continue
        data = getattr(sample, modality)
        spec = Modality.get(modality)
        if data.shape[-1] != spec.num_bands:
            raise ValueError(
                f"{modality} has {data.shape[-1]} bands (data shape "
                f"{tuple(data.shape)}); expected {spec.num_bands}, in the order "
                f"{spec.band_order}. Pass missing bands as NaN in their place."
            )
        mask_name = sample.get_masked_modality_name(modality)
        band_sets = tokenization.get_bandset_indices(modality)
        input_mask = getattr(sample, mask_name)
        mask, missing = _split_mask(data, input_mask, band_sets, modality)
        if mask is not input_mask:
            updates[mask_name] = mask

        nan = data.isnan()
        missing = nan if missing is None else missing | nan
        bands = missing_bands.get(modality, [])
        if bands:
            unknown = sorted(set(bands) - set(spec.band_order))
            if unknown:
                raise ValueError(
                    f"Unknown band(s) {unknown} for {modality!r}; "
                    f"valid bands: {spec.band_order}."
                )
            missing = missing.clone()
            missing[..., [spec.band_order.index(band) for band in bands]] = True
        if not missing.any():
            continue

        if copy_bands:
            before = missing
            data, missing = _copy_nearest_bands(data, missing, modality)
            worst = int((before & ~missing).sum(dim=-1).max())
            if worst / spec.num_bands > MAX_COPIED_FRACTION:
                warnings.warn(
                    f"Up to {worst} of the {spec.num_bands} {modality} bands were "
                    "copied from other bands at the same pixel and timestep. The model "
                    "was not trained with missing bands, so its outputs may degrade.",
                    UserWarning,
                    stacklevel=3,
                )
        data = data.masked_fill(missing, 0)
        updates[modality] = data

        trained_with_dropout = not copy_bands and (
            encoder.band_dropout_modalities is None
            or modality in encoder.band_dropout_modalities
        )
        allowed = encoder.band_dropout_rate if trained_with_dropout else 0.0
        mask = mask.clone()
        for index, band_set in enumerate(band_sets):
            num_missing = missing[..., band_set].sum(dim=-1)
            mask[..., index] = mask[..., index].masked_fill(
                num_missing == len(band_set), MaskValue.MISSING.value
            )
            partial = (num_missing > 0) & (num_missing < len(band_set))
            worst = int(num_missing[partial].max()) if partial.any() else 0
            if worst / len(band_set) > allowed:
                names = [spec.band_order[i] for i in band_set]
                how = "could not be copied and were" if copy_bands else "were"
                warnings.warn(
                    f"Up to {worst} of the {len(band_set)} {modality} bands {names} "
                    f"{how} set to 0 in the same token. The model was not trained "
                    f"with that much of a token dropped (band dropout rate "
                    f"{allowed}), so its outputs for these tokens may degrade.",
                    UserWarning,
                    stacklevel=3,
                )
        updates[mask_name] = mask

    return sample._replace(**updates) if updates else sample


def fill_missing_bands(
    sample: MaskedOlmoEarthSample,
    missing_bands: Mapping[str, Sequence[str]],
    model: nn.Module,
) -> MaskedOlmoEarthSample:
    """Mark bands as missing everywhere in a normalized sample and handle them.

    The encoder already handles bands marked with NaN or a per-band mask; use this to
    name missing bands instead, or to inspect what the encoder will see.

    Example:
        >>> sample = fill_missing_bands(
        ...     sample,  # already normalized, see ``data.normalize.Normalizer``
        ...     missing_bands={Modality.SENTINEL2_L2A.name: ["B01", "B09"]},
        ...     model=model,
        ... )

    Args:
        sample: Sample whose data has already been normalized.
        missing_bands: Modality name to names of its missing bands, using the names
            in ``Modality.<MODALITY>.band_order``.
        model: The model the sample is for (an ``Encoder``, or a model with one at
            ``.encoder``).

    Returns:
        A new sample; the input sample is not modified.
    """
    return handle_missing_bands(sample, _get_encoder(model), missing_bands)
