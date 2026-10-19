"""Handle inputs whose modalities are missing some bands.

Missing values are marked with NaN in the data (``Normalizer`` keeps NaN and turns
``MISSING_VALUE`` into NaN) or with ``MaskValue.MISSING`` in a per-band mask.

``Normalizer.normalize(..., fill_missing_bands=True)`` fills each missing band from
the present band nearest to it in wavelength (:func:`copy_nearest_bands`), as
olmoearth_pretrain's evals do. This is recommended for v1, which was not trained with
band dropout.

The encoder calls :func:`handle_missing_bands` on every input. The encoder embeds each
band set of a modality as one token, so missing data can be excluded from attention
only a whole band set at a time:

* A band set with every band missing is masked.
* A partially missing band set raises a warning. Models trained with band dropout
  (v1.1 onward) saw dropped bands as 0 in normalized space, so the missing bands are
  set to 0. Models trained without band dropout (v1) never saw zeroed bands, so the
  whole band set is treated as missing.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

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


def copy_nearest_bands(
    modality: str, data: np.ndarray, missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill each missing band with the present band nearest to it in wavelength.

    This is how olmoearth_pretrain's evals fill bands a dataset lacks (B01 from B02,
    B09 from B8A, B10 from B11), applied to raw (unnormalized) values at the same
    pixel and timestep. Ties go to the shorter wavelength. Bands are only copied
    within a group of ``SPECTRAL_GROUPS``.

    Returns:
        The filled data and the mask of values that are still missing.
    """
    groups = SPECTRAL_GROUPS.get(modality)
    if not groups:
        return data, missing
    band_order = Modality.get(modality).band_order
    filled, still_missing = data.copy(), missing.copy()
    for group in groups:
        for target, wavelength in group.items():
            sources = sorted(
                (band for band in group if band != target),
                key=lambda band: (abs(group[band] - wavelength), group[band]),
            )
            t = band_order.index(target)
            for source in sources:
                s = band_order.index(source)
                copy = still_missing[..., t] & ~missing[..., s]
                filled[..., t] = np.where(copy, data[..., s], filled[..., t])
                still_missing[..., t] &= ~copy
    worst = int((missing & ~still_missing).sum(axis=-1).max(initial=0))
    if worst / len(band_order) > MAX_COPIED_FRACTION:
        warnings.warn(
            f"Up to {worst} of the {len(band_order)} {modality} bands were copied "
            "from other bands at the same pixel and timestep, so model outputs may "
            "degrade.",
            UserWarning,
            stacklevel=3,
        )
    return filled, still_missing


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


def handle_missing_bands(
    sample: MaskedOlmoEarthSample, encoder: Encoder
) -> MaskedOlmoEarthSample:
    """Zero missing bands and mask missing band sets; see the module docstring.

    Also checks that each modality has the expected number of bands and a mask of a
    supported shape, and converts masks to one entry per band set. Samples with no
    missing bands and masks already per band set are returned unchanged.

    Args:
        sample: Sample whose data has already been normalized.
        encoder: The encoder the sample is for.

    Returns:
        A new sample; the input sample is not modified.

    Raises:
        ValueError: If a modality has the wrong number of bands or a mask has an
            unexpected shape.
    """
    zero_partial_band_sets = encoder.band_dropout_rate > 0
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
        band_sets = encoder.tokenization_config.get_bandset_indices(modality)
        input_mask = getattr(sample, mask_name)
        mask, missing = _split_mask(data, input_mask, band_sets, modality)
        if mask is not input_mask:
            updates[mask_name] = mask

        nan = data.isnan()
        missing = nan if missing is None else missing | nan
        if not missing.any():
            continue

        mask = mask.clone()
        partial_band_sets = []
        for index, band_set in enumerate(band_sets):
            num_missing = missing[..., band_set].sum(dim=-1)
            set_missing = num_missing == len(band_set)
            partial = (num_missing > 0) & ~set_missing
            if partial.any():
                partial_band_sets.append([spec.band_order[i] for i in band_set])
                if not zero_partial_band_sets:
                    set_missing = set_missing | partial
                    missing[..., band_set] |= partial.unsqueeze(-1)
            mask[..., index] = mask[..., index].masked_fill(
                set_missing, MaskValue.MISSING.value
            )
        if partial_band_sets:
            if zero_partial_band_sets:
                action = "their missing bands were set to 0"
            else:
                action = (
                    "the model was not trained with band dropout, so these band sets "
                    "were treated as fully missing. Pass fill_missing_bands=True to "
                    "Normalizer.normalize to fill missing bands from other bands"
                )
            warnings.warn(
                f"{modality} band sets {partial_band_sets} are partially missing at "
                f"some pixels and timesteps; {action}.",
                UserWarning,
                stacklevel=3,
            )
        updates[modality] = data.masked_fill(missing, 0)
        updates[mask_name] = mask

    return sample._replace(**updates) if updates else sample
