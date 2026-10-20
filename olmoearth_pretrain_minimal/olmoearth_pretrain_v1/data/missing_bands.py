"""Fill missing bands from other bands of the same modality.

Used by ``NormalizerV1`` (see ``data.normalize``): v1 was not trained with band
dropout, so a missing band is filled from the present band nearest to it in
wavelength, as olmoearth_pretrain's evals do.
"""

from __future__ import annotations

import warnings

import numpy as np

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

# Warn when more than this fraction of a modality's bands are filled (copied or zeroed)
# at a pixel and timestep. Matches the band dropout rate of v1.1 and v1.2.
MAX_FILLED_FRACTION = 0.2

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
    if worst / len(band_order) > MAX_FILLED_FRACTION:
        warnings.warn(
            f"Up to {worst} of the {len(band_order)} {modality} bands were copied "
            "from other bands at the same pixel and timestep, so model outputs may "
            "degrade.",
            UserWarning,
            stacklevel=3,
        )
    return filled, still_missing
