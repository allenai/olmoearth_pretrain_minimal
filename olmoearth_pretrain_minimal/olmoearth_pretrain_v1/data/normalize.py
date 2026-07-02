"""Normalizer for the OlmoEarth Pretrain dataset."""

from __future__ import annotations

import functools
import json
import logging
from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
    Modality,
    ModalitySpec,
)

if TYPE_CHECKING:
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
        MaskedOlmoEarthSample,
    )

logger = logging.getLogger(__name__)


def load_computed_config() -> dict[str, dict]:
    """Load the computed config.

    The normalization config maps from modality -> band name to a dictionary with mean
    and std keys.
    """
    with (
        files("olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.norm_configs")
        / "computed.json"
    ).open() as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _cached_computed_config() -> dict[str, dict]:
    """Cache the computed config so per-forward normalization avoids re-reading it."""
    return load_computed_config()


def normalize_sample(
    sample: MaskedOlmoEarthSample, std_multiplier: float = 2.0
) -> MaskedOlmoEarthSample:
    """Normalize a sample's raw modality tensors using the pretrained statistics.

    Mirrors :meth:`Normalizer.normalize` but operates directly on the torch tensors of
    a :class:`MaskedOlmoEarthSample` (each ``(B, H, W, T, C)`` with bands last, in the
    modality's canonical ``band_order``). Every present modality that has entries in the
    computed config is normalized to roughly ``[0, 1]`` via
    ``(x - (mean - k*std)) / (2*k*std)``; modalities without stats (e.g.
    ``openstreetmap_raster``, ``worldcereal``) and non-modality fields (timestamps,
    latlon) are left untouched.

    The inputs must be raw, un-normalized values (Sentinel-1 already in decibels), the
    same expectation as the model itself, which does not normalize internally.

    Args:
        sample: the sample whose modality tensors should be normalized.
        std_multiplier: the ``k`` in ``mean ± k*std`` defining the normalization range.

    Returns:
        A new sample (via ``NamedTuple._replace``) with normalized modality tensors.
    """
    norm_config = _cached_computed_config()
    updates: dict[str, torch.Tensor] = {}
    for name in sample.modalities:
        if name not in norm_config:
            continue
        band_order = Modality.get(name).band_order
        stats = norm_config[name]
        missing = [band for band in band_order if band not in stats]
        if missing:
            logger.warning(
                "Skipping normalization for modality '%s': missing stats for bands %s",
                name,
                missing,
            )
            continue
        data = getattr(sample, name)
        means = torch.tensor(
            [stats[band]["mean"] for band in band_order],
            dtype=data.dtype,
            device=data.device,
        )
        stds = torch.tensor(
            [stats[band]["std"] for band in band_order],
            dtype=data.dtype,
            device=data.device,
        )
        min_vals = means - std_multiplier * stds
        max_vals = means + std_multiplier * stds
        # Bands are the last dim, so the (C,) min/max broadcast over (B, H, W, T, C).
        updates[name] = (data - min_vals) / (max_vals - min_vals)

    if not updates:
        return sample
    return sample._replace(**updates)


class Normalizer:
    """Normalize data using computed mean and standard deviation values.

    The normalizer uses pre-computed statistics from the training dataset to normalize
    input data. Values are normalized to a range based on mean ± (std_multiplier * std),
    which typically covers ~90% of the data distribution.

    Example:
        >>> from olmoearth_pretrain_minimal import Normalizer
        >>> from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        >>> import numpy as np
        >>>
        >>> normalizer = Normalizer(std_multiplier=2.0)
        >>> sentinel2_modality = Modality.SENTINEL2_L2A
        >>> # Data shape: (batch, height, width, time, bands)
        >>> # Bands must be in order: ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09']
        >>> data = np.random.rand(1, 128, 128, 12, 12).astype(np.float32)
        >>> normalized_data = normalizer.normalize(sentinel2_modality, data)
    """

    def __init__(
        self,
        std_multiplier: float = 2.0,
    ) -> None:
        """Initialize the normalizer.

        Args:
            std_multiplier: The multiplier for the standard deviation when computing
                the normalization range. Default is 2.0, which covers approximately
                95% of the data (assuming normal distribution). The normalization
                range is [mean - std_multiplier * std, mean + std_multiplier * std].
        """
        self.std_multiplier = std_multiplier
        self.norm_config = load_computed_config()

    def normalize(self, modality: ModalitySpec, data: np.ndarray) -> np.ndarray:
        """Normalize the data using computed mean and standard deviation values.

        Args:
            modality: The ModalitySpec that defines the modality and band order.
            data: The data to normalize. Shape should be (..., bands) where the last
                dimension corresponds to bands in the order specified by
                `modality.band_order`.

        Returns:
            The normalized data with the same shape as input. Values are normalized
            to approximately [0, 1] range based on mean ± (std_multiplier * std).

        Raises:
            ValueError: If a band in the modality's band_order is not found in the
                normalization config.
        """
        # Get the band order for this modality
        modality_bands = modality.band_order
        modality_norm_values = self.norm_config[modality.name]

        mean_vals = []
        std_vals = []
        for band in modality_bands:
            if band not in modality_norm_values:
                raise ValueError(
                    f"Band '{band}' not found in normalization config for modality '{modality.name}'. "
                    f"Available bands: {list(modality_norm_values.keys())}"
                )
            mean_val = modality_norm_values[band]["mean"]
            std_val = modality_norm_values[band]["std"]
            mean_vals.append(mean_val)
            std_vals.append(std_val)

        # Compute min and max values based on mean ± (std_multiplier * std)
        min_vals = np.array(mean_vals) - self.std_multiplier * np.array(std_vals)
        max_vals = np.array(mean_vals) + self.std_multiplier * np.array(std_vals)

        # Normalize: (data - min) / (max - min)
        return (data - min_vals) / (max_vals - min_vals)  # type: ignore
