"""Normalizer for the OlmoEarth Pretrain dataset."""

import json
import logging
import warnings
from importlib.resources import files

import numpy as np

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.missing_bands import (
    MAX_FILLED_FRACTION,
    copy_nearest_bands,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
    MISSING_VALUE,
    ModalitySpec,
)

logger = logging.getLogger(__name__)

# Model types the Normalizer can fill missing bands for. Every model after v1 was
# trained with band dropout, so they all fill missing bands the same way.
MODEL_TYPES = ("v1", "v1.1", "v1.2", "v1.3")


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
        model_type: str | None = None,
    ) -> None:
        """Initialize the normalizer.

        Args:
            std_multiplier: The multiplier for the standard deviation when computing
                the normalization range. Default is 2.0, which covers approximately
                95% of the data (assuming normal distribution). The normalization
                range is [mean - std_multiplier * std, mean + std_multiplier * std].
            model_type: The model the data is normalized for, which sets how missing
                bands (NaN or ``MISSING_VALUE``) are filled. ``"v1"`` copies each
                missing band from the present band nearest to it in wavelength,
                since v1 was not trained with band dropout. ``"v1.1"``, ``"v1.2"`` and
                ``"v1.3"`` set missing bands to 0 after normalization, which is what
                those models saw for dropped bands. ``None`` leaves missing values
                as they are. See ``NormalizerV1`` and ``NormalizerV1_1``.
        """
        if model_type is not None and model_type not in MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {MODEL_TYPES} or None; got {model_type!r}"
            )
        self.std_multiplier = std_multiplier
        self.model_type = model_type
        self.norm_config = load_computed_config()

    def normalize(self, modality: ModalitySpec, data: np.ndarray) -> np.ndarray:
        """Normalize the data using computed mean and standard deviation values.

        Args:
            modality: The ModalitySpec that defines the modality and band order.
            data: The data to normalize. Shape should be (..., bands) where the last
                dimension corresponds to bands in the order specified by
                `modality.band_order`. With a ``model_type``, missing values can
                be NaN or ``MISSING_VALUE`` and are filled as described in
                ``__init__``.

        Returns:
            The normalized data with the same shape as input. Values are normalized
            to approximately [0, 1] range based on mean ± (std_multiplier * std).

        Raises:
            ValueError: If a band in the modality's band_order is not found in the
                normalization config, or the data has the wrong number of bands.
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

        if data.shape[-1] != len(modality_bands):
            raise ValueError(
                f"{modality.name} data has {data.shape[-1]} bands; expected "
                f"{len(modality_bands)}, in the order {modality_bands}. Pass missing "
                "bands as NaN in their place."
            )

        if self.model_type is None:
            # Normalize: (data - min) / (max - min)
            return (data - min_vals) / (max_vals - min_vals)  # type: ignore

        missing = np.isnan(data) | (data == MISSING_VALUE)
        if self.model_type == "v1" and missing.any():
            data, missing = copy_nearest_bands(modality.name, data, missing)

        # Normalize: (data - min) / (max - min). Missing values that were not copied
        # from other bands are set to 0.
        normalized = (data - min_vals) / (max_vals - min_vals)
        if missing.any():
            self._warn_zeroed(modality, missing)
        return np.where(missing, 0.0, normalized)  # type: ignore

    def _warn_zeroed(self, modality: ModalitySpec, zeroed: np.ndarray) -> None:
        """Warn when zeroed bands are likely to degrade model outputs."""
        worst = int(zeroed.sum(axis=-1).max())
        num_bands = len(modality.band_order)
        if self.model_type != "v1" and worst / num_bands <= MAX_FILLED_FRACTION:
            return
        warnings.warn(
            f"Up to {worst} of the {num_bands} {modality.name} bands were missing and "
            "set to 0 at a pixel and timestep, so model outputs may degrade. Mask band "
            "sets that have no data (MaskValue.MISSING) or leave them out.",
            UserWarning,
            stacklevel=3,
        )


class NormalizerV1(Normalizer):
    """``Normalizer`` for v1 models, which fills missing bands from other bands."""

    def __init__(self, std_multiplier: float = 2.0) -> None:
        """Initialize the normalizer; see ``Normalizer``."""
        super().__init__(std_multiplier, model_type="v1")


class NormalizerV1_1(Normalizer):
    """``Normalizer`` for v1.1 models, which sets missing bands to 0."""

    def __init__(self, std_multiplier: float = 2.0) -> None:
        """Initialize the normalizer; see ``Normalizer``."""
        super().__init__(std_multiplier, model_type="v1.1")


class NormalizerV1_2(NormalizerV1_1):
    """``Normalizer`` for v1.2 models; identical to ``NormalizerV1_1``."""


class NormalizerV1_3(NormalizerV1_1):
    """``Normalizer`` for v1.3 models; identical to ``NormalizerV1_1``."""
