"""OlmoEarth Pretrain v1 model initialization.

This module provides a simple interface to initialize OlmoEarth v1 models.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.latent_mim import (
    LatentMIMConfig,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

# Model size configurations matching the official OlmoEarth v1 models
MODEL_SIZE_CONFIGS = {
    "nano_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 128,
        "decoder_embedding_size": 128,
        "encoder_depth": 4,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "small_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 384,
        "decoder_embedding_size": 384,
        "encoder_depth": 12,
        "encoder_num_heads": 6,
        "decoder_num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "tiny_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
}

# Default modalities used in OlmoEarth v1 training
DEFAULT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

V1_SUPPORTED_SIZES = ("nano", "tiny", "base", "large")
# v1.1 adds a per-pixel hidden layer before patchification, band dropout, and
# uses the linear patch embed. The hidden size differs by model size.
V1_1_PATCH_EMBED_HIDDEN_SIZES = {
    "nano": [12],
    "tiny": [64],
    "base": [64],
}
V1_1_BAND_DROPOUT_RATE = 0.2
V1_1_BAND_DROPOUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.LANDSAT.name,
]
V1_1_SUPPORTED_SIZES = ("nano", "tiny", "base")

# v1.2 replaces the sequence and spatial absolute encodings with rope encodings
V1_2_POS_ENCODING = "rope_3d_mixed"
V1_2_ROPE_MIXED_BASE = 10.0
V1_2_ROPE_TEMPORAL_COORDINATE_SCALE = 1.0 / 30.0
V1_2_SUPPORTED_SIZES = ("nano", "small", "tiny", "base")


class OlmoEarthPretrain_v1(torch.nn.Module):
    """OlmoEarth Pretrain v1 model.

    This class provides a simple interface to initialize OlmoEarth v1 models
    directly from the repository. Models are initialized with random weights.

    """

    def __init__(
        self,
        model_size: Literal["nano", "tiny", "small", "base", "large"] = "nano",
        model_version: Literal["v1", "v1.1", "v1.2"] = "v1.2",
        supported_modality_names: list[str] | None = None,
        max_patch_size: int = 8,
        max_sequence_length: int = 12,
        drop_path: float = 0.1,
    ) -> None:
        """Initialize an OlmoEarth Pretrain v1 model.

        Args:
            model_size: Size of the model. Options: "nano", "tiny", "base", "large"
                ("large" is v1 only).
            model_version: Which model version to build.
            supported_modality_names: List of modality names to support. If None,
                uses the default modalities from OlmoEarth v1 training.
            max_patch_size: Maximum patch size for the encoder.
            max_sequence_length: Maximum sequence length.
            drop_path: Drop path rate for regularization.
        """
        super().__init__()

        # Map user-facing model size to internal config key with shallow_decoder suffix
        config_key = f"{model_size}_shallow_decoder"
        if config_key not in MODEL_SIZE_CONFIGS:
            raise ValueError(
                f"Invalid model_size: {model_size}. "
                f"Must be one of {['nano', 'small', 'tiny', 'base', 'large']}"
            )

        if model_version == "v1" and model_size not in V1_SUPPORTED_SIZES:
            raise ValueError(
                f"model_size {model_size!r} is not available for v1 "
                f"Must be one of {list(V1_SUPPORTED_SIZES)}"
            )

        if model_version == "v1.1" and model_size not in V1_1_SUPPORTED_SIZES:
            raise ValueError(
                f"model_size {model_size!r} is not available for v1.1. "
                f"Must be one of {list(V1_1_SUPPORTED_SIZES)}"
            )

        if model_version == "v1.2" and model_size not in V1_2_SUPPORTED_SIZES:
            raise ValueError(
                f"model_size {model_size!r} is not available for v1.2. "
                f"Must be one of {list(V1_2_SUPPORTED_SIZES)}"
            )

        if supported_modality_names is None:
            supported_modality_names = DEFAULT_MODALITIES

        model_config = MODEL_SIZE_CONFIGS[config_key]

        encoder_extra_kwargs: dict[str, Any] = {}
        if model_version in ["v1.1", "v1.2"]:
            encoder_extra_kwargs = {
                "use_linear_patch_embed": True,
                "patch_embed_hidden_sizes": V1_1_PATCH_EMBED_HIDDEN_SIZES[model_size],
                "band_dropout_rate": V1_1_BAND_DROPOUT_RATE,
                "random_band_dropout": True,
                "band_dropout_modalities": V1_1_BAND_DROPOUT_MODALITIES,
            }
            if model_version == "v1.2":
                encoder_extra_kwargs.update(
                    {
                        "position_encoding": V1_2_POS_ENCODING,
                        "rope_mixed_base": V1_2_ROPE_MIXED_BASE,
                        "rope_temporal_coordinate_scale": V1_2_ROPE_TEMPORAL_COORDINATE_SCALE,
                    }
                )

        # Build encoder config
        encoder_config = EncoderConfig(
            embedding_size=int(model_config["encoder_embedding_size"]),
            num_heads=int(model_config["encoder_num_heads"]),
            depth=int(model_config["encoder_depth"]),
            mlp_ratio=model_config["mlp_ratio"],
            supported_modality_names=supported_modality_names,
            max_patch_size=max_patch_size,
            drop_path=drop_path,
            max_sequence_length=max_sequence_length,
            **encoder_extra_kwargs,
        )

        # Build decoder config
        decoder_config = PredictorConfig(
            encoder_embedding_size=int(model_config["encoder_embedding_size"]),
            decoder_embedding_size=int(model_config["decoder_embedding_size"]),
            depth=int(model_config["decoder_depth"]),
            mlp_ratio=model_config["mlp_ratio"],
            num_heads=int(model_config["decoder_num_heads"]),
            supported_modality_names=supported_modality_names,
            max_sequence_length=max_sequence_length,
        )

        # Build model config and initialize the model
        model_config_obj = LatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

        self.model = model_config_obj.build()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the model."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
