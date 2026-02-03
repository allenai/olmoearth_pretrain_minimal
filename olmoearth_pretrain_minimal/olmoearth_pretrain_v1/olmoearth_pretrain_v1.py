"""OlmoEarth Pretrain v1 model initialization.

This module provides a simple interface to initialize OlmoEarth v1 models.
"""

from __future__ import annotations

from typing import Literal

import torch

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.latent_mim import LatentMIM, LatentMIMConfig

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


class OlmoEarthPretrain_v1(torch.nn.Module):
    """OlmoEarth Pretrain v1 model.

    This class provides a simple interface to initialize OlmoEarth v1 models
    directly from the repository. Models are initialized with random weights.

    """

    def __init__(
        self,
        model_size: Literal["nano", "tiny", "base", "large"] = "nano",
        supported_modality_names: list[str] | None = None,
        max_patch_size: int = 8,
        max_sequence_length: int = 12,
        drop_path: float = 0.1,
    ) -> None:
        """Initialize an OlmoEarth Pretrain v1 model.

        Args:
            model_size: Size of the model. Options: "nano", "tiny", "base", "large".
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
                f"Must be one of {['nano', 'tiny', 'base', 'large']}"
            )

        if supported_modality_names is None:
            supported_modality_names = DEFAULT_MODALITIES

        model_config = MODEL_SIZE_CONFIGS[config_key]

        # Build encoder config
        encoder_config = EncoderConfig(
            embedding_size=model_config["encoder_embedding_size"],
            num_heads=model_config["encoder_num_heads"],
            depth=model_config["encoder_depth"],
            mlp_ratio=model_config["mlp_ratio"],
            supported_modality_names=supported_modality_names,
            max_patch_size=max_patch_size,
            drop_path=drop_path,
            max_sequence_length=max_sequence_length,
        )

        # Build decoder config
        decoder_config = PredictorConfig(
            encoder_embedding_size=model_config["encoder_embedding_size"],
            decoder_embedding_size=model_config["decoder_embedding_size"],
            depth=model_config["decoder_depth"],
            mlp_ratio=model_config["mlp_ratio"],
            num_heads=model_config["decoder_num_heads"],
            supported_modality_names=supported_modality_names,
            max_sequence_length=max_sequence_length,
        )

        # Build model config and initialize the model
        model_config_obj = LatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

        self.model = model_config_obj.build()

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

