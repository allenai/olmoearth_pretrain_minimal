"""Load the OlmoEarth models from Hugging Face.

This module supports loading models by version (v1, v2, etc.).
Currently only v1 is supported.

The weights are converted to pth file from distributed checkpoint like this:

    import json
    from pathlib import Path

    import torch

    from olmo_core.config import Config
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    checkpoint_path = Path("/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000")
    with (checkpoint_path / "config.json").open() as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    model = model_config.build()

    train_module_dir = checkpoint_path / "model_and_optim"
    load_model_and_optim_state(str(train_module_dir), model)
    torch.save(model.state_dict(), "OlmoEarth-v1-Nano.pth")
"""

import json
from enum import StrEnum
from os import PathLike

import torch
from huggingface_hub import hf_hub_download
from upath import UPath

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pth"


class ModelID(StrEnum):
    """OlmoEarth pre-trained model ID."""

    OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"
    OLMOEARTH_V1_TINY = "OlmoEarth-v1-Tiny"
    OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"
    OLMOEARTH_V1_LARGE = "OlmoEarth-v1-Large"

    def repo_id(self) -> str:
        """Return the Hugging Face repo ID for this model."""
        return f"allenai/{self.value}"

    def version(self) -> str:
        """Return the model version (e.g., 'v1')."""
        if self.value.startswith("OlmoEarth-v1-"):
            return "v1"
        raise ValueError(f"Unknown version for model {self.value}")


def load_model_from_id(
    model_id: ModelID, load_weights: bool = True, version: str | None = None
) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from Hugging Face.

    Args:
        model_id: the model ID to load.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
            the config.json will still be downloaded from Hugging Face.
        version: the model version (e.g., 'v1'). If None, will be inferred from model_id.
    """
    if version is None:
        version = model_id.version()

    config_fpath = _resolve_artifact_path(model_id, CONFIG_FILENAME)
    model = _load_model_from_config(config_fpath, version)

    if not load_weights:
        return model

    state_dict_fpath = _resolve_artifact_path(model_id, WEIGHTS_FILENAME)
    state_dict = _load_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def load_model_from_path(
    model_path: PathLike | str,
    load_weights: bool = True,
    version: str | None = None,
) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from a path.

    Args:
        model_path: the path to the model.
        load_weights: whether to load the weights. Set false to skip loading the
            weights and leave them randomly initialized. Note that the config.json
            will still be loaded.
        version: the model version (e.g., 'v1'). If None, will be inferred from config.
    """
    config_fpath = _resolve_artifact_path(model_path, CONFIG_FILENAME)

    if version is None:
        version = _infer_version_from_config(config_fpath)

    model = _load_model_from_config(config_fpath, version)

    if not load_weights:
        return model

    state_dict_fpath = _resolve_artifact_path(model_path, WEIGHTS_FILENAME)
    state_dict = _load_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def _resolve_artifact_path(
    model_id_or_path: ModelID | PathLike | str, filename: str
) -> UPath:
    """Resolve the artifact file path for the specified model ID or path, downloading it from Hugging Face if necessary."""
    if isinstance(model_id_or_path, ModelID):
        return UPath(
            hf_hub_download(repo_id=model_id_or_path.repo_id(), filename=filename)  # nosec
        )
    base = UPath(model_id_or_path)
    return base / filename


def _infer_version_from_config(path: UPath) -> str:
    """Infer the model version from the config file.

    Args:
        path: Path to the config.json file.

    Returns:
        The model version (e.g., 'v1').

    Raises:
        ValueError: If the version cannot be inferred.
    """
    with path.open() as f:
        config_dict = json.load(f)

    # Try to get version from config
    if "version" in config_dict:
        return config_dict["version"]

    # Try to infer from model class name in config
    model_config = config_dict.get("model", {})
    if isinstance(model_config, dict):
        class_name = model_config.get("_CLASS_", "")
        if "v1" in class_name.lower() or "olmoearth_pretrain_v1" in class_name.lower():
            return "v1"

    # Default to v1 if we can't determine
    return "v1"


def _load_model_from_config(path: UPath, version: str) -> torch.nn.Module:
    """Load the model config from the specified path and build the model.

    Args:
        path: Path to the config.json file.
        version: The model version (e.g., 'v1').

    Returns:
        The initialized model.

    Raises:
        ValueError: If the version is not supported.
    """
    if version == "v1":
        return _load_v1_model_from_config(path)
    else:
        raise ValueError(f"Unsupported model version: {version}. Currently only 'v1' is supported.")


def _load_v1_model_from_config(path: UPath) -> torch.nn.Module:
    """Load a v1 model from config.

    This can either load from a full config.json (with model config) or
    use the OlmoEarthPretrain_v1 class directly if the config has model_size info.
    """
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.config import Config
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.latent_mim import LatentMIMConfig

    with path.open() as f:
        config_dict = json.load(f)

    # Check if we have a full model config
    if "model" in config_dict:
        model_config_dict = config_dict["model"]
        cleaned = None
        # First, try to use _clean_data to resolve _CLASS_ fields
        try:
            # _clean_data will resolve _CLASS_ fields and return the proper config instance
            cleaned = Config._clean_data(model_config_dict)
            # If _clean_data resolved a config class, it should be an instance with build()
            if hasattr(cleaned, "build") and callable(getattr(cleaned, "build")):
                return cleaned.build()
        except (TypeError, AttributeError, KeyError, NotImplementedError):
            pass
        
        # If _clean_data didn't resolve to a config instance, try LatentMIMConfig.from_dict
        try:
            # Use the cleaned dict if it's a dict, otherwise use the original
            config_to_use = cleaned if (cleaned is not None and isinstance(cleaned, dict)) else model_config_dict
            model_config = LatentMIMConfig.from_dict(config_to_use)
            return model_config.build()
        except (TypeError, AttributeError, KeyError, NotImplementedError):
            # Fall through to try model_size approach
            pass

    # Try to infer model_size from config or model_id
    model_size = None
    if "model_size" in config_dict:
        model_size = config_dict["model_size"]
    elif "model" in config_dict and isinstance(config_dict["model"], dict):
        if "model_size" in config_dict["model"]:
            model_size = config_dict["model"]["model_size"]

    # If we have model_size, use OlmoEarthPretrain_v1
    if model_size is not None:
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1 import OlmoEarthPretrain_v1

        return OlmoEarthPretrain_v1(model_size=model_size)

    # Last resort: try to build from LatentMIMConfig
    try:
        model_config = LatentMIMConfig.from_dict(config_dict.get("model", config_dict))
        return model_config.build()
    except (TypeError, AttributeError, KeyError) as e:
        raise ValueError(
            f"Could not load model from config. Config must contain either a 'model' key "
            f"with a valid model config, or a 'model_size' key. Error: {e}"
        ) from e


def _load_state_dict(path: UPath) -> dict[str, torch.Tensor]:
    """Load the model state dict from the specified path."""
    with path.open("rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    return state_dict

