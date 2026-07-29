"""Load the OlmoEarth models from Hugging Face.

This module works with or without olmo-core installed:
- Without olmo-core: inference-only mode (loading pre-trained models)
- With olmo-core: full functionality including training

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

from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.config import Config
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pth"


class ModelID(StrEnum):
    """OlmoEarth pre-trained model ID."""

    OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"
    OLMOEARTH_V1_TINY = "OlmoEarth-v1-Tiny"
    OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"
    OLMOEARTH_V1_LARGE = "OlmoEarth-v1-Large"

    OLMOEARTH_V1_1_NANO = "OlmoEarth-v1_1-Nano"
    OLMOEARTH_V1_1_TINY = "OlmoEarth-v1_1-Tiny"
    OLMOEARTH_V1_1_BASE = "OlmoEarth-v1_1-Base"

    OLMOEARTH_V1_2_NANO = "OlmoEarth-v1_2-Nano"
    OLMOEARTH_V1_2_TINY = "OlmoEarth-v1_2-Tiny"
    OLMOEARTH_V1_2_SMALL = "OlmoEarth-v1_2-Small"
    OLMOEARTH_V1_2_BASE = "OlmoEarth-v1_2-Base"

    def repo_id(self) -> str:
        """Return the Hugging Face repo ID for this model."""
        return f"allenai/{self.value}"


# The modalities each pretrained encoder was trained to ingest. The remaining
# modalities in the checkpoint config were decode-only during pretraining
# (only_decode_modalities in the training config): the decoder learned to predict
# them, but the encoder never saw them, so passing them to the encoder degrades
# results. Every ModelID must have an entry here; loading an unregistered model
# raises so that new models explicitly declare their encoder input modalities
# instead of silently having modalities they support filtered out.
_V1_FAMILY_ENCODER_INPUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]
ENCODER_INPUT_MODALITY_NAMES: dict[ModelID, list[str]] = {
    ModelID.OLMOEARTH_V1_NANO: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_TINY: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_BASE: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_LARGE: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_1_NANO: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_1_TINY: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_1_BASE: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_2_NANO: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_2_TINY: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_2_SMALL: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
    ModelID.OLMOEARTH_V1_2_BASE: _V1_FAMILY_ENCODER_INPUT_MODALITIES,
}


def _get_encoder_input_modalities(model_id: ModelID) -> list[str]:
    """Return the modalities the given pretrained model was trained to encode."""
    if model_id not in ENCODER_INPUT_MODALITY_NAMES:
        raise ValueError(
            f"{model_id} has no registered encoder input modalities. Add an entry "
            "to ENCODER_INPUT_MODALITY_NAMES listing the modalities its encoder was "
            "trained on; any other modality is dropped from the encoder input."
        )
    return ENCODER_INPUT_MODALITY_NAMES[model_id]


def _restrict_encoder_inputs(model: torch.nn.Module, modality_names: list[str]) -> None:
    """Restrict the model's encoders to the modalities they were trained on."""
    model.encoder.restrict_input_modalities(modality_names)
    target_encoder = getattr(model, "target_encoder", None)
    if target_encoder is not None:
        target_encoder.restrict_input_modalities(modality_names)


def load_model_from_id(model_id: ModelID, load_weights: bool = True) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from Hugging Face.

    Args:
        model_id: the model ID to load.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
            the config.json will still be downloaded from Hugging Face.
    """
    encoder_input_modalities = _get_encoder_input_modalities(model_id)
    config_fpath = _resolve_artifact_path(model_id, CONFIG_FILENAME)
    model = _load_model_from_config(config_fpath)
    _restrict_encoder_inputs(model, encoder_input_modalities)

    if not load_weights:
        return model

    state_dict_fpath = _resolve_artifact_path(model_id, WEIGHTS_FILENAME)
    state_dict = _load_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def load_model_from_path(
    model_path: PathLike | str, load_weights: bool = True
) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from a path.

    Args:
        model_path: the path to the model.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
    """
    config_fpath = _resolve_artifact_path(model_path, CONFIG_FILENAME)
    config_dict = _read_config(config_fpath)
    model = _build_model_from_config(config_dict)
    _restrict_encoder_inputs(model, _encoder_input_modalities_from_config(config_dict))

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


def _load_model_from_config(path: UPath) -> torch.nn.Module:
    """Load the model config from the specified path."""
    return _build_model_from_config(_read_config(path))


def _read_config(path: UPath) -> dict:
    """Read the full training config from the specified path."""
    with path.open() as f:
        return json.load(f)


def _build_model_from_config(config_dict: dict) -> torch.nn.Module:
    """Build the model from a full training config dictionary."""
    model_config = Config.from_dict(config_dict["model"])
    return model_config.build()


def _encoder_input_modalities_from_config(config_dict: dict) -> list[str]:
    """Derive the modalities the encoder was trained on from the training config.

    The training config records which modalities were decode-only (never seen by
    the encoder) under masking_config.strategy_config.only_decode_modalities; the
    encoder input modalities are the supported modalities minus those. Raises if
    the config does not record this, in which case the accepted modalities must be
    explicitly determined for the model (see ENCODER_INPUT_MODALITY_NAMES).
    """
    decode_only: list[str] | None = None
    for section in ("train_module", "data_loader"):
        decode_only = (
            config_dict.get(section, {})
            .get("masking_config", {})
            .get("strategy_config", {})
            .get("only_decode_modalities")
        )
        if decode_only is not None:
            break
    if decode_only is None:
        raise ValueError(
            "Could not determine which modalities this model was trained to encode: "
            "the config does not record masking_config.strategy_config."
            "only_decode_modalities. Explicitly mark the model's accepted encoder "
            "input modalities (see ENCODER_INPUT_MODALITY_NAMES in model_loader.py) "
            "to avoid passing the encoder modalities it was not trained on."
        )
    supported = config_dict["model"]["encoder_config"]["supported_modality_names"]
    return [modality for modality in supported if modality not in decode_only]


def _load_state_dict(path: UPath) -> dict[str, torch.Tensor]:
    """Load the model state dict from the specified path."""
    with path.open("rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    return state_dict
