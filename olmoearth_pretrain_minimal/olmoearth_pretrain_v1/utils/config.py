"""Standalone config handling for olmoearth_pretrain_minimal.

This module provides a minimal Config class for inference-only mode.
It does not depend on olmo-core and supports loading models from JSON configs.

Usage:
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.config import Config

    @dataclass
    class MyConfig(Config):
        ...
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from typing import Any, TypeVar

# olmo-core is not used in the minimal package
OLMO_CORE_AVAILABLE = False


C = TypeVar("C", bound="_StandaloneConfig")


@dataclass
class _StandaloneConfig:
    """Minimal Config for inference-only mode without olmo-core.

    This provides just enough functionality to deserialize model configs from JSON
    and build models. It intentionally does NOT support:
    - OmegaConf-based merging
    - CLI overrides via dotlist
    - YAML loading
    - Validation beyond what dataclasses provide

    For full functionality, install olmo-core.
    """

    CLASS_NAME_FIELD = "_CLASS_"

    @classmethod
    def _resolve_class(cls, class_name: str) -> type | None:
        """Resolve a fully-qualified class name to a class object."""
        if "." not in class_name:
            return None
        
        # Map old package paths to new ones for compatibility
        # Handle both "helios" (old name) and "olmoearth_pretrain" package names
        if class_name.startswith("helios."):
            class_name = class_name.replace("helios.", "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.", 1)
            # Fix common typos in config files
            class_name = class_name.replace("flexihelios", "flexi_vit")
        elif class_name.startswith("olmoearth_pretrain."):
            class_name = class_name.replace("olmoearth_pretrain.", "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.", 1)
        
        *modules, cls_name = class_name.split(".")
        module_name = ".".join(modules)
        try:
            module = import_module(module_name)
            return getattr(module, cls_name)
        except (ImportError, AttributeError):
            return None

    @classmethod
    def _clean_data(cls, data: Any) -> Any:
        """Recursively clean data, resolving _CLASS_ fields to actual instances."""
        if isinstance(data, dict):
            # Check if this dict represents a config class
            class_name = data.get(cls.CLASS_NAME_FIELD)
            
            # First, recursively clean all nested values
            # This will resolve nested configs that have _CLASS_ fields
            cleaned = {}
            for k, v in data.items():
                if k != cls.CLASS_NAME_FIELD:
                    cleaned_value = cls._clean_data(v)
                    cleaned[k] = cleaned_value

            if class_name is not None:
                resolved_cls = cls._resolve_class(class_name)
                if resolved_cls is not None and is_dataclass(resolved_cls):
                    # Get the field names for this dataclass
                    field_names = {f.name for f in fields(resolved_cls)}
                    # Filter to only include valid fields
                    valid_kwargs = {
                        k: v for k, v in cleaned.items() if k in field_names
                    }
                    # Ensure nested dicts that should be Config instances are resolved
                    # The recursive _clean_data() should have resolved them, but resolve any remaining dicts
                    for key, value in list(valid_kwargs.items()):
                        if isinstance(value, dict) and not is_dataclass(value):
                            # Try to resolve as Config using from_dict
                            if cls.CLASS_NAME_FIELD in value:
                                nested_class_name = value[cls.CLASS_NAME_FIELD]
                                nested_resolved_cls = cls._resolve_class(nested_class_name)
                                if nested_resolved_cls is not None and is_dataclass(nested_resolved_cls):
                                    nested_dict = {k: v for k, v in value.items() if k != cls.CLASS_NAME_FIELD}
                                    valid_kwargs[key] = nested_resolved_cls.from_dict(nested_dict)
                                else:
                                    raise ValueError(
                                        f"Could not resolve nested config class '{nested_class_name}' for field '{key}'"
                                    )
                    try:
                        return resolved_cls(**valid_kwargs)
                    except TypeError as e:
                        raise TypeError(
                            f"Failed to instantiate {class_name}: {e}"
                        ) from e
                # If class resolution failed, keep _CLASS_ field in dict for from_dict() to retry
                cleaned[cls.CLASS_NAME_FIELD] = class_name
            return cleaned

        elif isinstance(data, list | tuple):
            cleaned_items = [cls._clean_data(item) for item in data]
            return type(data)(cleaned_items)

        else:
            return data

    @classmethod
    def from_dict(
        cls: type[C], data: dict[str, Any], overrides: list[str] | None = None
    ) -> C:
        """Deserialize from a dictionary, handling nested _CLASS_ fields.

        Args:
            data: Dictionary representation of the config.
            overrides: Ignored in standalone mode (requires olmo-core for dotlist support).

        Returns:
            An instance of the config class.

        Note:
            The `overrides` parameter is accepted for API compatibility but ignored.
            Install olmo-core for full override support.
        """
        if overrides:
            warnings.warn(
                "Config overrides are not supported in standalone mode. "
                "Install olmo-core for full functionality.",
                UserWarning,
                stacklevel=2,
            )

        cleaned = cls._clean_data(data)

        # If _clean_data resolved a config class instance (from _CLASS_ field), return it directly
        if is_dataclass(cleaned) and not isinstance(cleaned, type):
            return cleaned
        elif isinstance(cleaned, cls):
            return cleaned
        elif isinstance(cleaned, dict):
            # Check if the dict has a _CLASS_ field that we should try to resolve
            if cls.CLASS_NAME_FIELD in cleaned:
                class_name = cleaned[cls.CLASS_NAME_FIELD]
                resolved_cls = cls._resolve_class(class_name)
                if resolved_cls is not None and is_dataclass(resolved_cls):
                    config_dict = {k: v for k, v in cleaned.items() if k != cls.CLASS_NAME_FIELD}
                    return resolved_cls.from_dict(config_dict)
                else:
                    raise ValueError(
                        f"Could not resolve class '{class_name}' from _CLASS_ field. "
                        f"Make sure the class exists and is importable."
                    )
            # No _CLASS_ field, try to create base Config instance
            field_names = {f.name for f in fields(cls)}
            valid_kwargs = {k: v for k, v in cleaned.items() if k in field_names}
            return cls(**valid_kwargs)
        else:
            raise TypeError(f"Expected dict or config instance, got {type(cleaned)}")

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        include_class_name: bool = False,
        json_safe: bool = False,
        recurse: bool = True,
    ) -> dict[str, Any]:
        """Convert to a dictionary.

        Args:
            exclude_none: Don't include values that are None.
            exclude_private_fields: Don't include private fields (starting with _).
            include_class_name: Include _CLASS_ field with fully-qualified class name.
            json_safe: Convert non-JSON-safe types to strings.
            recurse: Recursively convert nested dataclasses.

        Returns:
            Dictionary representation of this config.
        """

        def convert(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                result = {}
                if include_class_name:
                    result[self.CLASS_NAME_FIELD] = (
                        f"{obj.__class__.__module__}.{obj.__class__.__name__}"
                    )
                for field in fields(obj):
                    if exclude_private_fields and field.name.startswith("_"):
                        continue
                    value = getattr(obj, field.name)
                    if exclude_none and value is None:
                        continue
                    if recurse:
                        value = convert(value)
                    result[field.name] = value
                return result
            elif isinstance(obj, dict):
                return {k: convert(v) if recurse else v for k, v in obj.items()}
            elif isinstance(obj, list | tuple | set):
                converted = [convert(item) if recurse else item for item in obj]
                if json_safe:
                    return converted
                return type(obj)(converted)
            elif obj is None or isinstance(obj, float | int | bool | str):
                return obj
            elif json_safe:
                return str(obj)
            else:
                return obj

        return convert(self)

    def as_config_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dictionary suitable for serialization.

        This is a convenience wrapper around as_dict() with settings appropriate
        for saving configs to JSON files.
        """
        return self.as_dict(
            exclude_none=True,
            exclude_private_fields=True,
            include_class_name=True,
            json_safe=True,
            recurse=True,
        )

    def validate(self) -> None:
        """Validate the config. Override in subclasses."""
        pass

    def build(self) -> Any:
        """Build the object this config represents.

        Subclasses must implement this method.

        Raises:
            NotImplementedError: Always, unless overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement build()")


# === The unified export ===
# Always use standalone config for minimal package (no olmo-core dependency)
Config = _StandaloneConfig


__all__ = ["Config", "OLMO_CORE_AVAILABLE"]
