"""Example code for initializing OlmoEarth models."""

from olmoearth_pretrain import OlmoEarthPretrain_v1

# Example 1: Initialize a nano model (default)
print("Initializing OlmoEarth-v1-Nano...")
model = OlmoEarthPretrain_v1(model_size="nano")
print(f"Model initialized: {type(model)}")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Example 2: Initialize different model sizes
print("\nInitializing OlmoEarth-v1-Tiny...")
model_tiny = OlmoEarthPretrain_v1(model_size="tiny")
print(f"Model initialized: {type(model_tiny)}")

print("\nInitializing OlmoEarth-v1-Base...")
model_base = OlmoEarthPretrain_v1(model_size="base")
print(f"Model initialized: {type(model_base)}")

print("\nInitializing OlmoEarth-v1-Large...")
model_large = OlmoEarthPretrain_v1(model_size="large")
print(f"Model initialized: {type(model_large)}")

# Example 3: Initialize with custom modalities
print("\nInitializing with custom modalities...")
custom_model = OlmoEarthPretrain_v1(
    model_size="nano",
    supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat"],
)
print(f"Model initialized with custom modalities: {type(custom_model)}")

