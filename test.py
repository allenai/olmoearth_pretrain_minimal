"""Example code for initializing OlmoEarth models using the model loader."""

from olmoearth_pretrain_minimal import ModelID, load_model_from_id, OlmoEarthPretrain_v1

# Example 1: Load a nano model from Hugging Face (without weights, randomly initialized)
print("Loading OlmoEarth-v1-Nano from Hugging Face (random weights)...")
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
print(f"Model initialized: {type(model)}")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Example 2: Load different model sizes
print("\nLoading OlmoEarth-v1-Tiny from Hugging Face (random weights)...")
model_tiny = load_model_from_id(ModelID.OLMOEARTH_V1_TINY, load_weights=False)
print(f"Model initialized: {type(model_tiny)}")
print(f"Model has {sum(p.numel() for p in model_tiny.parameters())} parameters")

print("\nLoading OlmoEarth-v1-Base from Hugging Face (random weights)...")
model_base = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=False)
print(f"Model initialized: {type(model_base)}")
print(f"Model has {sum(p.numel() for p in model_base.parameters())} parameters")

print("\nLoading OlmoEarth-v1-Large from Hugging Face (random weights)...")
model_large = load_model_from_id(ModelID.OLMOEARTH_V1_LARGE, load_weights=False)
print(f"Model initialized: {type(model_large)}")
print(f"Model has {sum(p.numel() for p in model_large.parameters())} parameters")

# Example 3: Load with pre-trained weights (if available)
print("\nLoading OlmoEarth-v1-Nano with pre-trained weights...")
try:
    model_with_weights = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=True)
    print(f"Model loaded with pre-trained weights: {type(model_with_weights)}")
except Exception as e:
    print(f"Could not load weights (this is expected if weights are not yet available): {e}")

# Example 4: Initialize with custom modalities (requires direct instantiation)
print("\nInitializing with custom modalities (direct instantiation)...")
custom_model = OlmoEarthPretrain_v1(
    model_size="nano",
    supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat"],
)
print(f"Model initialized with custom modalities: {type(custom_model)}")

