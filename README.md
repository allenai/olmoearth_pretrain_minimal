# OlmoEarth Pretrain Minimal

A minimal package for loading and initializing OlmoEarth v1 models. This package contains only the code necessary to load models from Hugging Face or initialize them with random weights, without training or evaluation dependencies.


## Installation

Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install dependencies:
```
git clone git@github.com:allenai/olmoearth_pretrain_minimal.git
cd olmoearth_pretrain_minimal
# Install with CPU-only PyTorch (works on all platforms including Linux)
uv sync --locked --python 3.12 --extra torch-cpu
# Or install with CUDA 12.8 PyTorch
uv sync --locked --python 3.12 --extra torch-cu128
```
uv installs everything into a venv, so to keep using python commands you can activate uv's venv: `source .venv/bin/activate`. Otherwise, swap to `uv run python`.

**Note:** You must specify either `--extra torch-cpu` or `--extra torch-cu128` to install PyTorch. This allows you to explicitly choose the CPU or GPU version regardless of your platform, which is especially useful for CI environments that need CPU-only builds on Linux.


## Model Summary

<img src="https://raw.githubusercontent.com/allenai/olmoearth_pretrain/main/assets/model.png" alt="Model Architecture Diagram" style="width: 800px; margin-left:'auto' margin-right:'auto' display:'block'"/>

The OlmoEarth models are trained on three satellite modalities (Sentinel 2, Sentinel 1 and Landsat) and six derived maps (OpenStreetMap, WorldCover, USDA Cropland Data Layer, SRTM DEM, WRI Canopy Height Map, and WorldCereal).
| Model Size | Weights | Encoder Params | Decoder Params |
| --- | --- | --- | --- |
| Nano | [link](https://huggingface.co/allenai/OlmoEarth-v1-Nano) | 1.4M | 800K |
| Tiny | [link](https://huggingface.co/allenai/OlmoEarth-v1-Tiny) | 6.2M | 1.9M |
| Base | [link](https://huggingface.co/allenai/OlmoEarth-v1-Base) | 89M | 30M |
| Large | [link](https://huggingface.co/allenai/OlmoEarth-v1-Large) | 308M | 53M |


## Usage

### Loading Models from Hugging Face

The recommended way to load models is using the model loader, which downloads the model configuration from Hugging Face:

```python
from olmoearth_pretrain_minimal import ModelID, load_model_from_id

# Load a model from Hugging Face with pre-trained weights
# - ModelID.OLMOEARTH_V1_NANO - 1.4M encoder params, 800K decoder params
# - ModelID.OLMOEARTH_V1_TINY - 6.2M encoder params, 1.9M decoder params
# - ModelID.OLMOEARTH_V1_BASE - 89M encoder params, 30M decoder params
# - ModelID.OLMOEARTH_V1_LARGE - 308M encoder params, 53M decoder params
model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=True)

# Load with randomly initialized weights 
model_with_weights = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=True)
```


### Direct Model Initialization (Custom Configuration)

For custom configurations (e.g., custom modalities), you can directly instantiate the model class:

```python
from olmoearth_pretrain_minimal import OlmoEarthPretrain_v1

# Initialize with custom modalities and settings
model = OlmoEarthPretrain_v1(
    model_size="nano",
    supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat"],
    max_patch_size=8,
    max_sequence_length=12,
    drop_path=0.1,
)
```

### Manual Weight Loading

If you have pre-trained weights in a separate file, you can load them manually:

```python
from olmoearth_pretrain_minimal import ModelID, load_model_from_id
import torch

# Load model without weights
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)

# Load pre-trained weights from a separate file
weights = torch.load("path/to/weights.pth")
model.load_state_dict(weights)
```

### Data Normalization

The model expects normalized input data. Use the `Normalizer` class to normalize your data before passing it to the model.

**Important:** Data must be provided with bands in the specific order expected by each modality. See the band order section below.

```python
import torch
import numpy as np

from olmoearth_pretrain_minimal import load_model_from_id, ModelID, Normalizer
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample

# Initialize normalizer
normalizer = Normalizer(std_multiplier=2.0)

# Prepare Sentinel-2 L2A data: (batch, height, width, time, bands)
# Bands must match Modality.SENTINEL2_L2A.band_order (12 bands)
sentinel2_data = np.random.rand(1, 128, 128, 12, 12).astype(np.float32)

# Normalize the data
normalized_sentinel2 = normalizer.normalize(Modality.SENTINEL2_L2A, sentinel2_data)

model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=True)
model.eval()

# Create minimal sample (timestamps required, month must be long for embedding)
timestamps = torch.zeros(1, 12, 3, dtype=torch.long)
timestamps[:, :, 1] = torch.arange(12, dtype=torch.long)  # months 0-11

sample = MaskedOlmoEarthSample(
    timestamps=timestamps,
    sentinel2_l2a=torch.from_numpy(normalized_sentinel2).float(),
    sentinel2_l2a_mask=torch.zeros(1, 128, 128, 12, dtype=torch.long),
)

with torch.no_grad():
    output = model.encoder(sample, patch_size=8, input_res=10, fast_pass=True)
```
```

### Expected Band Orders

The model expects data with bands in a specific order for each modality. Use `Modality.<MODALITY_NAME>.band_order` to get the correct order:

```python
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

# Sentinel-2 L2A band order (12 bands)
print(Modality.SENTINEL2_L2A.band_order)
# ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09']

# Sentinel-1 band order (2 bands)
print(Modality.SENTINEL1.band_order)
# ['vv', 'vh']

# Landsat band order (11 bands)
print(Modality.LANDSAT.band_order)
# ['B8', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9', 'B10', 'B11']

# WorldCover band order (1 band)
print(Modality.WORLDCOVER.band_order)
# ['B1']

# SRTM band order (1 band)
print(Modality.SRTM.band_order)
# ['srtm']
```

**Key points:**
- The last dimension of your data array must match the band order exactly
- For multitemporal modalities (Sentinel-2, Sentinel-1, Landsat), data shape is `(batch, height, width, time, bands)`
- For single-temporal modalities (WorldCover, SRTM, etc.), data shape is `(batch, height, width, bands)`
- Always normalize data using the `Normalizer` before passing to the model

### Note

For the full package with training and evaluation capabilities, see the main `olmoearth_pretrain` package.

