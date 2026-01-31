# OlmoEarth Pretrain Minimal

A minimal package for initializing OlmoEarth v1 models directly from the repository. This package contains only the code necessary to initialize models without training or evaluation dependencies. Models are initialized with random weights and can be used for inference or fine-tuning.

## Installation

Install using `uv`:

```bash
uv pip install -e .
```

## Usage

### Initializing a Model

```python
from olmoearth_pretrain import OlmoEarthPretrain_v1

# Initialize a nano model (default)
model = OlmoEarthPretrain_v1(model_size="nano")

# Available model sizes:
# - "nano" - 1.4M encoder params, 800K decoder params
# - "tiny" - 6.2M encoder params, 1.9M decoder params
# - "base" - 89M encoder params, 30M decoder params
# - "large" - 308M encoder params, 53M decoder params

# Initialize different model sizes
model_tiny = OlmoEarthPretrain_v1(model_size="tiny")
model_base = OlmoEarthPretrain_v1(model_size="base")
model_large = OlmoEarthPretrain_v1(model_size="large")
```

### Custom Configuration

You can customize the model initialization:

```python
from olmoearth_pretrain import OlmoEarthPretrain_v1

# Initialize with custom modalities
model = OlmoEarthPretrain_v1(
    model_size="nano",
    supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat"],
    max_patch_size=8,
    max_sequence_length=12,
    drop_path=0.1,
)
```

### Loading Pre-trained Weights

If you have pre-trained weights, you can load them after initialization:

```python
from olmoearth_pretrain import OlmoEarthPretrain_v1
import torch

model = OlmoEarthPretrain_v1(model_size="nano")

# Load pre-trained weights
weights = torch.load("path/to/weights.pth")
model.load_state_dict(weights)
```

### Note

For the full package with training and evaluation capabilities, see the main `olmoearth_pretrain` package.

