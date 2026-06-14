# Google Colab Setup for TimesFM Training

## Quick Start

### 1. Upload Data to Google Drive
```bash
# Upload your processed data folder to Google Drive
# Structure: My Drive/stockvoli-research/data/processed/
```

### 2. Create Colab Notebook

```python
# Install dependencies
!pip install -q transformers peft torch pandas numpy pyyaml

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set paths
import sys
sys.path.append('/content/drive/MyDrive/stockvoli-research')
import os
os.chdir('/content/drive/MyDrive/stockvoli-research')

# Run training
!python src/model_training_fixed.py
```

### 3. GPU Check
```python
# Check available GPU
!nvidia-smi

# Should show T4 (16GB) or better
```

## Expected Colab Performance

| **Metric** | **Laptop (RTX 4060)** | **Colab (T4)** |
|------------|----------------------|----------------|
| Batch time | 2-5 seconds | 0.5-1 seconds |
| Epoch time | 40-95 minutes | 6-12 minutes |
| 50 epochs | 33-79 hours | 5-10 hours |

## Colab Tips

1. **Use T4 GPU** (Free tier)
2. **Save checkpoints** to Google Drive frequently
3. **Monitor GPU usage** with `!nvidia-smi`
4. **Use Pro tier** for L4/A100 (10x faster)
