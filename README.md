# Machine Learning & Deep Learning Environment

A comprehensive Python virtual environment setup with all essential machine learning and deep learning libraries.

## 🚀 Quick Start

### 1. Setup Environment
Run the setup script to create and configure the virtual environment:
```bash
./setup_ml_env.sh
```

### 2. Activate Environment
```bash
source ml_env/bin/activate
```

### 3. Test Installation
```bash
python test_environment.py
```

### 4. Deactivate Environment
```bash
deactivate
```

## 📦 Included Libraries

### Core Data Science
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scipy** - Scientific computing

### Machine Learning
- **scikit-learn** - General-purpose ML library
- **xgboost** - Gradient boosting framework
- **lightgbm** - Fast gradient boosting
- **catboost** - Categorical boosting

### Deep Learning
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision for PyTorch
- **torchaudio** - Audio processing for PyTorch
- **pytorch-lightning** - High-level PyTorch wrapper

### NLP & Transformers
- **transformers** - Hugging Face transformers library
- **tokenizers** - Fast tokenizers
- **datasets** - Hugging Face datasets
- **accelerate** - Training acceleration

### Computer Vision
- **opencv-python** - Computer vision library
- **Pillow** - Image processing
- **albumentations** - Image augmentation

### Visualization
- **matplotlib** - Plotting library
- **seaborn** - Statistical visualization
- **plotly** - Interactive plots
- **bokeh** - Interactive visualization

### Development Tools
- **jupyter** - Jupyter notebooks
- **jupyterlab** - JupyterLab interface
- **ipywidgets** - Interactive widgets

### MLOps & Deployment
- **mlflow** - ML lifecycle management
- **wandb** - Experiment tracking
- **tensorboard** - Visualization toolkit
- **fastapi** - Web API framework
- **gradio** - ML app interfaces
- **streamlit** - Data app framework

### Additional Libraries
- **optuna** - Hyperparameter optimization
- **ray[tune]** - Distributed hyperparameter tuning
- **librosa** - Audio analysis
- **networkx** - Network analysis
- **prophet** - Time series forecasting
- **statsmodels** - Statistical modeling

## 🔧 Manual Installation

If you prefer to install packages manually:

```bash
# Create virtual environment
python3 -m venv ml_env

# Activate environment
source ml_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages
pip install -r requirements.txt
```

## 🧪 Testing

The `test_environment.py` script will verify that all major libraries are installed correctly and provide information about your PyTorch/CUDA setup.

## 💡 Usage Examples

### Basic Data Science
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

### Deep Learning with PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
```

### NLP with Transformers
```python
from transformers import AutoTokenizer, AutoModel
import datasets
```

### Computer Vision
```python
import cv2
from PIL import Image
import albumentations as A
```

## 🛠️ Troubleshooting

### CUDA Issues
If you encounter CUDA-related issues:
1. Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Install CPU-only PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### Memory Issues
If you run out of memory during installation:
1. Install packages one by one
2. Use `pip install --no-cache-dir package_name`

### Package Conflicts
If you encounter version conflicts:
1. Create a fresh virtual environment
2. Install packages in the order specified in requirements.txt

## 📝 Notes

- This environment includes both CPU and GPU support for PyTorch
- All packages are set to recent stable versions
- The environment is suitable for research, development, and production use
- Regular updates are recommended to stay current with the latest versions

## 🤝 Contributing

Feel free to suggest additional libraries or improvements to this setup! 