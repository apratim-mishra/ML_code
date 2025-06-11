#!/bin/bash

# Setup script for ML/DL virtual environment
echo "🚀 Setting up Machine Learning Environment..."

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ml_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other packages
echo "📚 Installing ML/DL libraries..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source ml_env/bin/activate"
echo ""
echo "To deactivate, run:"
echo "deactivate" 