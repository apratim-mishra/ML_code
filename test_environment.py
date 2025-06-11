#!/usr/bin/env python3
"""
Test script to verify ML/DL environment setup
"""

import sys
from importlib import import_module

def test_import(module_name, description=""):
    """Test if a module can be imported"""
    try:
        import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description} (Error: {e})")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - {description} (Warning: {e})")
        return True

def main():
    """Test all important ML/DL libraries"""
    print("🧪 Testing Machine Learning Environment")
    print("=" * 50)
    
    # Core libraries
    print("\n📊 Core Data Science Libraries:")
    test_import("pandas", "Data manipulation and analysis")
    test_import("numpy", "Numerical computing")
    test_import("scipy", "Scientific computing")
    
    # Machine Learning
    print("\n🤖 Machine Learning Libraries:")
    test_import("sklearn", "Scikit-learn")
    test_import("xgboost", "XGBoost")
    test_import("lightgbm", "LightGBM")
    test_import("catboost", "CatBoost")
    
    # Deep Learning
    print("\n🔥 Deep Learning Libraries:")
    test_import("torch", "PyTorch")
    test_import("torchvision", "PyTorch Vision")
    test_import("torchaudio", "PyTorch Audio")
    test_import("pytorch_lightning", "PyTorch Lightning")
    
    # NLP and Transformers
    print("\n🗣️  NLP Libraries:")
    test_import("transformers", "Hugging Face Transformers")
    test_import("tokenizers", "Tokenizers")
    test_import("datasets", "Hugging Face Datasets")
    
    # Computer Vision
    print("\n👁️  Computer Vision:")
    test_import("cv2", "OpenCV")
    test_import("PIL", "Pillow")
    test_import("albumentations", "Image augmentations")
    
    # Visualization
    print("\n📈 Visualization Libraries:")
    test_import("matplotlib", "Matplotlib")
    test_import("seaborn", "Seaborn")
    test_import("plotly", "Plotly")
    
    # Development Tools
    print("\n🛠️  Development Tools:")
    test_import("jupyter", "Jupyter")
    test_import("IPython", "IPython")
    
    # PyTorch specific tests
    print("\n🔍 PyTorch Specific Tests:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except:
        print("   Could not test PyTorch details")
    
    print("\n" + "=" * 50)
    print("✅ Environment test completed!")

if __name__ == "__main__":
    main() 