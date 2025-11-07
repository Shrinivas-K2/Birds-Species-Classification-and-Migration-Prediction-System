#!/usr/bin/env python3
"""
Script to check if all required libraries are installed.
Run this script to verify your backend dependencies.
"""

import sys
from importlib import import_module

# Required libraries and their import names
REQUIRED_LIBRARIES = {
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'python-multipart': None,  # No direct import, checked via FastAPI
    'torch': 'torch',
    'torchvision': 'torchvision',
    'tensorflow': 'tensorflow',
    'Pillow': 'PIL',
    'numpy': 'numpy',
    'pydantic': 'pydantic',
}

# Optional libraries (nice to have)
OPTIONAL_LIBRARIES = {
    'python-multipart': 'multipart',
}

def check_library(lib_name, import_name=None):
    """Check if a library is installed."""
    if import_name is None:
        import_name = lib_name
    
    try:
        import_module(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    return version.major >= 3 and version.minor >= 8

def main():
    print("=" * 60)
    print("Backend Library Installation Checker")
    print("=" * 60)
    print()
    
    # Check Python version
    print("Checking Python version...")
    if check_python_version():
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Need Python 3.8+")
    print()
    
    # Check required libraries
    print("Checking required libraries...")
    print("-" * 60)
    
    all_installed = True
    missing_libs = []
    
    for lib_name, import_name in REQUIRED_LIBRARIES.items():
        if import_name is None:
            # Special case for python-multipart
            # Check if FastAPI can handle multipart (it requires this)
            try:
                from fastapi import File, UploadFile
                installed, error = True, None
            except ImportError:
                installed, error = False, "FastAPI multipart support not available"
        else:
            installed, error = check_library(lib_name, import_name)
        
        if installed:
            # Try to get version
            try:
                if import_name:
                    mod = import_module(import_name)
                    version = getattr(mod, '__version__', 'unknown')
                    print(f"✅ {lib_name:20s} - Installed (version: {version})")
                else:
                    print(f"✅ {lib_name:20s} - Installed")
            except:
                print(f"✅ {lib_name:20s} - Installed")
        else:
            print(f"❌ {lib_name:20s} - NOT INSTALLED")
            if error:
                print(f"   Error: {error}")
            missing_libs.append(lib_name)
            all_installed = False
    
    print()
    print("=" * 60)
    
    if all_installed:
        print("✅ All required libraries are installed!")
        print()
        print("You can now start the backend server with:")
        print("  uvicorn main:app --reload")
    else:
        print("❌ Some libraries are missing!")
        print()
        print("Missing libraries:")
        for lib in missing_libs:
            print(f"  - {lib}")
        print()
        print("To install missing libraries, run:")
        print("  pip install -r requirements.txt")
        print()
        print("Or install individually:")
        for lib in missing_libs:
            if lib == 'python-multipart':
                print(f"  pip install {lib}")
            elif lib == 'torch' or lib == 'torchvision':
                print(f"  pip install {lib} --index-url https://download.pytorch.org/whl/cpu")
            else:
                print(f"  pip install {lib}")
    
    print("=" * 60)
    
    # Additional checks
    print()
    print("Additional Information:")
    print("-" * 60)
    
    # Check if running in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment (recommended to use venv)")
    
    # Check PyTorch device
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available (GPU support)")
        else:
            print("ℹ️  PyTorch CPU only (no GPU)")
    except:
        pass
    
    # Check TensorFlow device
    try:
        import tensorflow as tf
        print(f"ℹ️  TensorFlow version: {tf.__version__}")
        if tf.config.list_physical_devices('GPU'):
            print("✅ TensorFlow GPU available")
        else:
            print("ℹ️  TensorFlow CPU only")
    except:
        pass

if __name__ == "__main__":
    main()

