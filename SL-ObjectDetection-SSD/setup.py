#!/usr/bin/env python3
"""
Setup script for SL-ObjectDetection-SSD

This script helps users set up the SSD object detection project.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    else:
        print(f"✓ Python {version.major}.{version.minor} is compatible")
        return True


def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("✗ pip not found. Please install pip first.")
        return False
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        success = run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                            "Installing requirements")
    else:
        print("✗ requirements.txt not found")
        return False
    
    return success


def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        "models/pretrained",
        "models/checkpoints", 
        "outputs",
        "sample_images",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True


def download_models():
    """Download pre-trained models"""
    print("\nSetting up pre-trained models...")
    
    if os.path.exists("download_models.py"):
        success = run_command(f"{sys.executable} download_models.py", 
                            "Downloading pre-trained models")
    else:
        print("✗ download_models.py not found")
        return False
    
    return success


def run_tests():
    """Run basic tests to verify installation"""
    print("\nRunning basic tests...")
    
    # Test imports
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (will use CPU)")
    
    return True


def run_example():
    """Run a simple example to verify everything works"""
    print("\nRunning example to verify installation...")
    
    if os.path.exists("examples/basic_detection.py"):
        success = run_command(f"{sys.executable} examples/basic_detection.py", 
                            "Running basic detection example")
    else:
        print("⚠ examples/basic_detection.py not found, skipping example")
        success = True
    
    return success


def main():
    """Main setup function"""
    print("SL-ObjectDetection-SSD Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\nSetup failed: Incompatible Python version")
        return False
    
    # Create directories
    if not create_directories():
        print("\nSetup failed: Could not create directories")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\nSetup failed: Could not install dependencies")
        return False
    
    # Download models
    if not download_models():
        print("\nSetup failed: Could not download models")
        return False
    
    # Run tests
    if not run_tests():
        print("\nSetup failed: Tests failed")
        return False
    
    # Run example
    if not run_example():
        print("\nSetup failed: Example failed")
        return False
    
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("=" * 50)
    print("\nYou can now use the SSD object detection system:")
    print("\nBasic usage:")
    print("  python examples/basic_detection.py")
    print("\nReal-time detection:")
    print("  python examples/real_time_detection.py")
    print("\nTraffic monitoring:")
    print("  python examples/traffic_monitoring.py")
    print("\nTutorial:")
    print("  python notebooks/ssd_tutorial.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
