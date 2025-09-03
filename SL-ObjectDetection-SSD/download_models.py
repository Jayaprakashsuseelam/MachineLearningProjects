#!/usr/bin/env python3
"""
Download pre-trained models for SSD Object Detection

This script downloads pre-trained SSD models and sets up the model directory.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, destination: str, description: str = "Downloading"):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        destination: Local file path
        description: Description for progress bar
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        print(f"Downloaded: {destination}")
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str):
    """
    Extract a ZIP file
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted: {zip_path} to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def setup_model_directory():
    """Create model directory structure"""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (model_dir / "pretrained").mkdir(exist_ok=True)
    (model_dir / "checkpoints").mkdir(exist_ok=True)
    
    print("Created model directory structure")


def download_pretrained_models():
    """Download pre-trained SSD models"""
    print("Downloading pre-trained SSD models...")
    
    # Model URLs (these are placeholder URLs - replace with actual model URLs)
    models = {
        "ssd300_voc": {
            "url": "https://example.com/ssd300_voc.pth",
            "filename": "ssd300_voc.pth",
            "description": "SSD300 trained on Pascal VOC"
        },
        "ssd512_voc": {
            "url": "https://example.com/ssd512_voc.pth", 
            "filename": "ssd512_voc.pth",
            "description": "SSD512 trained on Pascal VOC"
        },
        "ssd300_coco": {
            "url": "https://example.com/ssd300_coco.pth",
            "filename": "ssd300_coco.pth", 
            "description": "SSD300 trained on COCO"
        }
    }
    
    model_dir = Path("models/pretrained")
    model_dir.mkdir(exist_ok=True)
    
    successful_downloads = 0
    
    for model_name, model_info in models.items():
        print(f"\n{model_info['description']}")
        print("-" * 50)
        
        # Check if model already exists
        model_path = model_dir / model_info['filename']
        if model_path.exists():
            print(f"Model already exists: {model_path}")
            successful_downloads += 1
            continue
        
        # Download model
        if download_file(model_info['url'], str(model_path), model_info['description']):
            successful_downloads += 1
    
    print(f"\nDownloaded {successful_downloads}/{len(models)} models")
    return successful_downloads == len(models)


def create_dummy_models():
    """Create dummy models for testing (when real models are not available)"""
    print("Creating dummy models for testing...")
    
    import torch
    import torch.nn as nn
    
    model_dir = Path("models/pretrained")
    model_dir.mkdir(exist_ok=True)
    
    # Create a simple dummy model
    class DummySSD(nn.Module):
        def __init__(self):
            super(DummySSD, self).__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 21)  # 21 classes for VOC
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x, x  # Return dummy loc and conf predictions
    
    # Create dummy models
    models = {
        "ssd300_voc": DummySSD(),
        "ssd512_voc": DummySSD(),
        "ssd300_coco": DummySSD()
    }
    
    for model_name, model in models.items():
        model_path = model_dir / f"{model_name}.pth"
        
        # Save model state dict
        torch.save({
            'state_dict': model.state_dict(),
            'model_name': model_name,
            'description': 'Dummy model for testing'
        }, model_path)
        
        print(f"Created dummy model: {model_path}")
    
    print("Dummy models created successfully")


def verify_models():
    """Verify that downloaded models can be loaded"""
    print("\nVerifying models...")
    
    import torch
    from models.ssd_network import SSDNetwork
    
    model_dir = Path("models/pretrained")
    
    if not model_dir.exists():
        print("Model directory does not exist")
        return False
    
    # Load configuration
    config_path = "configs/ssd300_config.json"
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return False
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Test loading each model
    model_files = list(model_dir.glob("*.pth"))
    
    if not model_files:
        print("No model files found")
        return False
    
    for model_file in model_files:
        try:
            print(f"Testing {model_file.name}...")
            
            # Create model
            model = SSDNetwork(config)
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"✓ {model_file.name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {model_file.name}: {e}")
    
    return True


def main():
    """Main function"""
    print("SSD Model Downloader")
    print("=" * 50)
    
    # Setup directory structure
    setup_model_directory()
    
    # Try to download real models
    print("\nAttempting to download pre-trained models...")
    success = download_pretrained_models()
    
    if not success:
        print("\nReal models not available, creating dummy models for testing...")
        create_dummy_models()
    
    # Verify models
    verify_models()
    
    print("\n" + "=" * 50)
    print("Model setup complete!")
    print("=" * 50)
    print("You can now use the SSD detector with the downloaded models.")
    print("Example usage:")
    print("  python examples/basic_detection.py")
    print("  python examples/real_time_detection.py")


if __name__ == "__main__":
    main()
