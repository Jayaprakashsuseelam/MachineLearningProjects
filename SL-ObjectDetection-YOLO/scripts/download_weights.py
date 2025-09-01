#!/usr/bin/env python3
"""
Download pre-trained YOLO weights
"""

import os
import requests
from pathlib import Path
import urllib.request

def download_file(url: str, filename: str):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

def main():
    """Download pre-trained weights"""
    # Create weights directory
    weights_dir = Path("data/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLOv5 pre-trained weights URLs
    weights_urls = {
        "yolov5n.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
        "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        "yolov5m.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
        "yolov5l.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
        "yolov5x.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt"
    }
    
    print("Downloading pre-trained YOLO weights...")
    print("=" * 50)
    
    for filename, url in weights_urls.items():
        filepath = weights_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists, skipping...")
        else:
            download_file(url, str(filepath))
    
    print("\n" + "=" * 50)
    print("✓ Download completed!")
    print(f"Weights saved to: {weights_dir.absolute()}")

if __name__ == "__main__":
    main()
