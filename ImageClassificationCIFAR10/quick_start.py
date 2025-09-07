#!/usr/bin/env python3
"""
Quick Start Script for CIFAR-10 Image Classification

This script provides a simple way to get started with the CIFAR-10 classification
project. It demonstrates basic usage and provides examples.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import subprocess
import argparse

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'seaborn', 'scikit-learn', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def setup_directories():
    """Create necessary directories."""
    directories = ['./data', './plots', './results', './logs', './models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🚀 Running quick test...")
    
    try:
        # Import our modules
        from models import get_model, count_parameters
        from data_utils import CIFAR10DataProcessor
        from visualization import CIFAR10Visualizer
        
        # Test model creation
        print("📊 Testing model creation...")
        model = get_model('cnn')
        param_count = count_parameters(model)
        print(f"   ✅ CNN model created with {param_count:,} parameters")
        
        # Test data processor
        print("📊 Testing data processor...")
        processor = CIFAR10DataProcessor()
        print("   ✅ Data processor initialized")
        
        # Test visualizer
        print("📊 Testing visualizer...")
        visualizer = CIFAR10Visualizer()
        print("   ✅ Visualizer initialized")
        
        print("\n🎉 All tests passed! The system is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_examples():
    """Show example usage commands."""
    print("\n📚 Example Usage Commands:")
    print("=" * 50)
    
    print("\n1. Quick Training (5 epochs):")
    print("   python train.py --experiment quick_test")
    
    print("\n2. Standard Training (50 epochs):")
    print("   python train.py --experiment standard_training")
    
    print("\n3. Train Specific Model:")
    print("   python train.py --model resnet --epochs 30")
    
    print("\n4. Train with Custom Settings:")
    print("   python train.py --model efficientnet --epochs 100 --batch_size 64 --lr 0.001")
    
    print("\n5. Train without Data Augmentation:")
    print("   python train.py --no_augmentation")
    
    print("\n6. Run Examples:")
    print("   python examples.py --example 1")
    print("   python examples.py --all")
    
    print("\n7. Full Training Script:")
    print("   python cifar10_classifier.py --model cnn --epochs 50")

def show_project_structure():
    """Show the project structure."""
    print("\n📁 Project Structure:")
    print("=" * 50)
    
    structure = """
ImageClassificationCIFAR10/
├── 📄 README.md                 # Comprehensive documentation
├── 📄 requirements.txt          # Python dependencies
├── 📄 config.py                 # Configuration settings
├── 📄 cifar10_classifier.py     # Main training script
├── 📄 train.py                  # Simple training script
├── 📄 examples.py               # Example usage scripts
├── 📄 models.py                 # Model architectures
├── 📄 data_utils.py            # Data processing utilities
├── 📄 visualization.py          # Visualization tools
├── 📁 data/                     # CIFAR-10 dataset (auto-downloaded)
├── 📁 plots/                    # Generated visualizations
├── 📁 results/                  # Training results
├── 📁 logs/                     # Training logs
└── 📁 models/                   # Saved model weights
    """
    
    print(structure)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='CIFAR-10 Quick Start')
    parser.add_argument('--setup', action='store_true',
                       help='Setup the project (check requirements, create directories)')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test to verify installation')
    parser.add_argument('--examples', action='store_true',
                       help='Show example usage commands')
    parser.add_argument('--structure', action='store_true',
                       help='Show project structure')
    parser.add_argument('--all', action='store_true',
                       help='Run all setup steps')
    
    args = parser.parse_args()
    
    print("🎯 CIFAR-10 Image Classification - Quick Start")
    print("=" * 60)
    
    if args.all or args.setup:
        print("\n🔧 Setting up the project...")
        
        # Check requirements
        if not check_requirements():
            print("\n❌ Setup failed. Please install missing packages.")
            return
        
        # Create directories
        setup_directories()
        
        print("\n✅ Project setup completed!")
    
    if args.all or args.test:
        # Run quick test
        if not run_quick_test():
            print("\n❌ Quick test failed. Please check your installation.")
            return
    
    if args.all or args.examples:
        show_examples()
    
    if args.all or args.structure:
        show_project_structure()
    
    if not any([args.setup, args.test, args.examples, args.structure, args.all]):
        # Show help if no arguments provided
        print("\n💡 Available options:")
        print("   --setup      : Setup the project")
        print("   --test       : Run quick test")
        print("   --examples   : Show example commands")
        print("   --structure  : Show project structure")
        print("   --all        : Run all setup steps")
        print("\n🚀 Quick start:")
        print("   python quick_start.py --all")
        print("   python train.py --experiment quick_test")

if __name__ == "__main__":
    main()
