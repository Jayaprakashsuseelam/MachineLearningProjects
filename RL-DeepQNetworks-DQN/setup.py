#!/usr/bin/env python3
"""
Setup script for DQN project

This script helps set up the DQN project environment and provides
quick start examples.
"""

import os
import sys
import subprocess
import argparse


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def test_installation():
    """Test if the installation works."""
    print("Testing installation...")
    try:
        # Test basic imports
        import torch
        import numpy as np
        import gym
        import matplotlib.pyplot as plt
        
        print("✓ Basic imports successful!")
        
        # Test DQN imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from dqn.agent import DQNAgent
        from environments.wrappers import make_env
        from training.config import DQNConfig
        
        print("✓ DQN imports successful!")
        
        # Test environment creation
        env = make_env("CartPole-v1")
        env.close()
        print("✓ Environment creation successful!")
        
        return True
    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        return False


def run_quick_demo():
    """Run a quick demo to verify everything works."""
    print("Running quick demo...")
    try:
        subprocess.check_call([sys.executable, "examples/cartpole_demo.py", "--quick"])
        print("✓ Quick demo completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Quick demo failed: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["models", "logs", "results", "notebooks"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="DQN Project Setup")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip package installation")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip installation test")
    parser.add_argument("--skip-demo", action="store_true",
                       help="Skip quick demo")
    parser.add_argument("--demo-only", action="store_true",
                       help="Only run the quick demo")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DQN PROJECT SETUP")
    print("=" * 60)
    
    if args.demo_only:
        run_quick_demo()
        return
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not args.skip_install:
        if not install_requirements():
            print("Setup failed during package installation.")
            return
    
    # Test installation
    if not args.skip_test:
        if not test_installation():
            print("Setup failed during installation test.")
            return
    
    # Run quick demo
    if not args.skip_demo:
        if not run_quick_demo():
            print("Setup failed during quick demo.")
            return
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("You can now:")
    print("1. Run training: python train_dqn.py --env CartPole-v1")
    print("2. Try examples: python examples/cartpole_demo.py")
    print("3. Explore notebooks: jupyter notebook notebooks/")
    print("4. Read the README.md for more information")
    print("=" * 60)


if __name__ == "__main__":
    main()
