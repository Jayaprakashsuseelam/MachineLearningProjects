#!/usr/bin/env python3
"""
Quick Start Script for Q-Learning Project
This script provides easy access to all components and a guided introduction.
"""

import os
import sys

def print_banner():
    """Print project banner"""
    print("=" * 70)
    print("🚀 Q-LEARNING REINFORCEMENT LEARNING PROJECT")
    print("=" * 70)
    print("A comprehensive implementation and tutorial for Q-Learning")
    print("=" * 70)

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def show_menu():
    """Show main menu"""
    print("\n📚 LEARNING OPTIONS:")
    print("1. 🎓 Interactive Tutorial (Recommended for beginners)")
    print("2. 🔬 Simple Example (2x2 grid world)")
    print("3. 🌍 Full Case Study (5x5 grid world with visualization)")
    print("4. 🧪 Run Tests (Verify everything works)")
    print("5. 📖 View Project Structure")
    print("6. 🚪 Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print("Please enter a number between 1 and 6.")
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            sys.exit(0)

def run_tutorial():
    """Run the interactive tutorial"""
    print("\n🎓 Starting Interactive Tutorial...")
    print("This tutorial will teach you Q-Learning step by step.")
    print("Press Ctrl+C at any time to return to the main menu.\n")
    
    try:
        from q_learning_tutorial import q_learning_tutorial
        q_learning_tutorial()
    except ImportError as e:
        print(f"❌ Error importing tutorial: {e}")
    except Exception as e:
        print(f"❌ Error running tutorial: {e}")

def run_simple_example():
    """Run the simple example"""
    print("\n🔬 Running Simple Example...")
    print("This demonstrates basic Q-Learning in a 2x2 grid world.\n")
    
    try:
        from simple_example import simple_q_learning_example, explain_q_learning
        simple_q_learning_example()
        explain_q_learning()
    except ImportError as e:
        print(f"❌ Error importing simple example: {e}")
    except Exception as e:
        print(f"❌ Error running simple example: {e}")

def run_full_case_study():
    """Run the full case study"""
    print("\n🌍 Running Full Case Study...")
    print("This showcases Q-Learning in a 5x5 grid world with visualization.\n")
    
    try:
        from q_learning import main
        main()
    except ImportError as e:
        print(f"❌ Error importing main implementation: {e}")
    except Exception as e:
        print(f"❌ Error running case study: {e}")

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running Tests...")
    print("This will verify that all components work correctly.\n")
    
    try:
        from test_q_learning import test_simple_q_learning, test_grid_world_import
        test_simple_q_learning()
        print("\n" + "="*40)
        test_grid_world_import()
    except ImportError as e:
        print(f"❌ Error importing test module: {e}")
    except Exception as e:
        print(f"❌ Error running tests: {e}")

def show_project_structure():
    """Show project structure and file descriptions"""
    print("\n📁 PROJECT STRUCTURE:")
    print("=" * 50)
    
    files = [
        ("README.md", "Comprehensive project documentation and learning guide"),
        ("requirements.txt", "Python package dependencies"),
        ("q_learning.py", "Main implementation with 5x5 grid world case study"),
        ("simple_example.py", "Simplified Q-Learning example for learning"),
        ("q_learning_tutorial.py", "Interactive tutorial script"),
        ("test_q_learning.py", "Test suite to verify implementation"),
        ("quick_start.py", "This script - easy access to all components")
    ]
    
    for filename, description in files:
        print(f"📄 {filename:<20} - {description}")
    
    print("\n🎯 RECOMMENDED LEARNING PATH:")
    print("1. Start with the Interactive Tutorial (option 1)")
    print("2. Try the Simple Example (option 2)")
    print("3. Explore the Full Case Study (option 3)")
    print("4. Run Tests to verify everything works (option 4)")

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies before continuing.")
        print("Run: pip install -r requirements.txt")
        return
    
    print("\n🎯 Welcome to the Q-Learning Project!")
    print("This project will teach you reinforcement learning through hands-on examples.")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            run_tutorial()
        elif choice == '2':
            run_simple_example()
        elif choice == '3':
            run_full_case_study()
        elif choice == '4':
            run_tests()
        elif choice == '5':
            show_project_structure()
        elif choice == '6':
            print("\n👋 Thank you for exploring Q-Learning!")
            print("Keep learning and experimenting! 🚀")
            break
        
        if choice in ['1', '2', '3', '4']:
            input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for exploring Q-Learning!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Please check your Python environment and dependencies.")
