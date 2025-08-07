#!/usr/bin/env python3
"""
Quick Random Forest Demo
A simplified demonstration of Random Forest Classifier features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def quick_random_forest_demo():
    """Quick demonstration of Random Forest features"""
    print("🌳 QUICK RANDOM FOREST DEMO 🌳")
    print("=" * 50)
    
    # 1. Load Data
    print("\n📊 Loading Iris Dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {list(target_names)}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,       # Max depth of each tree
        random_state=42,    # For reproducibility
        n_jobs=-1          # Use all CPU cores
    )
    
    rf.fit(X_train, y_train)
    
    # 4. Make Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    
    # 5. Feature Importance
    print("\n📈 Feature Importance:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # 6. Visualize Feature Importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], rotation=45)
    plt.title('Random Forest Feature Importance', fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('quick_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 8. Demonstrate Ensemble Properties
    print(f"\n🌳 Ensemble Properties:")
    print(f"  • Number of trees: {rf.n_estimators}")
    print(f"  • Each tree sees different data (bootstrap sampling)")
    print(f"  • Each tree considers random subset of features")
    print(f"  • Final prediction: majority vote of all trees")
    
    # 9. Show Probability Estimates
    print(f"\n🎯 Probability Estimates (first 5 test samples):")
    proba = rf.predict_proba(X_test[:5])
    for i, (pred, prob) in enumerate(zip(y_pred[:5], proba[:5])):
        print(f"  Sample {i+1}: Predicted {target_names[pred]} "
              f"(Confidence: {max(prob):.3f})")
    
    # 10. Demonstrate Robustness
    print(f"\n🛡️ Robustness Demo:")
    print(f"  • Adding noise to test data...")
    
    # Add noise to test data
    X_test_noisy = X_test + np.random.normal(0, 0.1, X_test.shape)
    y_pred_noisy = rf.predict(X_test_noisy)
    accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
    
    print(f"  • Original accuracy: {accuracy:.4f}")
    print(f"  • Accuracy with noise: {accuracy_noisy:.4f}")
    print(f"  • Robustness: Random Forest handles noise well!")
    
    print(f"\n🎉 Demo Complete!")
    print(f"Key Random Forest Features:")
    print(f"  ✅ Ensemble Learning")
    print(f"  ✅ Feature Importance")
    print(f"  ✅ Probability Estimates")
    print(f"  ✅ Robustness to Noise")
    print(f"  ✅ No Feature Scaling Required")
    print(f"  ✅ Handles Missing Values")
    print(f"  ✅ Parallel Processing")

if __name__ == "__main__":
    quick_random_forest_demo()
