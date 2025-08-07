#!/usr/bin/env python3
"""
Simple Random Forest Example
Basic demonstration of Random Forest Classifier concepts
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def simple_random_forest_example():
    """Simple example showing Random Forest basics"""
    
    print("🌳 Simple Random Forest Example")
    print("=" * 40)
    
    # 1. Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Classes: {list(target_names)}")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        random_state=42,    # For reproducibility
        n_jobs=-1          # Use all CPU cores
    )
    
    rf.fit(X_train, y_train)
    
    # 4. Make predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # 5. Show feature importance
    print(f"\nFeature Importance:")
    importances = rf.feature_importances_
    for i, (feature, importance) in enumerate(zip(feature_names, importances)):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # 6. Show probability estimates
    proba = rf.predict_proba(X_test[:5])
    print(f"\nProbability Estimates (first 5 samples):")
    for i, (pred, prob) in enumerate(zip(y_pred[:5], proba[:5])):
        confidence = max(prob)
        print(f"  Sample {i+1}: {target_names[pred]} (confidence: {confidence:.3f})")
    
    # 7. Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"\n✅ Random Forest successfully trained and evaluated!")
    print(f"Key features demonstrated:")
    print(f"  • Ensemble of 100 decision trees")
    print(f"  • Automatic feature importance ranking")
    print(f"  • Probability estimates for predictions")
    print(f"  • High accuracy without feature scaling")

if __name__ == "__main__":
    simple_random_forest_example()
