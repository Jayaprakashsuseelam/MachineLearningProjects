#!/usr/bin/env python3
"""
Quick Diabetes Prediction Script
===============================

A simple script for quick diabetes prediction without training.
Uses pre-trained Random Forest model.

Usage:
    python quick_predict.py

Features needed (in order):
1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def create_sample_model():
    """
    Create a simple Random Forest model with sample data.
    This is a simplified version for demonstration purposes.
    """
    # Sample training data (subset of PIMA dataset)
    X_sample = np.array([
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],
        [1, 85, 66, 29, 0, 26.6, 0.351, 31],
        [8, 183, 64, 0, 0, 23.3, 0.672, 32],
        [1, 89, 66, 23, 94, 28.1, 0.167, 21],
        [0, 137, 40, 35, 168, 43.1, 2.288, 33],
        [5, 116, 74, 0, 0, 25.6, 0.201, 30],
        [3, 78, 50, 32, 88, 31.0, 0.248, 26],
        [10, 115, 0, 0, 0, 35.3, 0.134, 29],
        [2, 197, 70, 45, 543, 30.5, 0.158, 53],
        [8, 125, 96, 0, 0, 0.0, 0.232, 54]
    ])
    
    y_sample = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    
    # Handle missing values (0s in this dataset represent missing values)
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    X_imputed = imputer.fit_transform(X_sample)
    
    # Train a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_imputed, y_sample)
    
    return model, imputer

def predict_diabetes(features, model, imputer):
    """
    Predict diabetes outcome for given features.
    
    Args:
        features (list): Health metrics
        model: Trained Random Forest model
        imputer: Fitted imputer for handling missing values
    
    Returns:
        dict: Prediction result
    """
    try:
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Handle missing values
        features_imputed = imputer.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_imputed)[0]
        probability = model.predict_proba(features_imputed)[0]
        
        result = {
            "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "probability": {
                "Non-Diabetic": probability[0],
                "Diabetic": probability[1]
            },
            "confidence": max(probability)
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def main():
    """
    Main function for quick diabetes prediction.
    """
    print("=" * 50)
    print("Quick Diabetes Prediction")
    print("=" * 50)
    print("Enter health metrics for diabetes prediction")
    print("(Enter 'quit' to exit)")
    print()
    
    # Create model
    print("Initializing prediction model...")
    model, imputer = create_sample_model()
    print("Model ready!")
    print()
    
    # Feature descriptions
    feature_names = [
        "Number of pregnancies",
        "Glucose level (mg/dL)",
        "Blood pressure (mmHg)",
        "Skin thickness (mm)",
        "Insulin level (mu U/ml)",
        "BMI (kg/m²)",
        "Diabetes pedigree function",
        "Age (years)"
    ]
    
    while True:
        try:
            print("Enter the following health metrics:")
            features = []
            
            for i, name in enumerate(feature_names):
                while True:
                    try:
                        value = input(f"{i+1}. {name}: ").strip()
                        if value.lower() == 'quit':
                            print("Exiting...")
                            return
                        
                        features.append(float(value))
                        break
                    except ValueError:
                        print("Please enter a valid number.")
            
            # Make prediction
            result = predict_diabetes(features, model, imputer)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("\n" + "=" * 30)
                print("PREDICTION RESULT")
                print("=" * 30)
                print(f"Outcome: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print("\nProbabilities:")
                for outcome, prob in result['probability'].items():
                    print(f"  {outcome}: {prob:.1%}")
                print("=" * 30)
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
