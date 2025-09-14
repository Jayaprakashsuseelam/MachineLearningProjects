#!/usr/bin/env python3
"""
Simple Diabetes Prediction Script
=================================

This script demonstrates how to use machine learning to predict diabetes
based on health metrics. It uses the Random Forest model trained on the
PIMA diabetes dataset.

Usage:
    python diabetes_predictor.py

Author: PIMA Diabetes Analysis Project
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle
import os

class DiabetesPredictor:
    """
    A simple diabetes prediction class that uses Random Forest model.
    """
    
    def __init__(self):
        self.model = None
        self.imputer = SimpleImputer(missing_values=0, strategy='mean')
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    
    def train_model(self, data_path='diabetes.csv'):
        """
        Train the Random Forest model on the diabetes dataset.
        
        Args:
            data_path (str): Path to the diabetes CSV file
        """
        try:
            # Load the dataset
            df = pd.read_csv(data_path)
            print(f"Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Prepare features and target
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
            
            # Handle missing values (represented as 0 in this dataset)
            X_imputed = self.imputer.fit_transform(X)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(n_estimators=200, random_state=42)
            self.model.fit(X_imputed, y)
            
            # Calculate training accuracy
            train_pred = self.model.predict(X_imputed)
            accuracy = np.mean(train_pred == y)
            print(f"Model trained successfully! Training accuracy: {accuracy:.3f}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Dataset file '{data_path}' not found.")
            print("Please ensure the diabetes.csv file is in the current directory.")
            return False
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def predict(self, features):
        """
        Predict diabetes outcome for given features.
        
        Args:
            features (list or array): Health metrics in the order:
                [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                 Insulin, BMI, DiabetesPedigreeFunction, Age]
        
        Returns:
            dict: Prediction result with probability
        """
        if self.model is None:
            return {"error": "Model not trained. Please train the model first."}
        
        try:
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Handle missing values
            features_imputed = self.imputer.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_imputed)[0]
            probability = self.model.predict_proba(features_imputed)[0]
            
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
    
    def save_model(self, filename='diabetes_model.pkl'):
        """Save the trained model to a file."""
        if self.model is None:
            print("No model to save. Please train the model first.")
            return False
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump({'model': self.model, 'imputer': self.imputer}, f)
            print(f"Model saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filename='diabetes_model.pkl'):
        """Load a pre-trained model from a file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.imputer = data['imputer']
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file '{filename}' not found.")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


def main():
    """
    Main function demonstrating the diabetes prediction use case.
    """
    print("=" * 60)
    print("PIMA Diabetes Prediction - Simple Use Case")
    print("=" * 60)
    
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Try to load existing model, otherwise train new one
    if not predictor.load_model():
        print("\nTraining new model...")
        if not predictor.train_model():
            print("Failed to train model. Exiting.")
            return
    
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    
    # Sample test cases
    test_cases = [
        {
            "name": "Healthy Person",
            "features": [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            "description": "Young person with normal glucose levels"
        },
        {
            "name": "High Risk Person",
            "features": [8, 183, 64, 0, 0, 23.3, 0.672, 32],
            "description": "Person with high glucose and multiple pregnancies"
        },
        {
            "name": "Elderly Person",
            "features": [1, 89, 66, 23, 94, 28.1, 0.167, 21],
            "description": "Person with normal glucose but higher BMI"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"Features: {case['features']}")
        
        result = predictor.predict(case['features'])
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probabilities:")
            for outcome, prob in result['probability'].items():
                print(f"  {outcome}: {prob:.3f}")
    
    print("\n" + "=" * 60)
    print("Interactive Prediction")
    print("=" * 60)
    print("Enter your health metrics for prediction:")
    print("(Enter 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nEnter features separated by commas (8 values): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            features = [float(x.strip()) for x in user_input.split(',')]
            
            if len(features) != 8:
                print("Please enter exactly 8 values.")
                continue
            
            result = predictor.predict(features)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nPrediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Probabilities:")
                for outcome, prob in result['probability'].items():
                    print(f"  {outcome}: {prob:.3f}")
        
        except ValueError:
            print("Please enter valid numbers separated by commas.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    
    print("\nThank you for using the Diabetes Predictor!")


if __name__ == "__main__":
    main()
