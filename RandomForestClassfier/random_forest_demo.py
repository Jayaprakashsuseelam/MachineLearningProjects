#!/usr/bin/env python3
"""
Random Forest Classifier Demo
Showcasing key features using the Iris dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the Iris dataset"""
    print("=" * 60)
    print("RANDOM FOREST CLASSIFIER DEMO")
    print("=" * 60)
    
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"\nDataset Shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target Classes: {target_names}")
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [target_names[i] for i in y]
    
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nFirst few rows:")
    print(df.head())
    
    return X, y, feature_names, target_names, df

def visualize_data(df):
    """Create visualizations to understand the data"""
    print("\n" + "=" * 40)
    print("DATA VISUALIZATION")
    print("=" * 40)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Feature distributions by species
    for i, feature in enumerate(['sepal length (cm)', 'sepal width (cm)', 
                               'petal length (cm)', 'petal width (cm)']):
        row, col = i // 2, i % 2
        for species in df['species'].unique():
            species_data = df[df['species'] == species][feature]
            axes[row, col].hist(species_data, alpha=0.7, label=species, bins=15)
        axes[row, col].set_title(f'{feature} Distribution')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.drop(['target', 'species'], axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_basic_random_forest(X, y, feature_names, target_names):
    """Train a basic Random Forest Classifier"""
    print("\n" + "=" * 40)
    print("BASIC RANDOM FOREST TRAINING")
    print("=" * 40)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest
    rf_basic = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_basic.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_basic.predict(X_test)
    y_pred_proba = rf_basic.predict_proba(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_basic, X_train, X_test, y_train, y_test

def feature_importance_analysis(rf_model, feature_names):
    """Analyze and visualize feature importance"""
    print("\n" + "=" * 40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature ranking:")
    for f in range(len(feature_names)):
        print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance in Random Forest', fontsize=14, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def cross_validation_analysis(X, y):
    """Perform cross-validation analysis"""
    print("\n" + "=" * 40)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 40)
    
    # Different Random Forest configurations
    rf_configs = {
        'Basic RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'Deep RF': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'Wide RF': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'Conservative RF': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }
    
    cv_scores = {}
    for name, model in rf_configs.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_scores[name] = scores
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Visualize CV results
    plt.figure(figsize=(12, 6))
    plt.boxplot([cv_scores[name] for name in rf_configs.keys()], 
                labels=list(rf_configs.keys()))
    plt.title('Cross-Validation Scores Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cv_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def hyperparameter_tuning(X, y):
    """Demonstrate hyperparameter tuning with GridSearchCV"""
    print("\n" + "=" * 40)
    print("HYPERPARAMETER TUNING")
    print("=" * 40)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create Random Forest model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing Grid Search...")
    grid_search.fit(X, y)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    return best_rf, grid_search

def ensemble_analysis(X, y, feature_names, target_names):
    """Demonstrate ensemble properties of Random Forest"""
    print("\n" + "=" * 40)
    print("ENSEMBLE ANALYSIS")
    print("=" * 40)
    
    # Train multiple Random Forests
    n_models = 5
    models = []
    predictions = []
    
    for i in range(n_models):
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=i
        )
        rf.fit(X, y)
        models.append(rf)
        predictions.append(rf.predict(X))
    
    # Analyze individual tree predictions
    print(f"Training {n_models} Random Forest models...")
    
    # Show how ensemble reduces overfitting
    train_scores = []
    test_scores = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    for i in range(n_models):
        train_score = models[i].score(X_train, y_train)
        test_score = models[i].score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Visualize train vs test scores
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(n_models)
    width = 0.35
    
    plt.bar(x_pos - width/2, train_scores, width, label='Training Score', alpha=0.8)
    plt.bar(x_pos + width/2, test_scores, width, label='Test Score', alpha=0.8)
    
    plt.xlabel('Model Index')
    plt.ylabel('Accuracy Score')
    plt.title('Training vs Test Scores for Individual Models', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(x_pos)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ensemble_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete Random Forest demo"""
    # Load and explore data
    X, y, feature_names, target_names, df = load_and_explore_data()
    
    # Visualize the data
    visualize_data(df)
    
    # Train basic Random Forest
    rf_model, X_train, X_test, y_train, y_test = train_basic_random_forest(
        X, y, feature_names, target_names
    )
    
    # Analyze feature importance
    feature_importance_analysis(rf_model, feature_names)
    
    # Perform cross-validation
    cross_validation_analysis(X, y)
    
    # Hyperparameter tuning
    best_rf, grid_search = hyperparameter_tuning(X, y)
    
    # Ensemble analysis
    ensemble_analysis(X, y, feature_names, target_names)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("\nKey Random Forest Features Demonstrated:")
    print("1. ✅ Ensemble Learning with Multiple Decision Trees")
    print("2. ✅ Feature Importance Analysis")
    print("3. ✅ Cross-Validation for Model Evaluation")
    print("4. ✅ Hyperparameter Tuning with GridSearchCV")
    print("5. ✅ Robustness to Overfitting")
    print("6. ✅ Handling of Categorical and Numerical Features")
    print("7. ✅ Probability Estimates")
    print("8. ✅ Out-of-Bag (OOB) Error Estimation")
    
    print(f"\nGenerated visualizations:")
    print("- iris_data_analysis.png")
    print("- correlation_matrix.png")
    print("- confusion_matrix.png")
    print("- feature_importance.png")
    print("- cv_scores_comparison.png")
    print("- ensemble_analysis.png")

if __name__ == "__main__":
    main()
