#!/usr/bin/env python3
"""
Advanced Random Forest Features Demo
Additional demonstrations of Random Forest capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

def demonstrate_oob_score():
    """Demonstrate Out-of-Bag (OOB) score feature"""
    print("=" * 50)
    print("OUT-OF-BAG (OOB) SCORE DEMONSTRATION")
    print("=" * 50)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Train Random Forest with OOB score enabled
    rf_oob = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=42
    )
    
    rf_oob.fit(X, y)
    
    print(f"OOB Score: {rf_oob.oob_score_:.4f}")
    print(f"OOB Score is an unbiased estimate of generalization error")
    print(f"It's calculated using samples not used in training each tree")
    
    # Compare with traditional train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    rf_oob.fit(X_train, y_train)
    test_score = rf_oob.score(X_test, y_test)
    
    print(f"\nOOB Score: {rf_oob.oob_score_:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Difference: {abs(rf_oob.oob_score_ - test_score):.4f}")

def demonstrate_partial_dependence():
    """Demonstrate partial dependence plots"""
    print("\n" + "=" * 50)
    print("PARTIAL DEPENDENCE PLOTS")
    print("=" * 50)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Create partial dependence plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Partial Dependence Plots', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(feature_names):
        row, col = i // 2, i % 2
        
        # Calculate partial dependence
        feature_values = np.linspace(X[:, i].min(), X[:, i].max(), 50)
        pdp_values = []
        
        for val in feature_values:
            X_temp = X.copy()
            X_temp[:, i] = val
            predictions = rf.predict_proba(X_temp)
            pdp_values.append(predictions.mean(axis=0))
        
        pdp_values = np.array(pdp_values)
        
        # Plot for each class
        for class_idx in range(3):
            axes[row, col].plot(feature_values, pdp_values[:, class_idx], 
                               label=f'Class {class_idx}', linewidth=2)
        
        axes[row, col].set_title(f'Partial Dependence: {feature}')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Partial Dependence')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('partial_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_tree_visualization():
    """Demonstrate individual tree visualization"""
    print("\n" + "=" * 50)
    print("INDIVIDUAL TREE VISUALIZATION")
    print("=" * 50)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Train a smaller Random Forest for visualization
    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    rf.fit(X, y)
    
    # Visualize the first tree
    plt.figure(figsize=(20, 10))
    plot_tree(rf.estimators_[0], 
              feature_names=feature_names,
              class_names=target_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('First Decision Tree in Random Forest', fontsize=14, fontweight='bold')
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualized the first tree from the Random Forest")
    print(f"Each tree makes different decisions due to:")
    print(f"1. Bootstrap sampling (different training data)")
    print(f"2. Random feature selection at each split")

def demonstrate_feature_interactions():
    """Demonstrate feature interaction analysis"""
    print("\n" + "=" * 50)
    print("FEATURE INTERACTION ANALYSIS")
    print("=" * 50)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Analyze feature interactions through permutation importance
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    
    # Create interaction heatmap
    plt.figure(figsize=(10, 8))
    interaction_matrix = np.zeros((len(feature_names), len(feature_names)))
    
    # Calculate pairwise interactions (simplified)
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            if i != j:
                # Create interaction feature (product)
                X_interaction = X.copy()
                X_interaction[:, i] = X_interaction[:, i] * X_interaction[:, j]
                
                # Train model with interaction
                rf_interaction = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_interaction.fit(X_interaction, y)
                
                # Compare performance
                original_score = rf.score(X, y)
                interaction_score = rf_interaction.score(X_interaction, y)
                interaction_matrix[i, j] = interaction_score - original_score
    
    # Plot interaction matrix
    sns.heatmap(interaction_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                xticklabels=feature_names,
                yticklabels=feature_names)
    plt.title('Feature Interaction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_interactions.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_robustness_analysis():
    """Demonstrate Random Forest robustness to noise"""
    print("\n" + "=" * 50)
    print("ROBUSTNESS TO NOISE ANALYSIS")
    print("=" * 50)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Add different levels of noise
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    scores = []
    
    for noise in noise_levels:
        # Add noise to features
        X_noisy = X + np.random.normal(0, noise, X.shape)
        
        # Train and evaluate
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_noisy, y)
        score = rf.score(X_noisy, y)
        scores.append(score)
        
        print(f"Noise level {noise:.1f}: Score = {score:.4f}")
    
    # Visualize robustness
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level (Standard Deviation)')
    plt.ylabel('Accuracy Score')
    plt.title('Random Forest Robustness to Noise', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('robustness_to_noise.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_ensemble_diversity():
    """Demonstrate ensemble diversity analysis"""
    print("\n" + "=" * 50)
    print("ENSEMBLE DIVERSITY ANALYSIS")
    print("=" * 50)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Train multiple Random Forests
    n_models = 10
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
    
    # Calculate diversity metrics
    predictions_array = np.array(predictions)
    
    # Agreement matrix
    agreement_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            agreement = np.mean(predictions_array[i] == predictions_array[j])
            agreement_matrix[i, j] = agreement
    
    # Visualize agreement matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, 
                annot=True, 
                cmap='Blues',
                vmin=0.5, 
                vmax=1.0,
                square=True)
    plt.title('Model Agreement Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Model Index')
    plt.ylabel('Model Index')
    plt.tight_layout()
    plt.savefig('ensemble_diversity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Agreement matrix shows how similar the models are")
    print(f"Lower agreement = higher diversity = better ensemble")

def main():
    """Run all advanced demonstrations"""
    print("ADVANCED RANDOM FOREST FEATURES DEMO")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_oob_score()
    demonstrate_partial_dependence()
    demonstrate_tree_visualization()
    demonstrate_feature_interactions()
    demonstrate_robustness_analysis()
    demonstrate_ensemble_diversity()
    
    print("\n" + "=" * 60)
    print("ADVANCED DEMO COMPLETED!")
    print("=" * 60)
    print("\nAdvanced Features Demonstrated:")
    print("1. ✅ Out-of-Bag (OOB) Score Estimation")
    print("2. ✅ Partial Dependence Plots")
    print("3. ✅ Individual Tree Visualization")
    print("4. ✅ Feature Interaction Analysis")
    print("5. ✅ Robustness to Noise")
    print("6. ✅ Ensemble Diversity Analysis")
    
    print(f"\nAdditional visualizations generated:")
    print("- partial_dependence_plots.png")
    print("- decision_tree_visualization.png")
    print("- feature_interactions.png")
    print("- robustness_to_noise.png")
    print("- ensemble_diversity.png")

if __name__ == "__main__":
    main()
