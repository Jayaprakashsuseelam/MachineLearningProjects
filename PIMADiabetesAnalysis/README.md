# PIMA Diabetes Analysis

This repository contains an in-depth analysis of the **PIMA Diabetes Dataset** using Python and data science techniques. The project explores key features, performs exploratory data analysis (EDA), and applies machine learning models to predict diabetes occurrences.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PIMADiabetesAnalysis.git
cd PIMADiabetesAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the quick prediction script:
```bash
python quick_predict.py
```

### Simple Use Cases

#### 1. Quick Prediction (No Training Required)
```bash
python quick_predict.py
```
This script provides an interactive interface for diabetes prediction using a pre-trained model.

#### 2. Full Analysis with Training
```bash
python diabetes_predictor.py
```
This script trains a Random Forest model on the full dataset and provides comprehensive analysis.

#### 3. Jupyter Notebook Analysis
Open `PIMADiabetesAnalysis.ipynb` for detailed exploratory data analysis and model comparison.

## 📊 Project Overview

The PIMA Indian Diabetes dataset is widely used in machine learning and healthcare analytics. This analysis includes:

- **Data preprocessing and cleaning** - Handling missing values and data quality issues
- **Exploratory Data Analysis (EDA)** - Comprehensive visualization and statistical analysis
- **Feature engineering** - Data transformation and scaling
- **Model training and evaluation** - Multiple algorithms (Random Forest, Decision Trees, XGBoost, SVM)
- **Performance metrics and visualization** - Accuracy, precision, recall, and confusion matrices

## 📁 Project Structure

```
PIMADiabetesAnalysis/
├── PIMADiabetesAnalysis.ipynb    # Main Jupyter notebook with complete analysis
├── diabetes_predictor.py         # Comprehensive prediction script with training
├── quick_predict.py             # Simple prediction script (no training required)
├── sample_data.csv              # Sample data for testing
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 Use Cases

### Healthcare Professionals
- Quick diabetes risk assessment for patients
- Understanding key risk factors for diabetes
- Model interpretation and feature importance analysis

### Data Scientists
- Learning machine learning workflow
- Understanding healthcare data preprocessing
- Model comparison and evaluation techniques

### Students and Researchers
- Educational resource for diabetes prediction
- Example of end-to-end ML project
- Healthcare analytics case study

## 📈 Dataset Information

**Source**: [PIMA Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

The dataset consists of several health-related variables for PIMA Indian women:

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0-17 |
| Glucose | Plasma glucose concentration (mg/dL) | 0-199 |
| BloodPressure | Diastolic blood pressure (mmHg) | 0-122 |
| SkinThickness | Triceps skin fold thickness (mm) | 0-99 |
| Insulin | 2-Hour serum insulin (mu U/ml) | 0-846 |
| BMI | Body mass index (kg/m²) | 0-67.1 |
| DiabetesPedigreeFunction | Diabetes pedigree function | 0.078-2.42 |
| Age | Age in years | 21-81 |
| Outcome | Diabetes presence (1=Yes, 0=No) | 0-1 |

**Dataset Statistics:**
- Total samples: 768
- Diabetic patients: 268 (34.9%)
- Non-diabetic patients: 500 (65.1%)

## 🔧 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 76.8% | 0.68 | 0.67 | 0.68 |
| Decision Tree | 73.2% | 0.62 | 0.65 | 0.64 |
| XGBoost | 74.0% | 0.64 | 0.66 | 0.65 |
| SVM | 74.0% | 0.70 | 0.49 | 0.58 |

**Best Model**: Random Forest with 76.8% accuracy

## 🎨 Key Visualizations

The analysis includes various visualizations:
- Distribution plots for all features
- Correlation heatmap
- Feature importance analysis
- Confusion matrices
- ROC curves

## 🚀 Getting Started Examples

### Example 1: Using the Quick Predictor
```python
from quick_predict import predict_diabetes, create_sample_model

# Create model
model, imputer = create_sample_model()

# Predict for a patient
features = [1, 85, 66, 29, 0, 26.6, 0.351, 31]  # Sample patient data
result = predict_diabetes(features, model, imputer)
print(f"Prediction: {result['prediction']}")
```

### Example 2: Using the Full Predictor
```python
from diabetes_predictor import DiabetesPredictor

# Initialize predictor
predictor = DiabetesPredictor()

# Train model (requires diabetes.csv)
predictor.train_model('diabetes.csv')

# Make prediction
result = predictor.predict([1, 85, 66, 29, 0, 26.6, 0.351, 31])
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📋 Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- MLxtend
- Missingno

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This project is for educational and research purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.