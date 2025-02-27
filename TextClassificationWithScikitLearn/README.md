# Text Classification with Scikit-learn
This repository contains a Jupyter Notebook that demonstrates text classification using various machine learning models. The goal is to classify text documents from the 20 Newsgroups dataset into two categories: sci.med and comp.graphics.

## Model
- The notebook implements and evaluates the following machine learning models for text classification:
- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Classifier (SVC)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
Each model is trained on the TF-IDF vectorized text data and evaluated using classification metrics such as precision, recall, and F1-score.


## Dataset Reference
The dataset used in this project is the 20 Newsgroups dataset, which is a collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. For this project, we focus on two categories:
- `sci.med`: Medical science-related documents.
- `comp.graphics`: Computer graphics-related documents.

The dataset is loaded using the fetch_20newsgroups function from sklearn.datasets.

## Installation
To run this project, install the required dependencies:
```bash
pip install torch transformers datasets scikit-learn numpy pandas matplotlib
```

## Results
The performance of each model is summarized below:

- *Logistic Regression*: Achieved an accuracy of 95%.
- *Multinomial Naive Bayes*: Achieved an accuracy of 97%.
- *Support Vector Classifier (SVC)*: Achieved an accuracy of 97%.
- *Random Forest Classifier*: Achieved an accuracy of 92%.
- *K-Nearest Neighbors (KNN)*: Achieved an accuracy of 94%.

The detailed classification reports for each model are available in the notebook.

## Dependencies
To run the notebook, you need the following Python libraries:

- `scikit-learn`
- `numpy`
- `pandas`