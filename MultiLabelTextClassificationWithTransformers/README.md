# Multi-Label Text Classification with Transformers

This repository contains code for fine-tuning transformer-based models for multi-label text classification using the `sem_eval_2018_task_1` dataset. The goal is to classify tweets into multiple emotion categories such as anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, and trust.

## Dataset

The dataset used is `sem_eval_2018_task_1`, specifically the `subtask5.english` subset. It contains tweets annotated with multiple emotion labels. The dataset is divided into training, validation, and test sets.

### Dataset Features
- **ID**: Unique identifier for each tweet.
- **Tweet**: The text of the tweet.
- **anger**: Boolean indicating if the tweet expresses anger.
- **anticipation**: Boolean indicating if the tweet expresses anticipation.
- **disgust**: Boolean indicating if the tweet expresses disgust.
- **fear**: Boolean indicating if the tweet expresses fear.
- **joy**: Boolean indicating if the tweet expresses joy.
- **love**: Boolean indicating if the tweet expresses love.
- **optimism**: Boolean indicating if the tweet expresses optimism.
- **pessimism**: Boolean indicating if the tweet expresses pessimism.
- **sadness**: Boolean indicating if the tweet expresses sadness.
- **surprise**: Boolean indicating if the tweet expresses surprise.
- **trust**: Boolean indicating if the tweet expresses trust.

## Models

The following transformer-based models are fine-tuned and evaluated:

1. **bert-base-uncased**
2. **bert-large-uncased**
3. **bert-base-cased**
4. **bert-base-multilingual-cased**

## Requirements

To run the code, you need the following Python packages:

- `transformers`
- `datasets`
- `scikit-learn`
- `numpy`
- `pandas`

You can install the required packages using pip:

```bash
pip install transformers datasets scikit-learn numpy pandas