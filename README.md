# CS-433 Project 1: Heart Attack Risk Prediction

## Project

This project uses lifestyle and health-survey data from the 2015 Behavioral
Risk Factor Surveillance System to predict whether a person has experienced a
myocardial infarction or coronary heart disease.

## Goal

The goal is to build and compare machine-learning models for this binary
classification problem, then produce predictions for the test set.

## Usage

Install Python and NumPy, place the dataset files in `data/dataset/`, and use
the functions in `implementations.py` to train a model:

```python
from helpers import load_csv_data, create_csv_submission

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")
# Train a model, then store predictions in y_pred.
create_csv_submission(test_ids, y_pred, "submission.csv")
```

The submission file contains the test-set IDs and predicted labels.

