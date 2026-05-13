# models/evaluate.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
import numpy as np


def evaluate_model(model, X_test, y_test, is_regression=False,
                   y_pred_override=None, y_test_override=None):
    """
    Evaluate model performance.

    y_pred_override / y_test_override:
        Used for regression when the model was trained on log(y).
        Pass the inverse-transformed predictions and targets here so
        metrics are reported on the original scale (more meaningful to users).
    """

    if y_pred_override is not None:
        y_pred = y_pred_override
    else:
        y_pred = model.predict(X_test)

    if y_test_override is not None:
        y_true = y_test_override
    else:
        y_true = y_test

    if is_regression:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "r2": r2,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
        }
    else:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }