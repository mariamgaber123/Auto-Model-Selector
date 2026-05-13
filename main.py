import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing.pipeline import build_preprocessor, split_data
from preprocessing.clean import clean_data
from models.model_factory import get_model
from models.train import train_model
from models.evaluate import evaluate_model
from sklearn.utils.multiclass import type_of_target


CLASSIFICATION_MODELS = {"Logistic Regression", "SVM", "KNN", "Random Forest"}
REGRESSION_MODELS = {"Linear Regression"}


def detect_problem_type(y):
    """
    Smart detection of problem type
    """
    if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
        return "regression"
    if pd.api.types.is_integer_dtype(y) and y.nunique() > 20:
        return "regression"

    target_type = type_of_target(y)
    if target_type in ["binary", "multiclass"]:
        return "classification"

    return "regression"


def full_pipeline(df, target_column, model_name, params=None):
    """
    Main training pipeline with proper preprocessing
    """
    df = clean_data(df, target_column=target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    problem_type = detect_problem_type(y)

    # Validate model choice
    if problem_type == "regression" and model_name in CLASSIFICATION_MODELS:
        raise ValueError(
            f"'{model_name}' is a classification model but target is regression. "
            "Please choose 'Linear Regression'."
        )

    if problem_type == "classification" and model_name in REGRESSION_MODELS:
        raise ValueError(
            f"'{model_name}' is a regression model but target is classification. "
            "Please choose a classification model."
        )

    # Log transform for regression (only if all positive)
    log_transformed = False
    if problem_type == "regression":
        if (y > 0).all():
            y = np.log1p(y)
            log_transformed = True

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build preprocessor (imputer + onehot + scaler)
    preprocessor = build_preprocessor(X_train)

    # Get model
    model = get_model(model_name, params)

    # Create full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # No extra scaler here - we removed the double scaling
        ("model", model)
    ])

    # Train
    trained_model = train_model(pipeline, X_train, y_train)

    # Evaluate
    if log_transformed:
        # Inverse transform for meaningful metrics
        y_pred_log = trained_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)
        
        metrics = evaluate_model(
            trained_model, X_test, y_test,
            is_regression=True,
            y_pred_override=y_pred,
            y_test_override=y_test_original
        )
    else:
        metrics = evaluate_model(
            trained_model, X_test, y_test,
            is_regression=(problem_type == "regression")
        )

    return trained_model, metrics, X.columns