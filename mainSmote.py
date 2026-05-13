from preprocessing.pipeline import build_preprocessor, split_data
from preprocessing.smote import apply_smote
from preprocessing.clean import clean_data
from models.model_factory import get_model
from models.train import train_model
from models.evaluate import evaluate_model


REGRESSION_MODELS = {"Linear Regression"}


def detect_problem_type(y):
    if y.dtype == "object":
        return "classification"

    unique_ratio = y.nunique() / len(y)
    return "classification" if unique_ratio < 0.05 else "regression"


def full_pipeline_S(df, target_column, model_name, params=None):

    df = clean_data(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(y)

    if problem_type == "regression":
        raise ValueError("SMOTE is not allowed for regression problems.")

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_sm, y_train_sm = apply_smote(X_train_processed, y_train)

    model = get_model(model_name, params)

    trained_model = train_model(model, X_train_sm, y_train_sm)

    metrics = evaluate_model(
        trained_model,
        X_test_processed,
        y_test,
        is_regression=False
    )

    return trained_model, metrics, X.columns