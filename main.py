from sklearn.pipeline import Pipeline
from preprocessing.pipeline import build_preprocessor, split_data
from models.model_factory import get_model
from models.train import train_model
from models.evaluate import evaluate_model

REGRESSION_MODELS = {"Linear Regression"}

def full_pipeline(df, target_column, model_name, params=None):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    preprocessor = build_preprocessor(X)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = get_model(model_name, params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    trained_model = train_model(pipeline, X_train, y_train)

    is_regression = model_name in REGRESSION_MODELS

    metrics = evaluate_model(
        trained_model,
        X_test,
        y_test,
        is_regression=is_regression
    )

    return trained_model, metrics, X.columns