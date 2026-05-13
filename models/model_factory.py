# models/model_factory.py
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def get_model(model_name: str, params: dict = None):
    """
    Return the correct sklearn model for the given name.
    Classification models end with 'Classifier' internally.
    Regression models end with 'Regressor' or are LinearRegression.
    """
    if params is None:
        params = {}

    if model_name == "Logistic Regression":
        return LogisticRegression(**params)

    elif model_name == "SVM":
        return SVC(**params)

    elif model_name == "SVR":
        return SVR(**params)

    elif model_name == "KNN":
        return KNeighborsClassifier(**params)

    elif model_name == "KNN Regressor":
        return KNeighborsRegressor(**params)

    elif model_name == "Random Forest":
        return RandomForestClassifier(**params)

    elif model_name == "Random Forest Regressor":
        return RandomForestRegressor(**params)

    elif model_name == "Linear Regression":
        return LinearRegression(**params)

    else:
        raise ValueError(f"Model '{model_name}' is not supported yet!")