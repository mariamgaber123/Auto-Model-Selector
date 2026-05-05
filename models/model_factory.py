# models/model_factory.py
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def get_model(model_name: str, params: dict = None):

    if params is None:
        params = {}

    if model_name == "Logistic Regression":
        return LogisticRegression(**params)

    elif model_name == "SVM":
        return SVC(**params)

    elif model_name == "KNN":
        return KNeighborsClassifier(**params)

    elif model_name == "Random Forest":
        return RandomForestClassifier(**params)

    elif model_name == "Linear Regression":
        return LinearRegression(**params)

    else:
        raise ValueError(f"Model '{model_name}' is not supported yet!")