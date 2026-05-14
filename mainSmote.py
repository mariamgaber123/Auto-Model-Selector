from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing.pipeline import build_preprocessor, split_data
from preprocessing.smote import apply_smote
from preprocessing.clean import clean_data
from models.model_factory import get_model
from models.train import train_model
from models.evaluate import evaluate_model
from sklearn.utils.multiclass import type_of_target
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


CLASSIFICATION_MODELS = {"Logistic Regression", "SVM", "KNN", "Random Forest","Neural Network (MLP)"}
REGRESSION_MODELS = {"Linear Regression"}


def detect_problem_type(y):
    if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
        return "regression"
    if pd.api.types.is_integer_dtype(y) and y.nunique() > 20:
        return "regression"

    target_type = type_of_target(y)
    if target_type in ["binary", "multiclass"]:
        return "classification"
    return "regression"


def full_pipeline_S(df, target_column, model_name, params=None, use_pca=False):

    df = clean_data(df, target_column=target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(y)

    if problem_type == "regression":
        raise ValueError("SMOTE is not allowed for regression problems.")

    if model_name in REGRESSION_MODELS:
        raise ValueError(f"'{model_name}' is a regression model but target is classification.")

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)   # fitted on X_train

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    #ِApplay PCA
    pca_model = None
    if use_pca:
        pca_model = PCA(n_components=0.95)
        X_train_processed = pca_model.fit_transform(X_train_processed)
        X_test_processed = pca_model.transform(X_test_processed)

    X_train_sm, y_train_sm = apply_smote(X_train_processed, y_train)

    model = get_model(model_name, params)



    # Train
    trained_model = train_model(model, X_train_sm, y_train_sm)

    # Evaluate
    metrics = evaluate_model(trained_model, X_test_processed, y_test, is_regression=False)

    # ==================== FIXED WRAPPER ====================
    class PreprocessedPipeline:
        def __init__(self, preprocessor, model, pca=None):
            self.preprocessor = preprocessor
            self.model = model
            self.pca = pca

        def _transform(self, X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.preprocessor.feature_names_in_ 
                               if hasattr(self.preprocessor, 'feature_names_in_') else None)
            
            X_out = self.preprocessor.transform(X)
            if self.pca:
                X_out = self.pca.transform(X_out)
            return X_out

        def predict(self, X):
            X_processed = self._transform(X)
            return self.model.predict(X_processed)

        def predict_proba(self, X):
            X_processed = self._transform(X)
            return self.model.predict_proba(X_processed)

    wrapped_model = PreprocessedPipeline(preprocessor, trained_model, pca=pca_model)

    return wrapped_model, metrics, X.columns