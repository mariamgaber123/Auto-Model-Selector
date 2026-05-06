import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):

    # check target column
    if target_column not in df.columns:
        raise ValueError("Target column not found in dataframe")

    # split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print("Data split done:")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test