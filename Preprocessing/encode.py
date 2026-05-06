import pandas as pd

def encode_data(df):
    df_encoded = df.copy()

    # get categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

    # apply one-hot encoding
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

    print(f"Encoded columns: {list(categorical_cols)}")

    return df_encoded