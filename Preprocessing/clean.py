import pandas as pd


def clean_data(df, target_column=None):
    """
    Clean the dataframe:
    1. Drop rows where ALL values are NaN
    2. Drop duplicate rows
    3. If target_column is given, drop rows where target is NaN
       (SMOTE and models can't handle NaN in y)
    """
    # Drop rows that are completely empty
    df = df.dropna(how='all')

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop rows where the target column has NaN
    if target_column is not None:
        before = len(df)
        df = df.dropna(subset=[target_column])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[clean_data] Dropped {dropped} rows with NaN in '{target_column}'")

    df = df.reset_index(drop=True)

    return df