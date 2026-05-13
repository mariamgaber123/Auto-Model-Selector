import pandas as pd
import numpy as np


def clean_data(df, target_column=None):
    """
    Clean the dataframe:
    1. Drop rows where ALL values are NaN
    2. Drop duplicate rows
    3. Drop rows where target is NaN
    4. Handle outliers in numeric feature columns using IQR capping (Winsorization)
    """
    df = df.copy()

    # Drop fully empty rows
    df = df.dropna(how='all')

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows where target is NaN
    if target_column is not None:
        before = len(df)
        df = df.dropna(subset=[target_column])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[clean_data] Dropped {dropped} rows with NaN in '{target_column}'")

    # Handle outliers using IQR capping (Winsorization)
    # Cap instead of remove → no data loss
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Never cap the target column — model must learn its full range
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue  # constant column, skip

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower=lower, upper=upper)

    df = df.reset_index(drop=True)

    return df