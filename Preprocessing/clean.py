import pandas as pd

# remove rows that are completely empty
def remove_empty_rows(df):
    before = len(df)
    df = df.dropna(how='all')
    after = len(df)
    return df, before - after

# remove duplicated rows
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    return df, before - after


def handle_missing_values(df):
    df_cleaned = df.copy()
    total_filled = 0

    # fill missing values in numeric columns with mean
    for col in df_cleaned.select_dtypes(include=['number']).columns:
        missing_count = df_cleaned[col].isnull().sum()

        if missing_count > 0:
            mean_val = df_cleaned[col].mean()
            df_cleaned[col] = df_cleaned[col].fillna(mean_val)
            total_filled += missing_count


    # fill missing values in categorical columns with mode
    for col in df_cleaned.select_dtypes(include=['object', 'category']).columns:

        # if the whole column is empty, drop it
        if df_cleaned[col].isnull().all():
            df_cleaned.drop(columns=[col], inplace=True)
            continue

        missing_count = df_cleaned[col].isnull().sum()

        if missing_count > 0:
            mode_series = df_cleaned[col].mode()

            if not mode_series.empty:
                mode_val = mode_series[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                total_filled += missing_count

    return df_cleaned, total_filled


def clean_data(df):
    # step 1: remove empty rows
    df, empty_removed = remove_empty_rows(df)

    # step 2: remove duplicates
    df, duplicates_removed = remove_duplicates(df)

    # step 3: handle missing values
    df, missing_filled = handle_missing_values(df)

    print(
        f"Cleaned: removed {empty_removed} empty rows, "
        f"{duplicates_removed} duplicates, "
        f"filled {missing_filled} missing values"
    )

    return df