from clean import clean_data
from encode import encode_data
from split import split_data

def preprocess_pipeline(df, target_column):
    """
    Full preprocessing pipeline:
    1. Clean data
    2. Encode categorical features
    3. Split into train/test
    """

    print("\n" + "="*50)
    print("STARTING FULL PREPROCESSING PIPELINE")
    print("="*50)

    # Step 1: Cleaning
    df = clean_data(df)

    # Step 2: Encoding
    df = encode_data(df)

    # Step 3: Splitting
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    print("="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

    return X_train, X_test, y_train, y_test