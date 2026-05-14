from sklearn.model_selection import train_test_split
from preprocessing.encode import encode_data
from sklearn.decomposition import PCA

def build_preprocessor(df):

    return encode_data(df)


def split_data(X, y):

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
