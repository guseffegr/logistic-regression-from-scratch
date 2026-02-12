"""
preprocessing.py

Utility functions for data preparation.
"""
import numpy as np

def fit_zscore(X):
    """ Compute mean and standard deviation for Z-score normalization. """
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return mu, std

def transform_zscore(X, mu, std):
    """ Compute Z-score normalization. """
    return (X - mu) / std


def train_val_test_split(X, y, train_ratio, val_ratio, random_state):
    """ Split dataset into training, validation, and test sets. """
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1"

    m = X.shape[0]

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(m)

    train_end = int(train_ratio * m)
    val_end = int((train_ratio + val_ratio) * m)

    X_train = X[indices[:train_end]]
    y_train = y[indices[:train_end]]

    X_val = X[indices[train_end:val_end]]
    y_val = y[indices[train_end:val_end]]

    X_test = X[indices[val_end:]]
    y_test = y[indices[val_end:]]

    return X_train, y_train, X_val, y_val, X_test, y_test