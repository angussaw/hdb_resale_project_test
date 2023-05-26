"""Module containing the function to split the data
"""
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_val_split(
    data: pd.DataFrame,
    labels: Union[pd.DataFrame, np.ndarray],
    train_size: float,
    test_size: float,
    random_state: int,
) -> dict:
    """Performs train-test (and validation) split for input data

    Args:
        data (pd.DataFrame): DataFrame containing all features
        labels (Union[pd.DataFrame, np.ndarray]): DataFrame or numpy array containing only labels
        train_size (float): Training set size to be obtained from raw data. Range between 0 to 1
        test_size (float): Test set size to be obtained from remaining raw data after splitting
                           train data. Only used when performing tain-test-val split.
                           Range between 0 to 1. If not provided, dataset will only
                           be split into train and test
        random_state (int): Random seed

    Raises:
        ValueError: Train size is not between 0 and 1

    Returns:
        dict: Dictionary containing the resulting datasets after splitting the data
    """

    if train_size < 0 or train_size > 1:
        raise ValueError("Train size must be between 0 to 1")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        data,
        labels,
        train_size=train_size,
        random_state=random_state,
    )
    if test_size:
        X_test, X_val, y_test, y_val = train_test_split(
            X_tmp,
            y_tmp,
            train_size=test_size,
            random_state=random_state,
        )
        datasets = {
            "train": {"X": X_train, "y": y_train},
            "val": {"X": X_val, "y": y_val},
            "test": {"X": X_test, "y": y_test},
        }
        return datasets
    else:
        datasets = {
            "train": {"X": X_train, "y": y_train},
            "test": {"X": X_tmp, "y": y_tmp},
        }
        return datasets
