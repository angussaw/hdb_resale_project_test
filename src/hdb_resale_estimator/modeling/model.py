"""Module containing the functions to make model predictions
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from src.hdb_resale_estimator.modeling.builder import ClassicalModelBuilder


def make_predictions(
    builder: ClassicalModelBuilder,
    data: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Predicts the target label from a given set of features

    Args:
        builder (ClassicalModelBuilder): ClassicalModelBuilder class object which contains fitted model
        data (Union[pd.DataFrame, np.ndarray]): Dataframe or numpy array consisting of the feature(s)

    Returns:
        np.ndarray: Array of predicted target labels
    """
    predictions = builder.model.predict(data)
    return predictions
