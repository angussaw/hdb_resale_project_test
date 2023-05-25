"""Module that defines the Builder constructor to consolidate the steps required to load a model
and prepare the data for training or inference
"""
from abc import ABC, abstractmethod
import logging

import hdb_resale_estimator as hdb_est

from interpret.glassbox import ExplainableBoostingRegressor
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)
logger.setLevel(10)

class Builder(ABC):
    """Builder class, not to be imported directly."""

    @abstractmethod
    def __init__(self):
        self.model = None
        self.objects = {}

    def process_inference_data(self, inference_data: pd.DataFrame) -> pd.DataFrame:
        """Function to perform binarizing and scaling of the inference data

        Args:
            inference_data (pd.DataFrame): Inference derived features dataframe

        Returns:
            pd.DataFrame: DataFrame containing processed features ready for inference
        """

        if "ordinal_encoder" in self.objects:
            columns = self.objects["ordinal_encoder"]["columns"]
            fitted_encoder = self.objects["ordinal_encoder"]["encoder"]
            inference_data = self._ordinal_encode_variables(
                inference_data, columns, fitted_encoder=fitted_encoder
            )
        if "one_hot_encoder" in self.objects:
            columns = self.objects["one_hot_encoder"]["columns"]
            fitted_encoder = self.objects["one_hot_encoder"]["encoder"]
            inference_data = self._one_hot_encode_cat_var(
                inference_data, columns, fitted_encoder=fitted_encoder
            )
        if "standard_scaler" in self.objects:
            scaler = self.objects["standard_scaler"]["scaler"]
            inference_data = self.scale_data(inference_data.sort_index(axis=1), scaler)

        return inference_data
    
    def _ordinal_encode_variables(
        self,
        feature_data: pd.DataFrame,
        columns: list,
        fitted_encoder: OrdinalEncoder = None,
    ) -> pd.DataFrame:
        """Performs ordinal encoding for single label categorical features before training or inference

        Args:
            feature_data (pd.DataFrame): Dataframe consisting of the feature(s)
            columns (list): List of columns that require ordinal encoding
            fitted_encoder (OrdinalEncoder): Encoder object to be used during inference. Defaults to None during training

        Returns:
            pd.DataFrame: pd.DataFrame with ordinal encoded features
        """

        existing_columns = self._check_binarize_columns(
            feature_data.columns, columns
        )

        table_to_encode = feature_data[existing_columns]

        feature_data.drop(columns=existing_columns, inplace=True)

        if fitted_encoder:
            encoder = fitted_encoder
            ord_table = pd.DataFrame(
                encoder.transform(table_to_encode),
                columns=encoder.get_feature_names_out(table_to_encode.columns),
                index=table_to_encode.index,
            )

        else:
            encoder = OrdinalEncoder()
            ord_table = pd.DataFrame(
                encoder.fit_transform(table_to_encode),
                columns=encoder.get_feature_names_out(table_to_encode.columns),
                index=table_to_encode.index,
            )
            self.objects["ordinal_encoder"] = {
                "columns": existing_columns,
                "encoder": encoder,
            }
        feature_data = feature_data.join(ord_table)

        return feature_data

    def _one_hot_encode_cat_var(
        self,
        feature_data: pd.DataFrame,
        columns: list = None,
        fitted_encoder: OneHotEncoder = None,
    ) -> pd.DataFrame:
        """Performs one-hot encoding for single label categorical features before training or inference

        Args:
            feature_data (pd.DataFrame): Dataframe consisting of the feature(s)
            columns (list): List of columns that require one hot encoding. Defaults to None during training
            fitted_encoder (OneHotEncoder): Encoder object to be used during inference. Defaults to None during training

        Returns:
            pd.DataFrame: pd.DataFrame with one-hot encoded features
        """
        if columns:
            existing_columns = self._check_binarize_columns(
                feature_data.columns, columns
            )
        else:
            existing_columns = feature_data.select_dtypes(include=['object']).columns.tolist()

        table_to_encode = feature_data[existing_columns]

        feature_data.drop(columns=existing_columns, inplace=True)

        if fitted_encoder:
            encoder = fitted_encoder
            ohc_table = pd.DataFrame(
                encoder.transform(table_to_encode).toarray(),
                columns=encoder.get_feature_names_out(table_to_encode.columns),
                index=table_to_encode.index,
            )

        else:
            encoder = OneHotEncoder()
            ohc_table = pd.DataFrame(
                encoder.fit_transform(table_to_encode).toarray(),
                columns=encoder.get_feature_names_out(table_to_encode.columns),
                index=table_to_encode.index,
            )
            self.objects["one_hot_encoder"] = {
                "columns": existing_columns,
                "encoder": encoder,
            }

        feature_data = feature_data.join(ohc_table)

        return feature_data

    def _check_binarize_columns(
        self, feature_data_columns: list, col_to_select: list
    ) -> list:
        """Checks column(s) to be selected for binarizing, whether they exist in 
        the dataframe columns

        Args:
            feature_data_columns (list): List of columns in that exist in the feature dataframe
            col_to_select (list): List of columns to be selected for binarizing

        Returns:
            list: List of columns to be selected for binarizing that 
            exists in the dataframe
        """
        existing_columns = feature_data_columns[
            feature_data_columns.isin(col_to_select)
        ]

        if existing_columns.to_list() != col_to_select:
            raise NameError(
                f"Columns to binarize not in dataset: {set(col_to_select) - set(existing_columns.unique())}"
            )

        return existing_columns

    def scale_data(
        self,
        feature_data: pd.DataFrame,
        fitted_scaler: StandardScaler=None,
    ) -> pd.DataFrame:
        """Performs standard scaling for features before training or inference

        Args:
            feature_data (pd.DataFrame): Dataframe consisting of the feature(s).
            fitted_scaler (StandardScaler): StandardScaler object to be used during inference. Defaults to None during training

        Returns:
            pd.DataFrame: Dataframe containing features with scaled values
        """

        if fitted_scaler:
            scaler = fitted_scaler
            feature_data = pd.DataFrame(
                scaler.transform(feature_data),
                columns=scaler.get_feature_names_out(feature_data.columns),
                index=feature_data.index,
            )

        else:
            scaler = StandardScaler()
            scale_columns = feature_data.select_dtypes(
                ["int64", "uint8", "float64"]
            ).columns.to_list()
            feature_data[scale_columns] = scaler.fit_transform(
                feature_data[scale_columns]
            )
            self.objects["standard_scaler"] = {
                "scaler": scaler,
            }

        return feature_data


class ClassicalModelBuilder(Builder):
    """A builder class with functions similar to Sklearn; also compatible with
    interpretML."""

    def __init__(self) -> None:
        super().__init__()

    def set_model(self, model_name: str, model_params: dict) -> 'ClassicalModelBuilder':
        """Initiatiates a model with the specified model parameters.

        Args:
            model_name (str): Name of the model
            model_params (dict): Parameters of the model

        Raises:
            NameError: Model name given was incorrect

        Returns:
            ClassicalModelBuilder: SklearnBuilder object with model params set
        """

        if model_name == "ebm":
            self.model = ExplainableBoostingRegressor().set_params(**model_params)

        elif model_name == "randforest":
            self.model = RandomForestRegressor().set_params(**model_params)

        elif model_name == "xgboost":
            self.model = XGBRegressor().set_params(**model_params)

        else:
            raise NameError(f"Incorrect model name, '{model_name}' was given")

        return self
