"""
## data_preparation should contains the steps required to perform the following tasks:
## 1. Data Cleaning
## 2. Feature Engineering
"""
import logging
from omegaconf import DictConfig
import pandas as pd

import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)


def data_prep_pipeline(
    config: DictConfig,
    raw_hdb_data: pd.DataFrame,
) -> None:
    """This is a wrapper function to clean the data and perform feature engineering to
    prep the data for model training"""

    logger.info("Initialising Data cleaner...")
    data_cleaner = hdb_est.data_prep.data_cleaning.DataCleaner(
        raw_hdb_data=raw_hdb_data,
        params=config,
    )

    logger.info("Cleaning data.....Please wait as this takes some time!!!")
    with hdb_est.utils.timer("Data Cleaning"):
        clean_hdb_data = data_cleaner.clean_data()
    logger.info("Data cleaning complete!")

    logger.info("Shape of clean hdb data: %s", clean_hdb_data.shape)

    # Initialize feature engineer
    logger.info("Initialising Feature Engineering...")
    feature_engineer = hdb_est.data_prep.feature_engineering.FeatureEngineer(
        params=config
    )

    # Engineer features
    logger.info(
        "Generating hdb derived features...Please wait as this takes some time!!!"
    )
    with hdb_est.utils.timer("hdb derived feature generation"):
        derived_data_hdb = feature_engineer.engineer_features(hdb_data=clean_hdb_data)
        logger.info("Shape of hdb derived features: %s", derived_data_hdb.shape)

    return derived_data_hdb
