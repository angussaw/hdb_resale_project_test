"""
## data_prep_pipeline.py retrieves the raw data and data prep configuration,
and initialises data preparation
"""
from hydra import compose, initialize
import logging

import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)


def main():
    with hdb_est.utils.timer("Data Preparation"):
        hdb_est.utils.setup_logging()
        with initialize(version_base=None, config_path="../conf"):
            data_prep_config = compose(config_name="data_prep")
            logger.info("Starting data preparation pipeline")
            logger.info("Retrieving raw data and data preparation config...")

            logger.info("Performing data preparation in training mode...")
            raw_hdb_data = hdb_est.utils.read_data(
                source=data_prep_config["files"]["training"]["raw_data"][
                    "read_from_source"
                ],
                params=data_prep_config["files"]["training"]["raw_data"]["params"],
            )
            logger.info("Shape of raw hdb data: %s", raw_hdb_data.shape)

            logger.info("Initialising data preparation...")
            hdb_preprocessed = hdb_est.data_prep.data_preparation.data_prep_pipeline(
                data_prep_config["data_prep"], raw_hdb_data
            )
            number_of_nulls = hdb_preprocessed.isna().sum().sum()

            logger.info(
                f"Data preparation completed. There are {number_of_nulls} null values present"
            )

        # Save clean and engineered mppa data
        logger.info(
            "Saving derived hdb features to %s...",
            data_prep_config["files"]["training"]["save_to_source"],
        )
        if data_prep_config["files"]["training"]["save_to_source"] == "postgres":
            db_engine = hdb_est.utils.create_postgres_engine()
            derived_features_table_name = data_prep_config["files"]["training"][
                "derived_features_table_name"
            ]
            hdb_est.utils.push_data_to_sql(
                db_engine, hdb_preprocessed, derived_features_table_name
            )
            logger.info(
                "Derived mppa features is saved in %s", derived_features_table_name
            )

        elif data_prep_config["files"]["training"]["save_to_source"] == "local":
            preprocessed_save_path = data_prep_config["files"]["training"][
                "preprocessed_save_path"
            ]
            hdb_preprocessed.to_csv(preprocessed_save_path)
            logger.info("Derived mppa features is saved in %s", preprocessed_save_path)


if __name__ == "__main__":
    main()
