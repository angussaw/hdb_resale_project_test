"""Module containing the function to train a model
"""
import logging
from hydra import compose, initialize
import hydra.core.global_hydra

import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="train_config.yaml", version_base=None)
def main(train_config):
    with hdb_est.utils.timer("Model training"):
        hdb_est.utils.setup_logging()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="../conf"):
            logger.info("Starting train pipeline...")
            logger.info(
                "Retrieving training data..."
            )

            read_from_source = train_config["files"]["derived_features"]["read_from_source"]
            read_params = train_config["files"]["derived_features"]["params"]

            derived_hdb_features = hdb_est.utils.read_data(source=read_from_source, params=read_params)

            logger.info("Initialising model training...")
            metric, model_uri = hdb_est.modeling.training.train_pipeline(
                train_config, derived_hdb_features
            )
            logger.info("Model training completed!!!")

    return metric


if __name__ == "__main__":
    main()