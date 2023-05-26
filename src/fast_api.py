"""
## fast_api.py contains the backend logic to process raw input data for inference,
make predictions and generate shap values
"""
from fastapi import FastAPI
import jsonpickle
import logging
import mlflow
import os
import pandas as pd
import sys
import yaml
import uvicorn

with open("conf/data_prep.yaml", "r") as file:
    config = yaml.safe_load(file)

sys.path.append("src")
import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)

# Load model
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model_uri = os.getenv("MODEL_URI")
run_id = os.getenv("RUN_ID")
builder = hdb_est.utils.retrieve_builder(run_id=run_id, model_uri=model_uri)

PRED_MODEL = builder.model
PRED_MODEL_FEATURES = builder.objects["features"]  # before encoding
if "explainer" in builder.objects.keys():
    PRED_MODEL_EXPLAINER = builder.objects["explainer"]
else:
    PRED_MODEL_EXPLAINER = None


app = FastAPI()


@app.post("/predict")
def predict_resale_value(hdb_flat_dict: dict):
    """Get model prediction of hdb resale value

    Args:
        hdb_flat_dict (dict): Dictionary containing derived hdb features post data prep
        (data cleaning + feature engineering)
    """

    hdb_flat_df = pd.DataFrame([hdb_flat_dict])[PRED_MODEL_FEATURES]
    processed_hdb_flat_df = builder.process_inference_data(inference_data=hdb_flat_df)
    result = PRED_MODEL.predict(processed_hdb_flat_df)

    return result.tolist()[0]


@app.post("/dataprep")
def prepare_raw_data(input_data: dict):
    """Clean and process raw input data

    Args:
        hdb_flat_dict (dict): Dictionary containing raw hdb features
    """

    input_data = pd.DataFrame([input_data])
    data_cleaner = hdb_est.data_prep.data_cleaning.DataCleaner(
        raw_hdb_data=input_data, params=config["data_prep"], inference_mode=True
    )
    clean_input_data = data_cleaner.clean_data()

    logger.info("Conducting Feature Engineering...")
    feature_engineer = hdb_est.data_prep.feature_engineering.FeatureEngineer(
        params=config["data_prep"], inference_mode=True
    )

    derived_input_data = feature_engineer.engineer_features(hdb_data=clean_input_data)

    return derived_input_data.iloc[0].to_dict()


@app.post("/explain")
def generate_shap_values(hdb_flat_dict: dict):
    """Generate shap values to explain each model prediction on a hdb flat

    Args:
        hdb_flat_dict (dict): Dictionary containing derived hdb features post data prep
        (data cleaning + feature engineering)
    """

    hdb_flat_df = pd.DataFrame([hdb_flat_dict])[PRED_MODEL_FEATURES]
    processed_hdb_flat_df = builder.process_inference_data(inference_data=hdb_flat_df)

    if PRED_MODEL_EXPLAINER:
        shap_values = PRED_MODEL_EXPLAINER(processed_hdb_flat_df)
        return jsonpickle.encode(shap_values[0])

    else:
        return


if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8500)
