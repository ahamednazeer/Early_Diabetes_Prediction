import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")

    MODEL_DIR_PATH = MODEL_DIR
    MODEL_DIR = str(MODEL_DIR_PATH)
    PREPROCESSOR_PATH = str(MODEL_DIR_PATH / "preprocessor.joblib")
    MODEL_PATH = str(MODEL_DIR_PATH / "diabetes_model.keras")
    METADATA_PATH = str(MODEL_DIR_PATH / "feature_metadata.json")
    SHAP_BACKGROUND_PATH = str(MODEL_DIR_PATH / "background_transformed.npy")
    PREDICTION_ENGINE = os.getenv("PREDICTION_ENGINE", "heuristic").lower()
    MODEL_PREDICT_TIMEOUT_SEC = float(os.getenv("MODEL_PREDICT_TIMEOUT_SEC", "2.0"))
    EXPLANATION_ENGINE = os.getenv("EXPLANATION_ENGINE", "rule_based").lower()
    SHAP_NSAMPLES = int(os.getenv("SHAP_NSAMPLES", "40"))
    SHAP_BACKGROUND_MAX_ROWS = int(os.getenv("SHAP_BACKGROUND_MAX_ROWS", "25"))
    PRELOAD_MODEL_ON_STARTUP = os.getenv("PRELOAD_MODEL_ON_STARTUP", "0") == "1"
