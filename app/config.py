import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"
MODEL_DIR = BASE_DIR / "models"


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{INSTANCE_DIR / 'app.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODEL_DIR_PATH = MODEL_DIR
    MODEL_DIR = str(MODEL_DIR_PATH)
    PREPROCESSOR_PATH = str(MODEL_DIR_PATH / "preprocessor.joblib")
    MODEL_PATH = str(MODEL_DIR_PATH / "diabetes_model.keras")
    METADATA_PATH = str(MODEL_DIR_PATH / "feature_metadata.json")
    SHAP_BACKGROUND_PATH = str(MODEL_DIR_PATH / "background_transformed.npy")

    DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "admin")
    DEFAULT_PASSWORD = os.getenv("DEFAULT_PASSWORD", "admin123")
