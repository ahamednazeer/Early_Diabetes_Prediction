from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import current_app

from .preprocessing import ALL_FEATURES, payload_to_frame

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


@dataclass
class PredictionResult:
    risk_score: float
    risk_label: str

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_label": self.risk_label,
        }


class DiabetesPredictionService:
    def __init__(self) -> None:
        self.model = None
        self.preprocessor = None
        self.metadata: dict[str, Any] = {}
        self.threshold = 0.5
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None and self.preprocessor is not None

    def load(self) -> None:
        if self._loaded:
            return

        if load_model is None:
            raise RuntimeError(
                "TensorFlow is not installed. Install dependencies from requirements.txt."
            )

        preprocessor_path = Path(current_app.config["PREPROCESSOR_PATH"])
        model_path = Path(current_app.config["MODEL_PATH"])
        metadata_path = Path(current_app.config["METADATA_PATH"])

        if not preprocessor_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run scripts/train_model.py first."
            )

        self.preprocessor = joblib.load(preprocessor_path)
        self.model = load_model(model_path)

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
                self.threshold = float(self.metadata.get("threshold", 0.5))
        else:
            self.metadata = {}

        self._loaded = True

    def _transform_payload(self, payload: dict) -> np.ndarray:
        frame = payload_to_frame(payload)
        transformed = self.preprocessor.transform(frame[ALL_FEATURES])
        return _to_dense(transformed)

    def predict_proba_preprocessed(self, transformed_input: np.ndarray) -> np.ndarray:
        data = np.asarray(transformed_input, dtype=np.float32)
        probabilities = self.model.predict(data, verbose=0).reshape(-1)
        return probabilities

    def predict_proba_frame(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame[ALL_FEATURES])
        transformed_dense = _to_dense(transformed)
        return self.predict_proba_preprocessed(transformed_dense)

    def predict_from_payload(self, payload: dict) -> PredictionResult:
        self.load()
        transformed = self._transform_payload(payload)
        probability = float(self.predict_proba_preprocessed(transformed)[0])
        label = "High Risk" if probability >= self.threshold else "Low Risk"
        return PredictionResult(risk_score=probability, risk_label=label)

    def transform_payload(self, payload: dict) -> np.ndarray:
        self.load()
        return self._transform_payload(payload)

    def processed_feature_names(self) -> list[str]:
        self.load()
        if hasattr(self.preprocessor, "get_feature_names_out"):
            return list(self.preprocessor.get_feature_names_out())
        return ALL_FEATURES


_prediction_service: DiabetesPredictionService | None = None


def get_prediction_service() -> DiabetesPredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = DiabetesPredictionService()
    return _prediction_service
