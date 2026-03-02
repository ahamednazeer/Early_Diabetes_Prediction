from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import current_app

from .preprocessing import ALL_FEATURES, payload_to_frame


_predict_executor = ThreadPoolExecutor(max_workers=1)


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _lazy_load_keras_model(model_path: Path):
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies from requirements.txt."
        ) from exc
    return load_model(model_path)


@dataclass
class PredictionResult:
    risk_score: float
    risk_label: str
    backend: str

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_label": self.risk_label,
            "backend": self.backend,
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

        preprocessor_path = Path(current_app.config["PREPROCESSOR_PATH"])
        model_path = Path(current_app.config["MODEL_PATH"])
        metadata_path = Path(current_app.config["METADATA_PATH"])

        if not preprocessor_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run scripts/train_model.py first."
            )

        self.preprocessor = joblib.load(preprocessor_path)
        self.model = _lazy_load_keras_model(model_path)

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
                self.threshold = float(self.metadata.get("threshold", 0.5))
        else:
            self.metadata = {}

        self._loaded = True

    @staticmethod
    def _heuristic_probability(payload: dict) -> float:
        def safe_float(key: str, default: float = 0.0) -> float:
            try:
                value = payload.get(key, default)
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        age = safe_float("age")
        bmi = safe_float("bmi")
        glucose = safe_float("glucose_level")
        hba1c = safe_float("hba1c")
        hypertension = safe_float("hypertension")
        heart_disease = safe_float("heart_disease")
        smoking = str(payload.get("smoking_status", "")).lower()

        score = (
            -7.4
            + 0.02 * age
            + 0.09 * max(0.0, bmi - 22)
            + 0.03 * max(0.0, glucose - 95)
            + 0.75 * max(0.0, hba1c - 5.5)
            + 0.45 * hypertension
            + 0.55 * heart_disease
            + (0.22 if smoking in {"current", "ever"} else 0.08 if smoking in {"former", "not current"} else 0.0)
        )
        probability = 1.0 / (1.0 + np.exp(-score))
        return float(np.clip(probability, 0.01, 0.99))

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

    def _predict_with_model_timeout(self, payload: dict) -> float:
        self.load()
        transformed = self._transform_payload(payload)
        timeout_seconds = float(current_app.config.get("MODEL_PREDICT_TIMEOUT_SEC", 2.0))
        future = _predict_executor.submit(self.predict_proba_preprocessed, transformed)
        return float(future.result(timeout=timeout_seconds)[0])

    def predict_from_payload(self, payload: dict) -> PredictionResult:
        engine = str(current_app.config.get("PREDICTION_ENGINE", "heuristic")).lower()

        if engine == "heuristic":
            probability = self._heuristic_probability(payload)
            label = "High Risk" if probability >= 0.5 else "Low Risk"
            return PredictionResult(risk_score=probability, risk_label=label, backend="heuristic")

        if engine == "model":
            probability = self._predict_with_model_timeout(payload)
            label = "High Risk" if probability >= self.threshold else "Low Risk"
            return PredictionResult(risk_score=probability, risk_label=label, backend="model")

        # auto mode: try model first and fallback if timeout/error.
        try:
            probability = self._predict_with_model_timeout(payload)
            label = "High Risk" if probability >= self.threshold else "Low Risk"
            return PredictionResult(risk_score=probability, risk_label=label, backend="model")
        except (TimeoutError, Exception):
            probability = self._heuristic_probability(payload)
            label = "High Risk" if probability >= 0.5 else "Low Risk"
            return PredictionResult(risk_score=probability, risk_label=label, backend="heuristic")

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
