from __future__ import annotations

from pathlib import Path

import numpy as np
from flask import current_app

from .model_service import DiabetesPredictionService
from .preprocessing import CATEGORICAL_FEATURES, extract_original_feature

_SHAP_MODULE = None
_SHAP_IMPORT_FAILED = False


def _load_shap_module():
    global _SHAP_MODULE, _SHAP_IMPORT_FAILED
    if _SHAP_MODULE is not None:
        return _SHAP_MODULE
    if _SHAP_IMPORT_FAILED:
        return None
    try:
        import shap as shap_module

        _SHAP_MODULE = shap_module
        return _SHAP_MODULE
    except Exception:  # pragma: no cover
        _SHAP_IMPORT_FAILED = True
        return None


class ShapExplanationService:
    def __init__(self, prediction_service: DiabetesPredictionService) -> None:
        self.prediction_service = prediction_service
        self._explainer = None

    def _build_explainer(self):
        shap_module = _load_shap_module()
        if shap_module is None:
            return None
        if self._explainer is not None:
            return self._explainer

        background_path = Path(current_app.config["SHAP_BACKGROUND_PATH"])
        if not background_path.exists():
            return None

        background = np.load(background_path)
        max_rows = int(current_app.config.get("SHAP_BACKGROUND_MAX_ROWS", 25))
        background = background[:max_rows] if len(background) > max_rows else background

        self._explainer = shap_module.KernelExplainer(
            self.prediction_service.predict_proba_preprocessed,
            background,
            link="logit",
        )
        return self._explainer

    def _shap_explain_from_payload(self, payload: dict) -> dict:
        transformed = self.prediction_service.transform_payload(payload)
        feature_names = self.prediction_service.processed_feature_names()
        explainer = self._build_explainer()

        if explainer is None:
            return {
                "available": False,
                "top_features": [],
                "message": "SHAP explanation is unavailable until the background sample is generated.",
            }

        nsamples = int(current_app.config.get("SHAP_NSAMPLES", 40))
        shap_values = explainer.shap_values(transformed, nsamples=nsamples)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        contribution_vector = shap_values[0]

        grouped: dict[str, float] = {}
        for feature_name, contribution in zip(feature_names, contribution_vector):
            original_name = extract_original_feature(feature_name, CATEGORICAL_FEATURES)
            grouped[original_name] = grouped.get(original_name, 0.0) + float(contribution)

        sorted_features = sorted(
            grouped.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top_features = [
            {"feature": feature, "impact": round(score, 4)}
            for feature, score in sorted_features[:6]
        ]

        return {
            "available": True,
            "top_features": top_features,
            "backend": "shap",
        }

    @staticmethod
    def _rule_based_explain_from_payload(payload: dict) -> dict:
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        age = safe_float(payload.get("age"), 0.0)
        bmi = safe_float(payload.get("bmi"), 0.0)
        glucose = safe_float(payload.get("glucose_level"), 0.0)
        hba1c = safe_float(payload.get("hba1c"), 0.0)
        hypertension = int(safe_float(payload.get("hypertension"), 0.0))
        heart_disease = int(safe_float(payload.get("heart_disease"), 0.0))
        smoking = str(payload.get("smoking_status", "")).lower().strip()

        contributions: dict[str, float] = {}

        if glucose >= 180:
            contributions["glucose_level"] = 1.0
        elif glucose >= 126:
            contributions["glucose_level"] = 0.8
        elif glucose >= 100:
            contributions["glucose_level"] = 0.35
        else:
            contributions["glucose_level"] = -0.2

        if hba1c >= 6.5:
            contributions["hba1c"] = 0.95
        elif hba1c >= 5.7:
            contributions["hba1c"] = 0.5
        else:
            contributions["hba1c"] = -0.2

        if bmi >= 30:
            contributions["bmi"] = 0.45
        elif bmi >= 25:
            contributions["bmi"] = 0.22
        else:
            contributions["bmi"] = -0.08

        contributions["age"] = 0.25 if age >= 45 else -0.05
        contributions["hypertension"] = 0.3 if hypertension == 1 else -0.04
        contributions["heart_disease"] = 0.35 if heart_disease == 1 else -0.03

        if smoking in {"current", "ever"}:
            contributions["smoking_status"] = 0.25
        elif smoking in {"former", "not current"}:
            contributions["smoking_status"] = 0.1
        else:
            contributions["smoking_status"] = -0.04

        sorted_features = sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top_features = [
            {"feature": feature, "impact": round(float(score), 4)}
            for feature, score in sorted_features[:6]
        ]

        return {
            "available": True,
            "top_features": top_features,
            "backend": "rule_based",
            "message": "Fast explanation mode enabled for low-latency prediction.",
        }

    def explain_from_payload(self, payload: dict) -> dict:
        engine = str(current_app.config.get("EXPLANATION_ENGINE", "rule_based")).lower()
        if engine == "rule_based":
            return self._rule_based_explain_from_payload(payload)

        if engine == "shap":
            return self._shap_explain_from_payload(payload)

        # auto mode: try SHAP first, fall back immediately on any failure.
        try:
            shap_result = self._shap_explain_from_payload(payload)
            if shap_result.get("available"):
                return shap_result
        except Exception:
            pass
        return self._rule_based_explain_from_payload(payload)


_shap_service: ShapExplanationService | None = None


def get_explanation_service(prediction_service: DiabetesPredictionService) -> ShapExplanationService:
    global _shap_service
    if _shap_service is None:
        _shap_service = ShapExplanationService(prediction_service)
    return _shap_service
