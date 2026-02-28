from __future__ import annotations

from pathlib import Path

import numpy as np
from flask import current_app

from .model_service import DiabetesPredictionService
from .preprocessing import CATEGORICAL_FEATURES, extract_original_feature

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


class ShapExplanationService:
    def __init__(self, prediction_service: DiabetesPredictionService) -> None:
        self.prediction_service = prediction_service
        self._explainer = None

    def _build_explainer(self):
        if shap is None:
            return None
        if self._explainer is not None:
            return self._explainer

        background_path = Path(current_app.config["SHAP_BACKGROUND_PATH"])
        if not background_path.exists():
            return None

        background = np.load(background_path)
        background = background[:80] if len(background) > 80 else background

        self._explainer = shap.KernelExplainer(
            self.prediction_service.predict_proba_preprocessed,
            background,
            link="logit",
        )
        return self._explainer

    def explain_from_payload(self, payload: dict) -> dict:
        transformed = self.prediction_service.transform_payload(payload)
        feature_names = self.prediction_service.processed_feature_names()
        explainer = self._build_explainer()

        if explainer is None:
            return {
                "available": False,
                "top_features": [],
                "message": "SHAP explanation is unavailable until the background sample is generated.",
            }

        shap_values = explainer.shap_values(transformed, nsamples=120)
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

        return {"available": True, "top_features": top_features}


_shap_service: ShapExplanationService | None = None


def get_explanation_service(prediction_service: DiabetesPredictionService) -> ShapExplanationService:
    global _shap_service
    if _shap_service is None:
        _shap_service = ShapExplanationService(prediction_service)
    return _shap_service
