from __future__ import annotations

from datetime import datetime

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for

from .services.explanation_service import get_explanation_service
from .services.model_service import get_prediction_service
from .services.preprocessing import ALL_FEATURES, coerce_prediction_payload
from .services.recommendation_service import RecommendationService


main_bp = Blueprint("main", __name__)
recommendation_service = RecommendationService()

# In-memory history only (no DB persistence).
RECENT_PREDICTIONS: list[dict] = []
MAX_RECENT = 25


def _payload_from_request() -> dict:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict()
    return coerce_prediction_payload(payload)


def _store_recent(payload: dict, result: dict) -> None:
    RECENT_PREDICTIONS.insert(
        0,
        {
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "risk_label": result["prediction"]["risk_label"],
            "risk_score": result["prediction"]["risk_score"],
            "input_payload": payload,
        },
    )
    if len(RECENT_PREDICTIONS) > MAX_RECENT:
        del RECENT_PREDICTIONS[MAX_RECENT:]


def _run_prediction(payload: dict) -> dict:
    prediction_service = get_prediction_service()
    prediction_result = prediction_service.predict_from_payload(payload)
    explanation_service = get_explanation_service(prediction_service)
    explanation = explanation_service.explain_from_payload(payload)
    recommendations = recommendation_service.generate(
        payload,
        risk_score=prediction_result.risk_score,
        risk_label=prediction_result.risk_label,
    )

    result = {
        "prediction": prediction_result.to_dict(),
        "explanation": explanation,
        "recommendations": recommendations,
    }
    _store_recent(payload, result)
    return result


@main_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=ALL_FEATURES)


@main_bp.route("/predict", methods=["POST"])
def predict():
    payload = _payload_from_request()
    try:
        result = _run_prediction(payload)
    except FileNotFoundError as error:
        flash(str(error), "warning")
        return redirect(url_for("main.index"))
    except Exception as error:
        flash(f"Prediction failed: {error}", "danger")
        return redirect(url_for("main.index"))

    if request.is_json:
        return jsonify(result)

    return render_template(
        "result.html",
        payload=payload,
        prediction=result["prediction"],
        explanation=result["explanation"],
        recommendations=result["recommendations"],
    )


@main_bp.route("/history", methods=["GET"])
def history():
    return render_template("history.html", records=RECENT_PREDICTIONS)


@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    payload = _payload_from_request()
    try:
        result = _run_prediction(payload)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Prediction failed: {error}"}), 500
    return jsonify(result)


@main_bp.route("/api/history", methods=["GET"])
def api_history():
    return jsonify(RECENT_PREDICTIONS)
