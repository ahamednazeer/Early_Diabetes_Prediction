from __future__ import annotations

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user

from .extensions import db
from .models import PredictionRecord, User
from .services.explanation_service import get_explanation_service
from .services.model_service import get_prediction_service
from .services.preprocessing import ALL_FEATURES, coerce_prediction_payload
from .services.recommendation_service import RecommendationService


main_bp = Blueprint("main", __name__)
recommendation_service = RecommendationService()


def _payload_from_request() -> dict:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict()
    return coerce_prediction_payload(payload)


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

    record = PredictionRecord(
        user_id=current_user.id,
        risk_score=prediction_result.risk_score,
        risk_label=prediction_result.risk_label,
        input_payload=payload,
        explanation=explanation,
        recommendations=recommendations,
    )
    db.session.add(record)
    db.session.commit()

    return {
        "prediction": prediction_result.to_dict(),
        "explanation": explanation,
        "recommendations": recommendations,
        "record_id": record.id,
    }


@main_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("main.index"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@main_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.login"))


@main_bp.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html", features=ALL_FEATURES)


@main_bp.route("/predict", methods=["POST"])
@login_required
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
@login_required
def history():
    records = (
        PredictionRecord.query.filter_by(user_id=current_user.id)
        .order_by(PredictionRecord.created_at.desc())
        .limit(25)
        .all()
    )
    return render_template("history.html", records=records)


@main_bp.route("/api/predict", methods=["POST"])
@login_required
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
@login_required
def api_history():
    records = (
        PredictionRecord.query.filter_by(user_id=current_user.id)
        .order_by(PredictionRecord.created_at.desc())
        .limit(50)
        .all()
    )
    return jsonify([record.to_dict() for record in records])
