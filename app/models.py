from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    predictions = db.relationship("PredictionRecord", backref="user", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class PredictionRecord(db.Model):
    __tablename__ = "prediction_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    risk_score = db.Column(db.Float, nullable=False)
    risk_label = db.Column(db.String(32), nullable=False)
    input_payload = db.Column(db.JSON, nullable=False)
    explanation = db.Column(db.JSON, nullable=True)
    recommendations = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "risk_score": round(float(self.risk_score), 4),
            "risk_label": self.risk_label,
            "input_payload": self.input_payload,
            "explanation": self.explanation or {},
            "recommendations": self.recommendations or [],
            "created_at": self.created_at.isoformat(),
        }
