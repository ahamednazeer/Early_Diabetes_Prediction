from __future__ import annotations


class RecommendationService:
    def generate(self, payload: dict, risk_score: float, risk_label: str) -> list[str]:
        recommendations: list[str] = []

        bmi = float(payload.get("bmi", 0) or 0)
        glucose = float(payload.get("glucose_level", 0) or 0)
        hba1c = float(payload.get("hba1c", 0) or 0)
        hypertension = int(float(payload.get("hypertension", 0) or 0))
        heart_disease = int(float(payload.get("heart_disease", 0) or 0))
        smoking_status = str(payload.get("smoking_status", "")).lower()

        if risk_label == "High Risk" or risk_score >= 0.65:
            recommendations.append(
                "Schedule a physician consultation and fasting blood sugar confirmation test."
            )
            recommendations.append("Track glucose and HbA1c trends weekly for early intervention.")

        if bmi >= 25:
            recommendations.append(
                "Target gradual weight reduction through a calorie-controlled diet and 150+ minutes of weekly exercise."
            )
        if glucose >= 126:
            recommendations.append(
                "Reduce refined sugars and high-glycemic foods; prioritize fiber-rich meals."
            )
        if hba1c >= 6.5:
            recommendations.append(
                "Adopt strict carbohydrate management and discuss medication options with a clinician."
            )
        if hypertension == 1:
            recommendations.append(
                "Monitor blood pressure weekly and follow a low-sodium, heart-healthy diet plan."
            )
        if heart_disease == 1:
            recommendations.append(
                "Coordinate diabetes prevention with cardiology follow-up due to elevated cardiovascular risk."
            )
        if smoking_status in {"current", "yes", "smoker", "ever", "not_current"}:
            recommendations.append(
                "Enroll in a smoking cessation program to lower insulin resistance and cardiovascular risk."
            )
        if smoking_status in {"no_info", "unknown", ""}:
            recommendations.append(
                "Capture complete smoking history during next screening to improve risk personalization."
            )

        if not recommendations:
            recommendations.append(
                "Maintain current lifestyle habits and continue periodic screening every 6-12 months."
            )

        return recommendations
