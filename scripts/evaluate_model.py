from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.preprocessing import ALL_FEATURES, TARGET_COLUMN, sanitize_training_dataframe

try:
    from tensorflow.keras.models import load_model
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TensorFlow is required to evaluate the model. Install requirements.txt first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained diabetes model.")
    parser.add_argument(
        "--data",
        default="data/diabetes_prediction_dataset.csv",
        help="Path to test CSV file.",
    )
    parser.add_argument(
        "--output",
        default="models/evaluation_report.json",
        help="Path for evaluation JSON report.",
    )
    return parser.parse_args()


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    model_dir = Path("models")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not (model_dir / "diabetes_model.keras").exists():
        raise FileNotFoundError("Model file missing. Run scripts/train_model.py first.")

    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = load_model(model_dir / "diabetes_model.keras")

    raw_df = pd.read_csv(data_path)
    data = sanitize_training_dataframe(raw_df)
    X = data[ALL_FEATURES]
    y = data[TARGET_COLUMN].values

    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    proba = model.predict(transformed, verbose=0).reshape(-1)

    results = evaluate(y, proba, threshold=0.5)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Evaluation report written to: {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
