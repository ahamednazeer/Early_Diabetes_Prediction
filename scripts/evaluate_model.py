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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override prediction threshold. Defaults to value from models/feature_metadata.json.",
    )
    return parser.parse_args()


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (proba >= threshold).astype(int)
    precision = float(precision_score(y_true, predictions, zero_division=0))
    recall = float(recall_score(y_true, predictions, zero_division=0))
    f1 = float(f1_score(y_true, predictions, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
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

    metadata_path = model_dir / "feature_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    raw_df = pd.read_csv(data_path)
    data = sanitize_training_dataframe(raw_df)
    X = data[ALL_FEATURES]
    y = data[TARGET_COLUMN].values

    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    proba = model.predict(transformed, verbose=0).reshape(-1)

    threshold = float(args.threshold) if args.threshold is not None else float(
        metadata.get("threshold", 0.5)
    )
    results = evaluate(y, proba, threshold=threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Evaluation report written to: {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
