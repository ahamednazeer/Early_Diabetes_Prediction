from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.preprocessing import ALL_FEATURES, TARGET_COLUMN, build_preprocessor, sanitize_training_dataframe

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TensorFlow is required to train the deep learning model. Install requirements.txt first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diabetes risk prediction model.")
    parser.add_argument(
        "--data",
        default="data/diabetes_prediction_dataset.csv",
        help="Path to training CSV (must include required columns).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


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
    seed = args.seed
    np.random.seed(seed)

    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Download a modern dataset first."
        )
    raw_df = pd.read_csv(dataset_path)
    data = sanitize_training_dataframe(raw_df)

    X = data[ALL_FEATURES]
    y = data[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=seed,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    model = build_model(X_train_processed.shape[1])
    stopper = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(
        X_train_processed,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[stopper],
        verbose=1,
    )

    proba_test = model.predict(X_test_processed, verbose=0).reshape(-1)
    metrics = evaluate(y_test, proba_test, threshold=0.5)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(model_dir / "diabetes_model.keras")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    background = X_train_processed[: min(120, len(X_train_processed))]
    np.save(model_dir / "background_transformed.npy", background)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "threshold": 0.5,
        "raw_features": ALL_FEATURES,
        "processed_features": list(preprocessor.get_feature_names_out()),
        "metrics": metrics,
    }
    with (model_dir / "feature_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Training complete.")
    print("Saved model artifacts in /models:")
    print(" - diabetes_model.keras")
    print(" - preprocessor.joblib")
    print(" - background_transformed.npy")
    print(" - feature_metadata.json")
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
