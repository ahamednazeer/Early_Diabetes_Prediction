from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
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
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.preprocessing import ALL_FEATURES, TARGET_COLUMN, build_preprocessor, sanitize_training_dataframe

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--optimize-for",
        choices=["f1", "recall", "balanced"],
        default="f1",
        help="Metric used for validation threshold tuning.",
    )
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
    precision = float(precision_score(y_true, predictions, zero_division=0))
    recall = float(recall_score(y_true, predictions, zero_division=0))
    f1 = float(f1_score(y_true, predictions, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _objective_score(metrics: dict, optimize_for: str) -> float:
    if optimize_for == "recall":
        return float(metrics["recall"])
    if optimize_for == "balanced":
        return float((metrics["precision"] + metrics["recall"]) / 2.0)
    return float(metrics["f1_score"])


def tune_threshold(y_true: np.ndarray, proba: np.ndarray, optimize_for: str) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = evaluate(y_true, proba, threshold=0.5)
    best_score = _objective_score(best_metrics, optimize_for)

    for threshold in np.arange(0.10, 0.91, 0.01):
        threshold = float(np.round(threshold, 2))
        metrics = evaluate(y_true, proba, threshold=threshold)
        score = _objective_score(metrics, optimize_for)
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def compute_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def print_metrics(title: str, metrics: dict) -> None:
    print(f"{title}:")
    print(json.dumps(metrics, indent=2))


def _to_dense(matrix: np.ndarray) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _clean_smoking_status(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized["smoking_status"] = normalized["smoking_status"].replace(
        {"no_info": "never"}
    )
    return normalized


def build_training_splits(
    data: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X = data[ALL_FEATURES]
    y = data[TARGET_COLUMN].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def persist_metadata(
    model_dir: Path,
    dataset_path: Path,
    threshold: float,
    args: argparse.Namespace,
    preprocessor,
    val_metrics: dict,
    test_metrics: dict,
    class_weights: dict[int, float],
) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "threshold": round(float(threshold), 4),
        "threshold_optimized_for": args.optimize_for,
        "split": {
            "test_size": args.test_size,
            "val_size": args.val_size,
            "seed": args.seed,
        },
        "class_weights": class_weights,
        "raw_features": ALL_FEATURES,
        "processed_features": list(preprocessor.get_feature_names_out()),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (model_dir / "feature_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


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
    data = _clean_smoking_status(data)

    X_train, X_val, X_test, y_train, y_val, y_test = build_training_splits(
        data=data,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=seed,
    )

    preprocessor = build_preprocessor()
    X_train_processed = _to_dense(preprocessor.fit_transform(X_train))
    X_val_processed = _to_dense(preprocessor.transform(X_val))
    X_test_processed = _to_dense(preprocessor.transform(X_test))

    class_weights = compute_weights(y_train)

    model = build_model(X_train_processed.shape[1])
    stopper = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[stopper],
        class_weight=class_weights,
        verbose=1,
    )

    proba_val = model.predict(X_val_processed, verbose=0).reshape(-1)
    threshold, val_metrics = tune_threshold(y_val, proba_val, optimize_for=args.optimize_for)

    proba_test = model.predict(X_test_processed, verbose=0).reshape(-1)
    test_metrics = evaluate(y_test, proba_test, threshold=threshold)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(model_dir / "diabetes_model.keras")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    background = X_train_processed[: min(120, len(X_train_processed))]
    np.save(model_dir / "background_transformed.npy", background)

    persist_metadata(
        model_dir=model_dir,
        dataset_path=dataset_path,
        threshold=threshold,
        args=args,
        preprocessor=preprocessor,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        class_weights=class_weights,
    )

    print("Training complete.")
    print("Saved model artifacts in /models:")
    print(" - diabetes_model.keras")
    print(" - preprocessor.joblib")
    print(" - background_transformed.npy")
    print(" - feature_metadata.json")
    print(f"Selected threshold ({args.optimize_for}): {threshold}")
    print_metrics("Validation metrics", val_metrics)
    print_metrics("Test metrics", test_metrics)


if __name__ == "__main__":
    main()
