from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.preprocessing import ALL_FEATURES, TARGET_COLUMN, sanitize_training_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned train/test pretrain dataset files."
    )
    parser.add_argument(
        "--data",
        default="data/diabetes_prediction_dataset.csv",
        help="Path to raw dataset CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Output directory for prepared datasets.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.data)
    out_dir = Path(args.out_dir)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {source_path}. Run ./scripts/download_dataset.sh first."
        )

    raw_df = pd.read_csv(source_path)
    clean_df = sanitize_training_dataframe(raw_df)

    train_df, test_df = train_test_split(
        clean_df[ALL_FEATURES + [TARGET_COLUMN]],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=clean_df[TARGET_COLUMN],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "diabetes_clean.csv"
    train_path = out_dir / "diabetes_train.csv"
    test_path = out_dir / "diabetes_test.csv"

    clean_df[ALL_FEATURES + [TARGET_COLUMN]].to_csv(clean_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved cleaned dataset: {clean_path}")
    print(f"Saved train split: {train_path}")
    print(f"Saved test split: {test_path}")
    print(f"Clean shape: {clean_df.shape}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("Class distribution (clean):")
    print(clean_df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict())


if __name__ == "__main__":
    main()
