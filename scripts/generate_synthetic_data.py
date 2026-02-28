from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(20, 80, size=rows)
    bmi = np.clip(rng.normal(27, 5.5, rows), 16, 45)
    glucose_level = np.clip(rng.normal(112, 28, rows), 70, 240)
    hba1c = np.clip(rng.normal(5.8, 1.1, rows), 3.8, 11.5)
    hypertension = rng.binomial(1, p=0.22, size=rows)
    heart_disease = rng.binomial(1, p=0.10, size=rows)

    gender = rng.choice(["male", "female"], size=rows)
    smoking_status = rng.choice(
        ["never", "former", "current", "not current", "ever", "no info"],
        size=rows,
        p=[0.50, 0.12, 0.15, 0.12, 0.06, 0.05],
    )

    logits = (
        -12.0
        + 0.025 * age
        + 0.09 * (bmi - 25)
        + 0.030 * (glucose_level - 100)
        + 0.50 * (hba1c - 5.7)
        + 0.75 * hypertension
        + 0.85 * heart_disease
        + 0.16 * (smoking_status == "current")
    )
    probabilities = 1 / (1 + np.exp(-logits))
    diabetes_risk = rng.binomial(1, np.clip(probabilities, 0.01, 0.99))

    return pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_status": smoking_status,
            "bmi": np.round(bmi, 2),
            "hba1c": np.round(hba1c, 2),
            "glucose_level": np.round(glucose_level, 2),
            "diabetes_risk": diabetes_risk,
        }
    )


def main() -> None:
    output_path = Path("data/diabetes_synthetic.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_dataset(rows=5000, seed=42)
    dataset.to_csv(output_path, index=False)
    print(f"Synthetic dataset generated at: {output_path}")
    print(dataset.head())


if __name__ == "__main__":
    main()
