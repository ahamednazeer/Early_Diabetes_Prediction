from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "bmi",
    "glucose_level",
    "hba1c",
    "hypertension",
    "heart_disease",
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_status",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "diabetes_risk"

FEATURE_ALIASES = {
    "age": ["age"],
    "gender": ["gender", "sex"],
    "hypertension": ["hypertension", "high_blood_pressure"],
    "heart_disease": ["heart_disease", "cardiovascular_disease"],
    "smoking_status": ["smoking_status", "smoking_history", "smoking"],
    "bmi": ["bmi", "body_mass_index"],
    "hba1c": ["hba1c", "hba1c_level", "hba1c_level_percent"],
    "glucose_level": ["glucose_level", "blood_glucose_level", "glucose", "blood_glucose"],
    TARGET_COLUMN: ["diabetes_risk", "diabetes", "outcome", "target", "label"],
}


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {_col: _normalize_column_name(_col) for _col in df.columns}
    return df.rename(columns=renamed)


def _build_alias_lookup() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        for alias in aliases:
            mapping[_normalize_column_name(alias)] = canonical
    return mapping


def canonicalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = standardize_dataframe_columns(df.copy())
    alias_lookup = _build_alias_lookup()

    renamed: dict[str, str] = {}
    seen_canonical: set[str] = set()

    for column in frame.columns:
        canonical = alias_lookup.get(column)
        if canonical and canonical not in seen_canonical:
            renamed[column] = canonical
            seen_canonical.add(canonical)

    return frame.rename(columns=renamed)


def sanitize_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frame = canonicalize_dataset_columns(df)
    missing = [feature for feature in ALL_FEATURES + [TARGET_COLUMN] if feature not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    for feature in CATEGORICAL_FEATURES:
        frame[feature] = frame[feature].astype(str).str.strip().str.lower()

    frame[TARGET_COLUMN] = (
        pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )
    frame = frame.dropna(subset=[TARGET_COLUMN])
    return frame


def _payload_get(payload: dict, canonical_key: str):
    if canonical_key in payload:
        return payload.get(canonical_key)
    for alias in FEATURE_ALIASES.get(canonical_key, []):
        if alias in payload:
            return payload.get(alias)
    return None


def coerce_prediction_payload(payload: dict) -> dict:
    normalized = {}
    for feature in NUMERIC_FEATURES:
        value = _payload_get(payload, feature)
        normalized[feature] = float(value) if value not in (None, "", "null") else np.nan
    for feature in CATEGORICAL_FEATURES:
        value = _payload_get(payload, feature)
        normalized[feature] = str(value).strip().lower()
    return normalized


def payload_to_frame(payload: dict) -> pd.DataFrame:
    normalized = coerce_prediction_payload(payload)
    return pd.DataFrame([normalized], columns=ALL_FEATURES)


def extract_original_feature(feature_name: str, categorical_features: Iterable[str]) -> str:
    if feature_name.startswith("num__"):
        return feature_name.removeprefix("num__")
    if feature_name.startswith("cat__"):
        raw_name = feature_name.removeprefix("cat__")
        for category in categorical_features:
            if raw_name == category or raw_name.startswith(f"{category}_"):
                return category
        return raw_name
    return feature_name
