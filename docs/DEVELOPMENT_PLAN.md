# Explainable Early Diabetes Prediction System

This document translates your module-wise flow into a production-oriented build plan using:
- Frontend: HTML + CSS + JavaScript
- Backend: Python (Flask)
- Database: SQLite
- ML/XAI: TensorFlow/Keras + SHAP

## 1) System Architecture

### High-Level Flow
1. User logs in and submits health profile.
2. Flask backend validates and preprocesses input.
3. Deep learning model predicts diabetes risk probability.
4. SHAP module explains top contributing factors.
5. Recommendation engine generates health guidance.
6. Prediction record is stored in SQLite.
7. Result + explanations + recommendations are displayed.

### Module Mapping
- Module 1: `data/*.csv` data sources.
- Module 2: `app/services/preprocessing.py`.
- Module 3: `scripts/train_model.py`.
- Module 4: `scripts/evaluate_model.py`.
- Module 5: `app/services/explanation_service.py`.
- Module 6: Flask app under `app/`.
- Module 7: templates and assets under `app/templates` and `app/static`.
- Module 8: `app/services/recommendation_service.py`.
- Module 9: `app/models.py` with SQLite persistence.

## 2) Dataset Strategy

The system supports any dataset that contains these required columns:
- `age`
- `gender`
- `hypertension`
- `heart_disease`
- `smoking_status` (or `smoking_history`)
- `bmi`
- `hba1c` (or `HbA1c_level`)
- `glucose_level`
- `diabetes_risk` (or `diabetes`, target: 0/1)

Current downloaded dataset:
- `data/diabetes_prediction_dataset.csv`
- Source: `https://huggingface.co/datasets/marianeft/diabetes_prediction_dataset/raw/main/diabetes_prediction_dataset.csv`

Preprocessing also maps common aliases (`blood_glucose_level` -> `glucose_level`, `HbA1c_level` -> `hba1c`, etc.).

## 3) Development Phases

### Phase 1: Requirement Analysis
- Define risk threshold policy (`0.5` default, configurable in metadata).
- Decide target user roles (single-user demo vs multi-user production).
- Lock non-functional goals:
  - Prediction latency target under 2 seconds.
  - Secure login and password hashing.
  - Auditable prediction history.

### Phase 2: Data Module
1. Collect data from selected source (UCI/NHANES/Kaggle/Hospital).
2. Standardize column names and data types.
3. Handle missing values:
  - Numeric: median imputation.
  - Categorical: mode imputation.
4. Encode categorical features with one-hot encoding.
5. Scale numeric features using standard scaling.
6. Split train/test set (80/20 stratified).

### Phase 3: Deep Learning Module
1. Build Keras model:
  - Input
  - Dense(64, ReLU)
  - Dense(32, ReLU)
  - Dense(16, ReLU)
  - Output Dense(1, Sigmoid)
2. Train with early stopping.
3. Save artifacts:
  - `models/diabetes_model.keras`
  - `models/preprocessor.joblib`
  - `models/background_transformed.npy`
  - `models/feature_metadata.json`

### Phase 4: Evaluation Module
Generate:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Saved via `scripts/evaluate_model.py` into `models/evaluation_report.json`.

### Phase 5: Explainable AI Module
1. Load transformed background samples.
2. Run SHAP KernelExplainer on current prediction.
3. Aggregate one-hot encoded SHAP values back to original feature names.
4. Return top impact features for UI rendering.

### Phase 6: Backend Module (Flask)
Core endpoints:
- `GET/POST /login`
- `POST /logout`
- `GET /` (input page)
- `POST /predict` (web prediction)
- `GET /history`
- `POST /api/predict` (JSON API)
- `GET /api/history`

### Phase 7: Frontend Module
Pages:
- Login page
- Assessment input page
- Result page with:
  - Risk label and probability
  - SHAP top factors
  - Recommendations
- History page with prior predictions

UI standards applied:
- Responsive layout
- Consistent design system (colors, typography, spacing)
- Clear hierarchy and fast form submission flow

### Phase 8: Recommendation Module
Rule-based engine generates personalized guidance from:
- BMI
- Glucose
- HbA1c
- Smoking
- Hypertension
- Heart disease

### Phase 9: Database Module
SQLite tables:
- `users`
- `prediction_records`

Stored fields:
- user
- risk score
- risk label
- input payload
- explanation payload
- recommendation list
- timestamp

## 4) Production Hardening Roadmap

1. Add role-based access control and password reset.
2. Add API rate limiting and request validation schemas.
3. Containerize with Docker and serve via Gunicorn + Nginx.
4. Add model versioning and experiment tracking.
5. Add CI pipeline for lint, unit tests, and integration tests.
