#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
OUT_FILE="${DATA_DIR}/diabetes_prediction_dataset.csv"
URL="https://huggingface.co/datasets/marianeft/diabetes_prediction_dataset/raw/main/diabetes_prediction_dataset.csv"

mkdir -p "${DATA_DIR}"
curl -L "${URL}" -o "${OUT_FILE}"
echo "Downloaded dataset to ${OUT_FILE}"
