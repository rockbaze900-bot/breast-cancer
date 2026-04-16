# 🧠 Breast Cancer Prediction (Advanced)

A production-style ML project with model comparison, evaluation, and deployment.

## Features
- Multiple ML models (RF, Logistic Regression)
- Feature scaling
- Confusion matrix & visualization
- REST API
- Bootstrap UI

## API Usage

POST /api/predict

{
  "features": [value1, value2, ...]
}

## Run

Train:
python model/train.py

Evaluate:
python model/evaluate.py

Run App:
cd app
python app.py
