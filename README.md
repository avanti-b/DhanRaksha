# 🛡️ Dhan₹aksha — Intelligent Fraud Detection System

Guarding every rupee with intelligent precision.

---

## Overview

Dhan₹aksha is an end-to-end machine learning–powered fraud detection system designed to identify suspicious financial transactions in real time. It integrates a trained ML model, a Flask backend API, and an interactive frontend dashboard to deliver predictions, analytics, and performance insights.

---

## Problem Statement

Financial fraud detection is a challenging problem due to extreme class imbalance, evolving fraud patterns, and the need for real-time prediction. In real-world datasets, fraudulent transactions account for less than 0.2% of all transactions, making it difficult for traditional systems to detect anomalies effectively.

Dhan₹aksha addresses this by leveraging machine learning models capable of learning hidden patterns and detecting fraud with high precision and recall.

---

## Dataset

This project uses the Credit Card Fraud Detection dataset from Kaggle:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions, out of which only 492 are fraudulent (~0.17%). Features V1–V28 are PCA-transformed to preserve confidentiality, while Time and Amount represent transaction details. The target variable “Class” indicates whether a transaction is fraudulent or legitimate.

Note: The dataset is not included in this repository due to GitHub file size limits.

---

## Features

* Machine learning-based fraud detection using Random Forest and Logistic Regression
* SMOTE-based class balancing to handle imbalanced data
* Real-time prediction via Flask API
* Interactive dashboard with metrics and visualizations
* Confusion matrix and performance comparison
* Risk classification (Low, Medium, High)
* Modular and scalable architecture

---

## Project Structure

```
DhanRaksha/
│
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # ML training script
│   ├── requirements.txt    # Python dependencies
│   └── model/              # Auto-created after training
│       ├── fraud_model.pkl # Saved Random Forest model
│       ├── scaler.pkl      # Saved scaler
│       └── metrics.json    # Model evaluation results
│
├── frontend/
│   └── index.html          # Dashboard UI
│
└── FraudShield_AI_Training.ipynb  # Colab notebook
```

## Setup Instructions

1. Clone the repository
   git clone [https://github.com/your-username/dhanraksha.git](https://github.com/your-username/dhanraksha.git)
   cd dhanraksha

2. Create a virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r backend/requirements.txt

4. Add dataset
   Download the dataset and place it inside the backend folder as:
   backend/creditcard.csv

5. Train the model
   cd backend
   python train_model.py

6. Run backend server
   python app.py

The backend will run at:
[http://localhost:5000](http://localhost:5000)

7. Open frontend
   Open frontend/index.html in your browser

---

## API Endpoints

POST /api/predict
Predict whether a transaction is fraudulent

Sample request:
{
"amount": 5000,
"hour": 2,
"v1": -5,
"v2": 3,
"v14": -4,
"v17": 2
}

Response:
{
"prediction": 1,
"fraud_probability": 92.3,
"label": "Fraud",
"risk_level": "High"
}

GET /api/metrics
Returns model performance metrics

GET /api/health
Returns backend status

---

## Model Performance

Random Forest (final model):

* Precision: ~0.92
* Recall: ~0.85
* F1 Score: ~0.88
* AUC-ROC: ~0.97

Logistic Regression (baseline):

* Precision: ~0.89
* Recall: ~0.80
* F1 Score: ~0.84
* AUC-ROC: ~0.95

---

## Highlights

* Handles extreme class imbalance using SMOTE
* Real-time ML inference via Flask API
* Clean and interactive dashboard UI
* Efficient and scalable design

---

## Limitations

* Dataset is anonymized, limiting interpretability
* Model threshold tuning can be improved
* Not deployed on cloud (local system only)

---

## Future Improvements

* Deep learning models (LSTM, Autoencoders)
* Cloud deployment (AWS / GCP)
* Real-time streaming integration
* Authentication and multi-user dashboard

---

## Author

Avantika Bhattacharya
