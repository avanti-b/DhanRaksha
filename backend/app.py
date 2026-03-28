"""
AI Fraud Detection - Flask Backend API
=====================================
Run this file to start the backend server.
Command: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app)

MODEL_PATH   = "model/fraud_model.pkl"
SCALER_PATH  = "model/scaler.pkl"
METRICS_PATH = "model/metrics.json"
DATASET_PATH = "creditcard.csv"      # used only by /api/sample_frauds

model   = None
scaler  = None
metrics = None

def load_artifacts():
    global model, scaler, metrics
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded. Expects {model.n_features_in_} features.")
    else:
        print("❌ model/fraud_model.pkl not found. Run train_model.py first.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded. Expects {scaler.n_features_in_} features.")
    else:
        print("❌ model/scaler.pkl not found. Run train_model.py first.")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        print("✅ Metrics loaded.")

load_artifacts()


# ─────────────────────────────────────────────
# HELPER — build a properly scaled feature array
# ─────────────────────────────────────────────
def build_feature_array(data: dict) -> np.ndarray:
    """
    Reconstruct the exact feature vector the model was trained on.

    Training column order (creditcard.csv after preprocessing):
        [Time, V1, V2, ..., V28, Amount]  →  30 columns total

    The scaler in train_model.py does:
        X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    So the scaler was fit on ONLY 2 columns (Time, Amount).
    V1-V28 are PCA features — already normalised, must NOT be re-scaled.
    """
    amount      = float(data.get("amount", 0.0))
    hour        = float(data.get("hour",   0.0))
    time_approx = hour * 3600.0   # approximate Time as seconds since midnight

    # V1–V28 in order — raw PCA values, no scaling needed
    v_features = [float(data.get(f"v{i}", 0.0)) for i in range(1, 29)]

    # Scale only Time and Amount (scaler fit on 2 columns)
    time_amount_scaled = scaler.transform([[time_approx, amount]])
    time_scaled        = float(time_amount_scaled[0][0])
    amount_scaled      = float(time_amount_scaled[0][1])

    # Final vector: [Time_scaled, V1..V28_raw, Amount_scaled]
    feature_vector = [time_scaled] + v_features + [amount_scaled]
    return np.array(feature_vector, dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "FraudShield AI backend is running ✅"})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict fraud probability for a transaction.
    Send JSON: { amount, hour, v1, v2, ..., v28 }
    Any missing V features default to 0.
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body sent."}), 400

    try:
        feature_array = build_feature_array(data)

        prediction  = int(model.predict(feature_array)[0])
        probability = float(model.predict_proba(feature_array)[0][1])

        # Print to terminal so you can watch live
        print(f"\n[PREDICT] amount={data.get('amount')}  hour={data.get('hour')}")
        print(f"          fraud_prob={probability:.4f}  → {'🚨 FRAUD' if prediction==1 else '✅ LEGIT'}")

        return jsonify({
            "prediction":        prediction,
            "fraud_probability": round(probability * 100, 2),
            "label":             "Fraud" if prediction == 1 else "Legitimate",
            "risk_level": (
                "High"   if probability > 0.7 else
                "Medium" if probability > 0.4 else
                "Low"
            )
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug", methods=["POST"])
def debug():
    """
    Same as /api/predict but also returns the exact 30-feature vector
    passed to the model. Use this to verify the input is correct.
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body."}), 400

    try:
        feature_array = build_feature_array(data)
        prediction    = int(model.predict(feature_array)[0])
        probability   = float(model.predict_proba(feature_array)[0][1])

        return jsonify({
            "prediction":                   prediction,
            "fraud_probability":            round(probability * 100, 2),
            "label":                        "Fraud" if prediction == 1 else "Legitimate",
            "feature_vector_sent_to_model": feature_array.tolist()[0],
            "scaler_n_features":            int(scaler.n_features_in_),
            "model_n_features":             int(model.n_features_in_),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample_frauds", methods=["GET"])
def sample_frauds():
    """
    Returns 5 real confirmed fraud rows from creditcard.csv.
    The frontend loads these to pre-fill the Predict form
    with values the model will definitely flag as fraud.
    Requires creditcard.csv to be present in the backend/ folder.
    """
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "creditcard.csv not found in backend folder."}), 404

    df     = pd.read_csv(DATASET_PATH)
    frauds = df[df["Class"] == 1].head(5)

    samples = []
    for _, row in frauds.iterrows():
        sample = {
            "amount": round(float(row["Amount"]), 2),
            "hour":   int(float(row["Time"]) // 3600 % 24),
        }
        for i in range(1, 29):
            sample[f"v{i}"] = round(float(row[f"V{i}"]), 4)
        samples.append(sample)

    return jsonify({"samples": samples})


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if metrics is None:
        return jsonify({"error": "Metrics not found. Run train_model.py first."}), 503
    return jsonify(metrics)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "model_loaded":      model  is not None,
        "scaler_loaded":     scaler is not None,
        "metrics_loaded":    metrics is not None,
        "scaler_n_features": int(scaler.n_features_in_) if scaler else None,
        "model_n_features":  int(model.n_features_in_)  if model  else None,
        "status":            "ok" if model is not None else "model_missing"
    })


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting FraudShield AI Backend...")
    print("📡 API:            http://localhost:5000")
    print("📊 Metrics:        http://localhost:5000/api/metrics")
    print("🔍 Health check:   http://localhost:5000/api/health")
    print("🧪 Fraud samples:  http://localhost:5000/api/sample_frauds\n")
    app.run(debug=True, host="0.0.0.0", port=5000)