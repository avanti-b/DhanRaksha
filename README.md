# 🛡️ FraudShield AI — Complete Setup Guide

## Project Structure

```
fraud_detection_project/
│
├── backend/
│   ├── app.py               ← Flask API server (run this last)
│   ├── train_model.py       ← ML training script (run this first)
│   ├── requirements.txt     ← Python libraries to install
│   └── model/               ← Auto-created after training
│       ├── fraud_model.pkl  ← Saved Random Forest model
│       ├── scaler.pkl       ← Saved data scaler
│       └── metrics.json     ← Model evaluation results
│
├── frontend/
│   └── index.html           ← Dashboard (open in browser)
│
└── FraudShield_AI_Training.ipynb  ← Google Colab version
```

---

## Step 1 — Get the Dataset

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in / create a free Kaggle account
3. Click **Download** → you get `creditcard.csv` (144 MB)
4. Place `creditcard.csv` inside the `backend/` folder

---

## Step 2 — Train the Model

### Option A: Google Colab (RECOMMENDED — faster, no install needed)

1. Go to https://colab.research.google.com
2. Click **File → Upload notebook** → upload `FraudShield_AI_Training.ipynb`
3. Click **Runtime → Run all**
4. When prompted, upload your `creditcard.csv`
5. At the end, it auto-downloads 3 files: `fraud_model.pkl`, `scaler.pkl`, `metrics.json`
6. Put those 3 files in your `backend/model/` folder (create the folder if needed)

### Option B: Your own computer

Open terminal/command prompt in the `backend/` folder:

```bash
# Install Python libraries
pip install -r requirements.txt

# Run training (takes 5-10 minutes)
python train_model.py
```

This saves the model to `backend/model/` automatically.

---

## Step 3 — Start the Flask Backend

In your terminal, inside the `backend/` folder:

```bash
python app.py
```

You should see:
```
🚀 Starting FraudShield AI Backend...
📡 API running at: http://localhost:5000
```

Test it by opening http://localhost:5000/api/metrics in your browser.
You should see JSON with model metrics.

**Keep this terminal window open while using the dashboard.**

---

## Step 4 — Open the Dashboard

Simply open `frontend/index.html` in your browser (double-click it).

The dashboard will:
- Connect to the Flask backend automatically
- Show a green "Backend connected" pill if working
- Show real metrics from YOUR trained model
- Let you predict fraud in the Predict tab

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/metrics` | GET | Returns model metrics from training |
| `POST /api/predict` | POST | Predicts if a transaction is fraud |
| `GET /api/health` | GET | Check if backend is running |

**Example predict request:**
```json
POST http://localhost:5000/api/predict
{
  "amount": 5000,
  "hour": 2,
  "v1": -3.5,
  "v2": 1.2,
  "v14": -2.1
}
```

**Response:**
```json
{
  "prediction": 1,
  "fraud_probability": 89.4,
  "label": "Fraud",
  "risk_level": "High"
}
```

---

## Common Errors

**"pip is not recognized"**
→ Install Python from https://python.org (check "Add to PATH")

**"Backend offline" in dashboard**
→ Make sure `python app.py` is running in a terminal

**"model/fraud_model.pkl not found"**
→ Run `train_model.py` first (or download from Colab)

**CORS error in browser console**
→ Already handled by flask-cors. If still happening, try opening index.html via a local server

**Training takes too long on your PC**
→ Use Google Colab instead (Option A above)

---

## Team
Khushi Agarwal · Avantika Bhattacharya · Shreyas Birje · Meenakshi S Nair · Saketh Kallakuri · Shreyansh Bhaik

Department of Data Science and Engineering — Section B
