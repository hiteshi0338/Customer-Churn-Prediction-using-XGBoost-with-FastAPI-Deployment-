# 🚀 Customer Churn Prediction System

An end-to-end machine learning system that predicts telecom customer churn, with **explainability (SHAP)**, **batch processing**, and a **real-time analytics dashboard**.

---

## 📌 Overview

This project goes beyond a simple ML model — it is a **full-stack ML application** covering:

* Data analysis & preprocessing
* Model training & optimization
* API development
* Explainability
* Batch inference
* Analytics dashboard

---

## 🏗️ Project Structure

```
churn/
├── app.py                 # FastAPI backend (single + batch prediction APIs)
├── index.html             # Frontend (single + batch prediction UI)
├── dashboard.html         # Analytics dashboard (charts + logs)
├── database.py            # SQLite + SQLAlchemy setup
├── churn_predictions.db   # Auto-generated prediction logs
├── xgb_model.pkl          # Trained XGBoost model
├── feature_columns.pkl    # Feature alignment
├── EDA.ipynb              # Data exploration
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Key Features

### 🔮 1. Single Prediction API

* Predicts churn probability for one customer
* Returns:

  * Probability
  * Prediction (Churn / No Churn)
  * Risk Level (Low / Medium / High)

---

### 📂 2. Batch Prediction

* Upload CSV file with multiple customers
* Returns downloadable CSV with:

  * Predictions
  * Risk levels
  * Top SHAP factors

---

### 🧠 3. SHAP Explainability

* Uses SHAP TreeExplainer
* Shows **top 5 features** influencing each prediction
* Visualized as a bar chart:

  * 🔴 Red → increases churn risk
  * 🟢 Green → reduces churn risk

---

### 📊 4. Analytics Dashboard

* Tracks all predictions in SQLite database
* Displays:

  * Risk distribution
  * Churn vs No-churn
  * Top influencing features
  * 14-day trend
  * Recent predictions log

---

### 💾 5. Prediction Logging

* Every API call is stored automatically
* Enables monitoring + future retraining

---

## 🧠 Model Details

* **Algorithm:** XGBoost (Optuna tuned)
* **Decision Threshold:** 0.3
* **Recall (Churn):** 0.917
* **Precision (Churn):** 0.442
* **F1 Score:** 0.584

### Why recall-focused?

Missing a churner is more costly than a false alert → threshold lowered intentionally.

---

## 🛠️ Tech Stack

| Layer          | Technology            |
| -------------- | --------------------- |
| ML Model       | XGBoost               |
| Backend        | FastAPI               |
| Frontend       | HTML, CSS, JavaScript |
| Explainability | SHAP                  |
| Database       | SQLite + SQLAlchemy   |
| Tracking       | MLflow                |
| Data           | pandas, scikit-learn  |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/hiteshi0338/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Setup environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Run backend

```bash
uvicorn app:app --reload
```

---

### 4. Run frontend

```bash
python -m http.server 5500
```

Open:

```
http://localhost:5500/index.html
```

---

## 🔌 API Endpoints

### `POST /predict`

Single prediction

### `POST /predict_batch`

Batch prediction via CSV upload

---

## 📈 Future Improvements

* [ ] Docker + cloud deployment (Render / Railway)
* [ ] React frontend for better UX
* [ ] Add remaining telecom features
* [ ] CI/CD with GitHub Actions
* [ ] Automated retraining pipeline

---

## ⭐ Project Highlights

* End-to-end ML pipeline
* Explainable AI (SHAP integration)
* Batch + real-time inference
* Data logging + analytics dashboard

---

## ⚠️ Notes

* `.db` file is auto-generated at runtime
* Ensure backend is running before using frontend
* Do not open HTML directly — serve via HTTP

---

## 📬 Author

**Hiteshi Jain**
B.Tech CSE | Machine Learning Enthusiast
