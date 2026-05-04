from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback
import shap
import numpy as np
import io
import json
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal, PredictionLog, create_tables

# Load model and features
model = joblib.load("xgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
explainer = shap.TreeExplainer(model)
app = FastAPI()

create_tables()

# ── DB session dependency ─────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "endpoints": {
            "/predict": "Single prediction",
            "/predict_batch": "Batch prediction"
        }
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    PaperlessBilling: str
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str

# Preprocessing function (must match training logic)
def preprocess(data: dict):
    df = pd.DataFrame([data])

    # Binary mapping
    df.replace({
        'Yes': 1, 'No': 0,
        'Male': 1, 'Female': 0
    }, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns with training
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


@app.post("/predict")
def predict(data: CustomerData, db: Session = Depends(get_db)):
    try:
        input_dict = data.dict()
        processed = preprocess(input_dict)

        proba = model.predict_proba(processed)[0][1]
        prediction = int(proba >= 0.3)

        if proba < 0.3:
            risk = "Low"
        elif proba < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        # SHAP
        shap_values = explainer.shap_values(processed)
        shap_row = shap_values[0]
        shap_dict = dict(zip(feature_columns, shap_row))
        top_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        shap_output = [
            {
                "feature": feat,
                "value": round(float(val), 4),
                "impact": "increases churn risk" if val > 0 else "decreases churn risk"
            }
            for feat, val in top_features
        ]

        # ── Log to database ───────────────────────────────
        top_factors_str = " | ".join(
            [f"{f}({'+' if v > 0 else ''}{round(v, 3)})"
             for f, v in top_features]
        )
        log = PredictionLog(
            gender            = data.gender,
            senior_citizen    = data.SeniorCitizen,
            partner           = data.Partner,
            dependents        = data.Dependents,
            tenure            = data.tenure,
            phone_service     = data.PhoneService,
            paperless_billing = data.PaperlessBilling,
            monthly_charges   = data.MonthlyCharges,
            total_charges     = data.TotalCharges,
            contract          = data.Contract,
            payment_method    = data.PaymentMethod,
            internet_service  = data.InternetService,
            churn_probability = round(float(proba), 4),
            prediction        = "Churn" if prediction == 1 else "No Churn",
            risk_level        = risk,
            top_factors       = top_factors_str
        )
        db.add(log)
        db.commit()
        # ─────────────────────────────────────────────────

        return {
            "churn_probability": round(float(proba), 4),
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "risk_level": risk,
            "top_factors": shap_output
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
    


@app.get("/dashboard/data")
def dashboard_data(db: Session = Depends(get_db)):
    try:
        logs = db.query(PredictionLog).order_by(
            PredictionLog.timestamp.desc()
        ).all()

        if not logs:
            return {
                "total": 0,
                "risk_distribution": {},
                "prediction_distribution": {},
                "top_features": {},
                "recent_logs": [],
                "daily_trend": {}
            }

        # ── Risk distribution ─────────────────────────────
        risk_dist = {"Low": 0, "Medium": 0, "High": 0}
        for log in logs:
            risk_dist[log.risk_level] = risk_dist.get(log.risk_level, 0) + 1

        # ── Prediction distribution ───────────────────────
        pred_dist = {"Churn": 0, "No Churn": 0}
        for log in logs:
            pred_dist[log.prediction] = pred_dist.get(log.prediction, 0) + 1

        # ── Top driving features across all predictions ───
        feature_counts = {}
        for log in logs:
            if log.top_factors:
                for part in log.top_factors.split(" | "):
                    # Extract feature name before the parenthesis
                    feat = part.split("(")[0].strip()
                    # Only count features that increase churn risk
                    if "(+" in part:
                        feature_counts[feat] = \
                            feature_counts.get(feat, 0) + 1

        top_features = dict(
            sorted(feature_counts.items(),
                   key=lambda x: x[1],
                   reverse=True)[:8]
        )

        # ── Daily prediction trend (last 14 days) ─────────
        daily = {}
        for log in logs:
            day = log.timestamp.strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"Churn": 0, "No Churn": 0}
            daily[day][log.prediction] += 1

        # Sort by date
        daily_trend = dict(sorted(daily.items())[-14:])

        # ── Recent 10 logs ────────────────────────────────
        recent = [
            {
                "id":           log.id,
                "timestamp":    log.timestamp.strftime("%Y-%m-%d %H:%M"),
                "contract":     log.contract,
                "internet":     log.internet_service,
                "tenure":       log.tenure,
                "monthly":      log.monthly_charges,
                "probability":  log.churn_probability,
                "prediction":   log.prediction,
                "risk":         log.risk_level,
                "top_factors":  log.top_factors
            }
            for log in logs[:10]
        ]

        return {
            "total":                    len(logs),
            "risk_distribution":        risk_dist,
            "prediction_distribution":  pred_dist,
            "top_features":             top_features,
            "recent_logs":              recent,
            "daily_trend":              daily_trend
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
    
# @app.post("/predict")
# def predict(data: CustomerData):
#     try:
#         input_dict = data.dict()
#         processed = preprocess(input_dict)

#         proba = model.predict_proba(processed)[0][1]
#         prediction = int(proba >= 0.3)

#         if proba < 0.3:
#             risk = "Low"
#         elif proba < 0.6:
#             risk = "Medium"
#         else:
#             risk = "High"

#         # ── SHAP explanation ──────────────────────────────
#         shap_values = explainer.shap_values(processed)

#         # shap_values is shape (1, n_features) — get row 0
#         shap_row = shap_values[0]

#         # Pair each feature with its SHAP value
#         shap_dict = dict(zip(feature_columns, shap_row))

#         # Sort by absolute impact, take top 5
#         top_features = sorted(
#             shap_dict.items(),
#             key=lambda x: abs(x[1]),
#             reverse=True
#         )[:5]

#         # Format for frontend
#         shap_output = [
#             {
#                 "feature": feat,
#                 "value": round(float(val), 4),
#                 "impact": "increases churn risk" if val > 0 else "decreases churn risk"
#             }
#             for feat, val in top_features
#         ]
#         # ─────────────────────────────────────────────────

#         return {
#             "churn_probability": round(float(proba), 4),
#             "prediction": "Churn" if prediction == 1 else "No Churn",
#             "risk_level": risk,
#             "top_factors": shap_output        # ← new field
#         }

#     except Exception as e:
#         return {"error": str(e), "trace": traceback.format_exc()}
    
# @app.post("/predict")
# def predict(data: CustomerData):
#     try:
#         input_dict = data.dict()
#         processed = preprocess(input_dict)

#         proba = model.predict_proba(processed)[0][1]
#         prediction = int(proba >= 0.3)

#         if proba < 0.3:
#             risk = "Low"
#         elif proba < 0.6:
#             risk = "Medium"
#         else:
#             risk = "High"

#         return {
#             "churn_probability": round(float(proba), 4),
#             "prediction": "Churn" if prediction == 1 else "No Churn",
#             "risk_level": risk
#         }

#     except Exception as e:
#         return {"error": str(e), "trace": traceback.format_exc()}
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        # ── Read uploaded CSV ─────────────────────────────
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))

        # ── Validate required columns ─────────────────────
        required_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "PaperlessBilling",
            "MonthlyCharges", "TotalCharges", "Contract",
            "PaymentMethod", "InternetService"
        ]
        missing = [c for c in required_cols if c not in df_input.columns]
        if missing:
            return {"error": f"Missing columns: {missing}"}

        # ── Keep a copy of original input for output ──────
        df_output = df_input.copy()

        # ── Preprocess & predict row by row ───────────────
        probabilities = []
        predictions = []
        risk_levels = []
        top_factors_list = []

        for _, row in df_input.iterrows():
            row_dict = row[required_cols].to_dict()

            # Type casting (CSV reads everything as str/float)
            row_dict["SeniorCitizen"] = int(row_dict["SeniorCitizen"])
            row_dict["tenure"] = int(row_dict["tenure"])
            row_dict["MonthlyCharges"] = float(row_dict["MonthlyCharges"])
            row_dict["TotalCharges"] = float(row_dict["TotalCharges"])

            processed = preprocess(row_dict)
            proba = model.predict_proba(processed)[0][1]
            pred = int(proba >= 0.3)

            if proba < 0.3:
                risk = "Low"
            elif proba < 0.6:
                risk = "Medium"
            else:
                risk = "High"

            # SHAP for this row
            shap_values = explainer.shap_values(processed)
            shap_row = shap_values[0]
            shap_dict = dict(zip(feature_columns, shap_row))
            top_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            top_factors = " | ".join(
                [f"{f}({'+' if v > 0 else ''}{v:.3f})" for f, v in top_features]
            )

            probabilities.append(round(float(proba), 4))
            predictions.append("Churn" if pred == 1 else "No Churn")
            risk_levels.append(risk)
            top_factors_list.append(top_factors)

        # ── Append results to output DataFrame ────────────
        df_output["Churn_Probability"] = probabilities
        df_output["Prediction"] = predictions
        df_output["Risk_Level"] = risk_levels
        df_output["Top_Factors"] = top_factors_list

        # ── Return as downloadable CSV ────────────────────
        output = io.StringIO()
        df_output.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=churn_predictions.csv"
            }
        )

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
    
# Run with: uvicorn app:app --reload


