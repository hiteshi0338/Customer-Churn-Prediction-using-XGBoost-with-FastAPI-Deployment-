from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback

# Load model and features
model = joblib.load("xgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = FastAPI()

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
def predict(data: CustomerData):
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

        return {
            "churn_probability": round(float(proba), 4),
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "risk_level": risk
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# Run with: uvicorn app:app --reload


# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import joblib

# import traceback


    
# # Load model and features
# model = joblib.load("xgb_model.pkl")
# feature_columns = joblib.load("feature_columns.pkl")

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # Define input schema (RAW user input)

# class CustomerData(BaseModel):
#     gender: str
#     SeniorCitizen: int
#     Partner: str
#     Dependents: str
#     tenure: int
#     PhoneService: str
#     PaperlessBilling: str
#     MonthlyCharges: float
#     TotalCharges: float
#     Contract: str
#     PaymentMethod: str
#     InternetService: str

# # Preprocessing function (must match training logic)
# def preprocess(data: dict):
#     df = pd.DataFrame([data])

#     # Binary mapping
#     df.replace({
#         'Yes': 1, 'No': 0,
#         'Male': 1, 'Female': 0
#     }, inplace=True)

#     # One-hot encoding
#     df = pd.get_dummies(df)

#     # Align columns with training
#     df = df.reindex(columns=feature_columns, fill_value=0)

#     return df


# @app.post("/predict")
# def predict(data: CustomerData):
#     try:
#         input_dict = data.dict()
#         processed = preprocess(input_dict)

#         proba = model.predict_proba(processed)[0][1]
#         prediction = int(proba >= 0.3)

#         risk = "Low" if proba < 0.3 else "Medium" if proba < 0.6 else "High"

#         return {
#             "churn_probability": round(float(proba), 4),
#             "prediction": "Churn" if prediction == 1 else "No Churn",
#             "risk_level": risk
#         }

#     except Exception as e:
#         return {"error": str(e), "trace": traceback.format_exc()}
    
# # Prediction endpoint
# @app.post("/predict")
# def predict(data: CustomerData):
#     input_dict = data.dict()

#     processed = preprocess(input_dict)

#     proba = model.predict_proba(processed)[0][1]
#     prediction = int(proba >= 0.3)   # your threshold

#     # Risk categorization
#     if proba < 0.3:
#         risk = "Low"
#     elif proba < 0.6:
#         risk = "Medium"
#     else:
#         risk = "High"


#     return {
#     "churn_probability": round(float(proba), 4),
#     "prediction": "Churn" if prediction == 1 else "No Churn",
#     "risk_level": risk
# }
    


#     # uvicorn app:app --reload