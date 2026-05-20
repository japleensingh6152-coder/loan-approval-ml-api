from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI(
    title="Loan Approval ML API",
    description="Machine Learning API for Loan Prediction",
    version="1.0"
)

# -----------------------------
# Load Model Safely
# -----------------------------
MODEL_PATH = os.path.join("model", "loan_model.pkl")
COLUMNS_PATH = os.path.join("model", "model_columns.pkl")

model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)

# -----------------------------
# Input Schema
# -----------------------------
class LoanData(BaseModel):
    ApplicantIncome: float
    LoanAmount: float
    Credit_History: float


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Loan Approval API Running Successfully 🚀"}


# -----------------------------
# Health Check (VERY IMPORTANT for Render)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: LoanData):

    # Convert input to dataframe
    df = pd.DataFrame([data.dict()])

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return {
        "Loan Approved": bool(prediction)
    }