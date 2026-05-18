from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("model/loan_model.pkl")
columns = joblib.load("model/model_columns.pkl")


# Input schema
class LoanData(BaseModel):
    ApplicantIncome: float
    LoanAmount: float
    Credit_History: float


@app.get("/")
def home():
    return {"message": "Loan Approval API Running"}


@app.post("/predict")
def predict(data: LoanData):

    df = pd.DataFrame([data.dict()])
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return {"Loan Approved": bool(prediction)}