import joblib
import pandas as pd

# Load model

model = joblib.load("model/loan_model.pkl")
columns = joblib.load("model/model_columns.pkl")

def predict_loan(data_dict):

    df = pd.DataFrame([data_dict])

    # match training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)

    return prediction[0]

sample = {
    "ApplicantIncome": 5000,
    "LoanAmount": 150,
    "Credit_History": 1
}

print(predict_loan(sample))    