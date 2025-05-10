from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load the model pipeline
try:
    model = joblib.load("fraud_model.joblib")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Define the input schema
class Transaction(BaseModel):
    cc_num: int
    merchant: str
    category: str
    amt: float
    city_pop: int
    state: str

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        input_df = pd.DataFrame([transaction.dict()])
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        return {
            "fraud_prediction": int(prediction[0]),
            "confidence_score": float(proba[0][1])  # Probability of fraud (class 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")