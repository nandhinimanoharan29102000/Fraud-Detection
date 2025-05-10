from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="Transaction Service")

class Transaction(BaseModel):
    cc_num: int
    merchant: str
    category: str
    amt: float
    city_pop: int
    state: str

@app.post("/process/")
async def process_transaction(transaction: Transaction):
    try:
        # Step 1: Send transaction to fraud detection service
        fraud_response = httpx.post("http://localhost:8050/predict", json=transaction.dict())
        if fraud_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Fraud service error")

        fraud_result = fraud_response.json()
        is_fraud = fraud_result["fraud_prediction"]
        confidence_score = fraud_result["confidence_score"]

        # Step 2: Prepare payload for DB insert
        db_payload = transaction.dict()
        db_payload["is_fraud"] = is_fraud
        db_payload["confidence_score"] = confidence_score

        # Step 3: Send to DB service
        db_response = httpx.post("http://localhost:8052/transactions/", json=db_payload)
        if db_response.status_code != 200:
            raise HTTPException(status_code=500, detail="DB insert failed")

        return {
            "message": "Transaction processed and stored",
            "is_fraud": is_fraud,
            "confidence_score": confidence_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))