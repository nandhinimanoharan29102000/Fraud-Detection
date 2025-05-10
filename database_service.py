from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector

app = FastAPI()

class Transaction(BaseModel):
    cc_num: int
    merchant: str
    category: str
    amt: float
    city_pop: int
    state: str
    is_fraud: int
    confidence_score: float

@app.post("/transactions/")
def insert_transaction(transaction: Transaction):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root@39",
            database="fraud_detection_db"
        )
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO transactions 
            (cc_num, merchant, category, amt, city_pop, state, is_fraud, confidence_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (
            transaction.cc_num,
            transaction.merchant,
            transaction.category,
            transaction.amt,
            transaction.city_pop,
            transaction.state,
            transaction.is_fraud,
            transaction.confidence_score
        )
        cursor.execute(insert_query, data)
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "Transaction saved in DB"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {e}")
