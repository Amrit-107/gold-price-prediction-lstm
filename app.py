from fastapi import FastAPI
from tensorflow import keras
import joblib
import yfinance as yf
import numpy as np

app = FastAPI()

model = keras.models.load_model("gold_lstm_model.keras")
scaler = joblib.load("gold_scaler.pkl")

@app.get("/")
def home():
    return {"message": "Gold Price Prediction API"}

@app.get("/predict/{day}")
def predict(day: int):

    if day < 1 or day > 7:
        return {"error": "Day must be between 1 and 7"}

    data = yf.download("GC=F", period="90d", progress=False)

    close_prices = data[['Close']].dropna()

    scaled = scaler.transform(close_prices)

    last_60 = scaled[-60:].reshape(1, 60, 1)

    future = model.predict(last_60, verbose=0)

    future = future.reshape(-1, 1)
    future = scaler.inverse_transform(future).flatten()

    return {
        "day": day,
        "predicted_gold_price": round(float(future[day - 1]), 2)
    }