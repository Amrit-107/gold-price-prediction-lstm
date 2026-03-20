# gold-price-prediction-lstm
Gold price prediction using LSTM neural network (time-series forecasting)

# 🪙 Gold Price Prediction using LSTM

## 📌 Project Overview

This project predicts gold prices using a Long Short-Term Memory (LSTM) neural network.
The model uses the past 60 days of gold prices to forecast the next 7 days.

---

## 📊 Dataset

* Source: Yahoo Finance
* Ticker: GC=F (Gold Futures)
* Feature Used: Closing Price
* Unit: USD per troy ounce

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* yfinance
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## 🧠 Model Details

* Input: Past 60 days
* Output: Next 7 days
* Architecture:

  * LSTM (64 units)
  * Dropout
  * LSTM (32 units)
  * Dense (7 outputs)

---

## 🚀 How It Works

1. Collect gold price data
2. Preprocess data (scaling, sequence creation)
3. Train LSTM model
4. Predict future prices
5. Evaluate using MAE

---

## 📉 Performance

* Metric Used: Mean Absolute Error (MAE)
* The model captures overall trends but struggles with sudden spikes due to limited features.
* Prediction accuracy decreases for longer forecast horizons.

---

## ⚠️ Limitations

* Uses only closing price
* No external factors included
* Multi-step prediction increases error

---

## 🔮 Future Improvements

* Add more features (volume, high, low)
* Try advanced models (GRU, Transformer)
* Hyperparameter tuning

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📁 Project Structure

main.py          # Final model code
model.ipynb      # Development notebook
report.pdf       # Project report

---

## 👤 Author

Amrit Behera
B.Tech in Computer Science and Engineering (AI & ML)

---

## ⭐ Note

This is a personal machine learning project built for learning and practice.
