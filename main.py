import numpy as np
import pandas as pd
import yfinance as yf
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("📥 Downloading gold price data...")
time.sleep(2)

data = yf.download("GC=F", start="2010-01-01", progress=False)
data = data[['Close']]
data = data.dropna()

print("⚙️ Processing data...")
time.sleep(2)

split = int(len(data) * 0.8)
train_data = data[:split]

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_data)

scaled_data = scaler.transform(data)

x = []
y = []

input_days = 60
output_days = 7

for i in range (len(scaled_data) - input_days - output_days) :
    x.append(scaled_data[i : i + input_days])
    y.append(scaled_data[i + input_days : i + input_days + output_days])

x = np.array(x)
y = np.array(y)

split = int(len(x) * 0.8)

x_train = x[:split]
y_train = y[:split]

x_test = x[split:]
y_test = y[split:]

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

print("🧠 Building model...")
time.sleep(2)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(input_days,1)),
    Dropout(0.1),
    LSTM(32),
    Dense(output_days)
])

model.compile(optimizer='adam', loss='mse')

print("🚀 Training model (please wait)...")
time.sleep(2)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)
print("✅ Model training completed!")

print("📊 Predicting future prices...")
time.sleep(2)

predictions = model.predict(x_test)

predictions_reshaped = predictions.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

predictions_inv = scaler.inverse_transform(predictions_reshaped)
y_test_inv = scaler.inverse_transform(y_test_reshaped)

predictions_inv = predictions_inv.reshape(predictions.shape)
y_test_inv = y_test_inv.reshape(y_test.shape)

mae = mean_absolute_error(y_test_inv.flatten(), predictions_inv.flatten())
print(f"👉 This means the model is off by approximately ${mae:.2f} on average.")

print("\n📈 Gold Price Prediction System")
print("--------------------------------")

last_60 = scaled_data[-60:].reshape(1, 60, 1)

future = model.predict(last_60, verbose=0)
future = future.reshape(-1,1)
future = scaler.inverse_transform(future).flatten()

while True:
    user_input = input("\n👉 Enter number of days ahead (1-7) or type 'stop': ")

    if user_input.lower() == "stop":
        print("\n👋 Exiting prediction system...")
        break

    try:
        day = int(user_input)

        if 1 <= day <= 7:
            print(f"\n💰 Predicted gold price after {day} days: ${future[day-1]:.2f} per troy ounce")
            print(f"👉 Model average error: ${mae:.2f}")
        else:
            print("❌ Enter value between 1-7")

    except:
        print("❌ Invalid input! Please enter a number or 'stop'")