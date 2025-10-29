import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf

# === 1. Download Stock Data ===
ticker = "AAPL"  # Change to "TSLA", "MSFT", etc.
data = yf.download(ticker, start="2022-01-01", end="2025-01-01")

# Ensure folders exist
os.makedirs("../data", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# Save the raw data
data.to_csv("../data/stock_data.csv")

# === 2. Prepare Features ===
data["Target"] = data["Close"].shift(-1)  # next day's closing price
data.dropna(inplace=True)

X = data[["Open", "High", "Low", "Volume"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Train Model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === 4. Evaluate ===
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"R²: {r2:.3f}, RMSE: {rmse:.2f}")

# === 5. Visualize Predictions ===
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual", linewidth=2)
plt.plot(preds[:100], label="Predicted", linestyle="--")
plt.title(f"{ticker} Stock Price Prediction (Linear Regression)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()

# Save chart to ../results (one folder up from /scripts)
plt.savefig("../results/price_comparison.png")
plt.show()
