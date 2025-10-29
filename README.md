## Stock Price Prediction using Linear Regression

This project uses real stock market data from Yahoo Finance to predict the next day’s closing price of a company’s stock.  
It applies machine learning with Linear Regression and visualizes the predicted vs. actual prices.

## Overview

- Dataset: Downloaded dynamically from Yahoo Finance (`yfinance` library)
- Model: Linear Regression (`scikit-learn`)
- Goal: Predict the next day's stock closing price based on Open, High, Low, and Volume
- Evaluation: R² and RMSE metrics
- Visualization: Matplotlib chart comparing actual vs. predicted prices


## How to Run

1. Clone this repo or open it in PyCharm:
   ```bash
   git clone https://github.com/Jacqjuarez/StockPricePrediction.git
   cd StockPricePrediction
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the model

bash
Copy code
python scripts/stock_prediction.py
Results

Data: data/stock_data.csv
Graph: results/price_comparison.png
