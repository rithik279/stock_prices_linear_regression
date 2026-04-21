#!/usr/bin/env python3
"""
Quickstart: Train OLS model and generate prediction plot in under 60 seconds.
Run with: python quickstart.py
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

TICKERS = ["AAPL", "^GSPC"]
TRAIN_START = "2024-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-01-31"


def main():
    print("Quickstart: AAPL Stock Price Prediction\n")

    # Download data
    print("[1/4] Downloading data...")
    df_train = yf.download(TICKERS, start=TRAIN_START, end=TRAIN_END)["Close"]
    df_test = yf.download(TICKERS, start=TEST_START, end=TEST_END)["Close"]

    # Feature engineering
    print("[2/4] Engineering features...")
    for ticker in TICKERS:
        df_train[f"{ticker}(t-1)"] = df_train[ticker].shift(1)
        df_test[f"{ticker}(t-1)"] = df_test[ticker].shift(1)

    df_train["Target"] = df_train["AAPL"].shift(-1)
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Train model
    print("[3/4] Training OLS model...")
    features = ["AAPL(t-1)", "^GSPC(t-1)"]
    X_train = sm.add_constant(df_train[features])
    y_train = df_train["Target"]
    model = sm.OLS(y_train, X_train).fit()

    # Predict on test
    X_test = sm.add_constant(df_test[features])
    predictions = model.predict(X_test)
    actuals = df_test["AAPL"]

    # Metrics
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # Plot
    print("[4/4] Generating plot...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actuals.index, actuals, label="Actual", color="black", linewidth=1.5)
    ax.plot(actuals.index, predictions, label="Predicted", color="red", linewidth=1.5, linestyle="--")
    ax.set_title(f"AAPL Stock Price Prediction (Jan 2025)\nR² = {r2:.4f}, RMSE = ${rmse:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/quickstart_prediction.png", dpi=150)
    plt.close()

    print(f"\n{'='*50}")
    print(f"Model Summary:\n{model.summary().tables[1]}")
    print(f"\nMetrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"\nPlot saved to: output/quickstart_prediction.png")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
