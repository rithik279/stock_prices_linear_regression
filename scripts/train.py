#!/usr/bin/env python3
"""
Train all configured models and save results.
Run with: python scripts/train.py or make train
"""

import os
import yaml
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

os.makedirs("output", exist_ok=True)

# Load configs
def load_config(name):
    with open(f"configs/{name}.yaml") as f:
        return yaml.safe_load(f)

tickers_cfg = load_config("tickers")
dates_cfg = load_config("dates")
models_cfg = load_config("models")

# Get all tickers
all_tickers = (
    [tickers_cfg["tickers"]["primary"]]
    + tickers_cfg["tickers"]["market_indices"]
    + tickers_cfg["tickers"]["additional"]
)

# Download and prepare data
print("Downloading training data...")
df = yf.download(all_tickers, start=dates_cfg["training"]["start"],
                 end=dates_cfg["training"]["end"])["Close"]

# Feature engineering
for ticker in all_tickers:
    df[f"{ticker}(t-1)"] = df[ticker].shift(1)
    df[f"{ticker}_MA_5"] = df[ticker].rolling(window=5).mean()

df["Target"] = df["AAPL"].shift(-1)
df = df.dropna()

print(f"Training data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")

# Train each model
results = {}
models = models_cfg["models"]

# OLS Model
if models["ols"]["enabled"]:
    print("\nTraining OLS model...")
    features = ["AAPL(t-1)", "^GSPC(t-1)"]
    X = df[features]
    y = df["Target"]
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()

    predictions = ols_model.predict(X_const)
    results["OLS"] = {
        "model": ols_model,
        "predictions": predictions,
        "r2": r2_score(y, predictions),
        "rmse": np.sqrt(mean_squared_error(y, predictions)),
        "features": features
    }
    print(f"  OLS R²: {results['OLS']['r2']:.4f}")

# Regularized models (use all features)
all_features = [f for f in all_tickers] + [f"{f}_MA_5" for f in all_tickers]
X_all = df[all_features]
y = df["Target"]

# Time series split
test_size = models_cfg["test_size"]
split_idx = int(len(df) * (1 - test_size))
X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Ridge
if models["ridge"]["enabled"]:
    print("\nTraining Ridge model...")
    ridge = Ridge(alpha=models["ridge"]["alpha"])
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    results["Ridge"] = {
        "model": ridge,
        "predictions": ridge_pred,
        "r2": r2_score(y_test, ridge_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, ridge_pred)),
        "features": all_features,
        "coefficients": dict(zip(all_features, ridge.coef_))
    }
    print(f"  Ridge R²: {results['Ridge']['r2']:.4f}")

# Lasso
if models["lasso"]["enabled"]:
    print("\nTraining Lasso model...")
    lasso = Lasso(alpha=models["lasso"]["alpha"])
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    results["Lasso"] = {
        "model": lasso,
        "predictions": lasso_pred,
        "r2": r2_score(y_test, lasso_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, lasso_pred)),
        "features": all_features,
        "coefficients": dict(zip(all_features, lasso.coef_))
    }
    print(f"  Lasso R²: {results['Lasso']['r2']:.4f}")

# Elastic Net
if models["elastic_net"]["enabled"]:
    print("\nTraining Elastic Net model...")
    en = ElasticNet(alpha=models["elastic_net"]["alpha"],
                   l1_ratio=models["elastic_net"]["l1_ratio"])
    en.fit(X_train, y_train)
    en_pred = en.predict(X_test)
    results["ElasticNet"] = {
        "model": en,
        "predictions": en_pred,
        "r2": r2_score(y_test, en_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, en_pred)),
        "features": all_features,
        "coefficients": dict(zip(all_features, en.coef_))
    }
    print(f"  ElasticNet R²: {results['ElasticNet']['r2']:.4f}")

# Save results summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<15} {'R²':>10} {'RMSE':>12}")
print("-"*60)
for name, res in results.items():
    print(f"{name:<15} {res['r2']:>10.4f} {res['rmse']:>12.4f}")

# Save to CSV
summary_df = pd.DataFrame({
    "Model": list(results.keys()),
    "R2": [r["r2"] for r in results.values()],
    "RMSE": [r["rmse"] for r in results.values()]
})
summary_df.to_csv("output/model_comparison.csv", index=False)
print(f"\nResults saved to output/model_comparison.csv")

# Save OLS model summary
if "OLS" in results:
    with open("output/ols_summary.txt", "w") as f:
        f.write(str(results["OLS"]["model"].summary()))
    print("OLS summary saved to output/ols_summary.txt")
