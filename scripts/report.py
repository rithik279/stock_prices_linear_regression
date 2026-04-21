#!/usr/bin/env python3
"""
Generate full diagnostic report with all plots.
Run with: python scripts/report.py or make report
"""

import os
import yaml
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import r2_score, mean_squared_error

os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

# Config
with open("configs/dates.yaml") as f:
    dates_cfg = yaml.safe_load(f)

TICKERS = ["AAPL", "^GSPC"]

def load_data():
    """Load and engineer features."""
    print("Loading data...")
    df = yf.download(TICKERS,
                     start=dates_cfg["training"]["start"],
                     end=dates_cfg["training"]["end"])["Close"]

    for ticker in TICKERS:
        df[f"{ticker}(t-1)"] = df[ticker].shift(1)

    df["Target"] = df["AAPL"].shift(-1)
    return df.dropna()

def train_model(df):
    """Train OLS model."""
    features = ["AAPL(t-1)", "^GSPC(t-1)"]
    X = sm.add_constant(df[features])
    y = df["Target"]
    model = sm.OLS(y, X).fit()
    return model, X, y

def plot_predictions(df, model, X, y):
    """Generate prediction vs actual plot."""
    fig, ax = plt.subplots(figsize=(14, 5))
    predictions = model.predict(X)

    ax.plot(df.index, y, label="Actual", color="black", linewidth=1)
    ax.plot(df.index, predictions, label="Predicted", color="red",
            linewidth=1, linestyle="--", alpha=0.8)
    ax.set_title("AAPL Stock: Actual vs Predicted (Training Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/figures/01_predictions.png", dpi=150)
    plt.close()

def plot_residuals(model):
    """Generate residual diagnostics."""
    residuals = model.resid
    fitted = model.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.5)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted (Homoscedasticity)")

    # Histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Residual Histogram (Normality)")

    # QQ Plot
    sm.qqplot(residuals, line="45", fit=True, ax=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normality)")

    # Scale-Location
    standardized = np.sqrt(np.abs(residuals / np.std(residuals)))
    axes[1, 1].scatter(fitted, standardized, alpha=0.5)
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel("√|Standardized Residuals|")
    axes[1, 1].set_title("Scale-Location")

    plt.tight_layout()
    plt.savefig("output/figures/02_residual_diagnostics.png", dpi=150)
    plt.close()

def plot_vif(X):
    """Calculate and plot VIF."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    vif_data = vif_data[vif_data["Feature"] != "const"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(vif_data["Feature"], vif_data["VIF"], color="steelblue")
    ax.axvline(x=10, color="red", linestyle="--", label="VIF = 10 threshold")
    ax.set_xlabel("VIF")
    ax.set_title("Variance Inflation Factor (Multicollinearity)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/figures/03_vif.png", dpi=150)
    plt.close()

    return vif_data

def plot_forecast():
    """Generate out-of-sample forecast plot."""
    print("Generating forecast on 2025 data...")
    df_train = load_data()
    model, _, _ = train_model(df_train)

    # Load test data
    df_test = yf.download(TICKERS,
                          start=dates_cfg["testing"]["start"],
                          end=dates_cfg["testing"]["end"])["Close"]

    for ticker in TICKERS:
        df_test[f"{ticker}(t-1)"] = df_test[ticker].shift(1)

    df_test = df_test.dropna()
    X_test = sm.add_constant(df_test[["AAPL(t-1)", "^GSPC(t-1)"]])

    predictions = model.predict(X_test)
    actuals = df_test["AAPL"]

    # Metrics
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actuals.index, actuals, label="Actual", color="black", linewidth=1.5)
    ax.plot(actuals.index, predictions, label="Predicted", color="red",
            linewidth=1.5, linestyle="--")
    ax.set_title(f"AAPL Stock: Out-of-Sample Forecast\n"
                 f"R² = {r2:.4f}, RMSE = ${rmse:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/figures/04_forecast.png", dpi=150)
    plt.close()

    return r2, rmse

def main():
    print("="*60)
    print("GENERATING DIAGNOSTIC REPORT")
    print("="*60)

    # Train model
    df = load_data()
    model, X, y = train_model(df)

    # Generate plots
    print("\nGenerating plots...")
    plot_predictions(df, model, X, y)
    plot_residuals(model)
    vif_data = plot_vif(X)
    r2, rmse = plot_forecast()

    # Durbin-Watson
    dw = durbin_watson(model.resid)

    # Save summary report
    report = f"""
================================================================================
STOCK PRICE REGRESSION - DIAGNOSTIC REPORT
================================================================================

MODEL SUMMARY
-------------
{model.summary()}

--------------------------------------------------------------------------------
RESIDUAL DIAGNOSTICS
--------------------------------------------------------------------------------

Durbin-Watson Statistic: {dw:.4f}
  (Values close to 2 indicate no autocorrelation; <1.5 or >2.5 may be concerning)

VIF (Variance Inflation Factor):
{vif_data.to_string(index=False)}

  VIF < 1:   No multicollinearity
  VIF < 10:  Moderate multicollinearity
  VIF >= 10: High multicollinearity

--------------------------------------------------------------------------------
OUT-OF-SAMPLE PERFORMANCE
--------------------------------------------------------------------------------
R² Score:     {r2:.4f}
RMSE:         ${rmse:.2f}

--------------------------------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------------------------------
Plots:
  - output/figures/01_predictions.png    : Actual vs Predicted
  - output/figures/02_residual_diagnostics.png : Residual plots, QQ, histogram
  - output/figures/03_vif.png           : VIF analysis
  - output/figures/04_forecast.png      : Out-of-sample forecast

================================================================================
"""

    with open("output/reports/diagnostic_report.txt", "w") as f:
        f.write(report)

    print(report)
    print("\nAll plots saved to output/figures/")
    print("Full report saved to output/reports/diagnostic_report.txt")

if __name__ == "__main__":
    main()
