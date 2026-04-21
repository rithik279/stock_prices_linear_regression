# Key Takeaways: Stock Price Prediction Using Linear Regression

## Executive Summary

This project applied multiple linear regression techniques to predict AAPL (Apple Inc.) next-day stock closing prices using historical market data. The analysis compared Ordinary Least Squares (OLS), Ridge, Lasso, and Elastic Net regression across training data (2020-2024) and evaluated out-of-sample performance on 2025 data. Key findings reveal that simple OLS models can achieve remarkable in-sample fit (R² = 0.992) but face challenges with autocorrelation in financial time series, while regularized models provide more robust generalization despite slightly lower training metrics.

---

## 1. Data & Methodology

### 1.1 Data Sources & Features

**Primary Data Source:** Yahoo Finance via `yfinance` API

**Tickers Used:**
- `AAPL` - Apple Inc. (primary prediction target)
- `^GSPC` - S&P 500 Index (market benchmark)
- `QQQ` - Nasdaq 100 ETF
- `AMZN` - Amazon.com Inc.
- `MSFT` - Microsoft Corporation

**Feature Engineering:**

Three categories of features were engineered:

1. **Lagged Prices (t-1):** Previous day's closing prices for all tickers. This captures the autoregressive nature of stock prices—today's price is strongly correlated with yesterday's price.

2. **Moving Averages (MA_5):** 5-day rolling average for each ticker. Moving averages smooth out daily volatility and capture short-term trend momentum.

3. **Target Variable:** AAPL next-day closing price, created using `.shift(-1)` to align features with their subsequent day's outcome.

### 1.2 Train-Test Split

| Dataset | Period | Purpose |
|---------|--------|---------|
| Training | 2020-01-01 to 2024-12-31 | Model fitting & diagnostics |
| Testing | 2025-01-01 to 2025-07-29 | Out-of-sample evaluation |

**Critical Note:** For time series data, we used `shuffle=False` in train-test splits to respect temporal ordering. Unlike i.i.d. data, shuffling time series would create data leakage where future information contaminates training data.

---

## 2. Models Compared

### 2.1 Ordinary Least Squares (OLS)

OLS minimizes the sum of squared residuals: Σ(yᵢ - ŷᵢ)²

**Configuration:**
- Features: `AAPL(t-1)`, `^GSPC(t-1)` only
- Rationale: Limiting features reduces multicollinearity concerns

**Key Results:**
| Metric | Training | Testing |
|--------|----------|----------------|
| R² | 0.992 | 0.907 |
| RMSE | $3.71 | $4.90 |

**Coefficients (Training Period):**
```
AAPL(t-1):   0.9841  (p < 0.001) ✓
^GSPC(t-1):  0.0008  (p = 0.035) ✓
const:      -0.7169
```

**Interpretation:**
- For every $1 increase in yesterday's AAPL price, today's predicted price increases by $0.98
- The S&P 500 has a small positive effect ($0.0008 per index point)
- Both coefficients are statistically significant (p < 0.05)

### 2.2 Ridge Regression (L2 Regularization)

Ridge adds λΣβ² to the loss function, penalizing large coefficients to reduce overfitting and multicollinearity.

**Configuration:**
- Alpha (λ): 0.1
- Features: All 10 features (5 lagged + 5 MAs)

**Key Results:**
| Metric | Testing |
|--------|----------------|
| R² | 0.944 |
| RMSE | $2.49 |

### 2.3 Lasso Regression (L1 Regularization)

Lasso adds λΣ|β| to the loss function, which can shrink coefficients exactly to zero for feature selection.

**Configuration:**
- Alpha (λ): 0.1
- Features: All 10 features

**Key Results:**
| Metric | Testing |
|--------|----------------|
| R² | 0.944 |
| RMSE | $2.50 |

### 2.4 Elastic Net (L1 + L2)

Elastic Net combines both penalties: λ₁Σ|β| + λ₂Σβ², balancing feature selection with stability.

**Configuration:**
- Alpha: 1.0
- L1 Ratio: 0.5
- Features: All 10 features

**Key Results:**
| Metric | Testing |
|--------|----------------|
| R² | 0.943 |
| RMSE | $2.52 |

---

## 3. Statistical Diagnostics

### 3.1 Multicollinearity (VIF)

Variance Inflation Factor measures how much the variance of a coefficient is inflated due to correlation with other predictors.

**Results:**
| Feature | VIF |
|---------|-----|
| AAPL(t-1) | 7.64 |
| ^GSPC(t-1) | 7.64 |

**Interpretation:**
- VIF < 5: Low concern
- VIF 5-10: Moderate concern
- VIF > 10: High multicollinearity

Our VIF of 7.64 indicates **moderate multicollinearity**. This is expected in financial data where all prices tend to rise and fall together during market movements. Despite this, the OLS coefficients remain interpretable and statistically significant.

### 3.2 Autocorrelation (Durbin-Watson)

The Durbin-Watson statistic tests for correlation between consecutive residuals.

**Result:** DW = 1.041

**Interpretation:**
- DW ≈ 2: No autocorrelation
- DW < 1.5: Possible positive autocorrelation
- DW > 2.5: Possible negative autocorrelation

Our DW of 1.041 indicates **positive autocorrelation** in residuals. This is extremely common in financial time series and represents a fundamental violation of OLS assumptions. The model is leaving predictable patterns in the residuals that could theoretically be exploited.

**Why does this happen?**
Stock prices exhibit momentum—positive returns tend to follow positive returns, and vice versa. Our simple model captures the level of prices but misses the momentum/mean-reversion dynamics.

### 3.3 Residual Normality (Q-Q Plot)

The Q-Q plot compares residual distribution against a theoretical normal distribution.

**Observation:** The Jarque-Bera test returned a probability of 1.11e-23, strongly rejecting the null hypothesis of normality.

**Implications:**
- OLS coefficients remain unbiased but standard errors may be unreliable
- Confidence intervals and hypothesis tests should be interpreted cautiously
- Financial returns often exhibit fat tails (excess kurtosis = 4.42)

### 3.4 Homoscedasticity

Residual vs. Fitted plot showed constant variance (homoscedastic), satisfying this OLS assumption.

---

## 4. Key Findings & Insights

### 4.1 OLS Achieves Exceptional In-Sample Fit

The OLS model achieved R² = 0.992 during training, explaining 99.2% of variance in next-day prices. This superficially suggests an excellent model, but:

**Why this happens:**
- Stock prices are highly autocorrelated
- Today's price is essentially yesterday's price plus a small random walk component
- The lagged AAPL feature captures most of this persistence

**The catch:**
- High R² in finance doesn't mean the model is "good" for prediction
- It means the model captures the random walk structure, not predictable patterns
- True alpha (excess returns over the market) remains elusive

### 4.2 Regularized Models Show Better Test Performance

Counter-intuitively, regularized models (Ridge, Lasso, ElasticNet) achieved **higher out-of-sample R²** than OLS when tested on 2025 data:

| Model | Test R² |
|-------|---------|
| OLS | 0.907 |
| Ridge | 0.944 |
| Lasso | 0.944 |
| ElasticNet | 0.943 |

This demonstrates the value of regularization for generalization, even when it appears to "underfit" the training data.

### 4.3 Feature Selection Matters

OLS used only 2 features (avoiding multicollinearity), while regularized models used all 10 features. Despite having more features:

- Ridge/Lasso/ElasticNet still achieved strong R²
- The L1 penalty in Lasso can zero out irrelevant features
- The L2 penalty in Ridge handles correlated features gracefully

**Lesson:** Don't over-concern yourself with feature count. Let regularization handle redundancy.

### 4.4 Financial Data Violates OLS Assumptions

This project highlighted that financial time series violate multiple OLS assumptions:

| Assumption | Violated? | Impact |
|------------|-----------|--------|
| No autocorrelation | **Yes** (DW = 1.04) | Inefficiency, predictable residuals |
| Normality of residuals | **Yes** (JB p < 0.001) | Unreliable inference |
| No perfect multicollinearity | **Mildly** (VIF = 7.64) | Inflated standard errors |

**Implication:** Use robust standard errors, HAC (heteroskedasticity-autocorrelation consistent) estimators, or switch to models designed for time series (ARIMA, GARCH, LSTM).

### 4.5 The S&P 500 Adds Minimal Value

The coefficient for `^GSPC(t-1)` was 0.0008 with p = 0.035. While statistically significant, its practical impact is minimal:

- S&P 500 moved ~500 points during the period
- Effect on AAPL prediction: 500 × 0.0008 = $0.40

AAPL's own lagged price dominates the prediction, reflecting stock-specific momentum over market-wide effects in this short-term horizon.

---

## 5. Why Linear Regression for Stock Prediction?

### 5.1 Strengths

1. **Interpretability:** Coefficients directly tell us feature importance and direction
2. **Transparency:** All assumptions are explicit and testable
3. **Speed:** Trains in seconds, runs in milliseconds
4. **Baseline:** Establishes a benchmark before trying complex models

### 5.2 Limitations

1. **Assumes linearity:** Stock prices have nonlinear dynamics (volatility clustering, regime changes)
2. **Assumes stationarity:** Financial returns often violate this
3. **No volatility modeling:** Heteroskedasticity (changing variance) is common
4. **Short memory:** Doesn't capture long-term dependencies

### 5.3 When to Use Alternatives

| Scenario | Recommended Model |
|----------|------------------|
| Capturing volatility clustering | GARCH |
| Long-term dependencies | LSTM, GRU |
| Regime changes | Hidden Markov Models |
| Non-linear relationships | Random Forest, XGBoost |
| Risk management | Copulas, VaR models |

---

## 6. Practical Lessons Learned

### 6.1 Configuration-Driven Code

This project moved from "class report" code to production-ready infrastructure:

```
configs/
├── tickers.yaml    # Change tickers without touching code
├── dates.yaml      # Adjust training periods
├── models.yaml     # Enable/disable models, tune hyperparameters
```

**Benefit:** Non-technical users can modify experiments without reading Python code.

### 6.2 Reproducibility

Using `Makefile` targets ensures identical execution:

```bash
make train    # Same data, same preprocessing, same results
make report   # Same plots, same metrics
```

**Benefit:** Research becomes replicable and auditable.

### 6.3 Diagnostic-First Approach

Running diagnostics before trusting predictions revealed:

- Autocorrelation patterns the model misses
- Non-normal residuals affecting inference
- Moderate VIF despite careful feature selection

**Benefit:** Avoids false confidence in model predictions.

---

## 7. Future Directions

### 7.1 Immediate Improvements

1. **Add more features:**
   - Volatility (rolling standard deviation)
   - Trading volume
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Macroeconomic variables (VIX, interest rates)

2. **Use HAC standard errors:**
   - Newey-West estimators for autocorrelation-robust inference

3. **Implement walk-forward validation:**
   - Train on expanding window
   - Test on rolling out-of-sample periods

### 7.2 Advanced Models to Explore

1. **ARIMA/GARCH:** Explicitly model time series dynamics and volatility
2. **Random Forest:** Capture non-linear relationships
3. **LSTM Neural Networks:** Long-term dependencies in sequences
4. **Ensemble Methods:** Combine multiple models for robustness

### 7.3 Production Deployment

1. **Streamlit dashboard:** Interactive visualization of predictions
2. **FastAPI service:** Real-time prediction endpoints
3. **Automated retraining:** Daily model updates via cron jobs
4. **Monitoring:** Alert on prediction degradation (concept drift)

---

## 8. Conclusion

This project demonstrated that linear regression remains a valuable tool for stock price prediction, particularly for establishing baselines and understanding feature relationships. The key takeaway is that **high R² in training doesn't guarantee useful predictions**—statistical diagnostics revealed autocorrelation violations that point to fundamental limitations.

The transition from notebook-based analysis to config-driven, production-ready code with `make` targets and YAML configuration represents a maturity shift in the project, enabling reproducible research and easier experimentation.

**Final Recommendations:**

1. Always perform diagnostic checks before trusting predictions
2. Use regularized models for better generalization
3. Consider financial time series properties (autocorrelation, volatility clustering)
4. Move beyond R² as the sole metric of model quality
5. Build toward ensemble methods that combine linear interpretability with nonlinear flexibility

The foundation is solid. The next phase is feature engineering, model ensembling, and rigorous backtesting to validate whether any alpha exists in this prediction framework.

---

## Appendix: Reproducing This Analysis

```bash
# Clone the repository
git clone https://github.com/rithik279/stock_prices_linear_regression.git
cd stock_prices_linear_regression

# Install dependencies
pip install -r requirements.txt

# Run quick demo (<60 seconds)
make quickstart

# Train all models
make train

# Generate full diagnostic report
make report
```

All code, configs, and outputs are available in the repository.

---

*Generated: April 2026*
*Author: Stock Price Linear Regression Project Team*
