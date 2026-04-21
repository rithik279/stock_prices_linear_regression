# Stock Price Linear Regression

Predicting AAPL next-day closing price using OLS, Ridge, Lasso, and Elastic Net regression with statistical diagnostics.

## Quick Start

### Option 1: Run Immediately (Windows)
```batch
run.bat quickstart
```

### Option 2: Run Immediately (Unix/Mac/Linux with make)
```bash
make quickstart
```

This generates a prediction plot in under 60 seconds.

## Available Commands

| Command | Description |
|---------|-------------|
| `make quickstart` / `run.bat quickstart` | Quick prediction plot in <60s |
| `make train` / `run.bat train` | Train all models, save results |
| `make report` / `run.bat report` | Full diagnostic report with plots |
| `make install` / `run.bat install` | Install dependencies |
| `make clean` / `run.bat clean` | Remove output files |

## Project Structure

```
.
├── configs/               # Configuration files
│   ├── tickers.yaml      # Ticker symbols to use
│   ├── dates.yaml        # Training/testing date ranges
│   ├── models.yaml       # Model hyperparameters
│   └── settings.yaml      # General settings
├── logic/                 # Core modules
│   ├── data_import.py    # Data loading & feature engineering
│   ├── ols_linear_regression.py
│   ├── ridge_regression.py
│   ├── lasso_regression.py
│   └── elastic_net_regression.py
├── scripts/              # Executable scripts
│   ├── train.py         # Train all models
│   └── report.py        # Generate diagnostic report
├── output/               # Generated outputs (gitignored)
│   ├── figures/         # Plots
│   ├── reports/         # Text reports
│   └── *.csv            # Model comparisons
├── quickstart.py         # Fast demo (<60s)
├── Makefile             # Unix make targets
├── run.bat              # Windows runner
└── requirements.txt     # Dependencies
```

## Configuration

Edit `configs/*.yaml` to customize:

**tickers.yaml**: Change tickers, features, target
```yaml
tickers:
  primary: AAPL
  market_indices: [QQQ, ^GSPC]
```

**dates.yaml**: Adjust training/testing periods
```yaml
training:
  start: "2020-01-01"
  end: "2024-12-31"
```

**models.yaml**: Enable/disable models, tune hyperparameters
```yaml
models:
  ridge:
    enabled: true
    alpha: 0.1
```

## Diagnostics Included

- **VIF** (Variance Inflation Factor) - Multicollinearity check
- **Durbin-Watson** - Autocorrelation test
- **Q-Q Plot** - Residual normality
- **Homoscedasticity** - Residual vs Fitted plot

## Models

| Model | Features Used | Purpose |
|-------|--------------|---------|
| OLS | Lagged prices only | Baseline, interpretable |
| Ridge | All features | L2 regularization |
| Lasso | All features | Feature selection |
| ElasticNet | All features | L1 + L2 balance |

## Requirements

- Python 3.8+
- See `requirements.txt`

Install with: `pip install -r requirements.txt`
