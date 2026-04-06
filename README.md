# Predicting S&P 500 Daily Market Direction

Predict whether the S&P 500 index will **close higher (Up) or lower (Down) than its opening price** on a given trading day. The project combines classical statistical modelling, time-series analysis, and modern machine-learning classifiers applied to 15 years of daily OHLCV data (2010–2024).

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Methods](#methods)
5. [How to Run](#how-to-run)
6. [Configuration](#configuration)
7. [Results Summary](#results-summary)

---

## Project Goal

Given the open price of the S&P 500 on a trading day, can we predict whether it will close higher (bullish) or lower (bearish)?

- **Target variable:** `Direction` — binary label (`1` = Close > Open, `0` otherwise).
- **Inputs:** Technical indicators derived from price/volume history (moving averages, volatility, RSI, Bollinger Bands).
- **Scope:** Daily frequency, 2010-01-01 through 2024-12-31.

---

## Dataset

**Primary source – Kaggle:**  
[Historical S&P 500 (^GSPC) Index Data (1927–2025)](https://www.kaggle.com/datasets/henryhan117/sp500-historical-data)

Download the CSV and place it at `data/sp500_index.csv`.

**Fallback – yfinance:**  
If `data/sp500_index.csv` is absent, Notebook 01 automatically downloads equivalent data via `yfinance` (ticker `^GSPC`).

Expected CSV columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

---

## Repository Structure

```
sp500-direction-prediction/
├── config.py                    # All paths, dates & hyperparameters
├── requirements.txt             # Python dependencies
├── README.md
├── data/
│   └── sp500_index.csv          # ← place Kaggle CSV here (see Dataset)
└── notebooks/
    ├── 01_data_loading_eda.ipynb          # Load data, binary target, EDA
    ├── 02_distribution_monte_carlo.ipynb  # Normal vs Student-t, Monte Carlo
    ├── 03_bayesian_estimation.ipynb       # Bayesian mean-return estimation
    ├── 04_ols_regression.ipynb            # OLS on log returns + diagnostics
    ├── 05_logistic_regression.ipynb       # Logistic regression baseline
    ├── 06_arima.ipynb                     # ARIMA: ADF, ACF/PACF, rolling forecast
    ├── 07_ml_classifiers.ipynb            # Random Forest & SVM classifiers
    ├── 08_pca.ipynb                       # PCA scree, variance, 2-D projection
    └── 09_bootstrapping.ipynb             # Bootstrap 95% CI for test accuracy
```

---

## Methods

| Notebook | Topic | Key Techniques |
|----------|-------|----------------|
| 01 | Data Loading & EDA | Price chart, returns, rolling volatility, correlation heatmap, class distribution |
| 02 | Distribution Fitting & Monte Carlo | Normal & Student-t MLE, Q-Q plots, KS test, 500-path Monte Carlo simulation |
| 03 | Bayesian Estimation | Conjugate Normal-Normal update, posterior mean/CI, prior sensitivity |
| 04 | OLS Regression | Technical-indicator features, residual diagnostics, Breusch-Pagan, Durbin-Watson |
| 05 | Logistic Regression | GLM baseline, confusion matrix, ROC-AUC curve |
| 06 | ARIMA | ADF stationarity test, ACF/PACF, ARIMA(1,0,1), rolling one-step forecast |
| 07 | ML Classifiers | Random Forest (200 trees) + SVM (RBF kernel), full metrics, feature importance |
| 08 | PCA | Scree plot, cumulative variance, 2-D projection coloured by direction |
| 09 | Bootstrapping | 1,000-resample bootstrap CI for accuracy, precision, recall, and F1 |

**Technical indicators used as features:**

| Feature | Description |
|---------|-------------|
| `MA_cross` | MA(5) − MA(20) crossover |
| `Volatility` | 20-day rolling std of log returns |
| `RSI` | 14-day Relative Strength Index |
| `BB_width` | Bollinger Band width (20-day, 2σ) |
| `BB_pct` | Price position within Bollinger Bands |
| `LogReturn` | Previous-day log return |

---

## How to Run

### 1. Clone & install dependencies

```bash
git clone https://github.com/virakyuthSRUN/sp500-direction-prediction.git
cd sp500-direction-prediction
pip install -r requirements.txt
```

### 2. Add the dataset

Option A – Kaggle CSV (recommended):
```bash
# Download from https://www.kaggle.com/datasets/henryhan117/sp500-historical-data
# then move to:
mv ~/Downloads/sp500_index.csv data/sp500_index.csv
```

Option B – automatic download via yfinance:  
Simply run Notebook 01 without the CSV; it will fetch data from Yahoo Finance automatically.

### 3. Run notebooks in order

```bash
jupyter notebook
```

Open and run notebooks `01` through `09` in sequence.  
**Important:** Notebook 01 must be run first — it saves `data/{asset_code}_processed.csv` (for example `sp500_processed.csv`, `nasdaq_processed.csv`, `gold_processed.csv`, `silver_processed.csv`) which all subsequent notebooks load via `config.PROCESSED_DATA_CSV`.

---

## Configuration

All configuration is centralised in [`config.py`](config.py):

```python
START_DATE    = "2010-01-01"   # analysis start
END_DATE      = "2024-12-31"   # analysis end
TEST_SIZE     = 0.20           # 20% hold-out test set
RANDOM_STATE  = 42
N_ESTIMATORS  = 200            # Random Forest trees
ARIMA_ORDER   = (1, 0, 1)      # ARIMA (p, d, q)
N_BOOTSTRAP   = 1000           # bootstrap resamples
PRIOR_MEAN    = 0.0            # Bayesian prior μ₀
PRIOR_STD     = 0.01           # Bayesian prior τ₀
```

Edit `config.py` to change date ranges, model hyperparameters, or file paths — no changes to the notebooks themselves are required.

---

## Results Summary

> Numbers will vary slightly depending on whether Kaggle CSV or yfinance data is used.

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~0.53 | ~0.54 |
| Random Forest | ~0.55 | ~0.56 |
| SVM (RBF) | ~0.54 | ~0.55 |

The target is inherently noisy (daily market direction is close to a random walk), so modest improvements over 50% chance are typical and meaningful. The bootstrap confidence intervals in Notebook 09 quantify the uncertainty around these estimates.

