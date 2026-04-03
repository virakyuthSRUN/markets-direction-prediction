"""
Project-wide configuration for S&P 500 Direction Prediction.
All paths, dates, and hyperparameters are defined here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Expected Kaggle CSV filename (place this file in data/)
KAGGLE_CSV = os.path.join(DATA_DIR, "sp500_index.csv")

# Fallback: yfinance ticker symbol used when the CSV is absent
YFINANCE_TICKER = "^GSPC"

# ── Date range ─────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# ── Target variable ────────────────────────────────────────────────────────
TARGET_COL = "Direction"   # 1 if Close > Open on that day, else 0

# ── Technical-indicator windows ────────────────────────────────────────────
MA_SHORT = 5       # short moving-average window (days)
MA_LONG = 20       # long  moving-average window (days)
VOL_WINDOW = 20    # rolling volatility window (days)
RSI_WINDOW = 14    # RSI look-back period (days)
BB_WINDOW = 20     # Bollinger Band window (days)
BB_STD = 2.0       # Bollinger Band standard-deviation multiplier

# ── Model hyperparameters ──────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20          # fraction of data held out for testing
N_ESTIMATORS = 200        # Random Forest trees
SVM_C = 1.0               # SVM regularisation parameter
SVM_KERNEL = "rbf"        # SVM kernel

# ── ARIMA order ────────────────────────────────────────────────────────────
ARIMA_ORDER = (1, 0, 1)   # (p, d, q) – d=0 because returns are stationary
ARIMA_ROLLING_WINDOW = 252  # ~1 trading year for rolling forecast

# ── Distribution fitting ───────────────────────────────────────────────────
STUDENT_T_DF_INIT = 5     # initial degrees-of-freedom guess for Student-t fit

# ── Bayesian estimation ────────────────────────────────────────────────────
PRIOR_MEAN = 0.0
PRIOR_STD = 0.01          # tight prior: expect ~0 daily return

# ── Bootstrapping ─────────────────────────────────────────────────────────
N_BOOTSTRAP = 1000        # number of bootstrap resamples
CI_LEVEL = 0.95           # confidence interval level

# ── PCA ───────────────────────────────────────────────────────────────────
PCA_N_COMPONENTS = None   # None → keep all components for scree plot

# ── Plotting ──────────────────────────────────────────────────────────────
FIG_SIZE = (12, 5)
FIG_DPI = 100
