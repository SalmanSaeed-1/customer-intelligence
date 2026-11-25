"""Single source of truth for paths, dates and random seed.

Never type a date or a path inside a notebook: import it from here.
"""
from pathlib import Path

import pandas as pd

# ---- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "results"

RAW_FILE = DATA_RAW / "online_retail_II.xlsx"
# Parquet copy of the raw Excel (both sheets), so we only pay the slow Excel read once.
RAW_CACHE = DATA_PROCESSED / "transactions_raw.parquet"
CLEAN_FILE = DATA_PROCESSED / "transactions_clean.parquet"
# Cancelled invoices (one row each), kept aside for the optional cancel_rate feature.
CANCEL_FILE = DATA_PROCESSED / "cancellations.parquet"
TRAIN_TABLE = DATA_PROCESSED / "customer_table_train.parquet"
TEST_TABLE = DATA_PROCESSED / "customer_table_test.parquet"

# ---- dates (time-based split) -------------------------------------------
DATA_START = pd.Timestamp("2009-12-01")
DATA_END = pd.Timestamp("2011-12-09")

# Same-length label window for both tables so targets are comparable.
HOLDOUT_DAYS = 90

# Train table: features before TRAIN_CUTOFF, labels in the 90 days after it.
TRAIN_CUTOFF = pd.Timestamp("2011-06-01")
# Test table:  features before TEST_CUTOFF,  labels in the 90 days after it.
TEST_CUTOFF = pd.Timestamp("2011-09-01")

# ---- reproducibility -----------------------------------------------------
RANDOM_STATE = 42

# ---- module colours (use the same colour for a module in every figure) ---
COLORS = {
    "segmentation": "#2a78d6",  # blue
    "churn": "#eb6834",         # orange
    "clv": "#1baf7a",           # green
    "neutral": "#8c8c8c",       # grey: data/EDA plots that belong to no module
    "ink": "#1f1f1f",           # dark text colour (text is never in a series colour)
}

# Make sure output folders exist.
for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
