"""Load and clean the raw Online Retail II data (used by notebook 01).

    load_raw()            -> read both Excel sheets, stack them, rename columns
    clean_transactions()  -> apply the cleaning rules (CLAUDE.md Section 7) in order
                             and record how many rows are left after each rule
    get_cancellations()   -> the cancelled invoices, kept aside for the optional
                             cancel_rate feature in notebook 02
"""
import pandas as pd

from src import config

# Raw Excel column name -> our snake_case name.
COLUMN_NAMES = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_raw(use_cache=True):
    """Return every raw transaction line from both Excel sheets as one DataFrame.

    Reading the Excel file takes 1-2 minutes, so the first call also saves a
    parquet copy (config.RAW_CACHE). Later calls read that copy in a second or two.
    Pass use_cache=False to force a fresh read of the Excel file.
    """
    if use_cache and config.RAW_CACHE.exists():
        return pd.read_parquet(config.RAW_CACHE)

    # sheet_name=None reads ALL sheets and returns a dict {sheet name: DataFrame}.
    sheets = pd.read_excel(config.RAW_FILE, sheet_name=None)
    raw = pd.concat(sheets.values(), ignore_index=True)
    raw = raw.rename(columns=COLUMN_NAMES)

    # Invoice and StockCode mix numbers (489434, 85048) and text (C489449, 85123A).
    # Store them as text so that .str methods work on every row and parquet can save
    # the column (parquet refuses a column that mixes ints and strings).
    raw["invoice"] = raw["invoice"].astype(str)
    raw["stock_code"] = raw["stock_code"].astype(str)
    raw["description"] = raw["description"].astype(str)  # missing values stay missing

    raw.to_parquet(config.RAW_CACHE, index=False)
    return raw


def clean_transactions(raw):
    """Apply the cleaning rules in order.

    Returns (clean, steps):
      clean - one row per valid purchase line, with a new `revenue` column
      steps - a small table with the rows left after each rule (goes in the report)
    """
    df = raw.copy()
    steps = [("Raw data (both sheets)", len(df))]

    # Rule 2: exact duplicate rows (every column identical).
    df = df.drop_duplicates()
    steps.append(("Drop exact duplicate rows", len(df)))

    # Rule 3: cancellations. Their invoice number starts with "C".
    is_cancellation = df["invoice"].str.startswith("C")
    df = df[~is_cancellation]
    steps.append(("Drop cancellations (invoice starts with C)", len(df)))

    # Rule 4: rows without a customer cannot be used for per-customer modelling.
    # customer_id loads as a float (because of the missing values); now that the
    # missing ones are gone it can become a whole number.
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int)
    steps.append(("Drop rows with missing customer_id", len(df)))

    # Rule 5: a real purchase has a positive quantity and a positive price.
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    steps.append(("Drop quantity <= 0 or price <= 0", len(df)))

    # Rule 6: real products have a stock code that starts with a digit.
    # This removes postage, fees, manual adjustments, samples, etc. (POST, DOT, M, ...).
    is_product = df["stock_code"].str.match(r"^\d")
    df = df[is_product]
    steps.append(("Drop non-product stock codes", len(df)))

    # Rules 7 and 8 add/convert columns but do not remove rows.
    df["revenue"] = df["quantity"] * df["price"]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df = df.reset_index(drop=True)

    # Turn the list of (step, rows) into a table and add how many rows each step removed.
    steps = pd.DataFrame(steps, columns=["step", "rows_left"])
    steps["rows_removed"] = (steps["rows_left"].shift() - steps["rows_left"]).fillna(0).astype(int)
    steps["pct_of_raw_left"] = (100 * steps["rows_left"] / steps["rows_left"].iloc[0]).round(1)
    return df, steps


def get_cancellations(raw):
    """One row per cancelled invoice of a known customer: customer_id, invoice, invoice_date.

    These rows are dropped from the clean table, but we keep them here so notebook 02
    can build the optional cancel_rate feature. cancel_rate must be computed per
    cutoff from cancellations BEFORE that cutoff only, so the dates are kept too
    (counting over the whole dataset would leak the future into the features).
    """
    df = raw.drop_duplicates()
    df = df[df["invoice"].str.startswith("C") & df["customer_id"].notna()]
    # A cancelled invoice has one row per product line; we only need one row per invoice.
    df = df.drop_duplicates(subset="invoice")[["customer_id", "invoice", "invoice_date"]]
    df["customer_id"] = df["customer_id"].astype(int)
    return df.reset_index(drop=True)
