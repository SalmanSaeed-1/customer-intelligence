"""Build the customer table (features X and targets y) for one cutoff date (notebook 02).

    build_rfm(tx, cutoff)              -> recency, frequency, monetary per customer
    build_extra_features(tx, cutoff)   -> the other features (tenure, basket size, ...)
    build_targets(tx, cutoff, days)    -> churned (0/1) and next_period_spend
    build_customer_table(tx, cutoff)   -> features + targets, one row per customer

LEAKAGE RULE (the most important idea in the project):
    features use ONLY transactions with invoice_date <  cutoff          (the past)
    targets  use ONLY transactions with cutoff <= invoice_date < cutoff + days  (the future)
Both filters live in the two small helpers below, so no function can forget them.

`tx` is always the clean transaction table from notebook 01 (config.CLEAN_FILE).
"""
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# The two time filters
# ---------------------------------------------------------------------------
def history_before(tx, cutoff):
    """Transactions strictly BEFORE the cutoff: the only rows features may use."""
    past = tx[tx["invoice_date"] < cutoff]
    assert past["invoice_date"].max() < cutoff, "a feature row is on/after the cutoff"
    return past


def label_window(tx, cutoff, days=config.HOLDOUT_DAYS):
    """Transactions in the `days`-day window starting AT the cutoff: the only rows targets may use."""
    end = cutoff + pd.Timedelta(days=days)
    return tx[(tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end)]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_rfm(tx, cutoff):
    """Recency, Frequency and Monetary value per customer, from the history before `cutoff`.

    recency_days - days from the customer's last purchase to the cutoff (small = bought recently)
    frequency    - number of distinct invoices (orders)
    monetary     - total revenue in £
    """
    past = history_before(tx, cutoff)
    rfm = past.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
    )
    rfm["recency_days"] = (cutoff - rfm["last_purchase"]).dt.days
    return rfm.reset_index()[["customer_id", "recency_days", "frequency", "monetary"]]


def build_extra_features(tx, cutoff, cancellations=None, recent_days=config.HOLDOUT_DAYS):
    """The non-RFM features per customer, from the history before `cutoff`.

    If `cancellations` (config.CANCEL_FILE) is given, the optional cancel_rate is added too.
    """
    past = history_before(tx, cutoff).sort_values("invoice_date")   # date order, needed for "last"
    by_customer = past.groupby("customer_id")

    feats = by_customer.agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
        total_items=("quantity", "sum"),
        n_unique_products=("stock_code", "nunique"),
        last_country=("country", "last"),   # rows are in date order, so this is the latest country
    )

    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    feats["avg_basket_size"] = feats["total_items"] / feats["frequency"]   # items per order
    feats["tenure_days"] = (cutoff - feats["first_purchase"]).dt.days

    # Mean gap between consecutive orders. The average of the gaps between n orders is simply
    # (last - first) / (n - 1), so we don't need to list every gap.
    # It is undefined for one-order customers: we flag them and fill the gap with their tenure
    # ("at least this many days without a second order").
    span_days = (feats["last_purchase"] - feats["first_purchase"]) / pd.Timedelta(days=1)
    feats["is_single_order"] = (feats["frequency"] == 1).astype(int)
    feats["mean_interpurchase_days"] = span_days / (feats["frequency"] - 1)
    feats.loc[feats["is_single_order"] == 1, "mean_interpurchase_days"] = feats["tenure_days"]

    # Activity in the last `recent_days` before the cutoff (same length as the label window,
    # so revenue_last_90d can be used directly as the "next 90 days" baseline in Module 3).
    recent = past[past["invoice_date"] >= cutoff - pd.Timedelta(days=recent_days)]
    recent_stats = recent.groupby("customer_id").agg(
        revenue_last_90d=("revenue", "sum"),
        frequency_last_90d=("invoice", "nunique"),
    )
    feats = feats.join(recent_stats)
    feats[["revenue_last_90d", "frequency_last_90d"]] = feats[["revenue_last_90d", "frequency_last_90d"]].fillna(0)
    feats["frequency_last_90d"] = feats["frequency_last_90d"].astype(int)

    feats["is_uk"] = (feats["last_country"] == "United Kingdom").astype(int)

    columns = ["tenure_days", "avg_order_value", "n_unique_products", "avg_basket_size",
               "mean_interpurchase_days", "is_single_order", "revenue_last_90d",
               "frequency_last_90d", "is_uk"]

    if cancellations is not None:
        # cancel_rate = cancelled orders / (cancelled + completed orders), counting ONLY
        # cancellations before the cutoff (cancellations after it would be future information).
        past_cancels = history_before(cancellations, cutoff)
        n_cancelled = past_cancels.groupby("customer_id")["invoice"].nunique()
        feats["n_cancelled"] = n_cancelled.reindex(feats.index).fillna(0)
        feats["cancel_rate"] = feats["n_cancelled"] / (feats["n_cancelled"] + feats["frequency"])
        columns.append("cancel_rate")

    return feats.reset_index()[["customer_id"] + columns]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def build_targets(tx, cutoff, days=config.HOLDOUT_DAYS):
    """Targets for every customer who bought at least once before `cutoff`.

    churned           - 1 if the customer has NO purchase in the window, else 0
    next_period_spend - total revenue in the window (0 if no purchase)
    """
    population = history_before(tx, cutoff)["customer_id"].unique()
    future = label_window(tx, cutoff, days)

    spend = future.groupby("customer_id")["revenue"].sum()
    targets = pd.DataFrame({"customer_id": population})
    # Customers with no purchase in the window are missing from `spend`, so they get 0.
    targets["next_period_spend"] = targets["customer_id"].map(spend).fillna(0.0)
    # Every clean line has revenue > 0, so "spend == 0" means exactly "no purchase in the window".
    targets["churned"] = (targets["next_period_spend"] == 0).astype(int)
    return targets[["customer_id", "churned", "next_period_spend"]]


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
def build_customer_table(tx, cutoff, cancellations=None, days=config.HOLDOUT_DAYS):
    """One row per customer who bought before `cutoff`: all features, then the two targets."""
    rfm = build_rfm(tx, cutoff)
    extra = build_extra_features(tx, cutoff, cancellations)
    targets = build_targets(tx, cutoff, days)

    table = rfm.merge(extra, on="customer_id").merge(targets, on="customer_id")
    assert table["customer_id"].is_unique
    assert len(table) == len(rfm) == len(targets), "every customer must have features AND targets"
    return table.sort_values("customer_id").reset_index(drop=True)
