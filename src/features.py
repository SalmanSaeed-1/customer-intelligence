"""Build the customer table (features X and targets y) for a given cutoff.

Functions to write (notebook 02):
    build_rfm(tx, cutoff)            -> Recency, Frequency, Monetary per customer,
                                        using ONLY tx[InvoiceDate < cutoff]
    build_extra_features(tx, cutoff) -> tenure, avg basket size, country, ...
    build_targets(tx, cutoff, days)  -> churned (0/1), next_period_spend,
                                        using ONLY cutoff <= InvoiceDate < cutoff + days
    build_customer_table(tx, cutoff) -> X joined with y, one row per customer
"""
