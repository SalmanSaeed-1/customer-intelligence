"""Module 1: customer segmentation on Recency, Frequency and Monetary value (notebook 03).

    make_preprocessor()            -> log1p + StandardScaler for the three RFM columns
    make_kmeans_pipeline(k)        -> preprocessor + K-means in one sklearn Pipeline
    compare_k(X, ks)               -> inertia, silhouette and Davies-Bouldin for each k
    gmm_bic(X, ns)                 -> BIC for each number of Gaussian mixture components
    profile_segments(table, labels)-> size, mean/median R, F, M and the targets per cluster
    name_clusters(profile)         -> a business name for each K-means cluster (rule based)
    add_segment(table, model, names) -> copy of a customer table with a `segment` column

The scaler and the clustering are FIT ON THE TRAIN TABLE ONLY. The test table only goes
through `.predict()`, so nothing about the test period shapes the segments.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]
SEGMENT_NAMES = list(config.SEGMENT_COLORS)   # display order: Champions ... Lost / One-time


# ---------------------------------------------------------------------------
# Preprocessing and the K-means model
# ---------------------------------------------------------------------------
def make_preprocessor():
    """log1p, then standardise (mean 0, std 1).

    log1p(x) = log(1 + x). Frequency and monetary are extremely right-skewed (a few
    wholesalers spend 100x the median), so without the log they would dominate every
    distance. Standardising then gives the three columns the same weight in K-means.
    """
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])


def make_kmeans_pipeline(k):
    """Preprocessing + K-means with `k` clusters, as one Pipeline.

    Calling .fit(train[RFM_COLUMNS]) fits the scaler AND the clusters on train;
    .predict(test[RFM_COLUMNS]) then reuses both without refitting.
    n_init=10 runs K-means from 10 random starts and keeps the best one.
    """
    return Pipeline([
        ("prep", make_preprocessor()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)),
    ])


# ---------------------------------------------------------------------------
# Choosing the number of clusters
# ---------------------------------------------------------------------------
def compare_k(X, ks):
    """Fit K-means for every k in `ks` on the (already scaled) matrix X and score it.

    inertia        - sum of squared distances to the nearest centre (lower = tighter; always
                     falls as k grows, so we look for the "elbow" where it stops falling fast)
    silhouette     - how much closer each customer is to its own cluster than to the next one,
                     from -1 to 1 (higher = better separated)
    davies_bouldin - average similarity of each cluster to its most similar other cluster
                     (lower = better separated)
    """
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
            "smallest_cluster": np.bincount(km.labels_).min(),
        })
    return pd.DataFrame(rows)


def gmm_bic(X, ns):
    """BIC of a Gaussian mixture (full covariance) for every number of components in `ns`.

    BIC = -2 * log-likelihood + (number of parameters) * log(n). Lower is better: it rewards
    fitting the data but charges a penalty for every extra parameter.
    """
    rows = []
    for n in ns:
        gmm = GaussianMixture(n_components=n, covariance_type="full", n_init=3,
                              random_state=config.RANDOM_STATE).fit(X)
        rows.append({"n_components": n, "bic": gmm.bic(X)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Describing and naming the clusters
# ---------------------------------------------------------------------------
def profile_segments(table, labels):
    """One row per cluster: size, mean and median R, F, M, share of revenue, and the two targets.

    The targets (churned, next_period_spend) were NOT used to build the clusters.
    We only look at them afterwards, to check that the segments mean something.
    """
    profile = table.assign(cluster=labels).groupby("cluster").agg(
        customers=("customer_id", "size"),
        recency_mean=("recency_days", "mean"),
        recency_median=("recency_days", "median"),
        frequency_mean=("frequency", "mean"),
        frequency_median=("frequency", "median"),
        monetary_mean=("monetary", "mean"),
        monetary_median=("monetary", "median"),
        total_monetary=("monetary", "sum"),
        churn_rate=("churned", "mean"),
        mean_next_spend=("next_period_spend", "mean"),
    )
    profile.insert(1, "share", profile["customers"] / profile["customers"].sum())
    # Share of all past revenue that this cluster brought in.
    profile["revenue_share"] = profile.pop("total_monetary") / table["monetary"].sum()
    return profile


def name_clusters(profile):
    """Give each of the 4 K-means clusters a business name, using its median R, F, M.

    K-means numbers its clusters arbitrarily (cluster 2 has no meaning), so we name them
    with simple rules instead of by hand. That way the names stay right even if the
    numbering changes:
        Champions       - the highest median spend (monetary)
        Lost / One-time - the longest time since the last purchase (recency)
        Promising       - of the other two, the one that bought more recently
        At-risk         - the remaining one: bought regularly before, but not lately
    """
    assert len(profile) == 4, "the naming rules are written for exactly 4 clusters"
    champions = int(profile["monetary_median"].idxmax())
    lost = int(profile["recency_median"].idxmax())
    assert champions != lost, "one cluster is both the biggest spender and the most lapsed"

    others = profile.drop(index=[champions, lost]).sort_values("recency_median")
    promising, at_risk = (int(c) for c in others.index)
    return {champions: "Champions", promising: "Promising", at_risk: "At-risk", lost: "Lost / One-time"}


def add_segment(table, model, names):
    """Return a copy of `table` with a `segment` column (the business name).

    `model` is the K-means Pipeline fitted on the TRAIN table; it is only used to
    predict here, so the same code works for the train and the test table.
    """
    clusters = model.predict(table[RFM_COLUMNS])
    out = table.copy()
    out["segment"] = pd.Series(clusters, index=table.index).map(names)
    return out
