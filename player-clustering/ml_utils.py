"""
ml_utils.py
-----------
Every machine-learning step lives here so app.py stays thin.
"""

import io
import json
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # draw to memory; no GUI window on the server
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

PLAYER_COL = "Player"

# These are the numeric columns the models see.
# Player name is an identifier, NOT a feature.
FEATURES = [
    "Matches", "Runs", "BattingAverage",
    "StrikeRate", "Wickets", "EconomyRate",
]

LABEL_LOGIC_TEXT = (
    "Rule-based labelling: for each cluster we compute a 'batting score' from its "
    "average Runs, BattingAverage and StrikeRate, and a 'bowling score' from its "
    "average Wickets and EconomyRate (we use NEGATIVE economy because a lower "
    "economy rate is better). Both scores are min-max scaled to 0-1 across clusters. "
    "Then: batting>=0.66 and bowling>=0.66 -> 'All-Rounders'; batting>=0.66 -> "
    "'Top Performers'; bowling>=0.66 -> 'Top Bowlers'; both <=0.34 -> "
    "'Needs Improvement'; anything in between -> 'Consistent Players'."
)


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure into a base64 PNG string (inline in HTML)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def basic_stats(df: pd.DataFrame) -> dict:
    """Cheap summary shown BEFORE any ML runs: shape, missing counts, preview."""
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "missing": missing,
        "total_missing": int(sum(missing.values())),
        "preview": json.loads(df.head(10).to_json(orient="records")),
    }


def clean_data(raw_df: pd.DataFrame):
    """Returns (clean_df, report_dict)."""
    report = {}
    df = raw_df.copy()
    report["original_rows"] = int(len(df))

    needed = [PLAYER_COL] + FEATURES
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing these required columns: {missing_cols}. "
            f"Expected columns: {needed}"
        )
    df = df[needed].copy()

    # errors='coerce' turns unparseable text or blank cells into NaN
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_values = 0
    for col in FEATURES:
        invalid = df[col] < 0
        invalid_values += int(invalid.sum())
        if col != "Matches":
            df.loc[invalid, col] = np.nan
    report["negative_values_handled"] = invalid_values

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    before = len(df)
    df = df[df[PLAYER_COL].notna() & (df[PLAYER_COL].astype(str).str.strip() != "")]
    df = df.dropna(subset=FEATURES, how="all")
    df = df[df["Matches"].isna() | (df["Matches"] > 0)]
    report["rows_dropped"] = int(before - len(df))

    # Cricket-aware zero handling
    mask = df["EconomyRate"].isna() & (df["Wickets"].fillna(0) == 0)
    df.loc[mask, "EconomyRate"] = 0.0

    mask = df["Wickets"].isna() & (df["EconomyRate"].fillna(-1) == 0)
    df.loc[mask, "Wickets"] = 0.0

    mask = df["BattingAverage"].isna() & (df["Runs"].fillna(0) == 0)
    df.loc[mask, "BattingAverage"] = 0.0

    mask = df["StrikeRate"].isna() & (df["Runs"].fillna(0) == 0)
    df.loc[mask, "StrikeRate"] = 0.0

    # Median imputation (median, not mean, because star players skew the mean up)
    missing_before = int(df[FEATURES].isna().sum().sum())
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    report["missing_filled"] = missing_before

    df[FEATURES] = df[FEATURES].fillna(0.0)
    df = df.reset_index(drop=True)
    report["final_rows"] = int(len(df))
    return df, report


def scale_features(clean_df: pd.DataFrame):
    """StandardScaler -> mean 0, std 1 for every feature."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df[FEATURES])
    scaled_df = pd.DataFrame(scaled, columns=FEATURES, index=clean_df.index)
    return scaled_df, scaler


def scaling_summary(clean_df: pd.DataFrame, scaled_df: pd.DataFrame) -> list:
    """Before/after table so you can SHOW that scaling worked."""
    rows = []
    for col in FEATURES:
        rows.append({
            "feature": col,
            "before_mean": round(float(clean_df[col].mean()), 2),
            "before_std": round(float(clean_df[col].std(ddof=0)), 2),
            "after_mean": round(float(scaled_df[col].mean()), 2),
            "after_std": round(float(scaled_df[col].std(ddof=0)), 2),
        })
    return rows


def safe_silhouette_score(X: np.ndarray, labels):
    """Return no score when the labels do not define a valid silhouette."""
    if not 1 < len(np.unique(labels)) < len(X):
        return None
    try:
        return float(silhouette_score(X, labels))
    except ValueError:
        return None


def internal_validation_metrics(X: np.ndarray, labels) -> dict:
    """Calculate complementary internal scores when the partition is valid."""
    silhouette = safe_silhouette_score(X, labels)
    if silhouette is None:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
    try:
        davies_bouldin = float(davies_bouldin_score(X, labels))
        calinski_harabasz = float(calinski_harabasz_score(X, labels))
    except ValueError:
        davies_bouldin = None
        calinski_harabasz = None
    return {
        "silhouette": round(silhouette, 3),
        "davies_bouldin": None if davies_bouldin is None else round(davies_bouldin, 3),
        "calinski_harabasz": None if calinski_harabasz is None else round(calinski_harabasz, 2),
    }


def elbow_and_silhouette(X: np.ndarray, k_max: int = 10):
    """Compute elbow and silhouette values for feasible cluster counts."""
    n_samples = X.shape[0]
    n_distinct = np.unique(X, axis=0).shape[0]
    max_clusters = min(int(k_max), n_samples - 1, n_distinct)
    if max_clusters < 2:
        raise ValueError("At least 3 usable rows with 2 distinct feature profiles are required.")

    ks = list(range(1, max_clusters + 1))
    inertias = []
    for k in ks:
        # n_init=10 -> try 10 random starts, keep the best; K-Means result
        # depends on centroid initialisation, this makes it stable.
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(float(km.inertia_))

    sil_ks = list(range(2, max_clusters + 1))
    silhouettes = []
    for k in sil_ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        silhouettes.append(safe_silhouette_score(X, labels))

    valid_scores = [(k, score) for k, score in zip(sil_ks, silhouettes) if score is not None]
    recommended_k = max(valid_scores, key=lambda pair: pair[1])[0] if valid_scores else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, inertias, marker="o", color="#0d6efd")
    axes[0].set_title("Elbow Method - Inertia vs k")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.3)

    plot_scores = [np.nan if score is None else score for score in silhouettes]
    axes[1].plot(sil_ks, plot_scores, marker="o", color="#198754")
    if recommended_k is not None:
        axes[1].axvline(recommended_k, color="#c9714a", linestyle="--", linewidth=1,
                        label=f"Best silhouette: k={recommended_k}")
        axes[1].legend(loc="best")
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score (higher is better)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    data = {
        "ks": ks,
        "inertia": [round(v, 2) for v in inertias],
        "sil_ks": sil_ks,
        "silhouette": [None if v is None else round(v, 3) for v in silhouettes],
        "recommended_k": recommended_k,
    }
    return fig_to_base64(fig), data


def dendrogram_plot(Z: np.ndarray, k: int) -> str:
    """Draw the dendrogram produced by Ward linkage."""
    n_samples = Z.shape[0] + 1

    cut_height = None
    try:
        cut_height = float((Z[n_samples - k - 1, 2] + Z[n_samples - k, 2]) / 2.0)
    except Exception:
        cut_height = None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    kwargs = dict(
        truncate_mode="lastp",   # show only last 30 merges; a full 100-leaf
        p=30,                    # dendrogram is unreadable
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    if cut_height is not None:
        kwargs["color_threshold"] = cut_height

    dendrogram(Z, ax=ax, **kwargs)

    if cut_height is not None:
        ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=1.2,
                   label=f"Cut here -> {k} clusters")
        ax.legend(loc="upper right")

    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)")
    ax.set_xlabel("Players (truncated view)")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    return fig_to_base64(fig)


def pca_2d(X: np.ndarray):
    """
    PCA -> 2 dimensions for plotting.
    WHY: players live in 6 dimensions. A scatter plot only has 2 axes.
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def scatter_comparison(coords, kmeans_labels, hier_labels, evr, k) -> str:
    """Side-by-side 2D scatter of the two clustering results."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    panels = [
        (axes[0], kmeans_labels, f"K-Means (k={k})"),
        (axes[1], hier_labels, f"Hierarchical - Ward (k={k})"),
    ]
    for ax, labels, title in panels:
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                   s=45, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% of variance)")
        ax.grid(alpha=0.25)

    fig.suptitle("Clusters visualised in 2D with PCA")
    fig.tight_layout()
    return fig_to_base64(fig)


def cluster_profile(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Average of every feature per cluster + how many players are in it."""
    tmp = df[FEATURES].copy()
    tmp["Cluster"] = labels
    profile = tmp.groupby("Cluster")[FEATURES].mean().round(2)
    profile["Size"] = tmp.groupby("Cluster").size()
    return profile


def profile_to_records(profile_df: pd.DataFrame) -> list:
    """Convert the profile table into plain JSON-safe dicts."""
    records = []
    for cluster_id, row in profile_df.iterrows():
        rec = {"Cluster": int(cluster_id), "Size": int(row["Size"])}
        for col in FEATURES:
            rec[col] = round(float(row[col]), 2)
        records.append(rec)
    return records


def ari_sentence(ari: float) -> str:
    """Plain-English description of the Adjusted Rand Index."""
    if ari > 0.75:
        strength = "almost the same grouping"
    elif ari > 0.40:
        strength = "broadly similar groupings"
    elif ari > 0.10:
        strength = "somewhat similar groupings"
    else:
        strength = "quite different groupings"

    return (
        f"The Adjusted Rand Index between the K-Means and Hierarchical labels is "
        f"{ari:.3f} (1.0 = identical clusters, 0.0 = random) - the two methods "
        f"produced {strength}."
    )


def _minmax(series: pd.Series) -> pd.Series:
    """Scale a series to 0-1 so we can compare 'batting' and 'bowling' fairly."""
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def label_clusters(profile_df: pd.DataFrame) -> dict:
    """Turn cluster numbers into plain English using rules on the centroids."""
    batting_score = (
        _minmax(profile_df["Runs"])
        + _minmax(profile_df["BattingAverage"])
        + _minmax(profile_df["StrikeRate"])
    ) / 3.0

    # NEGATIVE economy: a low economy rate is GOOD for a bowler
    bowling_score = (
        _minmax(profile_df["Wickets"])
        + _minmax(-profile_df["EconomyRate"])
    ) / 2.0

    labels = {}
    for cluster_id in profile_df.index:
        bat = float(batting_score[cluster_id])
        bowl = float(bowling_score[cluster_id])

        if bat >= 0.66 and bowl >= 0.66:
            name = "All-Rounders"
        elif bat >= 0.66:
            name = "Top Performers"
        elif bowl >= 0.66:
            name = "Top Bowlers"
        elif bat <= 0.34 and bowl <= 0.34:
            name = "Needs Improvement"
        else:
            name = "Consistent Players"

        labels[cluster_id] = name
    return labels


def run_full_pipeline(raw_df: pd.DataFrame, k: int = 4, k_max: int = 10):
    """Run everything; return (JSON-safe dict, final clustered DataFrame)."""
    result = {}

    # 1. Clean
    clean_df, clean_report = clean_data(raw_df)
    result["cleaning"] = clean_report

    if len(clean_df) < k + 1:
        raise ValueError(
            f"Only {len(clean_df)} usable rows after cleaning - not enough for k={k}. "
            f"Try a smaller k or a bigger dataset."
        )

    # 2. Scale
    scaled_df, _scaler = scale_features(clean_df)
    X = scaled_df.values
    max_feasible_k = min(len(clean_df) - 1, np.unique(X, axis=0).shape[0])
    if k < 2 or k > max_feasible_k:
        raise ValueError(
            f"Choose k between 2 and {max_feasible_k} for the usable, distinct player profiles."
        )
    result["scaling"] = scaling_summary(clean_df, scaled_df)
    result["rows_used"] = int(len(clean_df))

    # 3. Elbow + silhouette
    elbow_img, elbow_data = elbow_and_silhouette(X, min(k_max, max_feasible_k))
    result["elbow_img"] = elbow_img
    result["elbow_data"] = elbow_data

    # 4. K-Means with chosen k
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    result["kmeans_inertia"] = round(float(kmeans.inertia_), 2)
    kmeans_metrics = internal_validation_metrics(X, kmeans_labels)
    result["kmeans_silhouette"] = kmeans_metrics["silhouette"]
    result["kmeans_davies_bouldin"] = kmeans_metrics["davies_bouldin"]
    result["kmeans_calinski_harabasz"] = kmeans_metrics["calinski_harabasz"]

    # 5. Hierarchical (Ward)
    Z = linkage(X, method="ward")
    result["dendro_img"] = dendrogram_plot(Z, k)
    # fcluster numbering starts at 1, so subtract 1 to match K-Means' 0-based labels
    hier_labels = fcluster(Z, t=k, criterion="maxclust") - 1
    hier_metrics = internal_validation_metrics(X, hier_labels)
    result["hier_silhouette"] = hier_metrics["silhouette"]
    result["hier_davies_bouldin"] = hier_metrics["davies_bouldin"]
    result["hier_calinski_harabasz"] = hier_metrics["calinski_harabasz"]

    # 6. PCA + scatter
    coords, evr = pca_2d(X)
    result["pca_img"] = scatter_comparison(coords, kmeans_labels, hier_labels, evr, k)
    result["pca_explained"] = [round(float(v) * 100, 1) for v in evr]

    # 7. Cluster profiles
    km_profile = cluster_profile(clean_df, kmeans_labels)
    hc_profile = cluster_profile(clean_df, hier_labels)
    result["kmeans_profile"] = profile_to_records(km_profile)
    result["hier_profile"] = profile_to_records(hc_profile)

    # 8. Agreement between the two methods
    ari = float(adjusted_rand_score(kmeans_labels, hier_labels))
    result["ari"] = round(ari, 3)
    result["ari_sentence"] = ari_sentence(ari)

    # 9. Plain-English labels
    km_label_map = label_clusters(km_profile)
    hc_label_map = label_clusters(hc_profile)
    result["kmeans_cluster_labels"] = {str(c): v for c, v in km_label_map.items()}
    result["hier_cluster_labels"] = {str(c): v for c, v in hc_label_map.items()}
    result["label_logic"] = LABEL_LOGIC_TEXT

    # 10. Final downloadable table
    final_df = clean_df.copy()
    final_df["KMeans_Cluster"] = kmeans_labels
    final_df["KMeans_Label"] = [km_label_map[c] for c in kmeans_labels]
    final_df["Hierarchical_Cluster"] = hier_labels
    final_df["Hierarchical_Label"] = [hc_label_map[c] for c in hier_labels]

    return result, final_df