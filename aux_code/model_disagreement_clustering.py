# -*- coding: utf-8 -*-
"""
Model- and region-clustering diagnostics for the disagreement on projected
RED severity changes at a given GWL: which (GCM, run) pairs tend to predict
opposite-sign severity changes at the same regions, whether those models
form consistent factions, and whether geographically distinct groups of
disagreeing regions are driven by the same factions or different ones.

Port of cells 12-21 of como24_group5/code_final/3.1.3 disagreements.ipynb.
Reuses supp_figs/suppfig7_agreement_variability.py's significance pipeline (load_indicators,
compute_severity_significance, agreement_fractions, GWL_LABELS) rather than
recomputing it.

Needs scikit-learn (KMeans, StandardScaler, silhouette_score,
adjusted_rand_score; written against/tested with 1.5.2). KMeans's own
n_init default changed from a fixed 10 (all versions before 1.4) to 'auto'
(-> 1 with the default k-means++ init) in scikit-learn 1.4+, which would
silently make clustering less stable across versions -- KMEANS_N_INIT below
pins it back to 10 everywhere.

Terminology: severity = frequency x duration x intensity (see supp_figs/suppfig7_agreement_variability.py).

Outputs (under --output_dir):
  model_disagreement_heatmap.png        - cell 15
  model_clustering_elbow_silhouette.png - cell 16's elbow/silhouette
                                          diagnostic (plt.show() only in the
                                          notebook; saved here)
  region_disagreement_clusters.png      - cell 18 (no savefig in the
                                          notebook; saved here)
Model cluster assignments and the Adjusted Rand Index comparisons between
region clusters' own model sub-clusterings (cells 16, 20-21) are printed,
not plotted, matching the notebook.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"] = config.ESMFMKFILE_XENV

import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from suppfig7_agreement_variability import load_indicators, compute_severity_significance, agreement_fractions, GWL_LABELS

KMEANS_N_INIT = 10


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Model- and region-clustering diagnostics for disagreement "
                    "on projected RED severity changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preprocessed_path", default=config.PATH_PREPROCESSED,
                        help="Root of the per-GCM wcf_agg_*/scf_agg_* aggregates "
                             "(see supp_figs/suppfig7_agreement_variability.py); the yearly indicators are always "
                             "rebuilt from these from scratch.")
    parser.add_argument("--gwl", default="GWL2",
                        help="GWL at which disagreement is analysed (vs --reference_gwl).")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ssp", default=config.SSP)
    parser.add_argument("--reference_gwl", default="GWL0-61")
    parser.add_argument("--disagreement_threshold", type=float, default=0.2,
                        help="A poly_idx counts as 'in disagreement' when both the "
                             "increasing- and decreasing-simulation shares exceed this.")
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH_LIGHT,
                        help="Shapefile matching the poly_idx of the rebuilt indicators.")
    parser.add_argument("--distance_matrix_txt",
                        default=os.path.join(os.path.dirname(config.SHARE_RENEWABLE_CSV),
                                             "distance_matrix_100sample.txt"),
                        help="Comma-delimited poly_idx x poly_idx distance matrix, used to "
                             "cluster disagreeing regions geographically. Default guesses "
                             "the socioeconomic_data folder next to config.SHARE_RENEWABLE_CSV "
                             "-- override if it lives elsewhere on the HPC.")
    parser.add_argument("--n_region_clusters", type=int, default=5)
    parser.add_argument("--n_model_subclusters", type=int, default=3,
                        help="k for the per-region-cluster model sub-clustering.")
    parser.add_argument("--model_cluster_ks", type=int, nargs=2, default=[2, 8],
                        help="Inclusive range of k to scan for the elbow/silhouette "
                             "diagnostic on the global model-disagreement matrix.")
    parser.add_argument("--output_dir", default=config.SUMMARY_FIGS_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# =============================================================================
# Model disagreement matrix
# =============================================================================

def build_model_disagreement_matrix(sig, gwl, incr, decr, threshold):
    """
    poly_idx where both the increasing- and decreasing-simulation shares
    exceed `threshold` count as "in disagreement". For each of those,
    collect which (GCM, run) pairs predicted an increase there and which
    predicted a decrease, then tally how often each pair of models lands on
    opposite sides across every such region.

    Returns (matrix, disagreement_idx, increase_by_idx, decrease_by_idx):
    matrix is a symmetric (GCM,run) x (GCM,run) co-occurrence count
    DataFrame; increase_by_idx/decrease_by_idx (keyed by poly_idx) are
    reused by model_subclusters_by_region_cluster() below.
    """
    disagreement_idx = np.where((incr.values > threshold) & (decr.values > threshold))[0]

    unique_models = list(dict.fromkeys(zip(sig.GCM.values, sig.run.values)))
    matrix = pd.DataFrame(0, index=unique_models, columns=unique_models)

    sig_gwl = sig.where(sig.GWL_comp == gwl, drop=True)
    increase_by_idx, decrease_by_idx = {}, {}
    for idx in disagreement_idx:
        region = sig_gwl.isel(poly_idx=idx)
        trend = region["severity_trend"]
        incr_models = set(zip(region.GCM.where(trend > 0, drop=True).values,
                              region.run.where(trend > 0, drop=True).values))
        decr_models = set(zip(region.GCM.where(trend < 0, drop=True).values,
                              region.run.where(trend < 0, drop=True).values))
        increase_by_idx[idx] = incr_models
        decrease_by_idx[idx] = decr_models
        for m1 in incr_models:
            for m2 in decr_models:
                if m1 in matrix.index and m2 in matrix.columns:
                    matrix.at[m1, m2] += 1
                if m2 in matrix.index and m1 in matrix.columns:
                    matrix.at[m2, m1] += 1

    matrix = matrix.reindex(
        index=sorted(matrix.index, key=lambda t: (t[0], t[1])),
        columns=sorted(matrix.columns, key=lambda t: (t[0], t[1])))
    return matrix, disagreement_idx, increase_by_idx, decrease_by_idx


def plot_model_disagreement_heatmap(matrix, output_path, dpi):
    labels = [f"{gcm} {run}" for gcm, run in matrix.index]
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="Reds", xticklabels=labels, yticklabels=labels,
               cbar_kws={"orientation": "vertical",
                         "label": "Number of regions with disagreement"})
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# =============================================================================
# Model clustering (elbow / silhouette)
# =============================================================================

def cluster_models_elbow_silhouette(matrix, ks, output_path, dpi):
    """
    KMeans elbow/silhouette diagnostic on the (standard-scaled)
    model-disagreement matrix, over k in range(ks[0], ks[1] + 1). Prints the
    best-k cluster assignment (by silhouette score) and saves the elbow +
    silhouette curves.
    """
    X = StandardScaler().fit_transform(matrix)
    # silhouette_score needs 2 <= n_clusters <= n_samples - 1; clamp the
    # requested range instead of failing outright on a smaller-than-expected
    # model ensemble.
    max_k = min(ks[1], X.shape[0] - 1)
    if max_k < 2:
        print(f"  Only {X.shape[0]} models -- not enough to cluster (need >= 3), skipping")
        return False
    k_values = list(range(max(2, ks[0]), max_k + 1))
    inertias, sil_scores = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=0).fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_values, inertias, "-o")
    axes[0].set_xlabel("n_clusters")
    axes[0].set_title("Elbow: inertia")
    axes[1].plot(k_values, sil_scores, "-o")
    axes[1].set_xlabel("n_clusters")
    axes[1].set_title("Silhouette score")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    best_k = k_values[int(np.argmax(sil_scores))]
    print(f"Best k by silhouette score: {best_k} (score = {max(sil_scores):.3f})")
    best_labels = KMeans(n_clusters=best_k, n_init=KMEANS_N_INIT, random_state=0).fit(X).labels_
    cluster_df = pd.DataFrame({"Model": matrix.index, "Cluster": best_labels})
    print(cluster_df.sort_values("Cluster").reset_index(drop=True).to_string(index=False))
    return True


# =============================================================================
# Region clustering
# =============================================================================

def cluster_disagreement_regions(dist_matrix, disagreement_idx, n_clusters, seed):
    """KMeans on the raw pairwise-distance submatrix of the disagreeing
    poly_idx (no scaling, matching the notebook)."""
    sub_dist = dist_matrix[np.ix_(disagreement_idx, disagreement_idx)]
    sub_dist = np.where(np.isnan(sub_dist), 0, sub_dist)
    labels = KMeans(n_clusters=n_clusters, n_init=KMEANS_N_INIT,
                    random_state=seed).fit(sub_dist).labels_
    return pd.DataFrame({"idx": disagreement_idx, "Cluster": labels})


def plot_region_disagreement_clusters(shapefile_path, df_clusters, n_clusters,
                                      gwl_label, output_path, dpi):
    shp = gpd.read_file(shapefile_path)
    shp = shp.merge(df_clusters, left_index=True, right_on="idx", how="left")
    cmap = ListedColormap(sns.color_palette("tab10", n_colors=n_clusters).as_hex())

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=1)
    shp.boundary.plot(ax=ax, color="black", linewidth=0.5, transform=ccrs.PlateCarree())
    shp.plot(ax=ax, column="Cluster", cmap=cmap, linewidth=0,
             transform=ccrs.PlateCarree(), legend=False)
    handles = [mpatches.Patch(color=cmap(i), label=f"Cluster {i}") for i in range(n_clusters)]
    ax.legend(handles=handles, title="Region clusters", loc="lower left")
    ax.set_title(f"Clusters of regions with model disagreement at {gwl_label}")
    ax.spines["geo"].set_visible(False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Model sub-clustering within each region cluster
# =============================================================================

def model_subclusters_by_region_cluster(sig, increase_by_idx, decrease_by_idx,
                                        df_clusters, n_clusters, n_subclusters):
    """
    For each region cluster, rebuild the model-disagreement matrix restricted
    to that cluster's own poly_idx, then KMeans-cluster the models on it
    (n_subclusters groups, standard-scaled). Returns a DataFrame with one
    'Cluster_<i>' column per region cluster, one row per (GCM, run).
    """
    unique_models = list(dict.fromkeys(zip(sig.GCM.values, sig.run.values)))
    df_disag = pd.DataFrame({"Model": unique_models})

    for cluster_id in range(n_clusters):
        matrix = pd.DataFrame(0, index=unique_models, columns=unique_models)
        for idx in df_clusters.loc[df_clusters["Cluster"] == cluster_id, "idx"]:
            for m1 in increase_by_idx[idx]:
                for m2 in decrease_by_idx[idx]:
                    if m1 in matrix.index and m2 in matrix.columns:
                        matrix.at[m1, m2] += 1
                    if m2 in matrix.index and m1 in matrix.columns:
                        matrix.at[m2, m1] += 1

        X = StandardScaler().fit_transform(matrix)
        labels = KMeans(n_clusters=n_subclusters, n_init=KMEANS_N_INIT,
                        random_state=0).fit(X).labels_
        col = f"Cluster_{cluster_id}"
        df_disag = df_disag.merge(
            pd.DataFrame({"Model": matrix.index, col: labels}), on="Model", how="left")
    return df_disag


def compare_region_cluster_model_groupings(df_disag):
    """Adjusted Rand Index between every pair of the per-region-cluster
    model sub-clusterings -- do different disagreeing-region groups share
    the same model factions, or does each group split the models differently?"""
    cluster_cols = [c for c in df_disag.columns if c != "Model"]
    for i in range(len(cluster_cols)):
        for j in range(i + 1, len(cluster_cols)):
            score = adjusted_rand_score(df_disag[cluster_cols[i]], df_disag[cluster_cols[j]])
            print(f"Adjusted Rand Index between {cluster_cols[i]} and "
                 f"{cluster_cols[j]}: {score:.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print(f"Rebuilding yearly RED indicators from {args.preprocessed_path}")
    ds = load_indicators(args.preprocessed_path)
    print("Computing severity trend significance")
    sig = compute_severity_significance(
        ds, gwls=[args.gwl], alpha=args.alpha, ssp=args.ssp,
        reference_gwl=args.reference_gwl, n_resamples=args.n_resamples, seed=args.seed,
    )
    incr, decr = agreement_fractions(sig, args.gwl)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building the model-disagreement co-occurrence matrix")
    matrix, disagreement_idx, increase_by_idx, decrease_by_idx = build_model_disagreement_matrix(
        sig, args.gwl, incr, decr, args.disagreement_threshold)
    print(f"  {len(disagreement_idx)} poly_idx in disagreement "
         f"(>{args.disagreement_threshold:.0%} increasing AND decreasing)")
    heatmap_path = os.path.join(args.output_dir, "model_disagreement_heatmap.png")
    plot_model_disagreement_heatmap(matrix, heatmap_path, args.dpi)
    print(f"  Saved {heatmap_path}")

    print("Clustering models on their disagreement pattern")
    elbow_path = os.path.join(args.output_dir, "model_clustering_elbow_silhouette.png")
    if cluster_models_elbow_silhouette(matrix, args.model_cluster_ks, elbow_path, args.dpi):
        print(f"  Saved {elbow_path}")

    if len(disagreement_idx) < args.n_region_clusters:
        print(f"Only {len(disagreement_idx)} disagreeing regions -- skipping region "
             f"clustering (needs >= {args.n_region_clusters})")
        return

    print(f"Loading distance matrix from {args.distance_matrix_txt}")
    dist_matrix = np.loadtxt(args.distance_matrix_txt, delimiter=",")
    df_clusters = cluster_disagreement_regions(
        dist_matrix, disagreement_idx, args.n_region_clusters, seed=args.seed)
    region_map_path = os.path.join(args.output_dir, "region_disagreement_clusters.png")
    plot_region_disagreement_clusters(
        args.shapefile, df_clusters, args.n_region_clusters,
        GWL_LABELS.get(args.gwl, args.gwl), region_map_path, args.dpi)
    print(f"  Saved {region_map_path}")

    print("Clustering models within each region cluster")
    df_disag = model_subclusters_by_region_cluster(
        sig, increase_by_idx, decrease_by_idx, df_clusters,
        args.n_region_clusters, args.n_model_subclusters)
    compare_region_cluster_model_groupings(df_disag)


if __name__ == "__main__":
    main()
