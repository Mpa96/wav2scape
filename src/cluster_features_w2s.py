"""
Cluster HRT segments using XLSR-53 features from extract_features.py.

Two distinct pipelines depending on feature type
-------------------------------------------------

xlsr_hidden / xlsr_quantized  (CSV input)
    Continuous features → suitable for euclidean distance.
    Pipeline: StandardScaler → PCA → UMAP (euclidean) → HDBSCAN → UMAP 2D → plots.

xlsr_codebook  (PT + meta CSV input)
    Probability distributions (sum to 1) → JSD is the natural distance.
    This pipeline replicates wav2scape exactly, but per file instead of per category:

      Per-file histograms (M × 102400)
              ↓  pairwise JSD similarity (vectorised)
      M × M similarity matrix
              ↓  PCA(3)               ← wav2scape-exact scatter
      M × 3 coordinates
              ↓  scatter plot (PC1 vs PC2)

    For clustering the files:
      M × M JSD distance matrix
              ↓  UMAP(precomputed)
              ↓  HDBSCAN
      cluster labels + UMAP 2D scatter

Usage
-----
# CSV input (hidden or quantized):
python src/cluster_features_w2s.py \\
    --features_csv  bop_results/analysis/hrt_features_xlsr_hidden_both.csv \\
    --feature_type  xlsr_hidden \\
    --out_dir       bop_results/analysis/clustering

# PT input (codebook — exact wav2scape method per file):
python src/cluster_features_w2s.py \\
    --features_pt   bop_results/analysis/hrt_features_xlsr_codebook_both.pt \\
    --features_meta bop_results/analysis/hrt_features_xlsr_codebook_meta_both.csv \\
    --feature_type  xlsr_codebook \\
    --out_dir       bop_results/analysis/clustering
"""

import argparse
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

META_COLS = ["participant_id", "chunk_id", "phase", "start_s", "end_s", "duration_s"]
PCA_SKIP_THRESHOLD = 100  # skip PCA in continuous pipeline below this n_features


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_features(
    features_csv: str | None,
    features_pt: str | None,
    features_meta: str | None,
    phase: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if features_pt is not None:
        tensor = torch.load(features_pt, weights_only=True)
        X = tensor.numpy().astype(np.float32)
        df = pd.read_csv(features_meta)
        if phase != "both":
            mask = (df["phase"] == phase).values
            df = df[mask].reset_index(drop=True)
            X = X[mask]
    else:
        df = pd.read_csv(features_csv)
        if phase != "both":
            df = df[df["phase"] == phase].reset_index(drop=True)
        feat_cols = [c for c in df.columns if c not in META_COLS]
        X = df[feat_cols].values.astype(np.float32)
        valid = ~np.isnan(X).any(axis=1)
        if (~valid).sum() > 0:
            print(f"Dropping {(~valid).sum()} rows with NaN features.")
            df = df[valid].reset_index(drop=True)
            X = X[valid]

    return df, X


# ---------------------------------------------------------------------------
# Continuous pipeline  (xlsr_hidden / xlsr_quantized)
# ---------------------------------------------------------------------------

def run_continuous_pipeline(
    df: pd.DataFrame,
    X: np.ndarray,
    n_pca: int,
    n_umap_cluster: int,
    min_cluster_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    StandardScaler → PCA → UMAP (euclidean) → HDBSCAN → UMAP 2D.
    Returns df with cluster/umap_x/umap_y columns, and PCA-reduced X.
    """
    print(f"  {len(df)} segments, {X.shape[1]} features")

    print("Normalising ...")
    X = StandardScaler().fit_transform(X)

    if X.shape[1] > PCA_SKIP_THRESHOLD:
        n = min(n_pca, X.shape[0] - 1, X.shape[1])
        print(f"PCA → {n} components ...")
        pca = PCA(n_components=n, random_state=42)
        X_pca = pca.fit_transform(X)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    else:
        print(f"Skipping PCA ({X.shape[1]} features < threshold {PCA_SKIP_THRESHOLD})")
        X_pca = X

    print(f"UMAP → {n_umap_cluster} dimensions (clustering) ...")
    X_umap_cluster = umap.UMAP(
        n_components=n_umap_cluster, n_neighbors=30, min_dist=0.0,
        metric="euclidean", random_state=random_state,
    ).fit_transform(X_pca)

    print(f"HDBSCAN (min_cluster_size={min_cluster_size}) ...")
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=5,
        metric="euclidean", cluster_selection_method="eom",
    ).fit_predict(X_umap_cluster)
    _print_label_stats(labels)

    print("UMAP → 2D (visualisation) ...")
    X_2d = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="euclidean", random_state=random_state,
    ).fit_transform(X_pca)

    df = df.copy()
    df["cluster"] = labels
    df["umap_x"] = X_2d[:, 0]
    df["umap_y"] = X_2d[:, 1]
    return df, X_pca


# ---------------------------------------------------------------------------
# Codebook pipeline  (xlsr_codebook — wav2scape-exact)
# ---------------------------------------------------------------------------

def jsd_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise JSD similarity matrix exactly as wav2scape does.

    wav2scape formula (natural log):
        JSD_div(P, Q)  = 0.5 * (KL(P||M) + KL(Q||M)),  M = (P+Q)/2
        similarity     = 1 - JSD_div

    Uses scipy cdist for vectorised computation instead of nested Python loops.
    scipy jensenshannon returns sqrt(JSD_nat), so JSD_div = dist².
    """
    print(f"  Computing {X.shape[0]}×{X.shape[0]} JSD similarity matrix ...")
    # jensenshannon with no base uses natural log → values in [0, sqrt(ln2)]
    dists = cdist(X, X, metric="jensenshannon").astype(np.float32)
    jsd_div = dists ** 2  # ∈ [0, ln2 ≈ 0.693]
    return (1.0 - jsd_div)  # similarity ∈ [~0.307, 1.0]


def run_codebook_pipeline(
    df: pd.DataFrame,
    X: np.ndarray,
    n_umap_cluster: int,
    min_cluster_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Exact wav2scape pipeline, applied per file instead of per category:

        histograms (M × 102400)
            → M × M JSD similarity matrix
            → PCA(3) on similarity matrix          (wav2scape-exact scatter)
            → UMAP(precomputed JSD) → HDBSCAN      (clustering)
            → UMAP 2D                               (visualisation)

    Returns df with cluster/umap_x/umap_y, and the PCA-reduced coords (M × 3).
    """
    print(f"  {len(df)} segments, {X.shape[1]}-dim codebook histograms")

    sim = jsd_similarity_matrix(X)

    # wav2scape PCA: on the M×M similarity matrix (not on raw histograms)
    n_pca_wav2scape = min(3, sim.shape[0] - 1)
    print(f"PCA({n_pca_wav2scape}) on {sim.shape[0]}×{sim.shape[0]} similarity matrix ...")
    pca = PCA(n_components=n_pca_wav2scape, random_state=42)
    X_pca = pca.fit_transform(sim)
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {v:.1%} variance")

    # JSD distance matrix for UMAP / HDBSCAN
    # sqrt(JSD) is a proper metric (jensenshannon distance)
    jsd_dist = np.sqrt(np.maximum(1.0 - sim, 0.0)).astype(np.float32)

    print(f"UMAP({n_umap_cluster}D, precomputed JSD) ...")
    X_umap_cluster = umap.UMAP(
        n_components=n_umap_cluster, n_neighbors=30, min_dist=0.0,
        metric="precomputed", random_state=random_state,
    ).fit_transform(jsd_dist)

    print(f"HDBSCAN (min_cluster_size={min_cluster_size}) ...")
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=5,
        metric="precomputed", cluster_selection_method="eom",
    ).fit_predict(jsd_dist)
    _print_label_stats(labels)

    print("UMAP → 2D (visualisation, precomputed JSD) ...")
    X_2d = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="precomputed", random_state=random_state,
    ).fit_transform(jsd_dist)

    df = df.copy()
    df["cluster"] = labels
    df["umap_x"] = X_2d[:, 0]
    df["umap_y"] = X_2d[:, 1]
    return df, X_pca


def _print_label_stats(labels: np.ndarray) -> None:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels):.1%})")
    for c in sorted(set(labels)):
        print(f"    {'noise' if c==-1 else f'cluster_{c}'}: {(labels==c).sum()} segments")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _cluster_palette(clusters: np.ndarray) -> dict:
    unique = sorted(set(clusters))
    colours = sns.color_palette("tab10", n_colors=max(sum(c != -1 for c in unique), 1))
    palette = {-1: "#aaaaaa"}
    for i, c in enumerate(c for c in unique if c != -1):
        palette[c] = colours[i]
    return palette


def _scatter_cluster(ax, x, y, df, palette):
    for c, grp in df.groupby("cluster"):
        idx = grp.index
        label = "noise" if c == -1 else f"cluster {c} (n={len(grp)})"
        ax.scatter(x[idx], y[idx], c=palette[c], s=4, alpha=0.6, label=label, rasterized=True)
    ax.legend(markerscale=3, fontsize=8)


def _scatter_participant(ax, x, y, df):
    participants = sorted(df["participant_id"].unique())
    colours = sns.color_palette("husl", n_colors=len(participants))
    for pid, colour in zip(participants, colours):
        idx = df.index[df["participant_id"] == pid]
        ax.scatter(x[idx], y[idx], c=[colour], s=4, alpha=0.6, label=str(pid), rasterized=True)
    ax.legend(markerscale=3, fontsize=7, ncol=2)


def plot_scatter(
    df: pd.DataFrame,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_label: str,
    y_label: str,
    title_prefix: str,
    out_path: Path,
    phase: str,
    feature_type: str,
) -> None:
    """Generic 2D scatter with cluster / participant / duration panels."""
    has_participant = "participant_id" in df.columns
    n_panels = 3 if has_participant else 2
    palette = _cluster_palette(df["cluster"].values)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    fig.suptitle(f"{title_prefix} — {feature_type} — phase: {phase}", fontsize=13)

    ax = axes[0]
    _scatter_cluster(ax, x_coords, y_coords, df, palette)
    ax.set_title("HDBSCAN clusters")
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)

    if has_participant:
        ax = axes[1]
        _scatter_participant(ax, x_coords, y_coords, df)
        ax.set_title("Coloured by participant")
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)

    ax = axes[-1]
    sc = ax.scatter(x_coords, y_coords, c=df["duration_s"],
                    cmap="coolwarm", s=4, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="duration (s)")
    ax.set_title("Coloured by duration")
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Cluster statistics
# ---------------------------------------------------------------------------

def print_cluster_stats(df: pd.DataFrame) -> None:
    print("\n--- Cluster statistics ---")
    for c in sorted(df["cluster"].unique()):
        grp = df[df["cluster"] == c]
        label = "noise" if c == -1 else f"cluster {c}"
        dur = grp["duration_s"]
        n_p = grp["participant_id"].nunique() if "participant_id" in grp.columns else "n/a"
        print(
            f"{label:12s}  n={len(grp):5d}  "
            f"dur mean={dur.mean():.3f}s  median={dur.median():.3f}s  "
            f"[{dur.min():.3f}–{dur.max():.3f}s]  participants={n_p}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster HRT segments using XLSR-53 features."
    )
    parser.add_argument("--features_csv",  default=None,
                        help="CSV feature file (xlsr_hidden or xlsr_quantized).")
    parser.add_argument("--features_pt",   default=None,
                        help=".pt tensor file (xlsr_codebook, shape N×102400).")
    parser.add_argument("--features_meta", default=None,
                        help="Companion _meta.csv for --features_pt.")
    parser.add_argument("--feature_type",     default="xlsr_hidden",
                        choices=["xlsr_hidden", "xlsr_quantized", "xlsr_codebook"])
    parser.add_argument("--out_dir",          default="bop_results/analysis/clustering")
    parser.add_argument("--phase",            default="both",
                        choices=["trial1", "trial2", "both"])
    parser.add_argument("--min_cluster_size", type=int, default=15)
    parser.add_argument("--n_pca",            type=int, default=40,
                        help="PCA components for continuous pipeline (hidden/quantized). "
                             "Not used for codebook (PCA is on M×M similarity matrix).")
    parser.add_argument("--n_umap_cluster",   type=int, default=15)
    parser.add_argument("--random_state",     type=int, default=42)
    args = parser.parse_args()

    if args.features_pt is None and args.features_csv is None:
        parser.error("Provide either --features_csv or --features_pt + --features_meta.")
    if args.features_pt is not None and args.features_meta is None:
        parser.error("--features_pt requires --features_meta.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features (phase={args.phase}) ...")
    df, X = load_features(
        args.features_csv, args.features_pt, args.features_meta, args.phase
    )

    ft = args.feature_type
    kw = dict(min_cluster_size=args.min_cluster_size,
              n_umap_cluster=args.n_umap_cluster,
              random_state=args.random_state)

    if ft == "xlsr_codebook":
        df, X_pca = run_codebook_pipeline(df, X, **kw)

        # wav2scape-style PCA scatter (PC1 vs PC2)
        plot_scatter(
            df, X_pca[:, 0], X_pca[:, 1],
            "PC1", "PC2", "PCA scatter (wav2scape-exact, per file)",
            out_dir / f"pca_scatter_{args.phase}_{ft}.png",
            args.phase, ft,
        )
        # PC1 vs PC3 (if available)
        if X_pca.shape[1] >= 3:
            plot_scatter(
                df, X_pca[:, 0], X_pca[:, 2],
                "PC1", "PC3", "PCA scatter (wav2scape-exact, per file)",
                out_dir / f"pca_scatter_13_{args.phase}_{ft}.png",
                args.phase, ft,
            )
    else:
        df, X_pca = run_continuous_pipeline(df, X, n_pca=args.n_pca, **kw)

        # PCA 2D scatter
        if X_pca.shape[1] >= 2:
            plot_scatter(
                df, X_pca[:, 0], X_pca[:, 1],
                "PC1", "PC2", "PCA scatter",
                out_dir / f"pca_scatter_{args.phase}_{ft}.png",
                args.phase, ft,
            )

    # UMAP scatter (both pipelines produce umap_x/umap_y)
    plot_scatter(
        df, df["umap_x"].values, df["umap_y"].values,
        "UMAP-1", "UMAP-2", "UMAP scatter",
        out_dir / f"umap_{args.phase}_{ft}.png",
        args.phase, ft,
    )

    print_cluster_stats(df)

    save_cols = [c for c in META_COLS if c in df.columns] + ["cluster", "umap_x", "umap_y"]
    out_csv = out_dir / f"hrt_clustered_{args.phase}_{ft}.csv"
    df[save_cols].to_csv(out_csv, index=False)
    print(f"Saved labelled segments → {out_csv}")


if __name__ == "__main__":
    main()
