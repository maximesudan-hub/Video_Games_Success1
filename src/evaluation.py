from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import os

import numpy as np
import pandas as pd
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "results/.mpl_cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    from sklearn.metrics import mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: Dict[str, float] = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def save_regression_metrics_table(
    metrics_by_model: Dict[str, Dict[str, float]],
    out_path: Path,
) -> pd.DataFrame:
    """Save regression metrics to CSV, sorted by RMSE."""
    df = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    df = df.sort_values(by="RMSE", ascending=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return df


def save_classification_metrics_table(
    metrics_by_model: Dict[str, Dict[str, float]],
    out_path: Path,
) -> pd.DataFrame:
    """Save classification metrics to CSV, sorted by F1."""
    df = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    df = df.sort_values(by="F1", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return df


def save_clustering_metrics(metrics: Dict[str, float], out_path: Path) -> pd.DataFrame:
    """Save clustering metrics to CSV."""
    df = pd.DataFrame([metrics])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Plot predicted vs true values."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Plot confusion matrix."""
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    disp.figure_.tight_layout()
    disp.figure_.savefig(out_path, dpi=150)
    plt.close(disp.figure_)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Plot ROC curve."""
    from sklearn.metrics import RocCurveDisplay

    disp = RocCurveDisplay.from_predictions(y_true, y_proba)
    disp.figure_.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    disp.figure_.tight_layout()
    disp.figure_.savefig(out_path, dpi=150)
    plt.close(disp.figure_)


def clustering_metrics(
    X_transformed: np.ndarray,
    labels: np.ndarray,
    model: Optional[object] = None,
) -> Dict[str, float]:
    """Compute clustering metrics (inertia + silhouette when possible)."""
    from sklearn.metrics import silhouette_score

    metrics: Dict[str, float] = {}
    if model is not None and hasattr(model, "inertia_"):
        metrics["Inertia"] = float(model.inertia_)
    if len(np.unique(labels)) > 1:
        metrics["Silhouette"] = float(silhouette_score(X_transformed, labels))
    return metrics


def plot_pca_clusters(
    X_2d: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Plot PCA scatter with cluster labels."""
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Plot top-N feature importances."""
    top = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"], color="#4c78a8")
    plt.xlabel("Importance")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_k_selection(
    k_values: list[int],
    silhouette_scores: list[float],
    inertias: list[float],
    out_path: Path,
    title: str,
) -> None:
    """Plot silhouette and inertia for K selection."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(k_values, silhouette_scores, marker="o", color="#4c78a8", label="Silhouette")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette")
    ax1.set_ylim(0, max(silhouette_scores) * 1.1 if silhouette_scores else 1)

    ax2 = ax1.twinx()
    ax2.plot(k_values, inertias, marker="s", color="#f58518", label="Inertia")
    ax2.set_ylabel("Inertia")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cluster_profile(
    profile_df: pd.DataFrame,
    out_path: Path,
    title: str,
    value_col: str = "mean_Global_Sales",
) -> None:
    """Plot a simple cluster profile bar chart."""
    plt.figure(figsize=(7, 4))
    plt.bar(profile_df["cluster"].astype(str), profile_df[value_col], color="#54a24b")
    plt.xlabel("Cluster")
    plt.ylabel(value_col.replace("_", " "))
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cluster_profile_multi(
    profile_df: pd.DataFrame,
    out_path: Path,
    title: str,
    metrics: list[str],
    labels: list[str],
) -> None:
    """Plot multiple cluster profile metrics in a 2x2 grid."""
    if len(metrics) != len(labels):
        raise ValueError("metrics and labels must have the same length")
    n = len(metrics)
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
    axes = axes.flatten()

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        ax.bar(profile_df["cluster"].astype(str), profile_df[metric], color="#4c78a8")
        ax.set_title(label)
        ax.set_xlabel("Cluster")
    # Hide any unused axes
    for j in range(n, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cluster_profile_summary(
    profile_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    """Plot a readable cluster summary: sales bars + top categories table."""
    fig = plt.figure(figsize=(9, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3])

    # Left: mean sales bar chart
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(
        profile_df["cluster"].astype(str),
        profile_df["mean_Global_Sales"],
        color="#4c78a8",
    )
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Mean Global Sales")
    ax_bar.set_title("Sales by Cluster")

    # Right: table with top categories
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    def _shorten(text: str, max_len: int = 28) -> str:
        text = str(text)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    table_df = profile_df[["cluster", "top_platforms", "top_genres", "top_publishers"]].copy()
    table_df["top_platforms"] = table_df["top_platforms"].map(_shorten)
    table_df["top_genres"] = table_df["top_genres"].map(_shorten)
    table_df["top_publishers"] = table_df["top_publishers"].map(_shorten)

    table = ax_tbl.table(
        cellText=table_df.values,
        colLabels=["Cluster", "Top Platforms", "Top Genres", "Top Publishers"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax_tbl.set_title("Top Categories", pad=6)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cluster_profile_metric(
    profile_df: pd.DataFrame,
    out_path: Path,
    title: str,
    metric: str,
    y_label: str,
) -> None:
    """Plot a single cluster metric as a clean bar chart."""
    plt.figure(figsize=(6, 4))
    plt.bar(profile_df["cluster"].astype(str), profile_df[metric], color="#4c78a8")
    plt.xlabel("Cluster")
    plt.ylabel(y_label)
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
