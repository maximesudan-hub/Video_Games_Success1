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
