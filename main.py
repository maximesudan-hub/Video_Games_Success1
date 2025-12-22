from pathlib import Path
import pandas as pd

from src.models import (
    ProjectSpec,
    make_regression_xy,
    make_classification_xy,
    make_feature_matrix,
    split_train_test,
    get_regression_models,
    get_classification_models,
    get_clustering_pipeline,
    get_pca_pipeline,
    train_and_predict_regression,
    train_and_predict_classifier,
    fit_predict_clusters,
)
from src.evaluation import (
    regression_metrics,
    save_regression_metrics_table,
    plot_pred_vs_true,
    classification_metrics,
    save_classification_metrics_table,
    plot_confusion_matrix,
    plot_roc_curve,
    clustering_metrics,
    save_clustering_metrics,
    plot_pca_clusters,
)


def main() -> None:
    data_path = Path("data/processed/games_modern.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/games_modern.csv. Run: python -m src.data_loader"
        )

    df = pd.read_csv(data_path)

    spec = ProjectSpec(target_col="Log_Sales", success_threshold=1.0, test_size=0.2, random_state=42)

    # --- Regression
    X_reg, y_reg = make_regression_xy(df, spec)
    X_train, X_test, y_train, y_test = split_train_test(X_reg, y_reg, spec)

    all_metrics = {}
    all_preds = {}
    for name, model in get_regression_models(random_state=spec.random_state).items():
        y_pred = train_and_predict_regression(model, X_train, y_train, X_test)
        all_preds[name] = y_pred
        all_metrics[name] = regression_metrics(y_test.to_numpy(), y_pred)

    metrics_path = Path("results/metrics/regression_metrics.csv")
    metrics_df = save_regression_metrics_table(all_metrics, metrics_path)
    print("\nRegression metrics (on Log_Sales):")
    print(metrics_df)

    # Save one diagnostic plot for the best RMSE model
    best_model_name = metrics_df.index[0]
    y_pred_best = all_preds[best_model_name]
    plot_pred_vs_true(
        y_test.to_numpy(),
        y_pred_best,
        out_path=Path("results/figures/regression_pred_vs_true.png"),
        title=f"Predicted vs True (Log_Sales) — {best_model_name}",
    )

    # --- Classification
    X_cls, y_cls = make_classification_xy(df, spec)
    X_train, X_test, y_train, y_test = split_train_test(X_cls, y_cls, spec, stratify=y_cls)

    cls_metrics = {}
    cls_preds = {}
    cls_probas = {}
    for name, model in get_classification_models(random_state=spec.random_state).items():
        y_pred, y_proba = train_and_predict_classifier(model, X_train, y_train, X_test)
        cls_preds[name] = y_pred
        cls_probas[name] = y_proba
        cls_metrics[name] = classification_metrics(y_test.to_numpy(), y_pred, y_proba)

    cls_metrics_path = Path("results/metrics/classification_metrics.csv")
    cls_df = save_classification_metrics_table(cls_metrics, cls_metrics_path)
    print("\nClassification metrics (success = Global_Sales > 1M):")
    print(cls_df)

    best_cls_name = cls_df.index[0]
    plot_confusion_matrix(
        y_test.to_numpy(),
        cls_preds[best_cls_name],
        out_path=Path("results/figures/classification_confusion_matrix.png"),
        title=f"Confusion Matrix — {best_cls_name}",
    )
    if cls_probas[best_cls_name] is not None:
        plot_roc_curve(
            y_test.to_numpy(),
            cls_probas[best_cls_name],
            out_path=Path("results/figures/classification_roc_curve.png"),
            title=f"ROC Curve — {best_cls_name}",
        )

    # --- Clustering
    X_cluster = make_feature_matrix(df)
    kmeans_pipe = get_clustering_pipeline(n_clusters=4, random_state=spec.random_state)
    labels = fit_predict_clusters(kmeans_pipe, X_cluster)
    X_transformed = kmeans_pipe.named_steps["preprocess"].transform(X_cluster)
    kmeans_model = kmeans_pipe.named_steps["model"]
    cluster_metrics = clustering_metrics(X_transformed, labels, model=kmeans_model)
    cluster_metrics_path = Path("results/metrics/clustering_metrics.csv")
    save_clustering_metrics(cluster_metrics, cluster_metrics_path)

    pca_pipe = get_pca_pipeline()
    X_pca = pca_pipe.fit_transform(X_cluster)
    plot_pca_clusters(
        X_pca,
        labels,
        out_path=Path("results/figures/clustering_pca.png"),
        title="PCA Clusters (KMeans)",
    )

    print("\nSaved:")
    print(f"- {metrics_path}")
    print("- results/figures/regression_pred_vs_true.png")
    print(f"- {cls_metrics_path}")
    print("- results/figures/classification_confusion_matrix.png")
    print("- results/figures/classification_roc_curve.png")
    print(f"- {cluster_metrics_path}")
    print("- results/figures/clustering_pca.png")


if __name__ == "__main__":
    main()
