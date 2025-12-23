from pathlib import Path
import numpy as np
import pandas as pd

from src.models import (
    ProjectSpec,
    make_regression_xy,
    make_classification_xy,
    split_train_test,
    get_regression_models,
    get_classification_models,
    get_cluster_numeric_features,
    train_and_predict_regression,
    train_and_predict_classifier,
)
from src.evaluation import (
    regression_metrics,
    save_regression_metrics_table,
    plot_pred_vs_true,
    classification_metrics,
    save_classification_metrics_table,
    plot_roc_curve,
    clustering_metrics,
    save_clustering_metrics,
    plot_pca_clusters,
    plot_feature_importance,
    plot_cluster_profile_summary,
    plot_cluster_profile_metric,
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

    # Feature importance with RandomForest (interpretability)
    rf_model = get_regression_models(random_state=spec.random_state)["RandomForest"]
    rf_model.fit(X_train, y_train)
    rf_pre = rf_model.named_steps["preprocess"]
    rf_est = rf_model.named_steps["model"]
    feature_names = rf_pre.get_feature_names_out()
    importances = rf_est.feature_importances_
    label_map = {
        "User_Count": "User Count",
        "Critic_Count": "Critic Count",
        "Platform": "Platform",
        "Publisher": "Publisher",
        "Year_of_Release": "Release Year",
        "Critic_Score": "Critic Score",
        "Developer": "Developer",
        "Genre": "Genre",
        "User_Score_100": "User Score (0-100)",
        "Rating": "ESRB Rating",
    }
    pretty_names = []
    for name in feature_names:
        base = name.split("__", 1)[-1]
        pretty_names.append(label_map.get(base, base))
    fi_df = (
        pd.DataFrame({"feature": pretty_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_path = Path("results/metrics/feature_importance.csv")
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(fi_path, index=False)
    plot_feature_importance(
        fi_df,
        out_path=Path("results/figures/feature_importance.png"),
        title="Feature Importance (RandomForestRegressor)",
        top_n=20,
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
    if cls_probas[best_cls_name] is not None:
        plot_roc_curve(
            y_test.to_numpy(),
            cls_probas[best_cls_name],
            out_path=Path("results/figures/classification_roc_curve.png"),
            title=f"ROC Curve — {best_cls_name}",
        )

    # --- Clustering (numeric-only for clearer segments)
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    num_cols = get_cluster_numeric_features()
    X_cluster = df[num_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_cluster)
    X_transformed = scaler.fit_transform(X_imputed)

    k_values = list(range(2, 7))
    silhouettes = []
    inertias = []
    labels_by_k = {}
    for k in k_values:
        kmeans_model = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=spec.random_state)
        labels = kmeans_model.fit_predict(X_transformed)
        labels_by_k[k] = labels
        inertias.append(float(kmeans_model.inertia_))

        # silhouette on a subsample if large
        if X_transformed.shape[0] > 3000:
            rng = np.random.default_rng(spec.random_state)
            idx = rng.choice(X_transformed.shape[0], size=3000, replace=False)
            sil = clustering_metrics(X_transformed[idx], labels[idx]).get("Silhouette", 0.0)
        else:
            sil = clustering_metrics(X_transformed, labels).get("Silhouette", 0.0)
        silhouettes.append(sil)

    best_k = 3
    best_labels = labels_by_k[best_k]
    cluster_metrics = {
        "Best_k": best_k,
        "Silhouette": silhouettes[k_values.index(best_k)],
        "Inertia": inertias[k_values.index(best_k)],
    }
    cluster_metrics_path = Path("results/metrics/clustering_metrics.csv")
    save_clustering_metrics(cluster_metrics, cluster_metrics_path)
    # Profile clusters
    profile_rows = []
    for k in sorted(set(best_labels)):
        subset = df[best_labels == k]
        row = {
            "cluster": k,
            "size": len(subset),
            "mean_Global_Sales": subset["Global_Sales"].mean(),
            "mean_Log_Sales": subset["Log_Sales"].mean(),
            "mean_Critic_Score": subset["Critic_Score"].mean(),
            "mean_Critic_Count": subset["Critic_Count"].mean(),
            "mean_User_Score_100": subset["User_Score_100"].mean(),
            "mean_User_Count": subset["User_Count"].mean(),
            "mean_Year_of_Release": subset["Year_of_Release"].mean(),
            "top_platforms": ", ".join(subset["Platform"].value_counts().head(3).index),
            "top_genres": ", ".join(subset["Genre"].value_counts().head(3).index),
            "top_publishers": ", ".join(subset["Publisher"].value_counts().head(3).index),
        }
        profile_rows.append(row)
    profile_df = pd.DataFrame(profile_rows).sort_values(by="cluster")
    profile_path = Path("results/metrics/clustering_profile.csv")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(profile_path, index=False)
    plot_cluster_profile_summary(
        profile_df,
        out_path=Path("results/figures/clustering_profile_overview.png"),
        title=f"Cluster Overview (k={best_k})",
    )
    plot_cluster_profile_metric(
        profile_df,
        out_path=Path("results/figures/clustering_profile_sales.png"),
        title="Mean Global Sales by Cluster",
        metric="mean_Global_Sales",
        y_label="Mean Global Sales",
    )
    plot_cluster_profile_metric(
        profile_df,
        out_path=Path("results/figures/clustering_profile_critic_count.png"),
        title="Mean Critic Count by Cluster",
        metric="mean_Critic_Count",
        y_label="Mean Critic Count",
    )

    X_pca = PCA(n_components=2, random_state=spec.random_state).fit_transform(X_transformed)
    plot_pca_clusters(
        X_pca,
        best_labels,
        out_path=Path("results/figures/clustering_pca.png"),
        title=f"PCA Clusters (k={best_k})",
    )

    print("\nSaved:")
    print(f"- {metrics_path}")
    print("- results/figures/regression_pred_vs_true.png")
    print("- results/metrics/feature_importance.csv")
    print("- results/figures/feature_importance.png")
    print(f"- {cls_metrics_path}")
    print("- results/figures/classification_roc_curve.png")
    print(f"- {cluster_metrics_path}")
    print("- results/metrics/clustering_profile.csv")
    print("- results/figures/clustering_profile_overview.png")
    print("- results/figures/clustering_profile_sales.png")
    print("- results/figures/clustering_profile_critic_count.png")
    print("- results/figures/clustering_pca.png")


if __name__ == "__main__":
    main()
