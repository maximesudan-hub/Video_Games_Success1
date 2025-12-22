from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np


@dataclass(frozen=True)
class ProjectSpec:
    target_col: str = "Log_Sales"
    success_threshold: float = 1.0
    test_size: float = 0.2
    random_state: int = 42


# --- Features we will use (no leakage)
NUM_FEATURES = [
    "Year_of_Release",
    "Critic_Score",
    "Critic_Count",
    "User_Score_100",
    "User_Count",
]

CAT_FEATURES = [
    "Platform",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
]


def _require_columns(df: "pd.DataFrame", columns: list[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {sorted(missing)}")


def make_feature_matrix(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return feature matrix used across tasks."""
    _require_columns(df, NUM_FEATURES + CAT_FEATURES)
    return df[NUM_FEATURES + CAT_FEATURES].copy()


def make_regression_xy(df: "pd.DataFrame", spec: ProjectSpec) -> Tuple["pd.DataFrame", "pd.Series"]:
    """Build X, y for regression. Avoid leakage columns."""
    _require_columns(df, NUM_FEATURES + CAT_FEATURES + [spec.target_col])
    X = make_feature_matrix(df)
    y = df[spec.target_col].copy()
    return X, y


def make_classification_xy(df: "pd.DataFrame", spec: ProjectSpec) -> Tuple["pd.DataFrame", "pd.Series"]:
    """Build X, y for classification (success = Global_Sales > threshold)."""
    _require_columns(df, NUM_FEATURES + CAT_FEATURES + ["Global_Sales"])
    X = make_feature_matrix(df)
    y = (df["Global_Sales"] > spec.success_threshold).astype(int)
    return X, y


def split_train_test(
    X: "pd.DataFrame",
    y: "pd.Series",
    spec: ProjectSpec,
    stratify: Optional["pd.Series"] = None,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.Series", "pd.Series"]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X,
        y,
        test_size=spec.test_size,
        random_state=spec.random_state,
        stratify=stratify,
    )


def _linear_preprocessor() -> Any:
    """Preprocess for OLS/Ridge: impute + scale numeric, one-hot categorical."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_FEATURES),
            ("cat", categorical_pipe, CAT_FEATURES),
        ]
    )


def _tree_preprocessor() -> Any:
    """
    Preprocess for tree/boosting:
    - we use OrdinalEncoder for categoricals so HistGradientBoosting can work efficiently.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_FEATURES),
            ("cat", categorical_pipe, CAT_FEATURES),
        ]
    )


def _cluster_preprocessor() -> Any:
    """Preprocess for KMeans/SVD: impute + one-hot, keep sparse output."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_FEATURES),
            ("cat", categorical_pipe, CAT_FEATURES),
        ]
    )


def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
    """Return model pipelines."""
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import Pipeline

    lin_pre = _linear_preprocessor()
    tree_pre = _tree_preprocessor()

    models: Dict[str, Any] = {
        "OLS": Pipeline(
            steps=[
                ("preprocess", lin_pre),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("preprocess", lin_pre),
                ("model", Ridge(alpha=1.0, random_state=None)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=None,
                    max_iter=500,
                    random_state=random_state,
                )),
            ]
        ),
    }
    return models


def get_classification_models(random_state: int = 42) -> Dict[str, Any]:
    """Return classification pipelines."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    lin_pre = _linear_preprocessor()
    tree_pre = _tree_preprocessor()

    models: Dict[str, Any] = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", lin_pre),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]
        ),
    }
    return models


def get_clustering_pipeline(n_clusters: int = 4, random_state: int = 42) -> Any:
    """Return preprocessing + MiniBatchKMeans pipeline."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocess", _cluster_preprocessor()),
            ("model", MiniBatchKMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)),
        ]
    )


def get_pca_pipeline() -> Any:
    """Return preprocessing + TruncatedSVD(2) pipeline for visualization."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocess", _cluster_preprocessor()),
            ("svd", TruncatedSVD(n_components=2, random_state=42)),
        ]
    )


def train_and_predict_regression(
    model: Any,
    X_train: "pd.DataFrame",
    y_train: "pd.Series",
    X_test: "pd.DataFrame",
) -> "np.ndarray":
    """Fit one model and return y_pred on the test set."""
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_and_predict_classifier(
    model: Any,
    X_train: "pd.DataFrame",
    y_train: "pd.Series",
    X_test: "pd.DataFrame",
) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    """Fit one model and return y_pred and y_proba (if available)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.shape[1] >= 2:
            y_proba = proba[:, 1]
    return y_pred, y_proba


def fit_predict_clusters(model: Any, X: "pd.DataFrame") -> "np.ndarray":
    """Fit KMeans pipeline and return cluster labels."""
    return model.fit_predict(X)
