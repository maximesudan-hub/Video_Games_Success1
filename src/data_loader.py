# src/data_loader.py
# RAW Kaggle (2016) -> games_modern.csv (clean + filter >=2005)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    raw_path: Path
    processed_path: Path


def load_raw_games(paths: DataPaths) -> pd.DataFrame:
    """Load raw Kaggle dataset."""
    df = pd.read_csv(paths.raw_path)
    return df


def process_raw_to_modern(df: pd.DataFrame, year_min: int = 2005, trim_top_quantile: float = 0.99) -> pd.DataFrame:
    """
    Clean raw Kaggle data and keep modern games (>= year_min).

    Steps:
    - drop missing Year_of_Release
    - filter Year_of_Release >= year_min
    - keep Global_Sales > 0
    - optional: remove extreme outliers (top 1% by Global_Sales)
    - convert scores/counts to numeric
    - create User_Score_100 and Log_Sales
    - drop rows missing essential metadata
    """
    df = df.copy()

    # --- Basic sanity / required columns
    required_cols = {"Name", "Platform", "Year_of_Release", "Genre", "Publisher", "Global_Sales"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    # --- Year filter
    df = df.dropna(subset=["Year_of_Release"])
    df = df[df["Year_of_Release"] >= year_min]

    # --- Sales filter
    df = df[df["Global_Sales"] > 0]

    # Remove extreme outliers (optional but consistent with your current approach)
    if 0 < trim_top_quantile < 1:
        cap = df["Global_Sales"].quantile(trim_top_quantile)
        df = df[df["Global_Sales"] < cap]

    # --- Convert rating-related columns to numeric (handles "tbd" etc.)
    for col in ["Critic_Score", "User_Score", "User_Count", "Critic_Count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # User score scaled to 0-100 if available
    if "User_Score" in df.columns:
        df["User_Score_100"] = df["User_Score"] * 10

    # Log target (useful for regression)
    df["Log_Sales"] = np.log1p(df["Global_Sales"])

    # --- Drop rows without essential metadata
    essential = ["Name", "Platform", "Genre", "Publisher", "Year_of_Release"]
    df = df.dropna(subset=essential)

    # (Optional) reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


def save_processed(df: pd.DataFrame, paths: DataPaths) -> None:
    """Save processed dataset."""
    paths.processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.processed_path, index=False)


def build_games_modern(
    raw_path: str | Path = Path("data/raw/Video_Games_Sales_as_at_22_Dec_2016.csv"),
    out_path: str | Path = Path("data/processed/games_modern.csv"),
    year_min: int = 2005,
    trim_top_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    End-to-end builder: raw -> processed modern csv.
    Returns the processed DataFrame.
    """
    paths = DataPaths(raw_path=Path(raw_path), processed_path=Path(out_path))
    df_raw = load_raw_games(paths)
    df_modern = process_raw_to_modern(df_raw, year_min=year_min, trim_top_quantile=trim_top_quantile)
    save_processed(df_modern, paths)
    return df_modern


if __name__ == "__main__":
    df_modern = build_games_modern()
    print(f"[OK] games_modern.csv created. shape={df_modern.shape}")
    print(df_modern[["Year_of_Release", "Global_Sales", "Log_Sales"]].describe())
