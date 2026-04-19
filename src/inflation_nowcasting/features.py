"""Feature engineering for inflation nowcasting."""

from __future__ import annotations

import pandas as pd

from inflation_nowcasting.config import Settings


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods) * 100


def build_training_table(source_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Create model-ready table with target and lagged features."""
    df = source_df.sort_index().copy()
    target_base = settings.target_base_column
    target_col = settings.target_column

    df[target_col] = _safe_pct_change(df[target_base], periods=1)

    # Create transformed features from non-target columns.
    predictor_cols = [c for c in df.columns if c not in {target_base, target_col}]
    for col in predictor_cols:
        df[f"{col}_mom"] = _safe_pct_change(df[col], periods=1)
        df[f"{col}_yoy"] = _safe_pct_change(df[col], periods=12)
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)

    # Lag the target itself to provide persistence signal.
    df[f"{target_col}_lag1"] = df[target_col].shift(1)
    df[f"{target_col}_lag3"] = df[target_col].shift(3)

    keep_cols = [target_col] + [c for c in df.columns if c not in {target_base, target_col}]
    model_df = df[keep_cols].dropna().copy()
    return model_df


def split_train_test(
    model_df: pd.DataFrame,
    target_col: str,
    test_months: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-aware split with the final `test_months` rows as holdout."""
    if test_months <= 0:
        raise ValueError("test_months must be positive")
    if len(model_df) <= test_months:
        raise ValueError("not enough rows for requested test split")

    X = model_df.drop(columns=[target_col])
    y = model_df[target_col]
    return X.iloc[:-test_months], X.iloc[-test_months:], y.iloc[:-test_months], y.iloc[-test_months:]
