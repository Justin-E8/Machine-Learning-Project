"""Model training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class ModelResult:
    name: str
    mae: float
    rmse: float


def _evaluate(estimator: object, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
    preds = estimator.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return mae, rmse


def train_and_select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[object, list[ModelResult], str]:
    """Train baseline models and return best model by holdout RMSE."""
    candidates: dict[str, object] = {
        "ridge": Ridge(alpha=1.0),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            random_state=42,
        ),
    }

    results: list[ModelResult] = []
    best_name = ""
    best_model: object | None = None
    best_rmse = float("inf")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        mae, rmse = _evaluate(model, X_test, y_test)
        results.append(ModelResult(name=name, mae=mae, rmse=rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("Model selection failed: no candidate model was trained.")

    return best_model, results, best_name


def save_model(model: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path: Path) -> object:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
