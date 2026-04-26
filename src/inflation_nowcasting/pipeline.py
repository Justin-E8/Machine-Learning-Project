"""High-level orchestration for training and forecasting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

from inflation_nowcasting.config import Settings
from inflation_nowcasting.data_loader import fetch_macro_data
from inflation_nowcasting.features import build_training_table, split_train_test
from inflation_nowcasting.modeling import load_model, save_model, train_and_select_model


@dataclass
class ForecastResult:
    target_date: pd.Timestamp
    prediction_mom_percent: float
    prediction_annualized_percent: float
    model_name: str


def run_training_pipeline(settings: Settings, refresh_data: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Execute the baseline training pipeline.

    Returns:
      metrics_df: metrics for all candidate models.
      best_model_name: selected model identifier.
    """
    raw = fetch_macro_data(settings=settings, refresh_data=refresh_data)
    table = build_training_table(raw, settings=settings)

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(settings.processed_table_path, index=True)

    X_train, X_test, y_train, y_test = split_train_test(
        table, target_col=settings.target_column, test_months=settings.test_months
    )
    best_model, results, best_name = train_and_select_model(X_train, y_train, X_test, y_test)

    save_model(best_model, settings.model_path)
    metrics_df = pd.DataFrame([asdict(r) for r in results]).sort_values("rmse")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.write_text(metrics_df.to_json(orient="records", indent=2), encoding="utf-8")

    return metrics_df, best_name


def generate_latest_forecast(settings: Settings, model_name: str) -> ForecastResult:
    """Generate one-step-ahead forecast from the latest feature row."""
    raw = fetch_macro_data(settings=settings, refresh_data=False)
    table = build_training_table(raw, settings=settings)
    latest_X = table.drop(columns=[settings.target_column]).iloc[[-1]]
    latest_date = latest_X.index[-1] + pd.offsets.MonthEnd(1)

    model = load_model(settings.model_path)
    prediction = float(model.predict(latest_X)[0])
    annualized = ((1.0 + prediction / 100.0) ** 12 - 1.0) * 100.0

    result = ForecastResult(
        target_date=pd.Timestamp(latest_date),
        prediction_mom_percent=prediction,
        prediction_annualized_percent=annualized,
        model_name=model_name,
    )

    settings.latest_forecast_path.write_text(
        json.dumps(
            {
                "target_date": result.target_date.date().isoformat(),
                "prediction_mom_percent": result.prediction_mom_percent,
                "prediction_annualized_percent": result.prediction_annualized_percent,
                "model_name": result.model_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result

