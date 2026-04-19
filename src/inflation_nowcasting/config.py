"""Configuration helpers for the inflation nowcasting project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    models_dir: Path
    raw_data_path: Path
    processed_table_path: Path
    model_path: Path
    metrics_path: Path
    latest_forecast_path: Path
    start_date: str = "1995-01-01"
    end_date: str | None = None
    target_base_column: str = "cpi_index"
    target_column: str = "cpi_mom"
    test_months: int = 24
    fred_series: dict[str, str] = field(
        default_factory=lambda: {
            "cpi_index": "CPIAUCSL",
            "unemployment_rate": "UNRATE",
            "payrolls": "PAYEMS",
            "fed_funds_rate": "FEDFUNDS",
            "ten_year_treasury": "DGS10",
            "two_year_treasury": "DGS2",
            "ppi": "PPIACO",
        }
    )


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    raw_data_dir = root / "data" / "raw"
    processed_data_dir = root / "data" / "processed"
    models_dir = root / "models"

    return Settings(
        root_dir=root,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        models_dir=models_dir,
        raw_data_path=raw_data_dir / "macro_data.csv",
        processed_table_path=processed_data_dir / "training_frame.csv",
        model_path=models_dir / "best_model.joblib",
        metrics_path=models_dir / "metrics.json",
        latest_forecast_path=models_dir / "latest_forecast.json",
    )
