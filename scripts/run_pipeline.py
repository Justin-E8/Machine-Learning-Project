"""Train model and immediately generate a forecast."""

from __future__ import annotations

import typer

from inflation_nowcasting.config import get_settings
from inflation_nowcasting.pipeline import generate_latest_forecast, run_training_pipeline

app = typer.Typer(add_completion=False)


@app.command()
def main(refresh_data: bool = typer.Option(False, help="Force redownload from FRED.")) -> None:
    settings = get_settings()
    metrics_df, best_model = run_training_pipeline(settings=settings, refresh_data=refresh_data)
    forecast = generate_latest_forecast(settings=settings, model_name=best_model)

    print("\nTraining complete. Validation metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSelected best model: {best_model}")
    print(
        f"Latest forecast month: {forecast.target_date.date()} | "
        f"predicted MoM CPI: {forecast.prediction_mom_percent:.2f}% | "
        f"annualized: {forecast.prediction_annualized_percent:.2f}%"
    )


if __name__ == "__main__":
    app()
