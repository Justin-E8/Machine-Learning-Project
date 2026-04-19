"""Generate a latest inflation forecast from trained model artifacts."""

from __future__ import annotations

import typer

from inflation_nowcasting.config import get_settings
from inflation_nowcasting.pipeline import generate_latest_forecast

app = typer.Typer(add_completion=False)


@app.command()
def main(model_name: str = typer.Option("best_model", help="Label stored in forecast artifact.")) -> None:
    settings = get_settings()
    forecast = generate_latest_forecast(settings=settings, model_name=model_name)
    print(
        f"Latest forecast month: {forecast.target_date.date()} | "
        f"predicted MoM CPI: {forecast.prediction_mom_percent:.2f}% | "
        f"annualized: {forecast.prediction_annualized_percent:.2f}%"
    )


if __name__ == "__main__":
    app()
