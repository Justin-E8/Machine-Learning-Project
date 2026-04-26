"""CLI entrypoint to run data prep and model training."""

from __future__ import annotations

import typer

from inflation_nowcasting.config import get_settings
from inflation_nowcasting.pipeline import run_training_pipeline

app = typer.Typer(add_completion=False)


@app.command()
def main(refresh_data: bool = typer.Option(False, help="Force redownload from FRED.")) -> None:
    settings = get_settings()
    metrics_df, best_model = run_training_pipeline(settings=settings, refresh_data=refresh_data)
    print("\nTraining complete. Validation metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSelected best model: {best_model}")
    print(f"Model saved to: {settings.model_path}")
    print(f"Metrics saved to: {settings.metrics_path}")


if __name__ == "__main__":
    app()
