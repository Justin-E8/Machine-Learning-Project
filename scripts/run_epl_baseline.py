"""Run Step 1 baseline for Premier League outcome prediction."""

from __future__ import annotations

from pathlib import Path

import typer

from premier_league_predictor.baseline import run_step1_baseline

app = typer.Typer(add_completion=False)


@app.command()
def main(lookback: int = typer.Option(5, min=2, help="Number of prior matches per team.")) -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_step1_baseline(project_root=project_root, lookback=lookback)


if __name__ == "__main__":
    app()
