"""Run Step 1 baseline for Premier League outcome prediction."""

from __future__ import annotations

from pathlib import Path

import typer

from premier_league_predictor.baseline import run_step1_baseline

app = typer.Typer(add_completion=False)


@app.command()
def main(
    lookback: int = typer.Option(5, min=3, help="Number of prior matches per team."),
    strength_window: int = typer.Option(20, min=5, help="Window size for persistent strength."),
    home_away_lookback: int = typer.Option(
        3,
        min=2,
        help="Recent home-only / away-only window size.",
    ),
    elo_season_decay: float = typer.Option(
        0.75,
        min=0.0,
        max=1.0,
        help="Season-to-season Elo carryover (0 reset, 1 full carry).",
    ),
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_step1_baseline(
        project_root=project_root,
        lookback=lookback,
        strength_window=strength_window,
        home_away_lookback=home_away_lookback,
        elo_season_decay=elo_season_decay,
    )


if __name__ == "__main__":
    app()
