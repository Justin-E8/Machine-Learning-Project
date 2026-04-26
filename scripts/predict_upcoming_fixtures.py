"""Train on completed EPL matches and predict upcoming fixtures."""

from __future__ import annotations

from pathlib import Path

import typer

from premier_league_predictor.upcoming import run_upcoming_predictions

app = typer.Typer(add_completion=False)


@app.command()
def main(
    lookback: int = typer.Option(5, min=3, help="Overall recent-form window."),
    strength_window: int = typer.Option(20, min=5, help="Persistent strength window."),
    home_away_lookback: int = typer.Option(
        2,
        min=2,
        help="Home-only / away-only recent window.",
    ),
    elo_season_decay: float = typer.Option(
        0.65,
        min=0.0,
        max=1.0,
        help="Season-to-season Elo carryover (0 reset, 1 full carry).",
    ),
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    run_upcoming_predictions(
        project_root=project_root,
        lookback=lookback,
        strength_window=strength_window,
        home_away_lookback=home_away_lookback,
        elo_season_decay=elo_season_decay,
    )


if __name__ == "__main__":
    app()
