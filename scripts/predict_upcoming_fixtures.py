"""Train on completed EPL matches and predict upcoming fixtures."""

from __future__ import annotations

from datetime import datetime
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
    from_date: str = typer.Option(
        "",
        help="Optional YYYY-MM-DD cutoff for upcoming fixtures; empty means include all unplayed fixtures.",
    ),
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    parsed_from_date = None
    if from_date:
        parsed_from_date = datetime.strptime(from_date, "%Y-%m-%d").date()
    run_upcoming_predictions(
        project_root=project_root,
        lookback=lookback,
        strength_window=strength_window,
        home_away_lookback=home_away_lookback,
        elo_season_decay=elo_season_decay,
        from_date=parsed_from_date,
    )


if __name__ == "__main__":
    app()
