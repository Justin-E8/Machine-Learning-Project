"""Benchmark EPL models with stronger non-odds features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from premier_league_predictor.baseline import (
    build_step1_features,
    load_epl_matches,
    train_baseline_model,
)

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
        help="Season-to-season Elo carryover (0 resets fully, 1 keeps full history).",
    ),
) -> None:
    project_root = Path(__file__).resolve().parents[1]

    matches = load_epl_matches()
    features = build_step1_features(
        matches=matches,
        lookback=lookback,
        strength_window=strength_window,
        home_away_lookback=home_away_lookback,
        elo_season_decay=elo_season_decay,
    )
    _, metrics, _ = train_baseline_model(features)

    leaderboard = pd.DataFrame(metrics["leaderboard"]).sort_values(
        ["log_loss", "accuracy"], ascending=[True, False]
    )

    output_path = project_root / "models" / "epl_model_benchmark.json"
    payload = {
        "config": {
            "lookback": lookback,
            "strength_window": strength_window,
            "home_away_lookback": home_away_lookback,
            "elo_season_decay": elo_season_decay,
        },
        "selected_model": metrics["selected_model"],
        "selected_accuracy": metrics["accuracy"],
        "selected_log_loss": metrics["log_loss"],
        "leaderboard": metrics["leaderboard"],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Model benchmark complete.")
    print(f"Benchmark artifact saved: {output_path}")
    print("\nLeaderboard:")
    print(leaderboard.to_string(index=False))
    print("\nSelected model metrics:")
    print(f"  selected_model: {metrics['selected_model']}")
    for key in ("accuracy", "log_loss", "draw_pred_rate_test", "draw_actual_rate_test"):
        print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    app()
