"""Compare direct, goals-first, and draw-aware outcome models."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from premier_league_predictor.baseline import (
    DEFAULT_ELO_SEASON_DECAY,
    DEFAULT_HOME_AWAY_LOOKBACK,
    DEFAULT_STRENGTH_WINDOW,
    build_step1_features,
    compare_outcome_vs_goal_models,
    load_epl_matches,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    lookback: int = typer.Option(5, min=3, help="Overall recent-form window."),
    strength_window: int = typer.Option(
        DEFAULT_STRENGTH_WINDOW,
        min=5,
        help="Window size for persistent strength features.",
    ),
    home_away_lookback: int = typer.Option(
        DEFAULT_HOME_AWAY_LOOKBACK,
        min=2,
        help="Home-only / away-only recent window.",
    ),
    elo_season_decay: float = typer.Option(
        DEFAULT_ELO_SEASON_DECAY,
        min=0.0,
        max=1.0,
        help="Season-to-season Elo carryover (0 reset, 1 full carry).",
    ),
    max_goals: int = typer.Option(
        10,
        min=4,
        help="Maximum goals per team for Poisson score-matrix truncation.",
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
    comparison = compare_outcome_vs_goal_models(features, max_goals=max_goals)

    output_path = project_root / "models" / "epl_model_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "lookback": lookback,
            "strength_window": strength_window,
            "home_away_lookback": home_away_lookback,
            "elo_season_decay": elo_season_decay,
            "max_goals": max_goals,
        },
        "summary_rows": comparison.summary_rows,
        "direct_logistic_metrics": comparison.logreg_metrics,
        "goal_poisson_metrics": comparison.goal_based_metrics,
        "draw_aware_metrics": comparison.draw_aware_metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Model comparison complete.")
    print(f"Comparison artifact saved: {output_path}")
    print("\nSummary:")
    for row in comparison.summary_rows:
        print(
            f"  {row['model']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"log_loss={row['log_loss']:.4f}, "
            f"draw_pred_rate={row['draw_pred_rate']:.4f}, "
            f"draw_actual_rate={row['draw_actual_rate']:.4f}"
        )


if __name__ == "__main__":
    app()
