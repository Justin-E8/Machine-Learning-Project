"""Run walk-forward backtesting to estimate true forward-looking accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from premier_league_predictor.baseline import (
    DEFAULT_ELO_SEASON_DECAY,
    DEFAULT_HOME_AWAY_LOOKBACK,
    DEFAULT_STRENGTH_WINDOW,
    build_step1_features,
    load_epl_matches,
    run_walk_forward_backtest,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    lookback: int = typer.Option(5, min=3, help="Overall recent-form window."),
    strength_window: int = typer.Option(
        DEFAULT_STRENGTH_WINDOW,
        min=5,
        help="Persistent strength window.",
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
    train_size: int = typer.Option(
        500,
        min=100,
        help="Rows in each rolling training window.",
    ),
    test_size: int = typer.Option(
        38,
        min=5,
        help="Rows in each rolling forward test window.",
    ),
    step_size: int = typer.Option(
        38,
        min=1,
        help="Rows to roll forward between windows.",
    ),
    metric: str = typer.Option(
        "accuracy",
        help="Model-selection metric per window: accuracy or log_loss.",
    ),
) -> None:
    """Build features once, then evaluate models across rolling time windows."""
    if metric not in {"accuracy", "log_loss"}:
        raise typer.BadParameter("metric must be one of: accuracy, log_loss")

    project_root = Path(__file__).resolve().parents[1]

    matches = load_epl_matches()
    features = build_step1_features(
        matches=matches,
        lookback=lookback,
        strength_window=strength_window,
        home_away_lookback=home_away_lookback,
        elo_season_decay=elo_season_decay,
    )

    summary = run_walk_forward_backtest(
        features=features,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        selection_metric=metric,
    )

    output_path = project_root / "models" / "epl_walk_forward_metrics.json"
    payload = {
        "config": {
            "lookback": lookback,
            "strength_window": strength_window,
            "home_away_lookback": home_away_lookback,
            "elo_season_decay": elo_season_decay,
            "train_size": train_size,
            "test_size": test_size,
            "step_size": step_size,
            "selection_metric": metric,
        },
        "window_count": summary.window_count,
        "summary_rows": summary.summary_rows,
        "window_rows": summary.window_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Walk-forward backtest complete.")
    print(f"Metrics saved: {output_path}")
    print(f"Windows evaluated: {summary.window_count}")
    print("\nModel summary:")
    for row in summary.summary_rows:
        print(
            f"  {row['model']}: "
            f"accuracy={row['accuracy_mean']:.4f} (+/- {row['accuracy_std']:.4f}), "
            f"log_loss={row['log_loss_mean']:.4f} (+/- {row['log_loss_std']:.4f}), "
            f"draw_pred_rate={row['draw_pred_rate_mean']:.4f}"
        )


if __name__ == "__main__":
    app()
