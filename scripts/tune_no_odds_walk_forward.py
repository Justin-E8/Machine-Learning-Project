"""Tune no-odds feature/Elo settings with walk-forward accuracy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import typer

from premier_league_predictor.baseline import (
    DEFAULT_ELO_K,
    DEFAULT_ELO_SEASON_DECAY,
    DEFAULT_HOME_AWAY_LOOKBACK,
    DEFAULT_HOME_ELO_ADVANTAGE,
    DEFAULT_STRENGTH_WINDOW,
    build_step1_features,
    load_epl_matches,
    run_walk_forward_backtest,
)

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ConfigCandidate:
    lookback: int
    strength_window: int
    home_away_lookback: int
    elo_season_decay: float
    elo_k: float
    home_elo_advantage: float


def _focused_candidates() -> list[ConfigCandidate]:
    """Return a compact, no-odds candidate set around the current defaults."""
    base = ConfigCandidate(
        lookback=5,
        strength_window=DEFAULT_STRENGTH_WINDOW,
        home_away_lookback=DEFAULT_HOME_AWAY_LOOKBACK,
        elo_season_decay=DEFAULT_ELO_SEASON_DECAY,
        elo_k=DEFAULT_ELO_K,
        home_elo_advantage=DEFAULT_HOME_ELO_ADVANTAGE,
    )
    return [
        base,
        ConfigCandidate(**{**asdict(base), "lookback": 4}),
        ConfigCandidate(**{**asdict(base), "lookback": 6}),
        ConfigCandidate(**{**asdict(base), "strength_window": 15}),
        ConfigCandidate(**{**asdict(base), "strength_window": 25}),
        ConfigCandidate(**{**asdict(base), "home_away_lookback": 3}),
        ConfigCandidate(**{**asdict(base), "elo_season_decay": 0.55}),
        ConfigCandidate(**{**asdict(base), "elo_season_decay": 0.75}),
        ConfigCandidate(**{**asdict(base), "elo_k": 20.0}),
        ConfigCandidate(**{**asdict(base), "elo_k": 28.0}),
        ConfigCandidate(**{**asdict(base), "home_elo_advantage": 70.0}),
        ConfigCandidate(**{**asdict(base), "home_elo_advantage": 90.0}),
    ]


@app.command()
def main(
    train_size: int = typer.Option(
        760,
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
    max_candidates: int = typer.Option(
        0,
        min=0,
        help="Optional cap for number of candidate configs (0 uses all).",
    ),
) -> None:
    """
    Search no-odds configurations by walk-forward mean accuracy.

    Ranking objective:
      1) highest walk-forward mean accuracy
      2) lower walk-forward mean log loss
    """
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "models" / "epl_no_odds_tuning.json"

    matches = load_epl_matches()
    candidates = _focused_candidates()
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    rows: list[dict[str, float | int | str]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] Evaluating: {candidate}")
        features = build_step1_features(
            matches=matches,
            lookback=candidate.lookback,
            strength_window=candidate.strength_window,
            home_away_lookback=candidate.home_away_lookback,
            elo_season_decay=candidate.elo_season_decay,
            elo_k=candidate.elo_k,
            home_elo_advantage=candidate.home_elo_advantage,
        )
        summary = run_walk_forward_backtest(
            features=features,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            selection_metric="accuracy",
        )

        best_model_row = sorted(
            summary.summary_rows,
            key=lambda r: (-float(r["accuracy_mean"]), float(r["log_loss_mean"])),
        )[0]
        rows.append(
            {
                "lookback": candidate.lookback,
                "strength_window": candidate.strength_window,
                "home_away_lookback": candidate.home_away_lookback,
                "elo_season_decay": candidate.elo_season_decay,
                "elo_k": candidate.elo_k,
                "home_elo_advantage": candidate.home_elo_advantage,
                "best_model": str(best_model_row["model"]),
                "accuracy_mean": float(best_model_row["accuracy_mean"]),
                "accuracy_std": float(best_model_row["accuracy_std"]),
                "log_loss_mean": float(best_model_row["log_loss_mean"]),
                "log_loss_std": float(best_model_row["log_loss_std"]),
                "draw_pred_rate_mean": float(best_model_row["draw_pred_rate_mean"]),
                "window_count": int(best_model_row["window_count"]),
            }
        )

    ranked_rows = sorted(rows, key=lambda r: (-float(r["accuracy_mean"]), float(r["log_loss_mean"])))
    payload = {
        "config": {
            "train_size": train_size,
            "test_size": test_size,
            "step_size": step_size,
            "candidate_count": len(candidates),
        },
        "best_candidate": ranked_rows[0] if ranked_rows else None,
        "leaderboard": ranked_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nNo-odds walk-forward tuning complete.")
    print(f"Results saved: {output_path}")
    print("\nTop candidates:")
    for row in ranked_rows[:5]:
        print(
            "  "
            f"acc={row['accuracy_mean']:.4f}, log_loss={row['log_loss_mean']:.4f}, "
            f"model={row['best_model']}, lookback={row['lookback']}, "
            f"strength={row['strength_window']}, home_away={row['home_away_lookback']}, "
            f"elo_decay={row['elo_season_decay']}, elo_k={row['elo_k']}, "
            f"home_adv={row['home_elo_advantage']}"
        )


if __name__ == "__main__":
    app()
