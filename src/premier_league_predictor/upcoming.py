"""Upcoming fixture prediction workflow for the Premier League baseline model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from premier_league_predictor.baseline import (
    DEFAULT_ELO,
    DEFAULT_ELO_SEASON_DECAY,
    DEFAULT_STRENGTH_WINDOW,
    ELO_K,
    FEATURE_COLUMNS,
    HOME_ELO_ADVANTAGE,
    _home_score,
    _points_for_side,
    build_step1_features,
    load_epl_matches,
    train_baseline_model,
)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


@dataclass(frozen=True)
class UpcomingPredictionArtifacts:
    fixtures_path: Path
    predictions_path: Path
    predictions_json_path: Path
    skipped_path: Path
    training_metrics_path: Path


def _expected_home_score(home_elo: float, away_elo: float) -> float:
    adjusted_home = home_elo + HOME_ELO_ADVANTAGE
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def _apply_elo_season_decay(team_elo: dict[str, float], decay: float) -> None:
    for team, elo in team_elo.items():
        team_elo[team] = DEFAULT_ELO + decay * (elo - DEFAULT_ELO)


def load_upcoming_fixtures(div_code: str = "E0") -> pd.DataFrame:
    """Load upcoming fixtures feed and filter to one division code."""
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    text = response.text.lstrip("\ufeff")
    fixtures = pd.read_csv(StringIO(text))
    if "ï»¿Div" in fixtures.columns:
        fixtures = fixtures.rename(columns={"ï»¿Div": "Div"})

    required_cols = {"Div", "Date", "Time", "HomeTeam", "AwayTeam"}
    missing = sorted(required_cols.difference(fixtures.columns))
    if missing:
        raise ValueError(f"Fixtures feed missing required columns: {missing}")

    fixtures = fixtures[fixtures["Div"] == div_code].copy()
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], dayfirst=True, errors="coerce")
    fixtures = fixtures.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    return fixtures.sort_values(["Date", "Time", "HomeTeam"]).reset_index(drop=True)


def _build_state_from_completed_matches(
    completed_matches: pd.DataFrame,
    elo_season_decay: float,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
    """Build team form history + Elo state from completed matches only."""
    team_history: dict[str, list[dict[str, float]]] = {}
    team_elo: dict[str, float] = {}
    current_season: str | None = None

    for _, match in completed_matches.iterrows():
        season_code = str(match["season_code"])
        if current_season is None:
            current_season = season_code
        elif season_code != current_season:
            _apply_elo_season_decay(team_elo, decay=elo_season_decay)
            current_season = season_code

        home = str(match["HomeTeam"])
        away = str(match["AwayTeam"])
        result = str(match["FTR"])
        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])

        home_elo_pre = team_elo.get(home, DEFAULT_ELO)
        away_elo_pre = team_elo.get(away, DEFAULT_ELO)

        team_history.setdefault(home, []).append(
            {
                "points": _points_for_side(result, "home"),
                "goals_for": home_goals,
                "goals_against": away_goals,
            }
        )
        team_history.setdefault(away, []).append(
            {
                "points": _points_for_side(result, "away"),
                "goals_for": away_goals,
                "goals_against": home_goals,
            }
        )

        expected_home = _expected_home_score(home_elo_pre, away_elo_pre)
        actual_home = _home_score(result)
        team_elo[home] = home_elo_pre + ELO_K * (actual_home - expected_home)
        team_elo[away] = away_elo_pre + ELO_K * ((1.0 - actual_home) - (1.0 - expected_home))

    return team_history, team_elo


def _build_features_for_upcoming_fixtures(
    fixtures: pd.DataFrame,
    team_history: dict[str, list[dict[str, float]]],
    team_elo: dict[str, float],
    lookback: int,
    strength_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create pre-match model features for upcoming fixtures."""
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    skipped: list[dict[str, str]] = []

    for _, fixture in fixtures.iterrows():
        home = str(fixture["HomeTeam"])
        away = str(fixture["AwayTeam"])
        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])

        if len(home_hist) < lookback or len(away_hist) < lookback:
            skipped.append(
                {
                    "date": fixture["Date"].date().isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "reason": "insufficient_history",
                }
            )
            continue

        home_recent = pd.DataFrame(home_hist[-lookback:])
        away_recent = pd.DataFrame(away_hist[-lookback:])
        home_strength = pd.DataFrame(home_hist[-strength_window:])
        away_strength = pd.DataFrame(away_hist[-strength_window:])

        home_elo_pre = team_elo.get(home, DEFAULT_ELO)
        away_elo_pre = team_elo.get(away, DEFAULT_ELO)

        rows.append(
            {
                "date": fixture["Date"],
                "time": fixture["Time"],
                "home_team": home,
                "away_team": away,
                "home_points_last5": float(home_recent["points"].mean()),
                "home_goals_for_last5": float(home_recent["goals_for"].mean()),
                "home_goals_against_last5": float(home_recent["goals_against"].mean()),
                "away_points_last5": float(away_recent["points"].mean()),
                "away_goals_for_last5": float(away_recent["goals_for"].mean()),
                "away_goals_against_last5": float(away_recent["goals_against"].mean()),
                "home_points_per_match_strength": float(home_strength["points"].mean()),
                "away_points_per_match_strength": float(away_strength["points"].mean()),
                "home_goal_diff_per_match_strength": float(
                    (home_strength["goals_for"] - home_strength["goals_against"]).mean()
                ),
                "away_goal_diff_per_match_strength": float(
                    (away_strength["goals_for"] - away_strength["goals_against"]).mean()
                ),
                "home_elo_pre": home_elo_pre,
                "away_elo_pre": away_elo_pre,
                "elo_diff_pre": home_elo_pre - away_elo_pre,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def run_upcoming_predictions(
    project_root: Path,
    lookback: int = 5,
    strength_window: int = DEFAULT_STRENGTH_WINDOW,
    elo_season_decay: float = DEFAULT_ELO_SEASON_DECAY,
    from_date: date | None = None,
) -> UpcomingPredictionArtifacts:
    """
    Train baseline model on completed matches and predict upcoming EPL fixtures.

    `from_date` defaults to today's date to keep output focused on future fixtures.
    """
    from_date = from_date or date.today()

    raw_fixtures_path = project_root / "data" / "raw" / "epl_upcoming_fixtures.csv"
    predictions_path = project_root / "data" / "processed" / "epl_upcoming_predictions.csv"
    predictions_json_path = project_root / "models" / "epl_upcoming_predictions.json"
    skipped_path = project_root / "data" / "processed" / "epl_upcoming_skipped.csv"
    training_metrics_path = project_root / "models" / "epl_upcoming_training_metrics.json"

    raw_fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    training_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    completed_matches = load_epl_matches()
    completed_features = build_step1_features(
        completed_matches,
        lookback=lookback,
        strength_window=strength_window,
        elo_season_decay=elo_season_decay,
    )
    model, metrics, _ = train_baseline_model(completed_features)
    training_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fixtures = load_upcoming_fixtures(div_code="E0")
    fixtures = fixtures[fixtures["Date"].dt.date >= from_date].copy()

    completed_fixture_keys = {
        (row.Date.date(), str(row.HomeTeam), str(row.AwayTeam))
        for row in completed_matches.itertuples(index=False)
    }
    fixtures = fixtures[
        ~fixtures.apply(
            lambda row: (row["Date"].date(), str(row["HomeTeam"]), str(row["AwayTeam"]))
            in completed_fixture_keys,
            axis=1,
        )
    ].copy()
    raw_fixtures_path.write_text(fixtures.to_csv(index=False), encoding="utf-8")

    team_history, team_elo = _build_state_from_completed_matches(
        completed_matches=completed_matches,
        elo_season_decay=elo_season_decay,
    )
    fixture_features, skipped = _build_features_for_upcoming_fixtures(
        fixtures=fixtures,
        team_history=team_history,
        team_elo=team_elo,
        lookback=lookback,
        strength_window=strength_window,
    )

    if fixture_features.empty:
        empty_columns = ["date", "time", "home_team", "away_team", "predicted_target"] + [
            f"pred_prob_{label}" for label in ("home_win", "draw", "away_win")
        ]
        pd.DataFrame(columns=empty_columns).to_csv(predictions_path, index=False)
        predictions_json_path.write_text("[]", encoding="utf-8")
        if skipped.empty:
            pd.DataFrame(columns=["date", "home_team", "away_team", "reason"]).to_csv(
                skipped_path, index=False
            )
        else:
            skipped.to_csv(skipped_path, index=False)
        print("No upcoming fixtures available after filtering.")
        print(f"Fixtures checked: {raw_fixtures_path}")
        return UpcomingPredictionArtifacts(
            fixtures_path=raw_fixtures_path,
            predictions_path=predictions_path,
            predictions_json_path=predictions_json_path,
            skipped_path=skipped_path,
            training_metrics_path=training_metrics_path,
        )

    probs = model.predict_proba(fixture_features[FEATURE_COLUMNS])
    preds = model.predict(fixture_features[FEATURE_COLUMNS])

    output = fixture_features[["date", "time", "home_team", "away_team"]].copy()
    output["predicted_target"] = preds
    output["prediction_confidence"] = probs.max(axis=1)
    for class_idx, class_name in enumerate(model.classes_):
        output[f"pred_prob_{class_name}"] = probs[:, class_idx]

    output = output.sort_values(["date", "time", "home_team"]).reset_index(drop=True)
    output.to_csv(predictions_path, index=False)
    predictions_json_path.write_text(output.to_json(orient="records", indent=2), encoding="utf-8")
    if skipped.empty:
        pd.DataFrame(columns=["date", "home_team", "away_team", "reason"]).to_csv(
            skipped_path, index=False
        )
    else:
        skipped.to_csv(skipped_path, index=False)

    print("Upcoming fixture prediction complete.")
    print(f"Upcoming fixtures saved: {raw_fixtures_path}")
    print(f"Predictions saved: {predictions_path}")
    print(f"Predictions JSON saved: {predictions_json_path}")
    print(f"Skipped fixtures saved: {skipped_path}")
    print(f"Training metrics saved: {training_metrics_path}")
    print("\nPreview:")
    preview_cols = [
        "date",
        "time",
        "home_team",
        "away_team",
        "predicted_target",
        "prediction_confidence",
    ]
    print(output[preview_cols].head(10).to_string(index=False))

    return UpcomingPredictionArtifacts(
        fixtures_path=raw_fixtures_path,
        predictions_path=predictions_path,
        predictions_json_path=predictions_json_path,
        skipped_path=skipped_path,
        training_metrics_path=training_metrics_path,
    )
