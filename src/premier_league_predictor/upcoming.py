"""Upcoming fixture prediction workflow using the current tuned model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from premier_league_predictor.baseline import (
    DEFAULT_ELO,
    DEFAULT_ELO_K,
    DEFAULT_ELO_SEASON_DECAY,
    DEFAULT_HOME_AWAY_LOOKBACK,
    DEFAULT_HOME_ELO_ADVANTAGE,
    DEFAULT_STRENGTH_WINDOW,
    FEATURE_COLUMNS,
    _apply_elo_season_decay,
    _expected_home_score,
    _goal_diff_mean_tail,
    _home_result_score,
    _mean_tail,
    _points_for_side,
    _safe_rest_days,
    build_step1_features,
    load_epl_matches,
    train_baseline_model,
)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
CURRENT_SEASON_CODE = "2526"
CURRENT_SEASON_URL = (
    f"https://www.football-data.co.uk/mmz4281/{CURRENT_SEASON_CODE}/E0.csv"
)
OPENFOOTBALL_SEASON_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/en.1.json"
)

TEAM_NAME_ALIASES = {
    "manchester united": "Man United",
    "manchester city": "Man City",
    "tottenham hotspur": "Spurs",
    "nottingham forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves",
    "west ham united": "West Ham",
    "newcastle united": "Newcastle",
    "brighton and hove albion": "Brighton",
    "leeds united": "Leeds",
}


@dataclass(frozen=True)
class UpcomingPredictionArtifacts:
    """Paths to files produced by the upcoming-fixtures pipeline."""

    fixtures_path: Path
    predictions_path: Path
    predictions_json_path: Path
    skipped_path: Path
    completed_predictions_path: Path
    training_metrics_path: Path


def _empty_fixtures_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "Time", "HomeTeam", "AwayTeam"])


def _canonical_team_key(name: str) -> str:
    value = str(name).strip().lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    tokens = [token for token in value.split() if token != "fc"]
    if tokens and tokens[0] == "afc":
        tokens = tokens[1:]
    return " ".join(tokens)


def _build_team_name_map(matches: pd.DataFrame) -> dict[str, str]:
    known_names = pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).dropna().astype(str).unique()
    mapping = {_canonical_team_key(name): name for name in known_names}
    for alias_key, target in TEAM_NAME_ALIASES.items():
        if target in set(known_names):
            mapping[alias_key] = target
    return mapping


def _normalize_team_name(name: str, team_name_map: dict[str, str]) -> str:
    raw_name = str(name).strip()
    if raw_name in team_name_map.values():
        return raw_name
    key = _canonical_team_key(raw_name)
    return team_name_map.get(key, raw_name)


def _normalize_fixture_team_names(
    fixtures: pd.DataFrame,
    team_name_map: dict[str, str],
) -> pd.DataFrame:
    """Normalize HomeTeam/AwayTeam names to training-data naming."""
    frame = fixtures.copy()
    if frame.empty:
        return frame
    frame["HomeTeam"] = frame["HomeTeam"].map(
        lambda value: _normalize_team_name(str(value), team_name_map)
    )
    frame["AwayTeam"] = frame["AwayTeam"].map(
        lambda value: _normalize_team_name(str(value), team_name_map)
    )
    return frame


def _normalize_fixture_columns(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Normalize fixture columns to Date/Time/HomeTeam/AwayTeam."""
    frame = fixtures.copy()
    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fixtures frame missing required columns: {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    frame = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    if "Time" not in frame.columns:
        frame["Time"] = ""
    frame["Time"] = frame["Time"].fillna("").astype(str)
    return frame[["Date", "Time", "HomeTeam", "AwayTeam"]].sort_values(
        ["Date", "Time", "HomeTeam"]
    ).reset_index(drop=True)


def _normalize_fixtures_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw fixtures feed and keep only EPL rows with valid dates."""
    frame = fixtures.copy()
    if "ï»¿Div" in frame.columns:
        frame = frame.rename(columns={"ï»¿Div": "Div"})

    required = {"Div", "Date", "Time", "HomeTeam", "AwayTeam"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fixtures feed missing required columns: {missing}")

    frame = frame[frame["Div"] == "E0"].copy()
    return _normalize_fixture_columns(frame)


def load_current_season_unplayed_fixtures() -> pd.DataFrame:
    """Fetch unplayed fixtures from current-season EPL CSV for full schedule coverage."""
    response = requests.get(CURRENT_SEASON_URL, timeout=30)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    frame = _normalize_fixture_columns(raw)

    has_result = (
        raw.get("FTR", pd.Series(index=raw.index, dtype=object))
        .astype(str)
        .str.strip()
        .isin({"H", "D", "A"})
    )
    home_goals = pd.to_numeric(raw.get("FTHG"), errors="coerce")
    away_goals = pd.to_numeric(raw.get("FTAG"), errors="coerce")
    has_score = home_goals.notna() & away_goals.notna()
    unplayed_mask = ~(has_result & has_score)

    frame = frame.loc[unplayed_mask].copy()
    return frame.sort_values(["Date", "Time", "HomeTeam"]).reset_index(drop=True)


def _openfootball_match_is_unplayed(score_value: object) -> bool:
    if not isinstance(score_value, dict):
        return True
    ft_value = score_value.get("ft")
    if isinstance(ft_value, list) and len(ft_value) == 2:
        return any(goal is None for goal in ft_value)
    return not bool(score_value)


def load_openfootball_unplayed_fixtures(team_name_map: dict[str, str]) -> pd.DataFrame:
    """Fetch remaining fixtures from OpenFootball full-season schedule."""
    response = requests.get(OPENFOOTBALL_SEASON_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    matches = payload.get("matches", [])
    if not matches:
        return _empty_fixtures_frame()

    rows: list[dict[str, object]] = []
    for match in matches:
        if not _openfootball_match_is_unplayed(match.get("score")):
            continue

        fixture_date = pd.to_datetime(match.get("date"), errors="coerce")
        if pd.isna(fixture_date):
            continue

        rows.append(
            {
                "Date": fixture_date,
                "Time": str(match.get("time") or ""),
                "HomeTeam": _normalize_team_name(str(match.get("team1") or ""), team_name_map),
                "AwayTeam": _normalize_team_name(str(match.get("team2") or ""), team_name_map),
            }
        )

    if not rows:
        return _empty_fixtures_frame()

    return _normalize_fixture_columns(pd.DataFrame(rows))


def load_upcoming_fixtures() -> pd.DataFrame:
    """Fetch EPL upcoming fixtures feed from football-data."""
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    text = response.text.lstrip("\ufeff")
    raw = pd.read_csv(StringIO(text))
    return _normalize_fixtures_frame(raw)


def _build_team_state(
    matches: pd.DataFrame,
    *,
    elo_season_decay: float,
    elo_k: float,
    home_elo_advantage: float,
) -> tuple[dict[str, list[dict[str, float | bool | pd.Timestamp]]], dict[str, float]]:
    """Build team histories and Elo ratings from completed matches."""
    team_history: dict[str, list[dict[str, float | bool | pd.Timestamp]]] = {}
    team_elo: dict[str, float] = {}
    current_season: str | None = None

    for _, match in matches.iterrows():
        season_code = str(match["season_code"])
        if current_season is None:
            current_season = season_code
        elif season_code != current_season:
            _apply_elo_season_decay(team_elo, decay=elo_season_decay)
            current_season = season_code

        match_date = pd.Timestamp(match["Date"])
        home = str(match["HomeTeam"])
        away = str(match["AwayTeam"])
        result = str(match["FTR"])
        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])

        home_elo_pre = float(team_elo.get(home, DEFAULT_ELO))
        away_elo_pre = float(team_elo.get(away, DEFAULT_ELO))

        team_history.setdefault(home, []).append(
            {
                "points": float(_points_for_side(result, "home")),
                "goals_for": home_goals,
                "goals_against": away_goals,
                "is_home": True,
                "date": match_date,
            }
        )
        team_history.setdefault(away, []).append(
            {
                "points": float(_points_for_side(result, "away")),
                "goals_for": away_goals,
                "goals_against": home_goals,
                "is_home": False,
                "date": match_date,
            }
        )

        expected_home = _expected_home_score(home_elo_pre, away_elo_pre, home_elo_advantage)
        actual_home = _home_result_score(result)
        team_elo[home] = home_elo_pre + elo_k * (actual_home - expected_home)
        team_elo[away] = away_elo_pre + elo_k * ((1.0 - actual_home) - (1.0 - expected_home))

    return team_history, team_elo


def _feature_row_for_fixture(
    fixture_date: pd.Timestamp,
    fixture_time: str,
    home: str,
    away: str,
    *,
    team_history: dict[str, list[dict[str, float | bool | pd.Timestamp]]],
    team_elo: dict[str, float],
    lookback: int,
    strength_window: int,
    home_away_lookback: int,
) -> dict[str, float | str | pd.Timestamp] | None:
    """Build a single pre-match feature row for a future fixture."""
    home_hist = team_history.get(home, [])
    away_hist = team_history.get(away, [])
    home_home_hist = [r for r in home_hist if bool(r["is_home"])]
    away_away_hist = [r for r in away_hist if not bool(r["is_home"])]

    if (
        len(home_hist) < lookback
        or len(away_hist) < lookback
        or len(home_home_hist) < home_away_lookback
        or len(away_away_hist) < home_away_lookback
    ):
        return None

    home_strength_n = min(strength_window, len(home_hist))
    away_strength_n = min(strength_window, len(away_hist))
    home_last_date = pd.Timestamp(home_hist[-1]["date"])
    away_last_date = pd.Timestamp(away_hist[-1]["date"])
    home_rest_days = _safe_rest_days(fixture_date, home_last_date)
    away_rest_days = _safe_rest_days(fixture_date, away_last_date)

    home_elo_pre = float(team_elo.get(home, DEFAULT_ELO))
    away_elo_pre = float(team_elo.get(away, DEFAULT_ELO))

    return {
        "date": fixture_date,
        "time": fixture_time,
        "home_team": home,
        "away_team": away,
        "home_points_last5": _mean_tail(home_hist, "points", lookback),
        "home_goals_for_last5": _mean_tail(home_hist, "goals_for", lookback),
        "home_goals_against_last5": _mean_tail(home_hist, "goals_against", lookback),
        "away_points_last5": _mean_tail(away_hist, "points", lookback),
        "away_goals_for_last5": _mean_tail(away_hist, "goals_for", lookback),
        "away_goals_against_last5": _mean_tail(away_hist, "goals_against", lookback),
        "home_home_points_lastN": _mean_tail(home_home_hist, "points", home_away_lookback),
        "home_home_goals_for_lastN": _mean_tail(home_home_hist, "goals_for", home_away_lookback),
        "home_home_goals_against_lastN": _mean_tail(
            home_home_hist, "goals_against", home_away_lookback
        ),
        "away_away_points_lastN": _mean_tail(away_away_hist, "points", home_away_lookback),
        "away_away_goals_for_lastN": _mean_tail(away_away_hist, "goals_for", home_away_lookback),
        "away_away_goals_against_lastN": _mean_tail(
            away_away_hist, "goals_against", home_away_lookback
        ),
        "home_points_last3": _mean_tail(home_hist, "points", 3),
        "away_points_last3": _mean_tail(away_hist, "points", 3),
        "home_goal_diff_last3": _goal_diff_mean_tail(home_hist, 3),
        "away_goal_diff_last3": _goal_diff_mean_tail(away_hist, 3),
        "home_points_per_match_strength": _mean_tail(home_hist, "points", home_strength_n),
        "away_points_per_match_strength": _mean_tail(away_hist, "points", away_strength_n),
        "home_goal_diff_per_match_strength": _goal_diff_mean_tail(home_hist, home_strength_n),
        "away_goal_diff_per_match_strength": _goal_diff_mean_tail(away_hist, away_strength_n),
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "rest_days_diff": home_rest_days - away_rest_days,
        "home_elo_pre": home_elo_pre,
        "away_elo_pre": away_elo_pre,
        "elo_diff_pre": home_elo_pre - away_elo_pre,
    }


def run_upcoming_predictions(
    project_root: Path,
    *,
    lookback: int = 5,
    strength_window: int = DEFAULT_STRENGTH_WINDOW,
    home_away_lookback: int = DEFAULT_HOME_AWAY_LOOKBACK,
    elo_season_decay: float = DEFAULT_ELO_SEASON_DECAY,
    elo_k: float = DEFAULT_ELO_K,
    home_elo_advantage: float = DEFAULT_HOME_ELO_ADVANTAGE,
    from_date: date | None = None,
) -> UpcomingPredictionArtifacts:
    """Train the baseline on completed matches and score upcoming fixtures."""
    raw_fixtures_path = project_root / "data" / "raw" / "epl_upcoming_fixtures.csv"
    predictions_path = project_root / "data" / "processed" / "epl_upcoming_predictions.csv"
    predictions_json_path = project_root / "models" / "epl_upcoming_predictions.json"
    skipped_path = project_root / "data" / "processed" / "epl_upcoming_skipped.csv"
    completed_predictions_path = project_root / "data" / "processed" / "epl_completed_predictions.csv"
    training_metrics_path = project_root / "models" / "epl_upcoming_training_metrics.json"

    raw_fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    training_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    matches = load_epl_matches()
    features = build_step1_features(
        matches=matches,
        lookback=lookback,
        strength_window=strength_window,
        home_away_lookback=home_away_lookback,
        elo_season_decay=elo_season_decay,
        elo_k=elo_k,
        home_elo_advantage=home_elo_advantage,
    )
    model, metrics, completed_predictions = train_baseline_model(features)
    completed_predictions.to_csv(completed_predictions_path, index=False)
    training_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    team_name_map = _build_team_name_map(matches)
    fixtures_feed = load_upcoming_fixtures()
    season_unplayed = load_current_season_unplayed_fixtures()
    try:
        openfootball_unplayed = load_openfootball_unplayed_fixtures(team_name_map)
    except Exception:
        # Keep workflow robust even if OpenFootball is temporarily unavailable.
        openfootball_unplayed = _empty_fixtures_frame()

    fixtures = pd.concat(
        [fixtures_feed, season_unplayed, openfootball_unplayed],
        ignore_index=True,
    )
    fixtures = _normalize_fixture_team_names(fixtures, team_name_map)
    fixtures = fixtures.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    fixtures = fixtures.sort_values(["Date", "Time", "HomeTeam"]).reset_index(drop=True)
    if from_date is not None:
        fixtures = fixtures[fixtures["Date"].dt.date >= from_date].copy()

    # Remove any fixtures already present in completed results.
    completed_keys = {
        (pd.Timestamp(row.Date).date(), str(row.HomeTeam), str(row.AwayTeam))
        for row in matches.itertuples(index=False)
    }
    fixtures = fixtures[
        ~fixtures.apply(
            lambda row: (pd.Timestamp(row["Date"]).date(), str(row["HomeTeam"]), str(row["AwayTeam"]))
            in completed_keys,
            axis=1,
        )
    ].copy()
    fixtures.to_csv(raw_fixtures_path, index=False)

    team_history, team_elo = _build_team_state(
        matches,
        elo_season_decay=elo_season_decay,
        elo_k=elo_k,
        home_elo_advantage=home_elo_advantage,
    )

    feature_rows: list[dict[str, float | str | pd.Timestamp]] = []
    skipped_rows: list[dict[str, str]] = []
    for _, fixture in fixtures.iterrows():
        fixture_date = pd.Timestamp(fixture["Date"])
        fixture_time = str(fixture.get("Time", ""))
        home = str(fixture["HomeTeam"])
        away = str(fixture["AwayTeam"])
        feature_row = _feature_row_for_fixture(
            fixture_date,
            fixture_time,
            home,
            away,
            team_history=team_history,
            team_elo=team_elo,
            lookback=lookback,
            strength_window=strength_window,
            home_away_lookback=home_away_lookback,
        )
        if feature_row is None:
            skipped_rows.append(
                {
                    "date": fixture_date.date().isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "reason": "insufficient_history",
                }
            )
            continue
        feature_rows.append(feature_row)

    if not feature_rows:
        empty = pd.DataFrame(
            columns=[
                "date",
                "time",
                "home_team",
                "away_team",
                "predicted_target",
                "prediction_confidence",
                "pred_prob_away_win",
                "pred_prob_draw",
                "pred_prob_home_win",
            ]
        )
        empty.to_csv(predictions_path, index=False)
        predictions_json_path.write_text("[]", encoding="utf-8")
    else:
        fixture_features = pd.DataFrame(feature_rows)
        probs = model.predict_proba(fixture_features[FEATURE_COLUMNS])
        preds = model.predict(fixture_features[FEATURE_COLUMNS])
        classes = list(model.classes_)
        class_to_idx = {name: idx for idx, name in enumerate(classes)}

        output = fixture_features[["date", "time", "home_team", "away_team"]].copy()
        output["predicted_target"] = preds
        output["prediction_confidence"] = np.round(probs.max(axis=1), 2)
        for label in ("away_win", "draw", "home_win"):
            if label in class_to_idx:
                output[f"pred_prob_{label}"] = np.round(probs[:, class_to_idx[label]], 2)
            else:
                output[f"pred_prob_{label}"] = 0.0

        output = output.sort_values(["date", "time", "home_team"]).reset_index(drop=True)
        output.to_csv(predictions_path, index=False)
        predictions_json_path.write_text(output.to_json(orient="records", indent=2), encoding="utf-8")

    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=["date", "home_team", "away_team", "reason"],
    )
    skipped_df.to_csv(skipped_path, index=False)

    return UpcomingPredictionArtifacts(
        fixtures_path=raw_fixtures_path,
        predictions_path=predictions_path,
        predictions_json_path=predictions_json_path,
        skipped_path=skipped_path,
        completed_predictions_path=completed_predictions_path,
        training_metrics_path=training_metrics_path,
    )
