"""Step 1 baseline pipeline for Premier League outcome prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEASON_CODES = ("1819", "1920", "2021", "2122", "2223", "2324", "2425")


@dataclass(frozen=True)
class BaselineArtifacts:
    raw_data_path: Path
    training_data_path: Path
    model_path: Path
    metrics_path: Path


def _season_url(season_code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"


def _result_to_label(result: str) -> str:
    mapping = {"H": "home_win", "D": "draw", "A": "away_win"}
    if result not in mapping:
        raise ValueError(f"Unexpected result label: {result}")
    return mapping[result]


def load_epl_matches(season_codes: tuple[str, ...] = SEASON_CODES) -> pd.DataFrame:
    """Download and combine EPL matches from football-data.co.uk."""
    frames: list[pd.DataFrame] = []
    for code in season_codes:
        response = requests.get(_season_url(code), timeout=30)
        response.raise_for_status()
        season_df = pd.read_csv(StringIO(response.text))
        season_df = pd.concat(
            [season_df, pd.Series(code, index=season_df.index, name="season_code")], axis=1
        ).copy()
        frames.append(season_df)

    df = pd.concat(frames, ignore_index=True)
    required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "season_code"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]).copy()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()
    return df.sort_values("Date").reset_index(drop=True)


def _points_for_side(result: str, side: str) -> int:
    if side == "home":
        return 3 if result == "H" else 1 if result == "D" else 0
    return 3 if result == "A" else 1 if result == "D" else 0


def build_step1_features(matches: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Build rolling team-form features using prior matches only."""
    team_history: dict[str, list[dict[str, float]]] = {}
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for _, match in matches.iterrows():
        home = str(match["HomeTeam"])
        away = str(match["AwayTeam"])
        result = str(match["FTR"])
        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])

        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])

        if len(home_hist) >= lookback and len(away_hist) >= lookback:
            home_recent = pd.DataFrame(home_hist[-lookback:])
            away_recent = pd.DataFrame(away_hist[-lookback:])
            rows.append(
                {
                    "date": match["Date"],
                    "season_code": match["season_code"],
                    "home_team": home,
                    "away_team": away,
                    "home_points_last5": float(home_recent["points"].mean()),
                    "home_goals_for_last5": float(home_recent["goals_for"].mean()),
                    "home_goals_against_last5": float(home_recent["goals_against"].mean()),
                    "away_points_last5": float(away_recent["points"].mean()),
                    "away_goals_for_last5": float(away_recent["goals_for"].mean()),
                    "away_goals_against_last5": float(away_recent["goals_against"].mean()),
                    "target": _result_to_label(result),
                }
            )

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

    if not rows:
        raise ValueError("No training rows were generated. Lower lookback or check input data.")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def train_baseline_model(
    features: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float | int], str]:
    """Train multinomial logistic regression and return model + metrics."""
    feature_cols = [
        "home_points_last5",
        "home_goals_for_last5",
        "home_goals_against_last5",
        "away_points_last5",
        "away_goals_for_last5",
        "away_goals_against_last5",
    ]
    X = features[feature_cols]
    y = features["target"]

    split_idx = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if X_test.empty:
        raise ValueError("Test set is empty. Need more rows or lower training split.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics: dict[str, float | int] = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_prob, labels=model.classes_)),
    }
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)
    return model, metrics, report


def run_step1_baseline(project_root: Path, lookback: int = 5) -> BaselineArtifacts:
    """Run Step 1 data/feature/model flow and write artifacts."""
    raw_path = project_root / "data" / "raw" / "epl_matches.csv"
    training_path = project_root / "data" / "processed" / "epl_baseline_features.csv"
    model_path = project_root / "models" / "epl_logreg_baseline.joblib"
    metrics_path = project_root / "models" / "epl_baseline_metrics.json"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    matches = load_epl_matches()
    matches.to_csv(raw_path, index=False)

    features = build_step1_features(matches=matches, lookback=lookback)
    features.to_csv(training_path, index=False)

    model, metrics, report = train_baseline_model(features)
    joblib.dump(model, model_path)
    metrics_with_report = {**metrics, "classification_report_text": report}
    metrics_path.write_text(json.dumps(metrics_with_report, indent=2), encoding="utf-8")

    print("Step 1 baseline complete.")
    print(f"Raw data saved: {raw_path}")
    print(f"Feature data saved: {training_path}")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("\nMetrics:")
    for key in ("train_rows", "test_rows", "accuracy", "log_loss"):
        print(f"  {key}: {metrics[key]}")

    return BaselineArtifacts(
        raw_data_path=raw_path,
        training_data_path=training_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
