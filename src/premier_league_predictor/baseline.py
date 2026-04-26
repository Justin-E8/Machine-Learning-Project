"""Model training pipeline for Premier League match outcome prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEASON_CODES = ("1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526")

DEFAULT_ELO = 1500.0
DEFAULT_ELO_K = 24.0
DEFAULT_HOME_ELO_ADVANTAGE = 80.0
DEFAULT_STRENGTH_WINDOW = 20
DEFAULT_HOME_AWAY_LOOKBACK = 2
DEFAULT_ELO_SEASON_DECAY = 0.65

CORE_FEATURE_COLUMNS = [
    "home_points_last5",
    "home_goals_for_last5",
    "home_goals_against_last5",
    "away_points_last5",
    "away_goals_for_last5",
    "away_goals_against_last5",
]

FEATURE_COLUMNS = [
    *CORE_FEATURE_COLUMNS,
    "home_home_points_lastN",
    "home_home_goals_for_lastN",
    "home_home_goals_against_lastN",
    "away_away_points_lastN",
    "away_away_goals_for_lastN",
    "away_away_goals_against_lastN",
    "home_points_last3",
    "away_points_last3",
    "home_goal_diff_last3",
    "away_goal_diff_last3",
    "home_points_per_match_strength",
    "away_points_per_match_strength",
    "home_goal_diff_per_match_strength",
    "away_goal_diff_per_match_strength",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_pre",
]


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


def _mean_tail(records: list[dict[str, float]], key: str, n: int) -> float:
    tail = records[-n:]
    return float(sum(float(r[key]) for r in tail) / len(tail))


def _goal_diff_mean_tail(records: list[dict[str, float]], n: int) -> float:
    tail = records[-n:]
    return float(sum(float(r["goals_for"] - r["goals_against"]) for r in tail) / len(tail))


def _safe_rest_days(current_date: pd.Timestamp, previous_date: pd.Timestamp) -> float:
    """Return non-negative rest days between matches, capped to avoid huge outliers."""
    rest = float((current_date - previous_date).days)
    # Cap very long offseason breaks so model focuses on meaningful cadence differences.
    return float(max(0.0, min(rest, 30.0)))


def _expected_home_score(home_elo: float, away_elo: float, home_advantage: float) -> float:
    adjusted_home = home_elo + home_advantage
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def _home_result_score(result: str) -> float:
    if result == "H":
        return 1.0
    if result == "D":
        return 0.5
    if result == "A":
        return 0.0
    raise ValueError(f"Unexpected result label: {result}")


def _apply_elo_season_decay(team_elo: dict[str, float], decay: float) -> None:
    for team, elo in list(team_elo.items()):
        team_elo[team] = DEFAULT_ELO + decay * (elo - DEFAULT_ELO)


def build_step1_features(
    matches: pd.DataFrame,
    lookback: int = 5,
    strength_window: int = DEFAULT_STRENGTH_WINDOW,
    home_away_lookback: int = DEFAULT_HOME_AWAY_LOOKBACK,
    elo_season_decay: float = DEFAULT_ELO_SEASON_DECAY,
    elo_k: float = DEFAULT_ELO_K,
    home_elo_advantage: float = DEFAULT_HOME_ELO_ADVANTAGE,
) -> pd.DataFrame:
    """
    Build leak-free pre-match features from historical match results.

    Includes:
      - overall recent form
      - home/away split recent form
      - momentum and strength-window trends
      - rest days
      - pre-match Elo ratings
    """
    if lookback < 3:
        raise ValueError("lookback must be at least 3")
    if strength_window < lookback:
        raise ValueError("strength_window must be >= lookback")
    if home_away_lookback < 2:
        raise ValueError("home_away_lookback must be at least 2")
    if not 0.0 <= elo_season_decay <= 1.0:
        raise ValueError("elo_season_decay must be between 0.0 and 1.0")

    team_history: dict[str, list[dict[str, float | bool | pd.Timestamp]]] = {}
    team_elo: dict[str, float] = {}
    rows: list[dict[str, float | str | pd.Timestamp]] = []

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

        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])
        home_home_hist = [r for r in home_hist if bool(r["is_home"])]
        away_away_hist = [r for r in away_hist if not bool(r["is_home"])]

        home_elo_pre = float(team_elo.get(home, DEFAULT_ELO))
        away_elo_pre = float(team_elo.get(away, DEFAULT_ELO))

        has_required_history = (
            len(home_hist) >= lookback
            and len(away_hist) >= lookback
            and len(home_home_hist) >= home_away_lookback
            and len(away_away_hist) >= home_away_lookback
        )

        if has_required_history:
            home_strength_n = min(strength_window, len(home_hist))
            away_strength_n = min(strength_window, len(away_hist))

            home_last_date = pd.Timestamp(home_hist[-1]["date"])
            away_last_date = pd.Timestamp(away_hist[-1]["date"])
            home_rest_days = float((match_date - home_last_date).days)
            away_rest_days = float((match_date - away_last_date).days)

            rows.append(
                {
                    "date": match_date,
                    "season_code": season_code,
                    "home_team": home,
                    "away_team": away,
                    "home_goals_actual": home_goals,
                    "away_goals_actual": away_goals,
                    "home_points_last5": _mean_tail(home_hist, "points", lookback),
                    "home_goals_for_last5": _mean_tail(home_hist, "goals_for", lookback),
                    "home_goals_against_last5": _mean_tail(home_hist, "goals_against", lookback),
                    "away_points_last5": _mean_tail(away_hist, "points", lookback),
                    "away_goals_for_last5": _mean_tail(away_hist, "goals_for", lookback),
                    "away_goals_against_last5": _mean_tail(away_hist, "goals_against", lookback),
                    "home_home_points_lastN": _mean_tail(home_home_hist, "points", home_away_lookback),
                    "home_home_goals_for_lastN": _mean_tail(
                        home_home_hist, "goals_for", home_away_lookback
                    ),
                    "home_home_goals_against_lastN": _mean_tail(
                        home_home_hist, "goals_against", home_away_lookback
                    ),
                    "away_away_points_lastN": _mean_tail(
                        away_away_hist, "points", home_away_lookback
                    ),
                    "away_away_goals_for_lastN": _mean_tail(
                        away_away_hist, "goals_for", home_away_lookback
                    ),
                    "away_away_goals_against_lastN": _mean_tail(
                        away_away_hist, "goals_against", home_away_lookback
                    ),
                    "home_points_last3": _mean_tail(home_hist, "points", 3),
                    "away_points_last3": _mean_tail(away_hist, "points", 3),
                    "home_goal_diff_last3": _goal_diff_mean_tail(home_hist, 3),
                    "away_goal_diff_last3": _goal_diff_mean_tail(away_hist, 3),
                    "home_points_per_match_strength": _mean_tail(home_hist, "points", home_strength_n),
                    "away_points_per_match_strength": _mean_tail(away_hist, "points", away_strength_n),
                    "home_goal_diff_per_match_strength": _goal_diff_mean_tail(
                        home_hist, home_strength_n
                    ),
                    "away_goal_diff_per_match_strength": _goal_diff_mean_tail(
                        away_hist, away_strength_n
                    ),
                    "home_rest_days": home_rest_days,
                    "away_rest_days": away_rest_days,
                    "rest_days_diff": home_rest_days - away_rest_days,
                    "home_elo_pre": home_elo_pre,
                    "away_elo_pre": away_elo_pre,
                    "elo_diff_pre": home_elo_pre - away_elo_pre,
                    "target": _result_to_label(result),
                }
            )

        # Update state after feature construction (leak-free).
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

    if not rows:
        raise ValueError("No training rows were generated. Lower lookback or check input data.")

    features = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    features["date"] = pd.to_datetime(features["date"])
    return features


def _attach_prediction_columns(
    features: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    classes: list[str],
    split_idx: int,
) -> pd.DataFrame:
    out = features.copy()
    out["split"] = "train"
    out.loc[split_idx:, "split"] = "test"
    out["predicted_target"] = preds
    out["prediction_correct"] = out["target"] == out["predicted_target"]

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    for label in ("away_win", "draw", "home_win"):
        if label in class_to_idx:
            out[f"pred_prob_{label}"] = probs[:, class_to_idx[label]]
        else:
            out[f"pred_prob_{label}"] = 0.0
    return out


def train_baseline_model(
    features: pd.DataFrame,
) -> tuple[object, dict[str, float | int | str], pd.DataFrame]:
    """
    Train candidate models and return best model by log loss.

    Candidates:
      - multinomial logistic regression
      - HistGradientBoostingClassifier
    """
    X = features
    y = features["target"]

    split_idx = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if X_test.empty:
        raise ValueError("Test set is empty. Need more rows or lower training split.")

    candidates: list[tuple[str, object, list[str]]] = [
        (
            "logistic_core",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=4000, random_state=42)),
                ]
            ),
            CORE_FEATURE_COLUMNS,
        ),
        (
            "logistic_enhanced",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=4000, random_state=42, C=1.5)),
                ]
            ),
            FEATURE_COLUMNS,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=5,
                max_iter=450,
                l2_regularization=0.05,
                random_state=42,
            ),
            FEATURE_COLUMNS,
        ),
    ]

    leaderboard: list[dict[str, float | str]] = []
    fitted: dict[str, tuple[object, list[str]]] = {}
    for name, model, cols in candidates:
        model.fit(X_train[cols], y_train)
        fitted[name] = (model, cols)
        probs_test = model.predict_proba(X_test[cols])
        preds_test = model.predict(X_test[cols])
        leaderboard.append(
            {
                "model": name,
                "feature_count": float(len(cols)),
                "accuracy": float(accuracy_score(y_test, preds_test)),
                "log_loss": float(log_loss(y_test, probs_test, labels=model.classes_)),
                "draw_pred_rate_test": float((pd.Series(preds_test) == "draw").mean()),
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(
        ["log_loss", "accuracy"], ascending=[True, False]
    )
    best_name = str(leaderboard_df.iloc[0]["model"])
    best_model, best_cols = fitted[best_name]

    all_probs = best_model.predict_proba(X[best_cols])
    all_preds = best_model.predict(X[best_cols])
    features_with_predictions = _attach_prediction_columns(
        features=features,
        probs=all_probs,
        preds=all_preds,
        classes=list(best_model.classes_),
        split_idx=split_idx,
    )

    best_test_probs = best_model.predict_proba(X_test[best_cols])
    best_test_preds = best_model.predict(X_test[best_cols])
    metrics: dict[str, float | int | str] = {
        "selected_model": best_name,
        "selected_feature_count": int(len(best_cols)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, best_test_preds)),
        "log_loss": float(log_loss(y_test, best_test_probs, labels=best_model.classes_)),
        "draw_pred_rate_test": float((pd.Series(best_test_preds) == "draw").mean()),
        "draw_actual_rate_test": float((y_test == "draw").mean()),
        "classification_report_text": classification_report(
            y_test, best_test_preds, digits=3, zero_division=0
        ),
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }
    return best_model, metrics, features_with_predictions


def run_step1_baseline(
    project_root: Path,
    lookback: int = 5,
    strength_window: int = DEFAULT_STRENGTH_WINDOW,
    home_away_lookback: int = DEFAULT_HOME_AWAY_LOOKBACK,
    elo_season_decay: float = DEFAULT_ELO_SEASON_DECAY,
) -> BaselineArtifacts:
    """Run baseline training pipeline and write model/data artifacts."""
    raw_path = project_root / "data" / "raw" / "epl_matches.csv"
    training_path = project_root / "data" / "processed" / "epl_baseline_features.csv"
    model_path = project_root / "models" / "epl_best_model.joblib"
    metrics_path = project_root / "models" / "epl_baseline_metrics.json"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    matches = load_epl_matches()
    matches.to_csv(raw_path, index=False)

    features = build_step1_features(
        matches=matches,
        lookback=lookback,
        strength_window=strength_window,
        home_away_lookback=home_away_lookback,
        elo_season_decay=elo_season_decay,
    )
    model, metrics, features_with_predictions = train_baseline_model(features)
    features_with_predictions.to_csv(training_path, index=False)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Step 1 baseline complete.")
    print(f"Raw data saved: {raw_path}")
    print(f"Feature data saved: {training_path}")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("\nSelected model:")
    print(f"  {metrics['selected_model']}")
    print("\nSelected-model metrics:")
    for key in ("train_rows", "test_rows", "accuracy", "log_loss", "draw_pred_rate_test"):
        print(f"  {key}: {metrics[key]}")
    print("\nLeaderboard:")
    for row in metrics["leaderboard"]:
        print(
            f"  {row['model']}: accuracy={row['accuracy']:.4f}, "
            f"log_loss={row['log_loss']:.4f}, draw_pred_rate={row['draw_pred_rate_test']:.4f}"
        )

    return BaselineArtifacts(
        raw_data_path=raw_path,
        training_data_path=training_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
