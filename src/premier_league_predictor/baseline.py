"""Model training pipeline for Premier League match outcome prediction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
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

OUTCOME_LABELS = ["away_win", "draw", "home_win"]


@dataclass(frozen=True)
class BaselineArtifacts:
    raw_data_path: Path
    training_data_path: Path
    model_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class ModelComparisonResult:
    """Container for direct-vs-goals-vs-draw-aware comparison outputs."""

    logreg_metrics: dict[str, float | int | str]
    goal_based_metrics: dict[str, float | int | str]
    draw_aware_metrics: dict[str, float | int | str]
    summary_rows: list[dict[str, float | int | str]]


@dataclass(frozen=True)
class WalkForwardBacktestResult:
    """Container for rolling walk-forward evaluation outputs."""

    window_count: int
    summary_rows: list[dict[str, float | int | str]]
    window_rows: list[dict[str, float | int | str]]


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
    """Return league points earned by one side for a match result."""
    if side == "home":
        return 3 if result == "H" else 1 if result == "D" else 0
    return 3 if result == "A" else 1 if result == "D" else 0


def _mean_tail(records: list[dict[str, float]], key: str, n: int) -> float:
    """Average the last ``n`` values for ``key`` from a history list."""
    tail = records[-n:]
    return float(sum(float(r[key]) for r in tail) / len(tail))


def _goal_diff_mean_tail(records: list[dict[str, float]], n: int) -> float:
    """Average goal-difference over the last ``n`` records."""
    tail = records[-n:]
    return float(sum(float(r["goals_for"] - r["goals_against"]) for r in tail) / len(tail))


def _safe_rest_days(current_date: pd.Timestamp, previous_date: pd.Timestamp) -> float:
    """Return non-negative rest days between matches, capped to avoid huge outliers."""
    rest = float((current_date - previous_date).days)
    # Cap very long offseason breaks so model focuses on meaningful cadence differences.
    return float(max(0.0, min(rest, 30.0)))


def _expected_home_score(home_elo: float, away_elo: float, home_advantage: float) -> float:
    """Compute expected home result score from Elo ratings."""
    adjusted_home = home_elo + home_advantage
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def _home_result_score(result: str) -> float:
    """Map match result to Elo outcome score from home perspective."""
    if result == "H":
        return 1.0
    if result == "D":
        return 0.5
    if result == "A":
        return 0.0
    raise ValueError(f"Unexpected result label: {result}")


def _apply_elo_season_decay(team_elo: dict[str, float], decay: float) -> None:
    """Regress each team's Elo toward DEFAULT_ELO at season boundaries."""
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
            home_rest_days = _safe_rest_days(match_date, home_last_date)
            away_rest_days = _safe_rest_days(match_date, away_last_date)

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
    """Attach split labels and prediction/probability columns to feature rows."""
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


def _candidate_models() -> list[tuple[str, object, list[str]]]:
    """Return the candidate model set used across split and walk-forward evaluations."""
    return [
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


def _align_probabilities(
    probs: np.ndarray,
    model_classes: list[str],
    labels: list[str] = OUTCOME_LABELS,
) -> np.ndarray:
    """Align model class probabilities to a fixed label order."""
    aligned = np.full((probs.shape[0], len(labels)), 1e-9, dtype=float)
    class_to_idx = {name: idx for idx, name in enumerate(model_classes)}
    for col_idx, label in enumerate(labels):
        if label in class_to_idx:
            aligned[:, col_idx] = probs[:, class_to_idx[label]]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _safe_multiclass_log_loss(y_true: pd.Series, probs: np.ndarray) -> float:
    """Compute multiclass log loss with stable handling for degenerate slices."""
    try:
        return float(log_loss(y_true, probs, labels=OUTCOME_LABELS))
    except ValueError:
        fallback = np.full_like(probs, 1.0 / probs.shape[1], dtype=float)
        return float(log_loss(y_true, fallback, labels=OUTCOME_LABELS))


def _evaluate_model_on_window(
    *,
    model: object,
    feature_cols: list[str],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, float]:
    """Fit one candidate on a train window and score the forward test window."""
    fitted_model = clone(model)
    fitted_model.fit(train_frame[feature_cols], train_frame["target"])
    probs = _align_probabilities(
        fitted_model.predict_proba(test_frame[feature_cols]),
        list(fitted_model.classes_),
    )
    preds = fitted_model.predict(test_frame[feature_cols])
    return {
        "accuracy": float(accuracy_score(test_frame["target"], preds)),
        "log_loss": _safe_multiclass_log_loss(test_frame["target"], probs),
        "draw_pred_rate": float((pd.Series(preds) == "draw").mean()),
    }


def run_walk_forward_backtest(
    *,
    features: pd.DataFrame,
    train_size: int = 500,
    test_size: int = 38,
    step_size: int = 38,
    selection_metric: str = "accuracy",
) -> WalkForwardBacktestResult:
    """
    Evaluate candidate models in rolling chronological windows.

    Each window trains on ``train_size`` rows and predicts the next ``test_size``
    rows. Windows advance by ``step_size`` rows. This mimics repeated future
    forecasting better than a single static split.
    """
    if selection_metric not in {"accuracy", "log_loss"}:
        raise ValueError("selection_metric must be 'accuracy' or 'log_loss'")
    if train_size < 100:
        raise ValueError("train_size must be at least 100 rows")
    if test_size < 5:
        raise ValueError("test_size must be at least 5 rows")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")

    ordered = features.sort_values("date").reset_index(drop=True).copy()
    total_rows = len(ordered)
    if total_rows < train_size + test_size:
        raise ValueError("Not enough rows for requested walk-forward window sizes")

    per_window_model_rows: list[dict[str, float | int | str]] = []
    start = 0
    window_idx = 0
    candidates = _candidate_models()
    while True:
        train_start = start
        train_end = train_start + train_size
        if train_end >= total_rows:
            break
        test_end = min(train_end + test_size, total_rows)
        if test_end <= train_end:
            break

        train_frame = ordered.iloc[train_start:train_end]
        test_frame = ordered.iloc[train_end:test_end]
        if test_frame.empty:
            break

        current_rows: list[dict[str, float | int | str]] = []
        for model_name, model, cols in candidates:
            metrics = _evaluate_model_on_window(
                model=model,
                feature_cols=cols,
                train_frame=train_frame,
                test_frame=test_frame,
            )
            current_rows.append(
                {
                    "window_idx": window_idx,
                    "model": model_name,
                    "train_start_idx": train_start,
                    "train_end_idx": train_end - 1,
                    "test_start_idx": train_end,
                    "test_end_idx": test_end - 1,
                    "test_rows": len(test_frame),
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "draw_pred_rate": metrics["draw_pred_rate"],
                }
            )

        window_df = pd.DataFrame(current_rows)
        if selection_metric == "accuracy":
            window_df = window_df.sort_values(["accuracy", "log_loss"], ascending=[False, True])
        else:
            window_df = window_df.sort_values(["log_loss", "accuracy"], ascending=[True, False])
        selected_model = str(window_df.iloc[0]["model"])

        for row in window_df.to_dict(orient="records"):
            row["selected_model"] = selected_model
            row["is_selected"] = bool(row["model"] == selected_model)
            per_window_model_rows.append(row)

        window_idx += 1
        start += step_size
        if start + train_size >= total_rows:
            break

    if not per_window_model_rows:
        raise ValueError("Walk-forward backtest produced no windows")

    window_rows = pd.DataFrame(per_window_model_rows)
    grouped = window_rows.groupby("model", sort=False)
    summary = grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        log_loss_mean=("log_loss", "mean"),
        log_loss_std=("log_loss", "std"),
        draw_pred_rate_mean=("draw_pred_rate", "mean"),
        selected_count=("is_selected", "sum"),
    ).reset_index()
    summary["window_count"] = int(window_idx)
    for col in ("accuracy_std", "log_loss_std"):
        summary[col] = summary[col].fillna(0.0)
    summary = summary.sort_values(["accuracy_mean", "log_loss_mean"], ascending=[False, True])

    return WalkForwardBacktestResult(
        window_count=int(window_idx),
        summary_rows=summary.to_dict(orient="records"),
        window_rows=window_rows.to_dict(orient="records"),
    )


def train_baseline_model(
    features: pd.DataFrame,
) -> tuple[object, dict[str, float | int | str], pd.DataFrame]:
    """
    Train candidate models and return the best model by holdout log loss.

    Uses an 80/20 chronological split and compares the shared model set.
    """
    X = features
    y = features["target"]

    split_idx = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if X_test.empty:
        raise ValueError("Test set is empty. Need more rows or lower training split.")

    leaderboard: list[dict[str, float | str]] = []
    fitted: dict[str, tuple[object, list[str]]] = {}
    for name, model, cols in _candidate_models():
        fitted_model = clone(model)
        fitted_model.fit(X_train[cols], y_train)
        fitted[name] = (fitted_model, cols)

        probs_test = _align_probabilities(
            fitted_model.predict_proba(X_test[cols]),
            list(fitted_model.classes_),
        )
        preds_test = fitted_model.predict(X_test[cols])
        leaderboard.append(
            {
                "model": name,
                "feature_count": float(len(cols)),
                "accuracy": float(accuracy_score(y_test, preds_test)),
                "log_loss": float(log_loss(y_test, probs_test, labels=OUTCOME_LABELS)),
                "draw_pred_rate_test": float((pd.Series(preds_test) == "draw").mean()),
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(
        ["log_loss", "accuracy"], ascending=[True, False]
    )
    best_name = str(leaderboard_df.iloc[0]["model"])
    best_model, best_cols = fitted[best_name]

    all_probs = _align_probabilities(
        best_model.predict_proba(X[best_cols]),
        list(best_model.classes_),
    )
    all_preds = best_model.predict(X[best_cols])
    features_with_predictions = _attach_prediction_columns(
        features=features,
        probs=all_probs,
        preds=all_preds,
        classes=OUTCOME_LABELS,
        split_idx=split_idx,
    )

    best_test_probs = _align_probabilities(
        best_model.predict_proba(X_test[best_cols]),
        list(best_model.classes_),
    )
    best_test_preds = best_model.predict(X_test[best_cols])
    metrics: dict[str, float | int | str] = {
        "selected_model": best_name,
        "selected_feature_count": int(len(best_cols)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, best_test_preds)),
        "log_loss": float(log_loss(y_test, best_test_probs, labels=OUTCOME_LABELS)),
        "draw_pred_rate_test": float((pd.Series(best_test_preds) == "draw").mean()),
        "draw_actual_rate_test": float((y_test == "draw").mean()),
        "classification_report_text": classification_report(
            y_test, best_test_preds, digits=3, zero_division=0
        ),
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }
    return best_model, metrics, features_with_predictions


def _poisson_prob_vector(mean_goals: float, max_goals: int) -> np.ndarray:
    """Return normalized Poisson probabilities for goals 0..max_goals."""
    safe_mean = max(float(mean_goals), 1e-8)
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = math.exp(-safe_mean)
    for goals in range(1, max_goals + 1):
        probs[goals] = probs[goals - 1] * safe_mean / goals

    total = probs.sum()
    if total <= 0:
        probs[:] = 1.0 / len(probs)
    else:
        probs /= total
    return probs


def _outcome_probs_from_goal_means(
    home_mean: float,
    away_mean: float,
    max_goals: int,
) -> tuple[float, float, float]:
    """
    Convert expected home/away goals into outcome probabilities.

    Returns:
      (p_away_win, p_draw, p_home_win)
    """
    home_probs = _poisson_prob_vector(home_mean, max_goals=max_goals)
    away_probs = _poisson_prob_vector(away_mean, max_goals=max_goals)
    score_matrix = np.outer(home_probs, away_probs)
    score_matrix /= score_matrix.sum()

    p_home_win = float(np.tril(score_matrix, k=-1).sum())
    p_draw = float(np.trace(score_matrix))
    p_away_win = float(np.triu(score_matrix, k=1).sum())
    return p_away_win, p_draw, p_home_win


def train_goal_based_model(
    features: pd.DataFrame,
    max_goals: int = 10,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """
    Train a goals-first model (Poisson home/away goals), then derive outcomes.

    The model predicts expected goals for each side and converts those expected
    goals into win/draw/loss probabilities using a Poisson score matrix.
    """
    if max_goals < 4:
        raise ValueError("max_goals must be at least 4 for stable outcome conversion")

    X = features[FEATURE_COLUMNS]
    y = features["target"]
    y_home_goals = features["home_goals_actual"]
    y_away_goals = features["away_goals_actual"]

    split_idx = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    yh_train = y_home_goals.iloc[:split_idx]
    ya_train = y_away_goals.iloc[:split_idx]
    if X_test.empty:
        raise ValueError("Test set is empty. Need more rows or lower training split.")

    home_goal_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.05, max_iter=500)),
        ]
    )
    away_goal_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.05, max_iter=500)),
        ]
    )
    home_goal_model.fit(X_train, yh_train)
    away_goal_model.fit(X_train, ya_train)

    classes = ["away_win", "draw", "home_win"]
    test_home_means = home_goal_model.predict(X_test)
    test_away_means = away_goal_model.predict(X_test)
    test_probs = np.array(
        [
            _outcome_probs_from_goal_means(hm, am, max_goals=max_goals)
            for hm, am in zip(test_home_means, test_away_means)
        ]
    )
    test_pred = np.array(classes)[test_probs.argmax(axis=1)]

    metrics: dict[str, float | int | str] = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "log_loss": float(log_loss(y_test, test_probs, labels=classes)),
        "goal_model_max_goals": max_goals,
    }
    metrics["classification_report_text"] = classification_report(
        y_test, test_pred, digits=3, zero_division=0
    )

    all_home_means = home_goal_model.predict(X)
    all_away_means = away_goal_model.predict(X)
    all_probs = np.array(
        [
            _outcome_probs_from_goal_means(hm, am, max_goals=max_goals)
            for hm, am in zip(all_home_means, all_away_means)
        ]
    )
    all_pred = np.array(classes)[all_probs.argmax(axis=1)]

    features_with_predictions = features.copy()
    features_with_predictions["split"] = "train"
    features_with_predictions.loc[split_idx:, "split"] = "test"
    features_with_predictions["predicted_target"] = all_pred
    features_with_predictions["prediction_correct"] = (
        features_with_predictions["target"] == features_with_predictions["predicted_target"]
    )
    features_with_predictions["pred_home_goals_mean"] = np.maximum(all_home_means, 0.0)
    features_with_predictions["pred_away_goals_mean"] = np.maximum(all_away_means, 0.0)
    features_with_predictions["pred_prob_away_win"] = all_probs[:, 0]
    features_with_predictions["pred_prob_draw"] = all_probs[:, 1]
    features_with_predictions["pred_prob_home_win"] = all_probs[:, 2]

    return metrics, features_with_predictions


def _build_draw_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Build extra draw-sensitive features from baseline pre-match inputs."""
    frame = features[FEATURE_COLUMNS].copy()
    frame["abs_elo_diff"] = frame["elo_diff_pre"].abs()
    frame["abs_points_diff_last5"] = (frame["home_points_last5"] - frame["away_points_last5"]).abs()
    frame["abs_goal_diff_strength"] = (
        frame["home_goal_diff_per_match_strength"] - frame["away_goal_diff_per_match_strength"]
    ).abs()
    frame["expected_goal_total_proxy"] = (
        frame["home_goals_for_last5"] + frame["away_goals_for_last5"]
    ) / 2.0
    frame["expected_goal_diff_proxy"] = (
        frame["home_goals_for_last5"] - frame["away_goals_for_last5"]
    ).abs()
    return frame


def _compose_outcome_probs(
    p_draw: np.ndarray,
    p_home_non_draw: np.ndarray,
) -> np.ndarray:
    """Compose away/draw/home probabilities from staged model probabilities."""
    p_home = (1.0 - p_draw) * p_home_non_draw
    p_away = (1.0 - p_draw) * (1.0 - p_home_non_draw)
    probs = np.column_stack([p_away, p_draw, p_home])
    probs = np.clip(probs, 1e-8, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _predict_with_draw_threshold(
    p_draw: np.ndarray,
    p_home_non_draw: np.ndarray,
    draw_threshold: float,
) -> np.ndarray:
    """Apply draw decision threshold, then classify remaining matches home/away."""
    pred = np.where(p_home_non_draw >= 0.5, "home_win", "away_win")
    pred = pred.astype(object)
    pred[p_draw >= draw_threshold] = "draw"
    return pred.astype(str)


def train_draw_aware_model(
    features: pd.DataFrame,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """
    Train a draw-aware two-stage model and return metrics + row-level predictions.

    Stage 1: predict draw vs non-draw.
    Stage 2: for non-draw matches, predict home-win vs away-win.
    A draw threshold is tuned on a validation window of the training period.
    """
    X = _build_draw_feature_frame(features)
    y = features["target"]
    split_idx = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if X_test.empty:
        raise ValueError("Test set is empty. Need more rows or lower training split.")

    train_fit_end = int(len(X_train) * 0.85)
    train_fit_end = max(train_fit_end, 50)
    if train_fit_end >= len(X_train):
        train_fit_end = len(X_train) - 1
    X_fit, X_val = X_train.iloc[:train_fit_end], X_train.iloc[train_fit_end:]
    y_fit, y_val = y_train.iloc[:train_fit_end], y_train.iloc[train_fit_end:]

    draw_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=6000,
                    random_state=42,
                    class_weight={0: 1.0, 1: 1.4},
                    C=1.0,
                ),
            ),
        ]
    )
    y_draw_fit = (y_fit == "draw").astype(int)
    draw_model.fit(X_fit, y_draw_fit)

    non_draw_fit = y_fit != "draw"
    home_away_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=6000, random_state=42, C=1.0)),
        ]
    )
    y_home_fit = (y_fit[non_draw_fit] == "home_win").astype(int)
    home_away_model.fit(X_fit[non_draw_fit], y_home_fit)

    val_p_draw = draw_model.predict_proba(X_val)[:, 1]
    val_p_home_non_draw = home_away_model.predict_proba(X_val)[:, 1]
    val_actual_draw_rate = float((y_val == "draw").mean())
    best_threshold = 0.30
    best_score = -float("inf")
    best_acc = 0.0
    best_gap = float("inf")
    for threshold in np.arange(0.20, 0.351, 0.005):
        val_pred = _predict_with_draw_threshold(val_p_draw, val_p_home_non_draw, float(threshold))
        val_acc = float(accuracy_score(y_val, val_pred))
        val_draw_rate = float((val_pred == "draw").mean())
        draw_gap = abs(val_draw_rate - val_actual_draw_rate)
        score = val_acc - 0.08 * draw_gap
        if score > best_score or (score == best_score and val_acc > best_acc):
            best_score = score
            best_acc = val_acc
            best_gap = draw_gap
            best_threshold = float(threshold)

    draw_model.fit(X_train, (y_train == "draw").astype(int))
    non_draw_train = y_train != "draw"
    y_home_train = (y_train[non_draw_train] == "home_win").astype(int)
    home_away_model.fit(X_train[non_draw_train], y_home_train)

    classes = ["away_win", "draw", "home_win"]
    test_p_draw = draw_model.predict_proba(X_test)[:, 1]
    test_p_home_non_draw = home_away_model.predict_proba(X_test)[:, 1]
    test_probs = _compose_outcome_probs(test_p_draw, test_p_home_non_draw)
    test_pred = _predict_with_draw_threshold(
        test_p_draw,
        test_p_home_non_draw,
        draw_threshold=best_threshold,
    )

    metrics: dict[str, float | int | str] = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "log_loss": float(log_loss(y_test, test_probs, labels=classes)),
        "draw_threshold": best_threshold,
        "validation_draw_rate_gap": best_gap,
        "test_draw_pred_rate": float((test_pred == "draw").mean()),
        "test_draw_actual_rate": float((y_test == "draw").mean()),
    }
    metrics["classification_report_text"] = classification_report(
        y_test, test_pred, digits=3, zero_division=0
    )

    all_p_draw = draw_model.predict_proba(X)[:, 1]
    all_p_home_non_draw = home_away_model.predict_proba(X)[:, 1]
    all_probs = _compose_outcome_probs(all_p_draw, all_p_home_non_draw)
    all_pred = _predict_with_draw_threshold(
        all_p_draw,
        all_p_home_non_draw,
        draw_threshold=best_threshold,
    )

    features_with_predictions = features.copy()
    features_with_predictions["split"] = "train"
    features_with_predictions.loc[split_idx:, "split"] = "test"
    features_with_predictions["predicted_target"] = all_pred
    features_with_predictions["prediction_correct"] = (
        features_with_predictions["target"] == features_with_predictions["predicted_target"]
    )
    features_with_predictions["pred_prob_away_win"] = all_probs[:, 0]
    features_with_predictions["pred_prob_draw"] = all_probs[:, 1]
    features_with_predictions["pred_prob_home_win"] = all_probs[:, 2]
    features_with_predictions["pred_draw_stage_prob"] = all_p_draw
    features_with_predictions["pred_home_non_draw_prob"] = all_p_home_non_draw

    return metrics, features_with_predictions


def compare_outcome_vs_goal_models(
    features: pd.DataFrame,
    max_goals: int = 10,
) -> ModelComparisonResult:
    """
    Compare direct outcome classification versus goals-first Poisson modeling.

    Returns both metrics dicts and a compact summary table payload.
    """
    # Reuse the same feature matrix so model comparisons are like-for-like.
    _, logreg_metrics, logreg_rows = train_baseline_model(features)
    goal_metrics, goal_rows = train_goal_based_model(features, max_goals=max_goals)
    draw_aware_metrics, draw_aware_rows = train_draw_aware_model(features)

    def _draw_metrics(metrics: dict[str, float | int | str], rows: pd.DataFrame) -> dict[str, float]:
        test_rows = rows[rows["split"] == "test"]
        draw_pred_rate = float((test_rows["predicted_target"] == "draw").mean())
        draw_actual_rate = float((test_rows["target"] == "draw").mean())
        return {
            "accuracy": float(metrics["accuracy"]),
            "log_loss": float(metrics["log_loss"]),
            "draw_pred_rate": draw_pred_rate,
            "draw_actual_rate": draw_actual_rate,
        }

    logreg_draw = _draw_metrics(logreg_metrics, logreg_rows)
    goal_draw = _draw_metrics(goal_metrics, goal_rows)
    draw_aware_draw = _draw_metrics(draw_aware_metrics, draw_aware_rows)

    summary_rows = [
        {
            "model": "direct_logistic",
            **logreg_draw,
        },
        {
            "model": "goal_poisson",
            **goal_draw,
        },
        {
            "model": "draw_aware_staged",
            **draw_aware_draw,
        },
    ]
    return ModelComparisonResult(
        logreg_metrics=logreg_metrics,
        goal_based_metrics=goal_metrics,
        draw_aware_metrics=draw_aware_metrics,
        summary_rows=summary_rows,
    )


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
