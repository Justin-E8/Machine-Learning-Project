"""Step 1 baseline pipeline for Premier League outcome prediction."""

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
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEASON_CODES = ("1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526")
DEFAULT_ELO = 1500.0
ELO_K = 20.0
HOME_ELO_ADVANTAGE = 65.0
DEFAULT_STRENGTH_WINDOW = 20
DEFAULT_ELO_SEASON_DECAY = 0.75
FEATURE_COLUMNS = [
    "home_points_last5",
    "home_goals_for_last5",
    "home_goals_against_last5",
    "away_points_last5",
    "away_goals_for_last5",
    "away_goals_against_last5",
    "home_points_per_match_strength",
    "away_points_per_match_strength",
    "home_goal_diff_per_match_strength",
    "away_goal_diff_per_match_strength",
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


@dataclass(frozen=True)
class ModelComparisonResult:
    logreg_metrics: dict[str, float | int | str]
    goal_based_metrics: dict[str, float | int | str]
    draw_aware_metrics: dict[str, float | int | str]
    summary_rows: list[dict[str, float | int | str]]


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


def _home_score(result: str) -> float:
    """Convert full-time result to home-team match score for Elo updates."""
    if result == "H":
        return 1.0
    if result == "D":
        return 0.5
    if result == "A":
        return 0.0
    raise ValueError(f"Unexpected result label: {result}")


def _expected_home_score(home_elo: float, away_elo: float) -> float:
    """Return expected home-team score using Elo formula with home advantage."""
    adjusted_home = home_elo + HOME_ELO_ADVANTAGE
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def _apply_elo_season_decay(team_elo: dict[str, float], decay: float) -> None:
    """Regress Elo ratings toward the default at each new season."""
    for team, elo in team_elo.items():
        team_elo[team] = DEFAULT_ELO + decay * (elo - DEFAULT_ELO)


def build_step1_features(
    matches: pd.DataFrame,
    lookback: int = 5,
    strength_window: int = DEFAULT_STRENGTH_WINDOW,
    elo_season_decay: float = DEFAULT_ELO_SEASON_DECAY,
) -> pd.DataFrame:
    """Build rolling form and capped persistent-strength features from prior matches only."""
    if strength_window < lookback:
        raise ValueError("strength_window must be >= lookback")
    if not 0.0 <= elo_season_decay <= 1.0:
        raise ValueError("elo_season_decay must be between 0.0 and 1.0")

    team_history: dict[str, list[dict[str, float]]] = {}
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

        home = str(match["HomeTeam"])
        away = str(match["AwayTeam"])
        result = str(match["FTR"])
        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])

        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])
        home_elo_pre = team_elo.get(home, DEFAULT_ELO)
        away_elo_pre = team_elo.get(away, DEFAULT_ELO)

        if len(home_hist) >= lookback and len(away_hist) >= lookback:
            home_recent = pd.DataFrame(home_hist[-lookback:])
            away_recent = pd.DataFrame(away_hist[-lookback:])
            home_strength = pd.DataFrame(home_hist[-strength_window:])
            away_strength = pd.DataFrame(away_hist[-strength_window:])
            rows.append(
                {
                    "date": match["Date"],
                    "season_code": season_code,
                    "home_team": home,
                    "away_team": away,
                    "home_goals_actual": home_goals,
                    "away_goals_actual": away_goals,
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

        expected_home = _expected_home_score(home_elo_pre, away_elo_pre)
        actual_home = _home_score(result)
        team_elo[home] = home_elo_pre + ELO_K * (actual_home - expected_home)
        team_elo[away] = away_elo_pre + ELO_K * ((1.0 - actual_home) - (1.0 - expected_home))

    if not rows:
        raise ValueError("No training rows were generated. Lower lookback or check input data.")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def train_baseline_model(
    features: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float | int], pd.DataFrame]:
    """Train multinomial logistic regression and return model, metrics, and row-level predictions."""
    feature_cols = [
        "home_points_last5",
        "home_goals_for_last5",
        "home_goals_against_last5",
        "away_points_last5",
        "away_goals_for_last5",
        "away_goals_against_last5",
        "home_points_per_match_strength",
        "away_points_per_match_strength",
        "home_goal_diff_per_match_strength",
        "away_goal_diff_per_match_strength",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
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

    metrics: dict[str, float | int | str] = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_prob, labels=model.classes_)),
    }
    metrics["classification_report_text"] = classification_report(
        y_test, y_pred, digits=3, zero_division=0
    )

    # Add row-level predictions so users can inspect actual vs predicted outcomes in Data Wrangler.
    all_pred = model.predict(X)
    all_prob = model.predict_proba(X)
    features_with_predictions = features.copy()
    features_with_predictions["split"] = "train"
    features_with_predictions.loc[split_idx:, "split"] = "test"
    features_with_predictions["predicted_target"] = all_pred
    features_with_predictions["prediction_correct"] = (
        features_with_predictions["target"] == features_with_predictions["predicted_target"]
    )
    for class_idx, class_name in enumerate(model.classes_):
        features_with_predictions[f"pred_prob_{class_name}"] = all_prob[:, class_idx]

    return model, metrics, features_with_predictions


def _poisson_prob_vector(mean_goals: float, max_goals: int) -> np.ndarray:
    """
    Return Poisson probabilities for goals 0..max_goals.

    The vector is normalized to handle small truncation mass.
    """
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
    yh_train, yh_test = y_home_goals.iloc[:split_idx], y_home_goals.iloc[split_idx:]
    ya_train, ya_test = y_away_goals.iloc[:split_idx], y_away_goals.iloc[split_idx:]
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
    """Build a draw-focused feature matrix from baseline pre-match features."""
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

    # Reserve a validation tail inside training for threshold tuning.
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

    # Tune draw threshold on validation to balance accuracy and draw-rate realism.
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

    # Refit staged models on full training window.
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
    elo_season_decay: float = DEFAULT_ELO_SEASON_DECAY,
) -> BaselineArtifacts:
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

    features = build_step1_features(
        matches=matches,
        lookback=lookback,
        strength_window=strength_window,
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
    print("\nMetrics:")
    for key in ("train_rows", "test_rows", "accuracy", "log_loss"):
        print(f"  {key}: {metrics[key]}")

    return BaselineArtifacts(
        raw_data_path=raw_path,
        training_data_path=training_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
