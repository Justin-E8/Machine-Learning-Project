"""Streamlit dashboard for Premier League prediction outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPCOMING_PATH = PROJECT_ROOT / "data" / "processed" / "epl_upcoming_predictions.csv"
COMPLETED_PATH = PROJECT_ROOT / "data" / "processed" / "epl_completed_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "epl_baseline_metrics.json"


@st.cache_data(show_spinner=False)
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_metrics(path: Path) -> dict[str, float | int | str]:
    if not path.exists():
        return {}
    try:
        metrics = pd.read_json(path, typ="series")
        return dict(metrics.to_dict())
    except Exception:
        return {}


def _show_sidebar(upcoming: pd.DataFrame, completed: pd.DataFrame) -> tuple[set[str], set[str], float]:
    st.sidebar.title("Dashboard filters")
    st.sidebar.caption("Use these to focus the tables.")

    teams: list[str] = []
    if not upcoming.empty:
        teams.extend(upcoming.get("home_team", pd.Series(dtype=str)).dropna().astype(str).tolist())
        teams.extend(upcoming.get("away_team", pd.Series(dtype=str)).dropna().astype(str).tolist())
    if not completed.empty:
        teams.extend(completed.get("home_team", pd.Series(dtype=str)).dropna().astype(str).tolist())
        teams.extend(completed.get("away_team", pd.Series(dtype=str)).dropna().astype(str).tolist())
    team_options = sorted(set(teams))

    selected_teams = set(
        st.sidebar.multiselect(
            "Filter by teams (home or away)",
            options=team_options,
            default=[],
            help="Leave empty to show all teams.",
        )
    )
    conf_threshold = float(
        st.sidebar.slider(
            "Minimum prediction confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="Applies to upcoming predictions.",
        )
    )

    split_options = ["all", "test", "train"]
    selected_splits = set(
        st.sidebar.multiselect(
            "Completed rows split filter",
            options=split_options,
            default=["all"],
            help="Pick 'all' or specific split groups.",
        )
    )

    st.sidebar.divider()
    st.sidebar.subheader("Data refresh")
    st.sidebar.markdown(
        "Run these in terminal to refresh files before reloading the page:"
        "\n\n"
        "`python3 scripts/run_epl_baseline.py`\n\n"
        "`python3 scripts/predict_upcoming_fixtures.py`"
    )
    return selected_teams, selected_splits, conf_threshold


def _filter_by_teams(df: pd.DataFrame, selected_teams: set[str]) -> pd.DataFrame:
    if df.empty or not selected_teams:
        return df
    if {"home_team", "away_team"}.issubset(df.columns):
        mask = df["home_team"].isin(selected_teams) | df["away_team"].isin(selected_teams)
        return df[mask].copy()
    return df


def _show_summary_cards(
    metrics: dict[str, float | int | str],
    upcoming: pd.DataFrame,
    completed: pd.DataFrame,
) -> None:
    st.subheader("Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", str(metrics.get("selected_model", "-")))
    c2.metric("Baseline test accuracy", f"{float(metrics.get('accuracy', 0.0)):.4f}")
    c3.metric("Upcoming fixtures shown", f"{len(upcoming)}")
    if {"split", "prediction_correct"}.issubset(completed.columns):
        test_rows = completed[completed["split"] == "test"]
        test_acc = float(test_rows["prediction_correct"].mean()) if not test_rows.empty else 0.0
        c4.metric("Completed test accuracy", f"{test_acc:.4f}")
    else:
        c4.metric("Completed rows shown", f"{len(completed)}")
    st.caption(
        f"Train rows: {int(metrics.get('train_rows', 0))} | "
        f"Test rows: {int(metrics.get('test_rows', 0))} | "
        f"Log loss: {float(metrics.get('log_loss', 0.0)):.4f}"
    )


def _show_upcoming(upcoming: pd.DataFrame) -> None:
    st.subheader("Upcoming fixture predictions")
    if upcoming.empty:
        st.info("No upcoming predictions found for current filters.")
        return

    display_cols = [
        col
        for col in [
            "date",
            "time",
            "home_team",
            "away_team",
            "predicted_target",
            "prediction_confidence",
            "pred_prob_home_win",
            "pred_prob_draw",
            "pred_prob_away_win",
        ]
        if col in upcoming.columns
    ]
    st.dataframe(upcoming[display_cols], use_container_width=True, hide_index=True)

    if {"predicted_target", "prediction_confidence"}.issubset(upcoming.columns):
        st.caption("Top 10 most confident upcoming predictions")
        top_conf = upcoming.sort_values("prediction_confidence", ascending=False).head(10)
        st.dataframe(
            top_conf[[c for c in display_cols if c in top_conf.columns]],
            use_container_width=True,
            hide_index=True,
        )


def _show_completed(completed: pd.DataFrame) -> None:
    st.subheader("Completed match predictions")
    if completed.empty:
        st.info("No completed predictions found for current filters.")
        return

    display_cols = [
        col
        for col in [
            "date",
            "home_team",
            "away_team",
            "target",
            "predicted_target",
            "prediction_correct",
            "split",
            "pred_prob_home_win",
            "pred_prob_draw",
            "pred_prob_away_win",
        ]
        if col in completed.columns
    ]
    st.dataframe(completed[display_cols], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="EPL Predictor Dashboard", layout="wide")
    st.title("Premier League Prediction Dashboard")
    st.caption("Cleaner view of no-odds baseline outputs and prediction tables.")

    upcoming = _load_csv(UPCOMING_PATH)
    completed = _load_csv(COMPLETED_PATH)
    metrics = _load_metrics(METRICS_PATH)

    selected_teams, selected_splits, conf_threshold = _show_sidebar(upcoming, completed)

    upcoming_filtered = _filter_by_teams(upcoming, selected_teams)
    if "prediction_confidence" in upcoming_filtered.columns:
        upcoming_filtered = upcoming_filtered[upcoming_filtered["prediction_confidence"] >= conf_threshold].copy()

    completed_filtered = _filter_by_teams(completed, selected_teams)
    if "all" not in selected_splits and "split" in completed_filtered.columns and selected_splits:
        completed_filtered = completed_filtered[completed_filtered["split"].isin(selected_splits)].copy()

    _show_summary_cards(metrics, upcoming_filtered, completed_filtered)
    st.divider()
    _show_upcoming(upcoming_filtered)
    st.divider()
    _show_completed(completed_filtered)


if __name__ == "__main__":
    main()
