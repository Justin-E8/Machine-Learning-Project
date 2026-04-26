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


def _show_sidebar() -> None:
    st.sidebar.title("Data refresh")
    st.sidebar.markdown(
        "Run these in terminal to refresh files before reloading the page:"
        "\n\n"
        "`python3 scripts/run_epl_baseline.py`\n\n"
        "`python3 scripts/predict_upcoming_fixtures.py`"
    )


def _show_metrics() -> None:
    st.subheader("Current baseline metrics")
    if not METRICS_PATH.exists():
        st.info("No baseline metrics found yet. Run `python3 scripts/run_epl_baseline.py`.")
        return
    metrics = pd.read_json(METRICS_PATH, typ="series")
    metric_cols = ["selected_model", "accuracy", "log_loss", "train_rows", "test_rows"]
    values = {col: metrics.get(col) for col in metric_cols}
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", str(values.get("selected_model", "-")))
    c2.metric("Test accuracy", f"{float(values.get('accuracy', 0.0)):.4f}")
    c3.metric("Test log loss", f"{float(values.get('log_loss', 0.0)):.4f}")
    st.caption(f"Train rows: {int(values.get('train_rows', 0))} | Test rows: {int(values.get('test_rows', 0))}")


def _show_upcoming() -> None:
    st.subheader("Upcoming fixture predictions")
    upcoming = _load_csv(UPCOMING_PATH)
    if upcoming.empty:
        st.info("No upcoming predictions found yet. Run `python3 scripts/predict_upcoming_fixtures.py`.")
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
        st.caption("Most confident predictions")
        top_conf = upcoming.sort_values("prediction_confidence", ascending=False).head(10)
        st.dataframe(
            top_conf[[c for c in display_cols if c in top_conf.columns]],
            use_container_width=True,
            hide_index=True,
        )


def _show_completed() -> None:
    st.subheader("Completed match predictions (latest run)")
    completed = _load_csv(COMPLETED_PATH)
    if completed.empty:
        st.info("No completed predictions found yet. Run `python3 scripts/predict_upcoming_fixtures.py`.")
        return

    if {"split", "prediction_correct"}.issubset(completed.columns):
        test_rows = completed[completed["split"] == "test"]
        if not test_rows.empty:
            acc = float(test_rows["prediction_correct"].mean())
            st.metric("Test-split accuracy from completed predictions", f"{acc:.4f}")

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
    st.caption("No-odds model outputs from the current baseline pipeline.")

    _show_sidebar()
    _show_metrics()
    st.divider()
    _show_upcoming()
    st.divider()
    _show_completed()


if __name__ == "__main__":
    main()
