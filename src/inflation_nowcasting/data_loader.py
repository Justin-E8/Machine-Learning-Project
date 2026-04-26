"""Data ingestion utilities for inflation nowcasting."""

from __future__ import annotations

from io import StringIO
from typing import Dict

import pandas as pd
import requests

from inflation_nowcasting.config import Settings


def _to_month_end(series_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to month-end timestamps."""
    frame = series_df.copy()
    frame.index = pd.to_datetime(frame.index).to_period("M").to_timestamp("M")
    return frame


def fetch_fred_series(
    series_map: Dict[str, str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download several FRED series and return one aligned dataframe."""
    frames: list[pd.DataFrame] = []
    for output_name, fred_code in series_map.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
        date_col = raw.columns[0]
        value_col = raw.columns[1]
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
        raw = raw.dropna(subset=[date_col, value_col]).set_index(date_col)
        if end is not None:
            data = raw.loc[start:end, [value_col]]
        else:
            data = raw.loc[start:, [value_col]]
        data = _to_month_end(data).rename(columns={value_col: output_name})
        frames.append(data)

    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.ffill().dropna()
    return combined


def fetch_macro_data(settings: Settings, refresh_data: bool = False) -> pd.DataFrame:
    """
    Fetch source macro data from FRED or load a cached copy.

    The cached copy is saved at `settings.raw_data_path`.
    """
    if settings.raw_data_path.exists() and not refresh_data:
        return pd.read_csv(settings.raw_data_path, index_col=0, parse_dates=True)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw = fetch_fred_series(settings.fred_series, settings.start_date, settings.end_date)
    raw.to_csv(settings.raw_data_path)
    return raw
