# Inflation Nowcasting App

An end-to-end machine learning starter project for forecasting next-month U.S.
inflation (CPI month-over-month change) using public macroeconomic time series.

This project is intentionally structured so you can build it step-by-step in
VSCode:
- Start as a Python ML project (data pipeline + model training).
- Upgrade to an API service.
- Optionally add a frontend and deploy.

## 1) Project Structure

```text
.
├── app/
│   └── main.py                        # FastAPI service
├── data/
│   ├── processed/.gitkeep
│   └── raw/.gitkeep
├── models/
│   └── .gitkeep
├── notebooks/                         # Optional EDA notebooks
├── scripts/
│   ├── run_training.py                # Download data + train/select model
│   ├── run_forecast.py                # Generate latest one-step forecast
│   └── run_pipeline.py                # Convenience wrapper (train + forecast)
├── src/
│   └── inflation_nowcasting/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── features.py
│       ├── modeling.py
│       └── pipeline.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## 2) Quickstart (VSCode-friendly)

Open this repo in VSCode and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the baseline pipeline:

```bash
python scripts/run_pipeline.py --refresh-data
```

You should get these generated artifacts:
- `data/raw/macro_data.csv`
- `data/processed/training_frame.csv`
- `models/best_model.joblib`
- `models/metrics.json`
- `models/latest_forecast.json`

Start the API:

```bash
uvicorn app.main:app --reload
```

Then open:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/forecast`

## 3) What the baseline pipeline does

1. Downloads macro series from FRED:
   - `CPIAUCSL` (target base series)
   - `UNRATE`, `PAYEMS`, `FEDFUNDS`, `DGS10`, `DGS2`, `PPIACO`
2. Builds lag-based and percent-change features.
3. Trains two baseline models:
   - Ridge Regression
   - HistGradientBoostingRegressor
4. Picks the best model by RMSE on a time-based holdout window.
5. Saves model, metrics, and latest forecast artifact.

## 4) Suggested next milestones

- **Milestone A (current):** baseline pipeline + API.
- **Milestone B:** richer feature engineering (term spread, rolling stats, YoY).
- **Milestone C:** experiment tracking (MLflow or lightweight CSV logging).
- **Milestone D:** frontend dashboard (React or Streamlit) + deployment.

## 5) Notes

- Data downloads require internet access to FRED endpoints.
- Baseline data ingestion uses FRED public CSV endpoints (no API key required).
- This starter prioritizes clean engineering and reproducible workflows.