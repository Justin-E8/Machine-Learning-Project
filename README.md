# Premier League Match Predictor (Step 1 Baseline)

This repository now focuses on a soccer project: predicting Premier League match
outcomes as one of three classes:

- `H`: Home win
- `D`: Draw
- `A`: Away win

Step 1 keeps things intentionally simple and easy to follow:
- download historical EPL results
- build recent-form features and persistent team-strength features
- train a baseline Logistic Regression model
- evaluate with accuracy and log loss

## 1) Project Structure

```text
.
├── data/
│   ├── processed/.gitkeep
│   └── raw/.gitkeep
├── models/
│   └── .gitkeep
├── scripts/
│   └── run_epl_baseline.py
├── src/
│   └── premier_league_predictor/
│       ├── __init__.py
│       └── baseline.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## 2) Quickstart (VSCode-friendly)

Run this inside your VSCode terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Then run Step 1 baseline:

```bash
python3 scripts/run_epl_baseline.py
```

Useful options:

```bash
python3 scripts/run_epl_baseline.py --lookback 5 --strength-window 20 --elo-season-decay 0.75
```

Predict upcoming fixtures (using completed matches through 2025/26 + fixtures feed):

```bash
python3 scripts/predict_upcoming_fixtures.py
```

Optional (date filter):

```bash
python3 scripts/predict_upcoming_fixtures.py --from-date 2026-04-19
```

## 3) What Step 1 does

1. Downloads EPL CSV data from football-data.co.uk for seasons 2018-19 through
   2025-26 (completed matches so far).
2. Builds pre-match features using each team's prior matches only:
   - rolling 5-match form (points/goals for/goals against)
   - capped-strength window stats (default last 20 matches)
   - pre-match Elo ratings and Elo difference with season-to-season decay
3. Trains multinomial Logistic Regression on a time-based split.
4. Prints baseline metrics and writes artifacts:
   - `data/raw/epl_matches.csv`
   - `data/processed/epl_baseline_features.csv`
   - `models/epl_logreg_baseline.joblib`
   - `models/epl_baseline_metrics.json`

## 4) Upcoming fixtures prediction

`scripts/predict_upcoming_fixtures.py` will:

1. Download historical/completed EPL results (including 2025/26 so far).
2. Train the same baseline model on completed matches.
3. Download upcoming EPL fixtures from `https://www.football-data.co.uk/fixtures.csv`.
4. Build pre-match features for each fixture from team state as-of now.
5. Save predictions to:
   - `data/processed/epl_upcoming_predictions.csv`
   - `models/epl_upcoming_predictions.json`
6. Refresh completed-match prediction snapshot:
   - `data/processed/epl_completed_predictions.csv`

## 5) Next steps after Step 1

- Step 2: add stronger features (home/away splits, Elo-style rating, rest days).
- Step 3: compare models (RandomForest/XGBoost) and calibration.
- Step 4: ship a website (Streamlit or FastAPI + frontend).