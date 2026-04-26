# Premier League Match Predictor (Step 1 Baseline)

This repository now focuses on a soccer project: predicting Premier League match
outcomes as one of three classes:

- `H`: Home win
- `D`: Draw
- `A`: Away win

Step 1 now focuses on improving out-of-sample accuracy (without bookmaker odds):
- download historical EPL results
- build stronger non-odds pre-match features (form, home/away splits, rest days, Elo)
- benchmark multiple model families
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

Benchmark multiple model families (logistic vs boosting) on the same feature set:

```bash
python3 scripts/benchmark_models.py
```

## 3) What Step 1 does

1. Downloads EPL CSV data from football-data.co.uk for seasons 2018-19 through
   2025-26 (completed so far).
2. Builds pre-match features using each team's prior matches only:
   - overall rolling form (points/goals for/goals against)
   - home-only and away-only rolling form windows
   - short-horizon momentum (last 3 matches)
   - capped strength window trends
   - rest-day features
   - pre-match Elo ratings with season decay
3. Benchmarks three model variants on a time-based split:
   - multinomial Logistic Regression
   - enhanced Logistic Regression (richer feature set)
   - HistGradientBoostingClassifier
4. Selects best model by holdout log loss, and reports leaderboard.
5. Writes artifacts:
   - `data/raw/epl_matches.csv`
   - `data/processed/epl_baseline_features.csv`
   - `models/epl_best_model.joblib`
   - `models/epl_baseline_metrics.json`
   - `models/epl_model_benchmark.json`

## 4) Next steps after Step 1

- Step 2: add richer match-intensity features (shots, shots on target, cards) where available.
- Step 3: add rolling time-series backtesting windows instead of one static split.
- Step 4: ship a website (Streamlit or FastAPI + frontend).