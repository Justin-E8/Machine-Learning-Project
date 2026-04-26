# Premier League Match Predictor

This repository is EPL-only and focused on one task:
predicting match outcomes (`home_win`, `draw`, `away_win`) from historical match data.

The current pipeline uses:
- rolling form features
- home/away split form
- rest-day and momentum signals
- Elo-based team strength with season decay

No inflation code remains in this branch.

## Project structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── scripts/
│   ├── run_epl_baseline.py
│   ├── benchmark_models.py
│   ├── compare_goal_model.py
│   ├── predict_upcoming_fixtures.py
│   └── run_walk_forward.py
├── src/premier_league_predictor/
│   ├── baseline.py
│   └── upcoming.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup (VSCode terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Run commands

### 1) Train baseline model and export training artifacts

```bash
python3 scripts/run_epl_baseline.py
```

Outputs:
- `data/raw/epl_matches.csv`
- `data/processed/epl_baseline_features.csv`
- `models/epl_best_model.joblib`
- `models/epl_baseline_metrics.json`

### 2) Benchmark model families (logistic core/enhanced + boosting)

```bash
python3 scripts/benchmark_models.py
```

Output:
- `models/epl_model_benchmark.json`

### 3) Compare outcome modeling strategies
(direct classifier vs goals-first Poisson vs draw-aware staged model)

```bash
python3 scripts/compare_goal_model.py
```

Output:
- `models/epl_model_comparison.json`

### 4) Predict upcoming fixtures

```bash
python3 scripts/predict_upcoming_fixtures.py
```

Optional date filter:

```bash
python3 scripts/predict_upcoming_fixtures.py --from-date 2026-04-26
```

Outputs:
- `data/raw/epl_upcoming_fixtures.csv`
- `data/processed/epl_upcoming_predictions.csv`
- `data/processed/epl_upcoming_skipped.csv`
- `data/processed/epl_completed_predictions.csv`
- `models/epl_upcoming_predictions.json`
- `models/epl_upcoming_training_metrics.json`

### 5) Walk-forward evaluation (best proxy for upcoming-match accuracy)

This backtest repeatedly trains on a rolling historical window and predicts the
next block of future matches. It is closer to real deployment than one static split.

```bash
python3 scripts/run_walk_forward.py
```

Optional example:

```bash
python3 scripts/run_walk_forward.py --train-size 760 --test-size 38
```

Outputs:
- `models/epl_walk_forward_metrics.json`

## Current tuned defaults

The code is tuned for stronger recent-season behavior:
- `lookback=5`
- `strength_window=20`
- `home_away_lookback=2`
- `elo_season_decay=0.65`
- `elo_k=24`
- `home_elo_advantage=80`

You can inspect all options with:

```bash
python3 scripts/run_epl_baseline.py --help
python3 scripts/predict_upcoming_fixtures.py --help
```