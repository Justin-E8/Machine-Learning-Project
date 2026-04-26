# Premier League Match Predictor

Predict EPL match outcomes (`home_win`, `draw`, `away_win`) using a single,
no-odds, feature-based machine-learning pipeline.

This repository is intentionally focused and cleaned up around one main model and one
dashboard workflow. It is designed so someone new to the project can run it quickly,
understand the data flow, and inspect outputs in both CSV and UI form.

---

## What this project does

1. Downloads historical EPL results.
2. Builds leak-free pre-match features (form, split form, strength, rest, Elo).
3. Trains one tuned logistic model on an 80/20 chronological split.
4. Exports row-level predictions for historical matches.
5. Predicts upcoming fixtures to the end of the current season.
6. Shows everything in a Streamlit dashboard.

---

## Repository structure

```text
.
├── dashboard/
│   └── app.py                              # Streamlit dashboard
├── data/
│   ├── raw/                                # downloaded raw CSVs
│   └── processed/                          # generated feature/prediction CSVs
├── models/
│   ├── epl_best_model.joblib               # trained model artifact
│   ├── epl_baseline_metrics.json           # baseline metrics summary
│   ├── epl_upcoming_predictions.json       # upcoming predictions in JSON
│   └── epl_upcoming_training_metrics.json  # metrics from upcoming run
├── scripts/
│   ├── run_epl_baseline.py                 # train + export baseline artifacts
│   └── predict_upcoming_fixtures.py        # train + predict upcoming fixtures
├── src/premier_league_predictor/
│   ├── __init__.py
│   ├── baseline.py                         # feature engineering + model training
│   └── upcoming.py                         # fixture sourcing + upcoming inference
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart (first run)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Then run:

```bash
python3 scripts/run_epl_baseline.py
python3 scripts/predict_upcoming_fixtures.py
python3 -m streamlit run dashboard/app.py
```

Open the Streamlit URL shown in terminal (usually `http://localhost:8501`).

---

## How to run the program (day-to-day)

Each new terminal session:

```bash
cd /path/to/Machine-Learning-Project
source .venv/bin/activate
```

Refresh artifacts:

```bash
python3 scripts/run_epl_baseline.py
python3 scripts/predict_upcoming_fixtures.py
```

Open dashboard:

```bash
python3 -m streamlit run dashboard/app.py
```

Optional upcoming cutoff:

```bash
python3 scripts/predict_upcoming_fixtures.py --from-date 2026-04-26
```

---

## Data sources

Historical and fixture data comes from public football-data feeds, with a fallback
full-season fixture schedule for broader remaining-match coverage:

- `football-data.co.uk` season CSVs (`mmz4281/.../E0.csv`) for historical matches
- `football-data.co.uk/fixtures.csv` for short-horizon upcoming fixtures
- OpenFootball (`football.json`) as a full-season fixture fallback

Team-name normalization is applied so these sources align with training-data team names.

---

## Core modeling approach

The project keeps only one tuned classifier path:

- Model: `LogisticRegression` in a `StandardScaler` pipeline
- Split: chronological 80% train / 20% test
- Features:
  - recent overall form (`last5`, `last3`)
  - home/away split form windows
  - capped persistent-strength windows
  - rest-day features
  - Elo pre-match ratings with season decay

No bookmaker odds are used.

---

## Generated artifacts

### Baseline run

Command:

```bash
python3 scripts/run_epl_baseline.py
```

Writes:

- `data/raw/epl_matches.csv`
- `data/processed/epl_baseline_features.csv`
- `models/epl_best_model.joblib`
- `models/epl_baseline_metrics.json`

`epl_baseline_features.csv` includes:
- `target` (actual result label)
- `predicted_target`
- `prediction_correct`
- `split` (`train` or `test`)
- `pred_prob_away_win`, `pred_prob_draw`, `pred_prob_home_win` (rounded to 2 decimals)

### Upcoming run

Command:

```bash
python3 scripts/predict_upcoming_fixtures.py
```

Writes:

- `data/raw/epl_upcoming_fixtures.csv`
- `data/processed/epl_upcoming_predictions.csv`
- `data/processed/epl_upcoming_skipped.csv`
- `data/processed/epl_completed_predictions.csv`
- `models/epl_upcoming_predictions.json`
- `models/epl_upcoming_training_metrics.json`

`epl_upcoming_predictions.csv` includes confidence and class probabilities rounded
to 2 decimals.

---

## Dashboard

Run:

```bash
python3 -m streamlit run dashboard/app.py
```

The dashboard provides:

- Upcoming predictions table
- Completed predictions table
- Team filters
- Confidence threshold filter
- Train/test split filter for completed rows
- Snapshot cards for key metrics

---

## Important concepts

- **target**: actual historical match outcome.
- **predicted_target**: model prediction.
- **prediction_correct**: whether prediction matched target.
- **train rows**: oldest 80% used to fit the model.
- **test rows**: newest 20% held out for evaluation metrics.

For project evaluation quality, use test rows.
For forecasting upcoming fixtures, the trained model is applied to future fixtures.

---

## Script options

Inspect all CLI options:

```bash
python3 scripts/run_epl_baseline.py --help
python3 scripts/predict_upcoming_fixtures.py --help
```

Current tuned defaults:

- `lookback=5`
- `strength_window=20`
- `home_away_lookback=2`
- `elo_season_decay=0.65`
- `elo_k=24`
- `home_elo_advantage=80`

---

## Troubleshooting

- If imports fail, re-run:
  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```
- If dashboard shows stale data, re-run both scripts, then refresh browser.
- If upcoming fixtures appear too short, run `predict_upcoming_fixtures.py` again;
  it merges multiple sources and removes already-completed matches.

---

## Code maintenance status

- Redundant legacy model-comparison scripts were removed.
- The repository now uses one primary model path and one upcoming-prediction path.
- README and in-file docstrings are aligned with current behavior.