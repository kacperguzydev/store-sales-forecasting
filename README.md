# Store Sales Forecasting

Time series forecasting of daily grocery sales for Corporación Favorita (Ecuador), built as an end-to-end ML project: EDA, feature engineering, model comparison, a deployed prediction API, tests, CI, and a dashboard.

- Competition: [Store Sales - Time Series Forecasting (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- Live API docs: https://store-sales-forecasting.onrender.com/docs (free tier, the first request after idle can take about a minute)
- Metric: RMSLE, horizon: 16 days, 54 stores x 33 product families (1,782 parallel series)

## Results

| Approach | Kaggle RMSLE |
|---|---|
| LightGBM, recursive, first version | 0.56084 |
| + Christmas rows, earthquake correction, zero-forcing, promo features, weighted LightGBM + XGBoost ensemble | 0.51303 |
| + 3-seed averaging | **0.50266** |

Local validation (last 16 days of training data, honest split without leakage) reached about 0.398, so there is a known gap of roughly 0.10 between validation and the leaderboard. I traced part of it (see "What went wrong along the way") but did not fully close it.

## Project structure

```
store-sales-forecasting/
├── notebooks/
│   ├── 01_eda.ipynb                    # exploration of all 7 source files
│   ├── 02_feature_engineering.ipynb    # calendar, holidays, oil, lags, rolling stats, promo features
│   ├── 03_baseline_model.ipynb         # baseline, zero-forcing, recursive validation, direct-horizon experiment
│   ├── 04_model_comparison.ipynb       # 8 models, ensembles, weight search, multi-seed
│   └── 05_export_for_api.ipynb         # exports models, metadata and data bundle for the API
├── api/                                # FastAPI service (main.py + small data bundle)
├── dashboard/                          # Streamlit app calling the API
├── models/                             # LightGBM + XGBoost models, metadata, drift reference
├── tests/                              # pytest suite for the API
├── .github/workflows/tests.yml         # CI: pytest on every push
├── Dockerfile
└── requirements.txt
```

## Approach

**Data.** Seven relational files (sales, stores, oil, holidays, transactions). Sales are heavily right-skewed with 31% zeros, so the model is trained on `log1p(sales)`, which makes the L2 loss equal to the competition metric.

**Feature engineering.**
- Calendar features and a holiday flag that handles transferred holidays, `Transfer`, `Bridge`, `Additional` and `Work Day` rows correctly
- Oil price, interpolated over weekends
- Lags (7, 14, 28, 364), rolling mean/std (7, 28 days), promotion relative to each series' 90-day norm
- Missing Christmas Day rows re-inserted, April 2016 earthquake window replaced with same-weekday averages from surrounding weeks
- `transactions.csv` deliberately excluded: it does not exist for the test period (data leakage)

**Validation.** Time-based splits only (`TimeSeriesSplit`, plus a held-out final 16 days), never random splits.

**Forecasting.** The test window is 16 days ahead with no ground truth, so lags for later days must use earlier predictions. Predictions are generated recursively, day by day, recomputing lag and rolling features from actuals plus previous predictions.

**Post-processing.** Series with no sales in the last 365 days are forced to exactly zero (65 store x family combinations).

**Models.** Eight models compared under the same recursive validation: XGBoost, LightGBM, CatBoost, HistGradientBoosting, Random Forest, Extra Trees (Ridge/ElasticNet were dropped after they diverged). The best result came from a weighted blend of the two best models (70% XGBoost, 30% LightGBM), and adding weaker models made the ensemble worse. Hyperparameters for LightGBM came from Optuna (30 trials).

## What went wrong along the way

- **Leakage in my own validation (twice).** First I validated a model that had been trained on the validation window, which gave an optimistic 0.39. After fixing it, a quick non-recursive check again used lag features computed from real sales. Both were caught by checking training/validation date overlap and recomputing lags in isolation.
- **NaN lags in the test period.** The first submission scored 2.18 because lags for the later test days pointed into the unknown test period. Recursive forecasting fixed it (0.56).
- **Seasonality that does not repeat.** `SCHOOL AND OFFICE SUPPLIES` (back-to-school ramp) and `GROCERY II` dominate the largest errors. Year-ago lags did not help because the ramp shifts and grows between years.
- **Validation vs leaderboard gap.** Added features (year-ago lags, day of year) and a direct per-horizon model were neutral on validation, and a large gap remained. I did not fully explain it.
- **Deployment memory.** The first Docker image used about 700 MiB, above the 512 MB limit of the free Render tier. Shortening the history bundle to 400 days and using a single seed brought it to about 320 MiB.

## API

`POST /predict`

```json
{ "store_nbr": 1, "family": "BEVERAGES", "date": "2017-08-20", "onpromotion": 5 }
```

Response:

```json
{
  "predicted_sales": 842.33,
  "store_nbr": 1,
  "family": "BEVERAGES",
  "date": "2017-08-20",
  "drift_warnings": []
}
```

The API builds lag, rolling and promotion features from a bundled 400-day history, applies zero-forcing, and returns a LightGBM + XGBoost blend.

**Logging.** Every prediction is logged as one JSON line (inputs, output, latency, drift warnings) to stdout, visible in the Render logs.

**Drift check.** Inputs (`onpromotion`, `rolling_mean_28` per family, `oil_price`) are compared with training statistics; a z-score above 4 adds a warning to the response and a WARNING log line.

**Known limitations.**
- The history ends on 2017-08-15. For dates more than about a week later, lag features cannot be computed and fall back to 0, so predictions become unreliable (the dashboard warns about this).
- The API predicts a single date and does not run the recursive multi-day loop used for the Kaggle submission.
- The API uses one seed, while the Kaggle score (0.50266) used a 3-seed average.

## Run it

```bash
pip install -r requirements.txt

# API
uvicorn api.main:app --reload

# tests
pytest tests/ -v

# dashboard (uses the deployed API by default)
streamlit run dashboard/app.py
# to use a local API instead:  API_URL=http://127.0.0.1:8000 streamlit run dashboard/app.py

# Docker
docker build -t store-sales-api .
docker run -p 8000:8000 store-sales-api
```

To rerun the notebooks, download the competition data from Kaggle into `data/` (not included in the repo).

## Stack

Python, pandas, LightGBM, XGBoost, CatBoost, scikit-learn, Optuna, SHAP, FastAPI, Streamlit, Plotly, pytest, Docker, GitHub Actions, Render.