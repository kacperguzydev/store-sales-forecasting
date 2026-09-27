from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Store Sales Forecasting API")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

# --- Load resources once at startup ---
with open(MODELS_DIR / "model_metadata.json") as f:
    metadata = json.load(f)

FEATURE_COLS = metadata["feature_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
SEEDS = metadata["seeds"]
WEIGHTS = metadata["ensemble_weights"]
ZERO_FORCE_COMBOS = set(tuple(c) for c in metadata["zero_force_combos"])

lgbm_models = {s: joblib.load(MODELS_DIR / f"lgbm_seed{s}.pkl") for s in SEEDS}
xgb_models = {s: joblib.load(MODELS_DIR / f"xgb_seed{s}.pkl") for s in SEEDS}

history = pd.read_parquet(BASE_DIR / "history.parquet")
history["date"] = pd.to_datetime(history["date"])

stores = pd.read_csv(BASE_DIR / "stores.csv")
oil = pd.read_csv(BASE_DIR / "oil_filled.csv", parse_dates=["date"])
holidays = pd.read_csv(BASE_DIR / "holidays.csv", parse_dates=["date"])

# Precompute the "is holiday / is work day" lookup once, using the same logic as training
real_holidays = holidays[(holidays["type"] == "Holiday") & (holidays["transferred"] == False)]
transfer_days = holidays[holidays["type"] == "Transfer"]
additional_days = holidays[holidays["type"].isin(["Additional", "Bridge"])]
non_working_days = pd.concat([real_holidays, transfer_days, additional_days])[["date"]].drop_duplicates()
non_working_dates = set(non_working_days["date"])

work_days = holidays[holidays["type"] == "Work Day"][["date"]].drop_duplicates()
work_dates = set(work_days["date"])

oil_lookup = oil.set_index("date")["oil_price"].to_dict()
stores_lookup = stores.set_index("store_nbr").to_dict(orient="index")

print("Model and data resources loaded successfully.")


# --- Request / response schemas ---
class PredictionRequest(BaseModel):
    store_nbr: int
    family: str
    date: str  # format: YYYY-MM-DD
    onpromotion: int = 0


class PredictionResponse(BaseModel):
    predicted_sales: float
    store_nbr: int
    family: str
    date: str


# --- Feature computation for a single request ---
def build_features(store_nbr: int, family: str, target_date: pd.Timestamp, onpromotion: int) -> pd.DataFrame:
    combo_history = history[
        (history["store_nbr"] == store_nbr) & (history["family"] == family) & (history["date"] < target_date)
    ].sort_values("date")

    if combo_history.empty and store_nbr not in stores_lookup:
        raise HTTPException(status_code=404, detail=f"Unknown store_nbr: {store_nbr}")

    def sales_on_lag(lag_days):
        lag_date = target_date - pd.Timedelta(days=lag_days)
        row = combo_history[combo_history["date"] == lag_date]
        return float(row["sales"].iloc[0]) if len(row) else np.nan

    def rolling_stat(window_days, stat="mean"):
        window_start = target_date - pd.Timedelta(days=window_days)
        window_data = combo_history[
            (combo_history["date"] >= window_start) & (combo_history["date"] < target_date)
        ]["sales"]
        if window_data.empty:
            return 0.0
        return float(window_data.mean()) if stat == "mean" else float(window_data.std() or 0.0)

    promo_window = combo_history[
        combo_history["date"] >= target_date - pd.Timedelta(days=90)
    ]["onpromotion"]
    promo_typical = float(promo_window.mean()) if not promo_window.empty else 0.0
    promo_relative = onpromotion / (promo_typical + 1)

    store_info = stores_lookup.get(store_nbr, {"city": "Unknown", "state": "Unknown", "type": "Unknown", "cluster": -1})

    oil_price = oil_lookup.get(target_date, None)
    if oil_price is None:
        # fall back to the nearest available date if the exact date isn't in our oil table
        nearest = min(oil_lookup.keys(), key=lambda d: abs((d - target_date).days))
        oil_price = oil_lookup[nearest]

    is_holiday = 1 if target_date in non_working_dates else 0
    is_work_day = 1 if target_date in work_dates else 0
    if is_work_day:
        is_holiday = 0

    row = {
        "store_nbr": store_nbr,
        "family": family,
        "onpromotion": onpromotion,
        "day_of_week": target_date.dayofweek,
        "day_of_month": target_date.day,
        "month": target_date.month,
        "year": target_date.year,
        "is_weekend": int(target_date.dayofweek in [5, 6]),
        "day_of_year": target_date.dayofyear,
        "is_holiday": is_holiday,
        "is_work_day": is_work_day,
        "oil_price": oil_price,
        "city": store_info["city"],
        "state": store_info["state"],
        "type": store_info["type"],
        "cluster": store_info["cluster"],
        "lag_7": sales_on_lag(7),
        "lag_14": sales_on_lag(14),
        "lag_28": sales_on_lag(28),
        "lag_364": sales_on_lag(364),
        "rolling_mean_7": rolling_stat(7, "mean"),
        "rolling_std_7": rolling_stat(7, "std"),
        "rolling_mean_28": rolling_stat(28, "mean"),
        "rolling_std_28": rolling_stat(28, "std"),
        "promo_typical": promo_typical,
        "promo_relative": promo_relative,
    }

    df = pd.DataFrame([row])
    for col in ["lag_7", "lag_14", "lag_28", "lag_364"]:
        df[col] = df[col].fillna(0)

    return df[FEATURE_COLS]


# --- Ensemble prediction ---
def predict_ensemble(features: pd.DataFrame, store_nbr: int, family: str) -> float:
    if (store_nbr, family) in ZERO_FORCE_COMBOS:
        return 0.0

    features_lgbm = features.copy()
    for col in CATEGORICAL_COLS:
        features_lgbm[col] = features_lgbm[col].astype("category")

    features_xgb = features.copy()
    for col in CATEGORICAL_COLS:
        features_xgb[col] = features_xgb[col].astype("category").cat.codes

    lgbm_preds = [np.expm1(lgbm_models[s].predict(features_lgbm)[0]) for s in SEEDS]
    xgb_preds = [np.expm1(xgb_models[s].predict(features_xgb)[0]) for s in SEEDS]

    lgbm_avg = np.clip(np.mean(lgbm_preds), 0, None)
    xgb_avg = np.clip(np.mean(xgb_preds), 0, None)

    final_pred = WEIGHTS["xgboost"] * xgb_avg + WEIGHTS["lightgbm"] * lgbm_avg
    return float(max(final_pred, 0.0))


# --- Endpoints ---
@app.get("/")
def health_check():
    """Simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Store Sales Forecasting API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predicts sales for a given store, product family, and date using
    a LightGBM + XGBoost multi-seed ensemble trained on historical data.
    """
    try:
        target_date = pd.Timestamp(request.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")

    if request.family not in history["family"].unique():
        raise HTTPException(status_code=404, detail=f"Unknown family: {request.family}")

    features = build_features(request.store_nbr, request.family, target_date, request.onpromotion)
    prediction = predict_ensemble(features, request.store_nbr, request.family)

    return PredictionResponse(
        predicted_sales=round(prediction, 2),
        store_nbr=request.store_nbr,
        family=request.family,
        date=request.date
    )