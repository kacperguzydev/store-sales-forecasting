import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://store-sales-forecasting.onrender.com")
HISTORY_PATH = Path(__file__).resolve().parent.parent / "api" / "history.parquet"

st.set_page_config(page_title="Store Sales Forecasting", layout="wide")
st.title("Store Sales Forecasting")
st.caption("Corporación Favorita | LightGBM + XGBoost ensemble served by a FastAPI service")


@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_parquet(HISTORY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


history = load_history()
families = sorted(history["family"].unique())
stores = sorted(int(s) for s in history["store_nbr"].unique())
last_date = history["date"].max()

with st.sidebar:
    st.header("Input")
    store_nbr = st.selectbox("Store", stores, index=0)
    family = st.selectbox(
        "Product family",
        families,
        index=families.index("BEVERAGES") if "BEVERAGES" in families else 0,
    )
    target_date = st.date_input("Date", value=(last_date + pd.Timedelta(days=5)).date())
    onpromotion = st.number_input("Items on promotion", min_value=0, value=0, step=1)
    run = st.button("Predict", type="primary")
    st.caption(f"Last date in history: {last_date.date()}")

result = None
if run:
    payload = {
        "store_nbr": int(store_nbr),
        "family": family,
        "date": target_date.isoformat(),
        "onpromotion": int(onpromotion),
    }
    try:
        # the free Render instance sleeps when idle, so the first call can take ~1 minute
        with st.spinner("Calling the API (the first request after idle can take up to a minute)..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")

series = (
    history[(history["store_nbr"] == store_nbr) & (history["family"] == family)]
    .sort_values("date")
    .tail(120)
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=series["date"], y=series["sales"], mode="lines", name="Actual sales"))

if result:
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(target_date)],
            y=[result["predicted_sales"]],
            mode="markers",
            marker=dict(size=14, symbol="star"),
            name="Prediction",
        )
    )

fig.update_layout(
    title=f"Store {store_nbr} | {family}: last 120 days of history",
    xaxis_title="Date",
    yaxis_title="Sales",
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

if result:
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted sales", f"{result['predicted_sales']:,.1f}")
    col2.metric("Store", result["store_nbr"])
    col3.metric("Date", result["date"])

    days_ahead = (pd.Timestamp(target_date) - last_date).days
    if days_ahead > 7:
        st.info(
            "The date is more than a week after the last known sales. Lag features "
            "(lag_7 and longer) are unavailable for it, so this prediction is less reliable."
        )

    if result["drift_warnings"]:
        st.warning("Possible data drift detected:\n\n" + "\n".join(f"- {w}" for w in result["drift_warnings"]))
    else:
        st.success("No data drift detected: inputs look like the training data.")