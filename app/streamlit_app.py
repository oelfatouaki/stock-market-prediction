"""Streamlit dashboard for the stock direction predictor.

Run from the project root with::

    streamlit run app/streamlit_app.py

Educational demo only - not financial advice.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import backtest as bt
from src import data as dmod
from src import models as mmod
from src.features import build_feature_list, make_features
from src.utils import FIGURES_DIR, format_pct, load_json

st.set_page_config(page_title="Stock Direction Predictor", page_icon="📈", layout="wide")

st.title("Stock Direction Predictor")
st.caption(
    "Machine-learning demo: next-day up/down classification with walk-forward validation, "
    "cost-aware backtesting and honest baselines."
)
st.warning(
    "**Not financial advice.** This model is a portfolio/learning project. Typical accuracy is "
    "around 51–55%, transaction costs and slippage are unavoidable, and past performance does "
    "not predict future returns.",
    icon="⚠️",
)


@st.cache_data(show_spinner="Loading prices...", ttl=60 * 60)
def cached_prices(ticker: str, start: str, end: str | None, refresh: bool) -> pd.DataFrame:
    return dmod.load_dataset(ticker, start=start, end=end, force_download=refresh)


@st.cache_data(show_spinner="Loading market context...", ttl=60 * 60)
def cached_market(start: str, end: str | None) -> pd.DataFrame:
    frames = []
    for t in dmod.MARKET_TICKERS:
        try:
            m = dmod.load_dataset(t, start=start, end=end)
            frames.append(m.add_prefix(f"{t.replace('^', '').replace('/', '_')}_"))
        except Exception:  # noqa: BLE001
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


@st.cache_resource(show_spinner="Loading model...")
def cached_model(path: str) -> dict:
    return mmod.load_artifact(path)


artifacts = mmod.list_artifacts()
tickers = sorted({a["ticker"] for a in artifacts} | set(dmod.DEFAULT_TICKERS))
model_options = {a["name"]: a for a in artifacts}

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)
    available = [a for a in artifacts if a["ticker"] == ticker]
    model_name = st.selectbox("Saved model", [a["name"] for a in available]) if available else None
    refresh = st.checkbox("Refresh data from Yahoo Finance", value=False)
    st.markdown("---")
    st.subheader("Backtest parameters")
    threshold = st.slider("Probability threshold", 0.40, 0.75, 0.55, 0.01)
    cost = st.slider("Cost per trade", 0.0, 0.005, 0.001, 0.0005, format="%.4f")
    mode = st.selectbox("Strategy", ["long", "long_short"], format_func=lambda x: "Long / cash" if x == "long" else "Long / short")
    st.markdown("---")
    st.caption(f"Artifacts found: {len(artifacts)}. Train one with `python -m src.cli train --ticker {ticker}`.")

if not model_name:
    st.info("No trained model for this ticker yet. Run `python -m src.cli train --ticker AAPL` to create one.")
    st.stop()

art = cached_model(model_options[model_name]["path"])
features: list[str] = art["features"]
needs_market = any(str(c).startswith("mkt_") for c in features)

try:
    prices = cached_prices(ticker, "2010-01-01", None, refresh)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load prices: {exc}")
    st.stop()

min_date, max_date = prices.index.min().date(), prices.index.max().date()
selected = st.sidebar.date_input(
    "Analysis period",
    value=(max(min_date, max_date - pd.Timedelta(days=365 * 3)), max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(selected, tuple) and len(selected) == 2:
    start_date, end_date = selected
else:
    start_date, end_date = min_date, max_date

market = cached_market("2010-01-01", None) if needs_market else None
featured_all = make_features(prices, market_df=market, dropna=False)
featured_all = featured_all.loc[featured_all.index >= pd.Timestamp(start_date)]
featured = featured_all.dropna(subset=features + ["next_ret"])
latest = featured_all.iloc[[-1]]

if featured.empty:
    st.error("Not enough data in the selected range for the model's lookback windows.")
    st.stop()

probs = mmod.predict_proba(art, featured)
latest_prob = float(mmod.predict_proba(art, latest)[-1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest close", f"${prices['Close'].iloc[-1]:,.2f}")
col2.metric(
    "P(up tomorrow)",
    f"{latest_prob:.1%}",
    delta=f"{(latest_prob - 0.5) * 100:+.1f} pp vs coin flip",
)
col3.metric("Signal", "LONG 📈" if latest_prob > threshold else "CASH 💤")
col4.metric(
    "Test accuracy (holdout)",
    format_pct(art.get("metrics", {}).get("accuracy")),
    help="Computed once on the untouched test period at training time",
)

st.caption(
    f"As of {prices.index[-1].date()} · model: `{art['name']}` · "
    f"trained on {art.get('train_range', ['?', '?'])[0]} → {art.get('train_range', ['?', '?'])[1]} · "
    f"walk-forward CV accuracy: {format_pct(art.get('walk_forward', {}).get('mean_accuracy'))}"
)

# --- Price chart -------------------------------------------------------------
st.subheader("Price, trend and volume")
window_prices = prices.loc[prices.index >= pd.Timestamp(start_date)].copy()
window_prices["sma_20"] = window_prices["Close"].rolling(20).mean()
window_prices["sma_50"] = window_prices["Close"].rolling(50).mean()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
fig.add_trace(
    go.Candlestick(
        x=window_prices.index,
        open=window_prices["Open"],
        high=window_prices["High"],
        low=window_prices["Low"],
        close=window_prices["Close"],
        name="OHLC",
    ),
    row=1,
    col=1,
)
for col, color in (("sma_20", "#ff9800"), ("sma_50", "#2196f3")):
    fig.add_trace(
        go.Scatter(x=window_prices.index, y=window_prices[col], mode="lines", name=col.upper(), line={"color": color}),
        row=1,
        col=1,
    )
colors = np.where(window_prices["Close"] >= window_prices["Open"], "#26a69a", "#ef5350")
fig.add_trace(
    go.Bar(x=window_prices.index, y=window_prices["Volume"], marker={"color": colors}, name="Volume", showlegend=False),
    row=2,
    col=1,
)
fig.update_layout(height=560, xaxis_rangeslider_visible=False, template="plotly_white", margin={"t": 20, "b": 20})
fig.update_yaxes(title_text="Price (split/dividend adjusted)", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
st.plotly_chart(fig)

# --- Backtest ----------------------------------------------------------------
st.subheader("Backtest on the selected period")
result = bt.backtest(featured, probs, threshold=threshold, cost=cost, mode=mode)
metrics = bt.backtest_table(result)
st.dataframe(metrics.style.format({"strategy": "{:.3f}", "buy_and_hold": "{:.3f}"}), width="stretch")

equity_fig = go.Figure()
equity_fig.add_trace(go.Scatter(x=result["equity"].index, y=result["equity"], name="Strategy", line={"color": "#1565c0"}))
equity_fig.add_trace(
    go.Scatter(x=result["benchmark"].index, y=result["benchmark"], name="Buy & hold", line={"color": "#616161", "dash": "dash"})
)
equity_fig.update_layout(
    title="Growth of $1 (transaction costs included)",
    yaxis_title="Equity",
    template="plotly_white",
    height=400,
    margin={"t": 40, "b": 20},
)
st.plotly_chart(equity_fig)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy total return", format_pct(result["metrics"]["total_return"]))
c2.metric("Sharpe ratio", f"{result['metrics']['sharpe']:.2f}")
c3.metric("Max drawdown", format_pct(result["metrics"]["max_drawdown"]))
c4.metric("Trades / exposure", f"{result['metrics']['trades']} / {result['metrics']['exposure']:.0%}")

# --- Interpretation ----------------------------------------------------------
with st.expander("Model interpretation"):
    imp = mmod.feature_importance(art, top_n=15)
    if imp is not None:
        st.bar_chart(imp.set_index("feature")["importance"])
    else:
        st.write("Feature importance is unavailable for this estimator.")

    if st.button("Explain the latest prediction with SHAP"):
        try:
            import shap

            X_row = latest[features]
            model = art["model"]
            if hasattr(model, "named_steps") and "clf" in model.named_steps:
                scaler = model.named_steps["scaler"]
                explainer = shap.LinearExplainer(model.named_steps["clf"], scaler.transform(featured[features]))
                values = explainer.shap_values(scaler.transform(X_row))
                x_plot = pd.DataFrame(scaler.transform(X_row), columns=features).iloc[0]
            else:
                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(X_row)
                if isinstance(values, list):
                    values = values[1]
                x_plot = X_row.iloc[0]
            plt.figure()
            shap.plots.waterfall(
                shap.Explanation(values=np.asarray(values).ravel(), base_value=0.5, data=x_plot, feature_names=features),
                max_display=12,
                show=False,
            )
            st.pyplot(plt.gcf(), width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"SHAP explanation unavailable: {exc}")

with st.expander("Data quality and methodology"):
    quality_path = ROOT / "data" / "processed" / f"{ticker}_quality.json"
    if quality_path.exists():
        st.json(load_json(quality_path))
    st.markdown(
        """
        - Features use only information available at day *t*'s close; `target` and `next_ret` are never inputs.
        - Chronological 80/20 split, expanding-window `TimeSeriesSplit` CV on the training set only.
        - Test set evaluated once; baselines are always-up and yesterday's direction.
        - Backtest charges a one-way cost on every position change; no slippage modelling.
        """
    )

st.caption("Built for learning and portfolio purposes. No return is guaranteed.")
