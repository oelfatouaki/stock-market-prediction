"""Feature engineering with strict point-in-time discipline.

Every feature at row ``t`` uses information available at the close of day
``t`` or earlier. ``target`` and ``next_ret`` are labels/metrics only and must
never be passed to a model.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TARGET = "target"
NEXT_RET = "next_ret"
EXCLUDED_COLUMNS = {
    TARGET,
    NEXT_RET,
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Adj Close",
    "ticker",
}
RETURN_WINDOWS = (1, 2, 5, 10, 20)
TREND_WINDOWS = (5, 10, 20, 50)
RETURN_COLS = [f"ret_{n}" for n in RETURN_WINDOWS]

try:  # ``ta`` is optional; pure-pandas fallbacks are provided below.
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import AverageTrueRange, BollingerBands

    HAS_TA = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_TA = False
    RSIIndicator = MACD = AverageTrueRange = BollingerBands = None  # type: ignore


def _wilder_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _fallback_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_ema(gain, window)
    avg_loss = _wilder_ema(loss, window)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def _fallback_macd(close: pd.Series) -> pd.DataFrame:
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal, "macd_diff": line - signal})


def _fallback_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return _wilder_ema(tr, window)


def _add_indicators(d: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series) -> None:
    if HAS_TA:
        d["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        d["macd"] = macd.macd()
        d["macd_signal"] = macd.macd_signal()
        d["macd_diff"] = macd.macd_diff()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        d["bb_width"] = bb.bollinger_wband()
        d["bb_pos"] = bb.bollinger_pband()
        d["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    else:
        d["rsi_14"] = _fallback_rsi(close)
        for name, values in _fallback_macd(close).items():
            d[name] = values
        middle = close.rolling(20).mean()
        sd = close.rolling(20).std(ddof=0)
        upper, lower = middle + 2 * sd, middle - 2 * sd
        d["bb_width"] = (upper - lower) / middle
        d["bb_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)
        d["atr_14"] = _fallback_atr(high, low, close)


def _add_market_context(d: pd.DataFrame, market_df: Optional[pd.DataFrame]) -> None:
    """Align external series (SPY/VIX) on the price index; no future data."""
    if market_df is None or market_df.empty:
        return
    aligned = market_df.reindex(d.index)
    for col in aligned.columns:
        if not str(col).endswith("_Close"):
            continue
        prefix = str(col)[: -len("_Close")]
        series = aligned[col].astype(float)
        d[f"mkt_{prefix}_ret_1"] = series.pct_change()
        d[f"mkt_{prefix}_vol_20"] = series.pct_change().rolling(20).std()
        vol_col = f"{prefix}_Volume"
        if vol_col in aligned.columns:
            vol = aligned[vol_col].astype(float).replace(0, np.nan)
            if vol.notna().sum() > 20:  # indices like ^VIX report no volume at all
                d[f"mkt_{prefix}_vol_chg"] = vol.pct_change()


def make_features(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the modelling frame. Returns and indicators are backward-looking.

    ``dropna=False`` keeps the newest row even though its label is not yet
    knowable; that row is for live inference only, never for evaluation.
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    d = d.sort_index()

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"make_features needs columns {sorted(missing)}")

    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    open_ = d["Open"].astype(float)
    volume = d["Volume"].astype(float)

    # --- Returns -----------------------------------------------------------
    d["ret_1"] = close.pct_change()
    for n in RETURN_WINDOWS[1:]:
        d[f"ret_{n}"] = close.pct_change(n)

    # --- Trend -------------------------------------------------------------
    for n in TREND_WINDOWS:
        d[f"sma_{n}"] = close.rolling(n).mean() / close - 1.0
        d[f"ema_{n}"] = close.ewm(span=n, adjust=False).mean() / close - 1.0

    # --- Momentum, volatility, volume, candles ----------------------------
    _add_indicators(d, close, high, low)
    d["vol_10"] = d["ret_1"].rolling(10).std()
    d["vol_20"] = d["ret_1"].rolling(20).std()
    d["vol_chg"] = volume.pct_change()
    d["vol_ratio"] = volume / volume.rolling(20).mean()
    d["hl_range"] = (high - low) / close
    d["oc_change"] = (close - open_) / open_
    d["gap"] = open_ / close.shift(1) - 1.0

    # --- Calendar ----------------------------------------------------------
    d["dow"] = d.index.dayofweek.astype(int)
    d["month"] = d.index.month.astype(int)

    # --- External context --------------------------------------------------
    _add_market_context(d, market_df)

    # --- Labels: strictly AFTER the features, never used as inputs --------
    d[TARGET] = (close.shift(-1) > close).astype(int)
    d[NEXT_RET] = close.pct_change().shift(-1)

    d = d.replace([np.inf, -np.inf], np.nan)
    return d.dropna() if dropna else d


def build_feature_list(df: pd.DataFrame) -> List[str]:
    """Return the model feature columns in a stable order."""
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]


def split_time_series(
    df: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split (never shuffle)."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(len(df) * train_fraction)
    if split < 10 or len(df) - split < 10:
        raise ValueError("Not enough rows for a train/test split")
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def feature_groups() -> Dict[str, List[str]]:
    """Human-readable feature groups for the README and the Streamlit app."""
    return {
        "returns": RETURN_COLS,
        "trend": [f"{kind}_{n}" for n in TREND_WINDOWS for kind in ("sma", "ema")],
        "momentum": ["rsi_14", "macd", "macd_signal", "macd_diff"],
        "volatility": ["vol_10", "vol_20", "bb_width", "bb_pos", "atr_14", "hl_range"],
        "volume": ["vol_chg", "vol_ratio"],
        "candles": ["oc_change", "gap"],
        "calendar": ["dow", "month"],
    }
