"""Data acquisition, validation and cleaning.

All prices are downloaded with ``auto_adjust=True`` so that splits and
dividends are already reflected in the OHLC series. Raw files are cached in
``data/raw``; cleaned files are written to ``data/processed``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .utils import RAW_DIR, PROCESSED_DIR, ensure_dirs, logger, save_json

DEFAULT_TICKERS: List[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "SPY",
]
MARKET_TICKERS: List[str] = ["SPY", "^VIX"]
DEFAULT_START = "2010-01-01"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance's single- and multi-level column layouts."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        wanted = {"open", "high", "low", "close", "volume", "adj close"}
        matched_level = None
        for level in range(df.columns.nlevels):
            values = {str(v).strip().lower() for v in df.columns.get_level_values(level)}
            if wanted & values:
                matched_level = level
                break
        df.columns = df.columns.get_level_values(
            matched_level if matched_level is not None else -1
        )
    df.columns = [str(c).strip() for c in df.columns]
    df.columns.name = None
    return df


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the canonical columns and convert them to numeric values."""
    df = _flatten_columns(df)
    rename = {"adj close": "Adj Close"}
    df = df.rename(columns={c: rename.get(c.lower(), c) for c in df.columns})
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded frame is missing columns: {missing}")
    df = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.sort_index()


def download_ticker(
    ticker: str,
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    raw_dir: Path | str = RAW_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Download one ticker and cache the raw data as CSV.

    Set ``force=True`` to refresh an existing cache entry.
    """
    ensure_dirs()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{ticker.replace('/', '_').replace('^', '^')}.csv"

    if cache_path.exists() and not force:
        return _standardise(pd.read_csv(cache_path, index_col=0, parse_dates=True))

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("yfinance is required to download data: pip install yfinance") from exc

    logger.info("Downloading %s from %s to %s", ticker, start, end or "today")
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} (check the ticker symbol)")

    df = _standardise(raw)
    df.to_csv(cache_path)
    logger.info("Saved %d rows to %s", len(df), cache_path)
    return df


def download_all(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Download a set of tickers; a failure for one does not stop the others."""
    out: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            out[ticker] = download_ticker(ticker, start=start, end=end, force=force)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.warning("Skipping %s: %s", ticker, exc)
    return out


def validate_data(df: pd.DataFrame, ticker: str = "unknown") -> Dict[str, object]:
    """Compute a data-quality report (no mutation)."""
    idx = pd.DatetimeIndex(df.index)
    expected_weekdays = pd.bdate_range(idx.min(), idx.max()) if len(idx) else idx
    report: Dict[str, object] = {
        "ticker": ticker,
        "rows": int(len(df)),
        "start": str(idx.min().date()) if len(idx) else None,
        "end": str(idx.max().date()) if len(idx) else None,
        "duplicate_dates": int(idx.duplicated().sum()),
        "missing_weekdays": int(len(expected_weekdays.difference(idx))),
        "nan_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "zero_volume_rows": int((df["Volume"].fillna(0) <= 0).sum()),
        "nonpositive_close": int((df["Close"].fillna(0) <= 0).sum()),
        "abs_return_gt_20pct": int((df["Close"].pct_change().abs() > 0.20).sum()),
    }
    return report


def clean_data(df: pd.DataFrame, allow_zero_volume: bool = False) -> pd.DataFrame:
    """Sort, de-duplicate and drop unusable rows.

    Index-like series such as ``^VIX`` legitimately report zero volume, so
    callers can opt out of that filter for them.
    """
    df = _standardise(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if not allow_zero_volume:
        df = df[df["Volume"] > 0]
    df = df[df["Close"] > 0]
    return df


def load_raw(ticker: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    path = raw_dir / f"{ticker.replace('/', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {ticker}: {path}. Run 'download' first.")
    return _standardise(pd.read_csv(path, index_col=0, parse_dates=True))


def load_dataset(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    raw_dir: Path | str = RAW_DIR,
    force_download: bool = False,
    default_start: str = DEFAULT_START,
) -> pd.DataFrame:
    """Get a clean OHLCV frame, downloading it if necessary."""
    raw_path = Path(raw_dir) / f"{ticker.replace('/', '_')}.csv"
    if raw_path.exists() and not force_download:
        df = load_raw(ticker, raw_dir)
    else:
        df = download_ticker(ticker, start=start or default_start, end=end, force=force_download)

    df = clean_data(df, allow_zero_volume=ticker.startswith("^"))
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError(f"No usable rows for {ticker} in the requested date range")
    return df


def save_processed(df: pd.DataFrame, ticker: str, processed_dir: Path | str = PROCESSED_DIR) -> Path:
    ensure_dirs()
    path = Path(processed_dir) / f"{ticker.replace('/', '_')}_features.csv"
    df.to_csv(path)
    return path


def prepare_dataset(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_market: bool = False,
    force_download: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a ticker, engineer features, write the processed CSV and save QC."""
    from .features import build_feature_list, make_features

    ensure_dirs()
    raw = load_dataset(ticker, start, end, force_download=force_download)
    save_json(validate_data(raw, ticker), PROCESSED_DIR / f"{ticker}_quality.json")

    market = None
    if with_market:
        market_frames = []
        for m_ticker in MARKET_TICKERS:
            try:
                m = load_dataset(m_ticker, start, end, force_download=force_download)
                market_frames.append(m.add_prefix(f"{m_ticker.replace('^', '').replace('/', '_')}_"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Market context feature %s unavailable: %s", m_ticker, exc)
        if market_frames:
            market = pd.concat(market_frames, axis=1)

    featured = make_features(raw, market_df=market)
    save_processed(featured, ticker)
    logger.info(
        "%s: %d rows, %d features (%s to %s)",
        ticker,
        len(featured),
        len(build_feature_list(featured)),
        featured.index.min().date(),
        featured.index.max().date(),
    )
    return featured, build_feature_list(featured)
