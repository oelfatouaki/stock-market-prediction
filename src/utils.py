"""Shared paths, logging, seeding and performance helpers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

TRADING_DAYS = 252


def ensure_dirs() -> None:
    """Create every output directory used by the project."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "stock_prediction") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def set_seed(seed: int = 42) -> None:
    """Seed all relevant random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_builtin(obj), fh, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    return obj


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound a return series into an annualised return."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(returns)) - 1.0


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    return float(returns.std(ddof=1) * np.sqrt(periods)) if len(returns) > 1 else 0.0


def sharpe_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return float("nan")
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def sortino_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return float("nan")
    downside = returns[returns < 0].std(ddof=1)
    if not np.isfinite(downside) or downside <= 0:
        return 0.0
    return float(returns.mean() / downside * np.sqrt(periods))


def max_drawdown(equity: pd.Series | Sequence[float]) -> float:
    """Return the worst peak-to-trough decline as a positive fraction."""
    equity = pd.Series(equity).dropna().astype(float)
    if equity.empty:
        return float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return float(abs(drawdown.min()))


def format_pct(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"
