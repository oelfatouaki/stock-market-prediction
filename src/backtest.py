"""Backtesting with transaction costs and honest performance metrics.

A signal computed from day ``t`` features is applied to the ``t -> t+1``
return stored in ``next_ret``. Costs are charged whenever the position changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; the Streamlit app renders the same figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import NEXT_RET
from .utils import (
    FIGURES_DIR,
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def performance_metrics(
    returns: pd.Series,
    initial_capital: float = 1.0,
    periods: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Standard performance summary for a daily return series."""
    r = pd.Series(returns).dropna().astype(float)
    equity = initial_capital * (1.0 + r).cumprod()
    active = r[r != 0]
    return {
        "total_return": float(equity.iloc[-1] / initial_capital - 1.0) if len(r) else float("nan"),
        "cagr": annualized_return(r, periods),
        "annualized_vol": annualized_volatility(r, periods),
        "sharpe": sharpe_ratio(r, periods),
        "sortino": sortino_ratio(r, periods),
        "max_drawdown": max_drawdown(equity),
        "win_rate_days": float((active > 0).mean()) if len(active) else float("nan"),
        "best_day": float(r.max()) if len(r) else float("nan"),
        "worst_day": float(r.min()) if len(r) else float("nan"),
        "n_days": int(len(r)),
    }


def backtest(
    df: pd.DataFrame,
    probs: np.ndarray | pd.Series,
    threshold: float = 0.5,
    cost: float = 0.001,
    mode: str = "long",
    initial_capital: float = 1.0,
) -> Dict[str, Any]:
    """Run a threshold-based strategy.

    Args:
        df: Feature frame containing ``next_ret``.
        probs: Predicted P(up), aligned by position or index.
        threshold: Go long/short when probability crosses this value.
        cost: One-way transaction cost as a fraction (0.001 = 0.1%).
        mode: ``"long"`` (long or cash) or ``"long_short"`` (long/short/cash).
        initial_capital: Starting equity, used only for the equity curve.

    Returns:
        Dict with signal, returns, equity, benchmark, metrics and trade counts.
    """
    if NEXT_RET not in df.columns:
        raise ValueError(f"Backtest needs a {NEXT_RET!r} column")

    p = pd.Series(np.asarray(probs, dtype=float), index=df.index)
    if mode == "long":
        signal = (p > threshold).astype(int)
    elif mode == "long_short":
        signal = np.where(p > threshold, 1, np.where(p < 1 - threshold, -1, 0)).astype(int)
        signal = pd.Series(signal, index=df.index)
    else:
        raise ValueError("mode must be 'long' or 'long_short'")

    future_ret = df[NEXT_RET].astype(float)
    turnover = signal.diff().fillna(signal).abs()  # cost paid when entering/exiting
    strategy_ret = signal * future_ret - turnover * cost
    benchmark_ret = future_ret.fillna(0.0)

    equity = initial_capital * (1.0 + strategy_ret.fillna(0.0)).cumprod()
    benchmark = initial_capital * (1.0 + benchmark_ret).cumprod()

    metrics = performance_metrics(strategy_ret.fillna(0.0), initial_capital)
    metrics.update(
        {
            "trades": int(turnover.sum()),
            "total_cost": float(turnover.sum() * cost),
            "exposure": float((signal != 0).mean()),
            "threshold": float(threshold),
            "cost_per_trade": float(cost),
            "mode": mode,
        }
    )
    benchmark_metrics = performance_metrics(benchmark_ret, initial_capital)
    return {
        "signal": signal,
        "returns": strategy_ret,
        "equity": equity,
        "benchmark_returns": benchmark_ret,
        "benchmark": benchmark,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
    }


def backtest_table(result: Dict[str, Any]) -> pd.DataFrame:
    """Strategy vs buy-and-hold, one row per metric."""
    strategy = result["metrics"]
    benchmark = result["benchmark_metrics"]
    keys = [
        "total_return",
        "cagr",
        "annualized_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "win_rate_days",
        "n_days",
    ]
    return pd.DataFrame(
        {
            "metric": keys,
            "strategy": [strategy.get(k, np.nan) for k in keys],
            "buy_and_hold": [benchmark.get(k, np.nan) for k in keys],
        }
    )


def plot_equity_curves(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path | str] = None,
    log_scale: bool = False,
):
    """Plot one or more strategy equity curves against buy-and-hold."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, result in results.items():
        ax.plot(result["equity"].index, result["equity"], label=f"{label} strategy", linewidth=1.6)
    first = next(iter(results.values()))
    ax.plot(
        first["benchmark"].index,
        first["benchmark"],
        label="Buy & hold",
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Backtest equity curves (transaction costs included)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0
