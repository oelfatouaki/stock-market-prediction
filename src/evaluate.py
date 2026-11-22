"""Baselines and report tables.

The baselines are deliberately simple: if a model cannot beat "always up"
or "yesterday's direction", it has no value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .features import NEXT_RET, TARGET


def baseline_predictions(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return predictions from the two classification baselines."""
    y = df[TARGET].astype(int)
    return {
        "always_up": np.ones(len(df), dtype=int),
        "yesterday_direction": y.shift(1).fillna(y.iloc[0]).astype(int).to_numpy(),
    }


def baseline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET].astype(int).to_numpy()
    rows = []
    for name, pred in baseline_predictions(df).items():
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "roc_auc": float("nan"),
                "n_samples": len(y),
            }
        )
    # Buy-and-hold is a return baseline, reported in the backtest table instead.
    return pd.DataFrame(rows)


def results_table(metrics_by_model: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Build a tidy comparison table from a {name: metrics} mapping."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "n_samples"]
    rows = []
    for name, metrics in metrics_by_model.items():
        row = {"model": name}
        for key in keys:
            row[key] = metrics.get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def honest_verdict(model_accuracy: float, baseline_accuracy: float, threshold: float = 0.01) -> str:
    """A deliberately blunt summary used in the CLI and README."""
    delta = model_accuracy - baseline_accuracy
    if delta >= threshold:
        return f"Model beats the always-up baseline by {delta * 100:.2f} pp"
    if delta <= -threshold:
        return f"Model is {abs(delta) * 100:.2f} pp WORSE than always-up"
    return "Model is statistically indistinguishable from the always-up baseline"


def error_analysis(df: pd.DataFrame, probs: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Accuracy by volatility bucket - useful to find regimes where the model fails."""
    d = df.copy()
    d["prob_up"] = np.asarray(probs, dtype=float)
    d["pred"] = (d["prob_up"] > 0.5).astype(int)
    d["correct"] = (d["pred"] == d[TARGET]).astype(int)
    if "vol_20" not in d.columns or d["vol_20"].notna().sum() < n_bins:
        return pd.DataFrame()
    d["vol_bucket"] = pd.qcut(d["vol_20"].rank(method="first"), n_bins, labels=False)
    out = (
        d.groupby("vol_bucket")
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            up_rate=(TARGET, "mean"),
            mean_vol=("vol_20", "mean"),
            mean_next_ret=(NEXT_RET, "mean"),
        )
        .reset_index()
    )
    out["vol_bucket"] = out["vol_bucket"].astype(int) + 1
    return out.rename(columns={"vol_bucket": "vol_quintile"})


def save_report_tables(
    ticker: str,
    classification: pd.DataFrame,
    backtest: Optional[pd.DataFrame] = None,
    walk_forward: Optional[pd.DataFrame] = None,
    error: Optional[pd.DataFrame] = None,
    reports_dir: str = "reports",
) -> List[str]:
    from pathlib import Path

    base = Path(reports_dir)
    base.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, table in (
        ("classification", classification),
        ("backtest", backtest),
        ("walk_forward", walk_forward),
        ("error_analysis", error),
    ):
        if table is None or table.empty:
            continue
        path = base / f"{ticker}_{name}.csv"
        table.to_csv(path, index=False)
        written.append(str(path))
    return written
