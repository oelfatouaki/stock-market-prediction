"""Model definitions, honest evaluation, tuning and artefact persistence."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .utils import MODELS_DIR, ensure_dirs, logger, save_json, set_seed

ARTIFACT_VERSION = 1

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # noqa: BLE001 - e.g. missing libomp.dylib on macOS
    HAS_XGB = False
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:  # noqa: BLE001 - e.g. missing libomp.dylib on macOS
    HAS_LGBM = False
    LGBMClassifier = None  # type: ignore


def available_models() -> List[str]:
    names = ["logistic", "random_forest"]
    if HAS_XGB:
        names.append("xgboost")
    if HAS_LGBM:
        names.append("lightgbm")
    return names


def build_model(name: str, **overrides: Any):
    """Create a fresh estimator. Shallow trees + low learning rate by default."""
    name = name.lower()
    if name == "logistic":
        params: Dict[str, Any] = {"clf__C": 0.1, "clf__max_iter": 2000, "clf__random_state": 42}
        params.update(overrides)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**_strip_prefix(params, "clf__"))),
            ]
        )
    if name == "random_forest":
        params = {
            "n_estimators": 400,
            "max_depth": 5,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
    elif name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        params = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
    elif name == "lightgbm":
        if not HAS_LGBM:
            raise RuntimeError("lightgbm is not installed")
        params = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {available_models()}")
    params.update(overrides)
    return {
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
    }[name](**params)


def _strip_prefix(params: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {k.removeprefix(prefix): v for k, v in params.items() if k.startswith(prefix)}


def search_spaces(name: str) -> Dict[str, Sequence[Any]]:
    """Small, conservative hyperparameter grids (training data only)."""
    spaces: Dict[str, Dict[str, Sequence[Any]]] = {
        "logistic": {
            "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [5, 10, 20, 40],
            "max_features": ["sqrt", 0.3],
        },
        "xgboost": {
            "n_estimators": [150, 300, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 5, 10],
        },
        "lightgbm": {
            "n_estimators": [150, 300, 500],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [7, 15, 31],
            "min_child_samples": [20, 50, 100],
            "subsample": [0.7, 0.8, 0.9],
        },
    }
    try:
        return spaces[name]
    except KeyError as exc:
        raise ValueError(f"No search space for {name!r}") from exc


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    probs: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Classification metrics; probs are P(up), one value per row."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    pred = (p > threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])),
        "positive_rate_pred": float(pred.mean()),
        "positive_rate_actual": float(y.mean()),
        "n_samples": int(len(y)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "report": {
            "0": {k: float(v) for k, v in report["0"].items()},
            "1": {k: float(v) for k, v in report["1"].items()},
        },
    }


def confident_subset_metrics(
    y_true: pd.Series | np.ndarray,
    probs: pd.Series | np.ndarray,
    low: float = 0.45,
    high: float = 0.55,
) -> Dict[str, Any]:
    """Accuracy on high-conviction predictions only (test set, used once)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    confident = (p >= high) | (p <= low)
    if not confident.any():
        return {"n": 0, "coverage": 0.0, "accuracy": float("nan")}
    pred = (p[confident] > 0.5).astype(int)
    return {
        "n": int(confident.sum()),
        "coverage": float(confident.mean()),
        "accuracy": float(accuracy_score(y[confident], pred)),
    }


def walk_forward_scores(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Expanding-window cross-validation on the training period only."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        est = clone(model)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = est.predict_proba(X.iloc[va_idx])[:, 1]
        metrics = evaluate_predictions(y.iloc[va_idx], p, threshold=threshold)
        rows.append(
            {
                "fold": fold,
                "train_size": len(tr_idx),
                "val_size": len(va_idx),
                "val_start": str(X.index[va_idx[0]].date()),
                "val_end": str(X.index[va_idx[-1]].date()),
                **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            }
        )
    scores = pd.DataFrame(rows)
    summary = {
        f"mean_{col}": float(scores[col].mean(skipna=True))
        for col in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }
    summary["std_accuracy"] = float(scores["accuracy"].std(ddof=1))
    return scores, summary


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    n_splits: int = 5,
    scoring: str = "accuracy",
    seed: int = 42,
):
    """Randomised search with chronological CV. Never sees the test set."""
    set_seed(seed)
    search = RandomizedSearchCV(
        estimator=build_model(name),
        param_distributions=search_spaces(name),
        n_iter=n_iter,
        scoring=scoring,
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=seed,
        n_jobs=1,  # XGBoost/LightGBM already parallelise internally
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)
    logger.info(
        "Tuned %s: best cv %s=%.4f with %s", name, scoring, search.best_score_, search.best_params_
    )
    return search


def make_artifact(
    name: str,
    ticker: str,
    model,
    features: Sequence[str],
    train_range: Sequence[str],
    metrics: Dict[str, Any],
    threshold: float = 0.5,
    tuned_params: Optional[Dict[str, Any]] = None,
    walk_forward: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    return {
        "version": ARTIFACT_VERSION,
        "name": name,
        "ticker": ticker,
        "model": model,
        "features": list(features),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "train_range": [str(v) for v in train_range],
        "threshold": float(threshold),
        "metrics": metrics,
        "tuned_params": tuned_params or {},
        "walk_forward": walk_forward or {},
    }


def save_artifact(artifact: Dict[str, Any], models_dir: Path | str = MODELS_DIR) -> Path:
    """Write a canonical (app-friendly) and a timestamped model artefact."""
    ensure_dirs()
    models_dir = Path(models_dir)
    stem = f"{artifact['ticker'].replace('/', '_')}_{artifact['name']}"
    canonical = models_dir / f"{stem}.joblib"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = models_dir / f"{stem}_{stamp}.joblib"
    joblib.dump(artifact, canonical)
    joblib.dump(artifact, versioned)

    metadata = {k: v for k, v in artifact.items() if k != "model"}
    save_json(metadata, models_dir / f"{stem}.json")
    logger.info("Saved model artefacts: %s, %s", canonical.name, versioned.name)
    return canonical


def load_artifact(path: Path | str) -> Dict[str, Any]:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
        raise ValueError(f"Not a valid model artefact: {path}")
    if artifact.get("version") != ARTIFACT_VERSION:
        logger.warning("Artefact %s has version %s, expected %s", path, artifact.get("version"), ARTIFACT_VERSION)
    return artifact


def list_artifacts(models_dir: Path | str = MODELS_DIR) -> List[Dict[str, Any]]:
    """Discover saved models for the Streamlit app and CLI reports."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    seen: set[tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for path in sorted(models_dir.glob("*.joblib")):
        stem = path.stem
        if "_" not in stem:
            continue
        ticker, name = stem.split("_", 1)
        if re.search(r"_\d{8}_\d{6}$", name):  # skip timestamped duplicates
            continue
        key = (ticker, name)
        if key in seen:
            continue
        seen.add(key)
        meta_path = models_dir / f"{stem}.json"
        meta = {}
        if meta_path.exists():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}
        out.append(
            {
                "path": str(path),
                "ticker": meta.get("ticker", ticker),
                "name": meta.get("name", name),
                "accuracy": meta.get("metrics", {}).get("accuracy"),
                "trained_at": meta.get("trained_at"),
                "features": meta.get("features", []),
            }
        )
    return out


def predict_proba(artifact: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Align columns to the training feature order, then predict P(up)."""
    features: List[str] = artifact["features"]
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing[:8]}")
    extra = [c for c in X.columns if c not in features]
    if extra:
        logger.debug("Ignoring %d extra columns at prediction time", len(extra))
    return np.asarray(artifact["model"].predict_proba(X.loc[:, features])[:, 1], dtype=float)


def feature_importance(artifact: Dict[str, Any], top_n: int = 20) -> Optional[pd.DataFrame]:
    model = artifact["model"]
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        model = model.named_steps["clf"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    imp = pd.DataFrame({"feature": artifact["features"], "importance": values})
    return imp.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
