"""Command-line entry point for the whole workflow.

Examples
--------
::

    python -m src.cli download --tickers AAPL MSFT SPY
    python -m src.cli prepare --tickers AAPL --market
    python -m src.cli train --ticker AAPL --model xgboost --tune
    python -m src.cli report --ticker AAPL
    python -m src.cli explain --ticker AAPL --model xgboost
    python -m src.cli app
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import backtest as bt
from . import data as data_module
from . import evaluate as ev
from . import models as model_module
from .features import build_feature_list, make_features, split_time_series
from .utils import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, logger, save_json, set_seed

DEFAULT_TICKERS = data_module.DEFAULT_TICKERS


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock direction prediction pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", default=data_module.DEFAULT_START)
        p.add_argument("--end", default=None)
        p.add_argument("--test-size", type=float, default=0.2)
        p.add_argument("--with-market", action="store_true", help="Add SPY/VIX context features")

    p_dl = sub.add_parser("download", help="Download raw OHLCV data")
    p_dl.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p_dl.add_argument("--start", default=data_module.DEFAULT_START)
    p_dl.add_argument("--end", default=None)
    p_dl.add_argument("--force", action="store_true")

    p_prep = sub.add_parser("prepare", help="Build processed feature files")
    p_prep.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p_prep.add_argument("--start", default=data_module.DEFAULT_START)
    p_prep.add_argument("--end", default=None)
    p_prep.add_argument("--market", action="store_true")
    p_prep.add_argument("--force", action="store_true")

    p_train = sub.add_parser("train", help="Train, evaluate once on test, and save a model")
    p_train.add_argument("--ticker", default="AAPL")
    p_train.add_argument("--model", default="xgboost", choices=model_module.available_models())
    add_common(p_train)
    p_train.add_argument("--tune", action="store_true")
    p_train.add_argument("--n-iter", type=int, default=20)
    p_train.add_argument("--cv-splits", type=int, default=5)
    p_train.add_argument("--threshold", type=float, default=0.5)
    p_train.add_argument("--cost", type=float, default=0.001)
    p_train.add_argument("--scoring", default="accuracy")
    p_train.add_argument("--seed", type=int, default=42)

    p_report = sub.add_parser("report", help="Compare all saved models for a ticker")
    p_report.add_argument("--ticker", default="AAPL")
    add_common(p_report)
    p_report.add_argument("--threshold", type=float, default=0.5)
    p_report.add_argument("--cost", type=float, default=0.001)

    p_explain = sub.add_parser("explain", help="Generate a SHAP summary plot")
    p_explain.add_argument("--ticker", default="AAPL")
    p_explain.add_argument("--model", default="xgboost")
    add_common(p_explain)

    sub.add_parser("app", help="Launch the Streamlit app")

    p_all = sub.add_parser("all", help="Download, prepare, train and report for all tickers")
    p_all.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p_all.add_argument("--start", default=data_module.DEFAULT_START)
    p_all.add_argument("--end", default=None)
    p_all.add_argument("--models", nargs="+", default=["xgboost"])
    p_all.add_argument("--tune", action="store_true")
    p_all.add_argument("--n-iter", type=int, default=15)
    p_all.add_argument("--market", action="store_true")
    p_all.add_argument("--cost", type=float, default=0.001)

    return parser.parse_args(argv)


def _load_featured(ticker: str, args) -> tuple[pd.DataFrame, List[str]]:
    """Use the cached processed file when it matches the requested settings."""
    path = PROCESSED_DIR / f"{ticker}_features.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df, build_feature_list(df)
    return data_module.prepare_dataset(
        ticker, start=args.start, end=args.end, with_market=getattr(args, "with_market", getattr(args, "market", False))
    )


def cmd_train(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed(args.seed)
    df, features = _load_featured(args.ticker, args)
    train, test = split_time_series(df, train_fraction=1 - args.test_size)
    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    logger.info(
        "%s | %s | train %s->%s (%d) | test %s->%s (%d) | %d features",
        args.ticker,
        args.model,
        train.index.min().date(),
        train.index.max().date(),
        len(train),
        test.index.min().date(),
        test.index.max().date(),
        len(test),
        len(features),
    )

    tuned_params: dict = {}
    wf_summary: dict = {}
    if args.tune:
        search = model_module.tune_model(
            args.model,
            X_train,
            y_train,
            n_iter=args.n_iter,
            n_splits=args.cv_splits,
            scoring=args.scoring,
            seed=args.seed,
        )
        tuned_params = search.best_params_
        model = clone(search.best_estimator_)
    else:
        model = model_module.build_model(args.model)

    wf_scores, wf_summary = model_module.walk_forward_scores(
        model, X_train, y_train, n_splits=args.cv_splits, threshold=args.threshold
    )
    save_json(
        {"folds": wf_scores.to_dict(orient="records"), "summary": wf_summary},
        REPORTS_DIR / f"{args.ticker}_{args.model}_walk_forward.json",
    )
    logger.info("Walk-forward CV accuracy: %.4f +/- %.4f", wf_summary["mean_accuracy"], wf_summary["std_accuracy"])

    model.fit(X_train, y_train)
    probs = model_module.predict_proba({**{"model": model, "features": features}}, X_test)
    metrics = model_module.evaluate_predictions(y_test, probs, threshold=args.threshold)
    confident = model_module.confident_subset_metrics(y_test, probs, low=0.45, high=0.55)
    metrics["confident_subset"] = confident

    artifact = model_module.make_artifact(
        name=args.model,
        ticker=args.ticker,
        model=model,
        features=features,
        train_range=[str(train.index.min().date()), str(train.index.max().date())],
        metrics=metrics,
        threshold=args.threshold,
        tuned_params=tuned_params,
        walk_forward=wf_summary,
    )
    path = model_module.save_artifact(artifact)

    # Backtest once, using the same test predictions.
    result = bt.backtest(test, probs, threshold=args.threshold, cost=args.cost)
    bt.plot_equity_curves({f"{args.ticker} ({args.model})": result}, FIGURES_DIR / f"{args.ticker}_{args.model}_equity.png")
    table = bt.backtest_table(result)
    table.to_csv(REPORTS_DIR / f"{args.ticker}_{args.model}_backtest.csv", index=False)

    baselines = ev.baseline_metrics(test)
    comparison = pd.concat(
        [baselines, pd.DataFrame([{"model": args.model, **{k: metrics[k] for k in baselines.columns if k in metrics}}])],
        ignore_index=True,
    )
    comparison.to_csv(REPORTS_DIR / f"{args.ticker}_{args.model}_classification.csv", index=False)

    err = ev.error_analysis(test, probs)
    if not err.empty:
        err.to_csv(REPORTS_DIR / f"{args.ticker}_{args.model}_error_analysis.csv", index=False)

    save_json(
        {
            "ticker": args.ticker,
            "model": args.model,
            "test_range": [str(test.index.min().date()), str(test.index.max().date())],
            "metrics": metrics,
            "walk_forward": wf_summary,
            "backtest": result["metrics"],
            "buy_and_hold": result["benchmark_metrics"],
            "artifact": str(path),
        },
        REPORTS_DIR / f"{args.ticker}_{args.model}_summary.json",
    )

    baseline_acc = float(baselines.loc[baselines["model"] == "always_up", "accuracy"].iloc[0])
    print("\n--- TEST SET (touched once) ---")
    print(comparison.to_string(index=False))
    print(ev.honest_verdict(metrics["accuracy"], baseline_acc))
    print("\n--- BACKTEST (with costs) ---")
    print(table.to_string(index=False))
    print(f"\nModel saved to {path}")


def cmd_report(args: argparse.Namespace) -> None:
    ensure_dirs()
    df, features = _load_featured(args.ticker, args)
    _, test = split_time_series(df, train_fraction=1 - args.test_size)
    artifacts = [a for a in model_module.list_artifacts() if a["ticker"] == args.ticker]
    if not artifacts:
        logger.warning("No saved models for %s. Run 'train' first.", args.ticker)
        return

    metrics_by_name: dict = {}
    equity_curves: dict = {}
    for art in artifacts:
        try:
            loaded = model_module.load_artifact(art["path"])
            probs = model_module.predict_proba(loaded, test[features])
            metrics_by_name[art["name"]] = model_module.evaluate_predictions(test["target"], probs, args.threshold)
            equity_curves[art["name"]] = bt.backtest(test, probs, args.threshold, args.cost)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not evaluate %s: %s", art["name"], exc)

    comparison = ev.results_table(metrics_by_name)
    baseline = ev.baseline_metrics(test).set_index("model")
    comparison = pd.concat([baseline, comparison])
    print("\n--- ALL MODELS vs BASELINES (test set) ---")
    print(comparison.to_string())

    if equity_curves:
        bt.plot_equity_curves(equity_curves, FIGURES_DIR / f"{args.ticker}_equity_all_models.png")
        frames = [bt.backtest_table(r).set_index("metric").add_prefix(f"{name}_") for name, r in equity_curves.items()]
        bt_table = pd.concat(frames, axis=1)
        bt_table.to_csv(REPORTS_DIR / f"{args.ticker}_backtest_all_models.csv")
        print("\n--- BACKTEST ---")
        print(bt_table.to_string())

    ev.save_report_tables(
        args.ticker,
        comparison.reset_index(),
        backtest=pd.concat(frames, axis=1).reset_index() if equity_curves else None,
        reports_dir=str(REPORTS_DIR),
    )


def cmd_explain(args: argparse.Namespace) -> None:
    ensure_dirs()
    path = MODELS_DIR / f"{args.ticker}_{args.model}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}; run 'train --model {args.model}' first")
    artifact = model_module.load_artifact(path)
    df, features = _load_featured(args.ticker, args)
    _, test = split_time_series(df, train_fraction=1 - args.test_size)
    X = test[features].sample(min(500, len(test)), random_state=42)

    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SHAP is required: pip install shap") from exc

    model = artifact["model"]
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        X_plot = pd.DataFrame(model.named_steps["scaler"].transform(X), columns=features)
        explainer = shap.LinearExplainer(model.named_steps["clf"], X_plot)
        shap_values = explainer.shap_values(X_plot)
    else:
        X_plot = X
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_plot)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    import matplotlib.pyplot as plt

    plt.figure()
    shap.summary_plot(shap_values, X_plot, show=False, max_display=15)
    fig = plt.gcf()
    fig.tight_layout()
    out = FIGURES_DIR / f"{args.ticker}_{args.model}_shap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP plot to %s", out)

    imp = model_module.feature_importance(artifact, top_n=20)
    if imp is not None:
        imp.to_csv(REPORTS_DIR / f"{args.ticker}_{args.model}_feature_importance.csv", index=False)
        print(imp.to_string(index=False))


def cmd_app(_: argparse.Namespace) -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


def cmd_all(args: argparse.Namespace) -> None:
    for ticker in args.tickers:
        try:
            data_module.download_ticker(ticker, start=args.start, end=args.end)
            data_module.prepare_dataset(
                ticker, start=args.start, end=args.end, with_market=args.market
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Data step failed for %s: %s", ticker, exc)
            continue
        for name in args.models:
            cmd_train(
                argparse.Namespace(
                    ticker=ticker,
                    model=name,
                    start=args.start,
                    end=args.end,
                    test_size=0.2,
                    with_market=args.market,
                    tune=args.tune,
                    n_iter=args.n_iter,
                    cv_splits=5,
                    threshold=0.5,
                    cost=args.cost,
                    scoring="accuracy",
                    seed=42,
                )
            )
        cmd_report(
            argparse.Namespace(
                ticker=ticker,
                start=args.start,
                end=args.end,
                test_size=0.2,
                with_market=args.market,
                threshold=0.5,
                cost=args.cost,
            )
        )


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    ensure_dirs()
    commands = {
        "download": lambda: print(
            {
                ticker: f"{len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})"
                for ticker, df in data_module.download_all(
                    args.tickers, args.start, args.end, args.force
                ).items()
            }
        ),
        "prepare": lambda: [
            data_module.prepare_dataset(t, args.start, args.end, args.market, args.force)
            for t in args.tickers
        ],
        "train": lambda: cmd_train(args),
        "report": lambda: cmd_report(args),
        "explain": lambda: cmd_explain(args),
        "app": lambda: cmd_app(args),
        "all": lambda: cmd_all(args),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
