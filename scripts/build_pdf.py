"""Build a detailed PDF walkthrough of the stock direction prediction project.

Run from the project root::

    venv/bin/python scripts/build_pdf.py

Output: reports/Stock_Direction_Predictor_Report.pdf
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
OUT = REPORTS / "Stock_Direction_Predictor_Report.pdf"

BLUE = colors.HexColor("#1565c0")
DARK = colors.HexColor("#0d47a1")
GREY = colors.HexColor("#546e7a")
LIGHT = colors.HexColor("#eef3f8")

styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body", parent=styles["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=6
)
H1 = ParagraphStyle(
    "h1", parent=styles["Heading1"], fontSize=15, leading=19, textColor=DARK,
    spaceBefore=14, spaceAfter=8,
)
H2 = ParagraphStyle(
    "h2", parent=styles["Heading2"], fontSize=11.5, leading=15, textColor=BLUE,
    spaceBefore=10, spaceAfter=5,
)
CAPTION = ParagraphStyle(
    "caption", parent=styles["Normal"], fontSize=8, leading=10,
    textColor=GREY, alignment=1, spaceAfter=10,
)
CODE = ParagraphStyle(
    "code", parent=styles["Code"], fontSize=7.4, leading=9.4,
    backColor=colors.HexColor("#f4f6f8"), borderColor=colors.HexColor("#d7dee5"),
    borderWidth=0.5, borderPadding=6, leftIndent=2, spaceAfter=8,
)
TITLE = ParagraphStyle(
    "title", parent=styles["Title"], fontSize=28, leading=33, textColor=DARK,
    alignment=1, spaceAfter=10,
)
SUBTITLE = ParagraphStyle(
    "subtitle", parent=styles["Normal"], fontSize=13, leading=18,
    textColor=GREY, alignment=1, spaceAfter=18,
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def P(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(item), leftIndent=12) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
        bulletFontSize=6,
    )


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE)


def table(rows: list[list[str]], widths=None, font: float = 8, center_from: int = 1) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 2),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8dc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]
    n_cols = len(rows[0])
    if center_from < n_cols:
        cmds.append(("ALIGN", (center_from, 1), (-1, -1), "CENTER"))
    for idx in range(2, len(rows), 2):
        cmds.append(("BACKGROUND", (0, idx), (-1, idx), LIGHT))
    t.setStyle(TableStyle(cmds))
    return t


def figure(path: Path, caption: str, width: float = 16.2 * cm, max_h: float = 9.5 * cm) -> list:
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    scale = min(width / iw, max_h / ih)
    img = Image(str(path), width=iw * scale, height=ih * scale)
    return [img, P(caption, CAPTION)]


def callout(text: str, bg=colors.HexColor("#fff8e1")) -> Table:
    t = Table([[P(text, BODY)]], colWidths=[16.2 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#ffb300")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.2f}%"


def num(x, digits: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Extra figures generated specifically for the report
# ---------------------------------------------------------------------------
def build_extra_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Workflow diagram
    phases = [
        ("0. Setup", "venv, requirements, folders"),
        ("1. Problem", "binary next-day direction"),
        ("2. Data", "yfinance, QC, cleaning"),
        ("3. EDA", "returns, volatility, balance"),
        ("4. Features", "34 point-in-time features"),
        ("5. Split", "chronological 80/20"),
        ("6. Baselines", "always-up, yesterday"),
        ("7. Models", "LR, RF, XGB, LGBM"),
        ("8. Validation", "walk-forward CV + tuning"),
        ("9. Backtest", "costs vs buy-and-hold"),
        ("10. Interpret", "importance, SHAP, errors"),
        ("11. Upgrades", "sentiment, macro, LSTM"),
        ("12. App", "Streamlit dashboard"),
        ("13. Docs", "README, results, deploy"),
    ]
    fig, ax = plt.subplots(figsize=(10, 7.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    for idx, (title, desc) in enumerate(phases):
        r, c = divmod(idx, 3)
        x = 0.3 + c * 3.3
        y = 9.6 - r * 1.9
        rect = FancyBboxPatch(
            (x, y - 0.55), 3.0, 1.35,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#e3f2fd" if idx % 2 == 0 else "#f1f8e9",
            edgecolor="#1565c0", linewidth=1.1,
        )
        ax.add_patch(rect)
        ax.text(x + 1.5, y + 0.35, title, ha="center", va="center", fontsize=10, weight="bold", color="#0d47a1")
        ax.text(x + 1.5, y - 0.1, desc, ha="center", va="center", fontsize=8, color="#37474f")
        if c < 2 and idx < len(phases) - 1:
            ax.annotate("", xy=(x + 3.25, y + 0.12), xytext=(x + 3.02, y + 0.12),
                        arrowprops={"arrowstyle": "->", "color": "#78909c", "lw": 1.2})
    ax.set_title("Project phases - from setup to deployment", fontsize=13, weight="bold", color="#0d47a1")
    fig.tight_layout()
    fig.savefig(FIGURES / "report_workflow.png", dpi=150)
    plt.close(fig)

    summaries = {}
    for path in REPORTS.glob("AAPL_*_summary.json"):
        d = load_json(path)
        summaries[d["model"]] = d

    # Walk-forward CV bar chart
    names = ["logistic", "random_forest", "xgboost", "lightgbm"]
    means = [summaries[n]["walk_forward"]["mean_accuracy"] for n in names]
    stds = [summaries[n]["walk_forward"]["std_accuracy"] for n in names]
    baseline = summaries[names[0]]["metrics"]["positive_rate_actual"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=["#90caf9", "#66bb6a", "#ffa726", "#ab47bc"])
    ax.axhline(baseline, color="#c62828", linestyle="--", lw=1.5, label=f"Always-up baseline = {baseline:.3f}")
    ax.set_ylim(0.47, 0.56)
    ax.set_ylabel("Accuracy")
    ax.set_title("AAPL walk-forward CV accuracy (training period only)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "report_cv_accuracy.png", dpi=150)
    plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, name in zip(axes, names):
        m = summaries[name]["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Act 0", "Act 1"], fontsize=7)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("AAPL test-set confusion matrices", fontsize=11, color="#0d47a1")
    fig.tight_layout()
    fig.savefig(FIGURES / "report_confusion.png", dpi=150)
    plt.close(fig)

    # Multi-ticker strategy vs buy-and-hold
    multi = []
    for path in REPORTS.glob("*_xgboost_summary.json"):
        d = load_json(path)
        multi.append((d["ticker"], d["backtest"]["total_return"], d["buy_and_hold"]["total_return"]))
    multi.sort()
    tickers = [m[0] for m in multi]
    strat = [m[1] * 100 for m in multi]
    bh = [m[2] * 100 for m in multi]
    x = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - 0.19, strat, 0.38, label="Strategy", color="#1565c0")
    ax.bar(x + 0.19, bh, 0.38, label="Buy & hold", color="#90a4ae")
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Total return (%)")
    ax.set_title("XGBoost strategy vs buy-and-hold by ticker (0.1% costs)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "report_multiticker.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------
def build() -> None:
    build_extra_figures()

    aapl = {d["model"]: d for d in
            [load_json(p) for p in REPORTS.glob("AAPL_*_summary.json")]}
    multi = sorted(
        [load_json(p) for p in REPORTS.glob("*_xgboost_summary.json")],
        key=lambda d: d["ticker"],
    )
    quality_path = ROOT / "data" / "processed" / "AAPL_quality.json"
    quality = load_json(quality_path) if quality_path.exists() else {}

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Stock Direction Predictor - Project Report",
        author="Oussama",
        subject="End-to-end machine learning project: next-day stock direction",
    )
    story: list = []

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cfd8dc"))
        canvas.line(2 * cm, 1.35 * cm, A4[0] - 2 * cm, 1.35 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, 0.95 * cm, "Stock Direction Predictor | Oussama | Learning project - not financial advice")
        canvas.drawRightString(A4[0] - 2 * cm, 0.95 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    # ---- Cover ------------------------------------------------------------
    story += [
        Spacer(1, 1.6 * cm),
        P("STOCK DIRECTION PREDICTOR", TITLE),
        P("An end-to-end machine learning project: prediction, honest evaluation, "
          "cost-aware backtesting and deployment", SUBTITLE),
        Spacer(1, 0.2 * cm),
    ]
    story += figure(FIGURES / "AAPL_equity_all_models.png", "Figure 1: Strategy equity curves versus buy-and-hold on the AAPL holdout period.", max_h=8.5 * cm)
    story += [
        Spacer(1, 0.5 * cm),
        P("<b>Author:</b> Oussama - Computer Science graduate, learning data science and AI", BODY),
        P("<b>Generated:</b> from the project code, saved model artefacts and result tables", BODY),
        P("<b>Repository:</b> github.com/oelfatouaki/stock-market-prediction", BODY),
        Spacer(1, 0.6 * cm),
        callout("<b>Important:</b> This document describes a portfolio/learning project, not financial advice. "
                "Next-day stock direction is close to random. The models here do not reliably beat trivial "
                "baselines, and no returns are guaranteed."),
        PageBreak(),
    ]

    # ---- Contents ---------------------------------------------------------
    story += [P("Contents", H1)]
    toc = [
        "1. Executive summary",
        "2. Problem definition and success criteria",
        "3. Project structure and environment setup",
        "4. Data collection, validation and cleaning",
        "5. Exploratory data analysis",
        "6. Feature engineering and leakage control",
        "7. Chronological split and baselines",
        "8. Modelling: from simple to complex",
        "9. Walk-forward validation and tuning",
        "10. Results on the untouched test set",
        "11. Backtesting with transaction costs",
        "12. Interpretation: importance, SHAP and errors",
        "13. Multi-ticker robustness check",
        "14. Streamlit application",
        "15. Optional upgrades and future work",
        "16. How to run the project",
        "17. Limitations, mistakes avoided and final verdict",
        "Appendix A: file and command reference",
    ]
    story += [bullets(toc), PageBreak()]

    # ---- 1 Executive summary ----------------------------------------------
    story += [
        P("1. Executive summary", H1),
        P("This project asks a deliberately modest question: can past price, volume, momentum, volatility "
          "and market-context features predict whether a stock will close higher or lower <b>tomorrow</b>? "
          "The answer, measured honestly, is <b>no reliable edge</b>. That outcome is the point of the project: "
          "the value is in the rigorous pipeline, not in a flattering accuracy number."),
        P("We downloaded adjusted daily OHLCV data for eight tickers from 2010 onward, built 34 strictly "
          "point-in-time features, split chronologically, evaluated four model families with expanding-window "
          "cross-validation, touched the test set once, and backtested a trading rule with a 0.1% one-way cost. "
          "The Streamlit app exposes the full workflow interactively."),
        P("Headline AAPL test results (840 days): always-up <b>53.81%</b>, Random Forest <b>52.74%</b>, "
          "XGBoost <b>51.43%</b>, Logistic Regression and LightGBM <b>50.48%</b>. All ROC-AUC values are at "
          "or below 0.50. The backtested strategies had lower drawdowns than buy-and-hold because they were "
          "often in cash, but none beat buy-and-hold on total return or Sharpe ratio."),
        callout("<b>Verdict:</b> A technically complete, leakage-safe, cost-aware ML system whose honest result "
                "is that next-day direction is essentially unpredictable with this feature set. "
                "Explaining <i>why</i> that is the case is the impressive part."),
    ]

    # ---- 2 Problem --------------------------------------------------------
    story += [
        P("2. Problem definition and success criteria", H1),
        P("<b>Target.</b> A binary label: 1 if the adjusted close on day t+1 is greater than the adjusted close "
          "on day t, otherwise 0. This is classification, not regression: we care about direction, not size."),
        P("<b>Assets.</b> AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA and SPY. SPY and VIX returns are also used "
          "as market-context features, so the model can react to broad-market conditions."),
        P("<b>Period.</b> 2010-01-04 to 2026-11-20, roughly 4,200 daily bars per long-history ticker. META "
          "starts in 2012 and TSLA in 2010; each ticker keeps its own available history."),
        P("<b>Success criteria.</b> A model must beat two trivial baselines (always predict up; repeat "
          "yesterday's direction) on accuracy, and its backtest must beat buy-and-hold after transaction "
          "costs. We also report precision, recall, F1, ROC-AUC, log loss, Sharpe, Sortino and maximum "
          "drawdown."),
        P("<b>Why this is hard.</b> Daily returns have near-zero autocorrelation and fat tails. If yesterday's "
          "return told us tomorrow's direction, a simple strategy would work - it does not. Feature-engineering "
          "and model choice can only extract weak, transient structure."),
    ]

    # ---- 3 Structure ------------------------------------------------------
    story += [
        P("3. Project structure and environment setup", H1),
        P("The repository separates data, research notebooks, reusable source modules, a dashboard, saved "
          "artefacts and reports. Raw data and trained joblib files are gitignored because they are large and "
          "reproducible; the JSON metadata and result tables are committed."),
        code("""
stock-market/
|-- data/
|   |-- raw/            # cached yfinance OHLCV CSVs (gitignored)
|   `-- processed/      # engineered features + quality JSON (gitignored)
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_features.ipynb
|   `-- 03_modeling.ipynb
|-- src/
|   |-- data.py         # download, validate, clean, prepare
|   |-- features.py     # point-in-time features and labels
|   |-- models.py       # estimators, CV, tuning, artefacts
|   |-- evaluate.py     # baselines, tables, error analysis
|   |-- backtest.py     # cost-aware strategy and metrics
|   |-- cli.py          # command-line workflow
|   `-- utils.py        # paths, logging, performance helpers
|-- app/streamlit_app.py
|-- models/             # *.joblib (gitignored) + *.json metadata
|-- reports/            # CSV tables, figures and this PDF
|-- requirements.txt
|-- Makefile
`-- README.md
"""),
        P("<b>Environment.</b> A Python 3.9 virtual environment is created with <font face='Courier'>python3 -m venv venv</font>. "
          "Dependencies are pinned in <font face='Courier'>requirements.txt</font> and installed with pip. On macOS, "
          "XGBoost and LightGBM wheels require OpenMP: <font face='Courier'>brew install libomp</font>."),
    ]
    story += figure(FIGURES / "report_workflow.png", "Figure 2: The full workflow, from setup to documentation.")

    # ---- 4 Data -----------------------------------------------------------
    qc_rows = [["Check", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in quality.items()]
    story += [
        P("4. Data collection, validation and cleaning", H1),
        P("Data comes from Yahoo Finance via <font face='Courier'>yfinance</font>, downloaded with "
          "<font face='Courier'>auto_adjust=True</font> so splits and dividends are already reflected in the "
          "OHLC series. The download helper flattens yfinance's single- and multi-index column layouts, keeps "
          "only Open/High/Low/Close/Volume, converts to numeric values, sorts by date and caches the raw CSV."),
        code("""
def download_ticker(ticker, start="2010-01-01", end=None, force=False):
    cache = RAW_DIR / f"{ticker}.csv"
    if cache.exists() and not force:
        return _standardise(pd.read_csv(cache, index_col=0, parse_dates=True))
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False, threads=False)
    df = _standardise(raw)            # flatten columns, numeric, sorted
    df.to_csv(cache)
    return df
"""),
        P("<b>Validation.</b> Before feature engineering we check row count, date range, duplicate dates, "
          "missing weekdays, NaNs, zero-volume rows, non-positive closes and implausible daily moves. The report "
          "is saved as <font face='Courier'>data/processed/&lt;TICKER&gt;_quality.json</font>."),
        table(qc_rows, widths=[7 * cm, 9.2 * cm], font=8, center_from=99),
        P("<b>Cleaning.</b> Rows are de-duplicated (keeping the last observation), sorted, stripped of "
          "non-finite values, and rows with missing OHLCV or zero volume are dropped. The VIX index is an "
          "exception: it legitimately reports zero volume, so that filter is skipped for tickers starting "
          "with '^'. This detail matters - without it, every VIX row would be discarded and the VIX context "
          "feature would be entirely NaN."),
        P("<b>Data quality findings.</b> Missing weekdays are expected market holidays. Large absolute returns "
          "are mostly split-adjusted legacy moves or high-volatility periods (for example 2008 and 2020), not "
          "data errors, because auto-adjustment changes historical prices."),
    ]

    # ---- 5 EDA ------------------------------------------------------------
    story += [
        P("5. Exploratory data analysis", H1),
        P("The EDA notebook plots price, volume, daily returns, rolling volatility, class balance and return "
          "autocorrelation. Four findings drive the design:"),
        bullets([
            "<b>Price and volume:</b> long-term uptrend with regime changes; volume spikes around earnings and crashes.",
            "<b>Return distribution:</b> clearly fat-tailed and non-normal. Extreme days are frequent enough that a Gaussian assumption would be misleading.",
            "<b>Rolling volatility:</b> strongly time-varying (volatility clustering). This motivates volatility features and regime-based error analysis.",
            "<b>Class balance:</b> up days are about 53-54% of observations. This number is the always-up baseline that every model must beat.",
            "<b>Autocorrelation:</b> returns at lags 1-5 are close to zero, which explains why next-day prediction is hard.",
        ]),
    ]
    story += figure(FIGURES / "aapl_eda.png", "Figure 3: AAPL adjusted close, volume, return distribution and 20-day rolling volatility.")
    story += figure(FIGURES / "feature_correlation.png", "Figure 4: Feature correlation heatmap. Correlated momentum features are expected; tree models tolerate them, linear models are regularised.")

    # ---- 6 Features -------------------------------------------------------
    feature_rows = [
        ["Group", "Features", "Purpose"],
        ["Returns", "ret_1, ret_2, ret_5, ret_10, ret_20", "Recent direction and momentum"],
        ["Trend", "sma/ema 5, 10, 20, 50 (distance from price)", "Where price sits vs its trend"],
        ["Momentum", "rsi_14, macd, macd_signal, macd_diff", "Overbought/oversold and trend shifts"],
        ["Volatility", "vol_10, vol_20, bb_width, bb_pos, atr_14, hl_range", "Risk regime and band position"],
        ["Volume", "vol_chg, vol_ratio", "Participation and unusual activity"],
        ["Candles", "oc_change, gap", "Intraday body and overnight gap"],
        ["Calendar", "dow, month", "Day-of-week and seasonal effects"],
        ["Market", "SPY/VIX returns, vol, SPY volume change", "Broad-market context"],
    ]
    story += [
        P("6. Feature engineering and leakage control", H1),
        P("The central rule: <b>a feature at row t may only use information available at the close of day t "
          "or earlier</b>. Rolling windows, EMAs and indicators all include the current bar but never the next "
          "one. The label uses <font face='Courier'>Close.shift(-1)</font> and the backtest return uses "
          "<font face='Courier'>pct_change().shift(-1)</font>; both live outside the feature list."),
        code("""
d["ret_1"] = close.pct_change()
for n in (5, 10, 20, 50):
    d[f"sma_{n}"] = close.rolling(n).mean() / close - 1
    d[f"ema_{n}"] = close.ewm(span=n, adjust=False).mean() / close - 1
d["rsi_14"] = RSIIndicator(close, 14).rsi()
d["vol_20"] = d["ret_1"].rolling(20).std()
d["target"]   = (close.shift(-1) > close).astype(int)   # LABEL ONLY
d["next_ret"] = close.pct_change().shift(-1)             # BACKTEST ONLY
features = [c for c in d if c not in {"target", "next_ret", "Open",
                                      "High", "Low", "Close", "Volume"}]
"""),
        table(feature_rows, widths=[2.4 * cm, 7.8 * cm, 6 * cm], font=7.6, center_from=99),
        P("The <font face='Courier'>ta</font> library supplies RSI, MACD, Bollinger Bands and ATR. Pure-pandas "
          "fallbacks exist if the library is unavailable. Market context is aligned by date from pre-downloaded "
          "SPY and VIX frames; any missing values are tolerated by dropping the first ~50 rows of each series. "
          "The final feature frame for AAPL has <b>4,198 rows and 34 features</b>."),
        P("<b>Leakage assertions in notebook 02:</b> (1) target equals the sign of next_ret; (2) next_ret at t "
          "equals ret_1 at t+1; (3) neither column appears in the feature list. These catch the most common "
          "and most dangerous mistake in financial ML."),
    ]

    # ---- 7 Split and baselines -------------------------------------------
    story += [
        P("7. Chronological split and baselines", H1),
        P("A random train/test split would leak future information into the training set. We use a simple "
          "80/20 chronological cut: train from 2010-03-16 to 2023-07-18 (3,358 rows), test from 2023-07-19 to "
          "2026-11-19 (840 rows). Scalers are fitted on training data only, inside a scikit-learn Pipeline."),
        code("""
split = int(len(d) * 0.8)
train, test = d.iloc[:split], d.iloc[split:]
X_train, y_train = train[features], train["target"]
X_test,  y_test  = test[features],  test["target"]
"""),
        P("<b>Baseline 1 - always up.</b> Accuracy equals the share of up days: 0.5381 on the AAPL test set. "
          "This is the single most important number in the project."),
        P("<b>Baseline 2 - yesterday's direction.</b> Accuracy 0.5048. Near-random, consistent with near-zero "
          "return autocorrelation."),
        P("<b>Baseline 3 - buy-and-hold.</b> Used in the backtest: +78.32% total return over the holdout, "
          "Sharpe 0.79, max drawdown 33.36%."),
    ]

    # ---- 8 Models ---------------------------------------------------------
    model_rows = [
        ["Model", "Key settings", "Why included"],
        ["Logistic Regression", "StandardScaler, C=0.1, max_iter=2000", "Simple, interpretable linear baseline"],
        ["Random Forest", "400 trees, max_depth=5, min_leaf=10", "Non-linear interactions, robust to noise"],
        ["XGBoost", "300 trees, depth=3, lr=0.03, subsample=0.8", "Strong tabular learner; tuned on train only"],
        ["LightGBM", "300 trees, 7 leaves, lr=0.03", "Alternative boosting implementation"],
    ]
    story += [
        P("8. Modelling: from simple to complex", H1),
        P("Models are trained in increasing complexity. Trees are deliberately shallow and learning rates low "
          "because financial data overfits easily. XGBoost is the only model tuned, using RandomizedSearchCV "
          "with 15 iterations on the training period; the best parameters were n_estimators=150, max_depth=4, "
          "learning_rate=0.05, subsample=0.7, colsample_bytree=0.8, min_child_weight=5."),
        code("""
model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8,
                      objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", random_state=42)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]   # P(up)
pred  = (probs > 0.5).astype(int)
"""),
        table(model_rows, widths=[3.6 * cm, 6.6 * cm, 6 * cm], font=7.8, center_from=99),
        P("Every saved artefact records the model version, ticker, exact feature order, training date range, "
          "metrics, threshold and any tuned parameters. At prediction time the input columns are re-ordered "
          "to the training order, which prevents silent feature mismatch in the app."),
    ]

    # ---- 9 Validation -----------------------------------------------------
    cv_rows = [["Model", "Mean CV accuracy", "Std", "Fallback baseline"]]
    for name in ["logistic", "random_forest", "xgboost", "lightgbm"]:
        wf = aapl[name]["walk_forward"]
        cv_rows.append([name, num(wf["mean_accuracy"]), num(wf["std_accuracy"]), num(0.538095)])
    story += [
        P("9. Walk-forward validation and tuning", H1),
        P("TimeSeriesSplit with five splits creates expanding training windows and forward-in-time validation "
          "folds. Each fold clones the estimator, fits on the earlier window and scores the later window. This "
          "respects causality and gives a more realistic estimate than a single holdout."),
        code("""
tscv = TimeSeriesSplit(n_splits=5)
for tr, va in tscv.split(X_train):
    est = clone(model)
    est.fit(X_train.iloc[tr], y_train.iloc[tr])
    p = est.predict_proba(X_train.iloc[va])[:, 1]
    print(accuracy_score(y_train.iloc[va], p > 0.5))
"""),
        table(cv_rows, widths=[5 * cm, 4 * cm, 3 * cm, 4.2 * cm], font=8),
        P("Mean CV accuracies of 0.509 to 0.525 are all below the 0.538 always-up baseline. Crucially, the "
          "test set was not used during tuning or fold selection; it was scored once at the end. A confident-subset "
          "analysis (probabilities below 0.45 or above 0.55) was also checked on the test set, but did not reveal "
          "a profitable filter."),
    ]
    story += figure(FIGURES / "report_cv_accuracy.png", "Figure 5: Walk-forward CV accuracy by model. The dashed line is the always-up baseline.")

    # ---- 10 Results -------------------------------------------------------
    res_rows = [["Model", "Acc", "Prec", "Recall", "F1", "AUC", "LogLoss"]]
    for name in ["always_up", "yesterday_direction", "logistic", "random_forest", "xgboost", "lightgbm"]:
        if name in ("always_up", "yesterday_direction"):
            m = {"accuracy": 0.538095 if name == "always_up" else 0.504762,
                 "precision": 0.538095 if name == "always_up" else 0.539823,
                 "recall": 1.0 if name == "always_up" else 0.539823,
                 "f1": 0.699690 if name == "always_up" else 0.539823,
                 "roc_auc": float("nan"), "log_loss": float("nan")}
        else:
            m = aapl[name]["metrics"]
        res_rows.append([
            name, num(m["accuracy"]), num(m["precision"]), num(m["recall"]),
            num(m["f1"]), num(m["roc_auc"]), num(m["log_loss"]),
        ])
    story += [
        P("10. Results on the untouched test set", H1),
        P("The table below is the honest scoreboard for AAPL. The models cluster around 0.50-0.53 accuracy, "
          "with Random Forest the best of the four but still 1.07 percentage points below always-up. ROC-AUC "
          "near 0.50 means the probability ranking is no better than chance."),
        table(res_rows, widths=[3.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm, 1.8 * cm, 2 * cm], font=7.4),
    ]
    story += figure(FIGURES / "report_confusion.png", "Figure 6: Test-set confusion matrices. The models mostly predict 'up', which is why they resemble the always-up baseline.")

    # ---- 11 Backtest ------------------------------------------------------
    bt_rows = [["Metric", "Logistic", "Random Forest", "XGBoost", "LightGBM", "Buy & hold"]]
    keys = ["total_return", "cagr", "annualized_vol", "sharpe", "sortino", "max_drawdown"]
    labels = ["Total return", "CAGR", "Annualised vol", "Sharpe", "Sortino", "Max drawdown"]
    for key, label in zip(keys, labels):
        row = [label]
        for name in ["logistic", "random_forest", "xgboost", "lightgbm"]:
            row.append(pct(aapl[name]["backtest"][key]) if key != "sharpe" and key != "sortino" else num(aapl[name]["backtest"][key], 2))
        row.append(pct(aapl["xgboost"]["buy_and_hold"][key]) if key not in ("sharpe", "sortino") else num(aapl["xgboost"]["buy_and_hold"][key], 2))
        bt_rows.append(row)
    story += [
        P("11. Backtesting with transaction costs", H1),
        P("A signal computed from features at day t is applied to the return from t to t+1 stored in "
          "<font face='Courier'>next_ret</font>. A one-way cost of 0.1% is charged whenever the position "
          "changes. In long/cash mode the signal is 1 when P(up) exceeds the threshold, else 0. In long/short "
          "mode, P(up) below 1-threshold opens a short."),
        code("""
signal  = (probs > threshold).astype(int)          # 1 = invested, 0 = cash
turnover = signal.diff().fillna(signal).abs()      # position changes
strat_ret = signal * test["next_ret"] - turnover * cost
equity    = (1 + strat_ret).cumprod()
bh        = (1 + test["next_ret"]).cumprod()
"""),
        table(bt_rows, widths=[3.4 * cm, 2.5 * cm, 2.9 * cm, 2.5 * cm, 2.5 * cm, 2.4 * cm], font=7.2),
        P("The strategies are invested only about 30-40% of the time. That naturally lowers volatility and "
          "drawdown, but it also means they participate in fewer up days. None beats buy-and-hold on total "
          "return or Sharpe. The lower drawdown is a cash-allocation effect, not evidence of skill."),
    ]
    story += figure(FIGURES / "AAPL_equity_all_models.png", "Figure 7: AAPL equity curves with transaction costs. The dashed line is buy-and-hold.")

    # ---- 12 Interpretation ------------------------------------------------
    imp_path = REPORTS / "AAPL_xgboost_feature_importance.csv"
    imp_rows = [["Feature", "Importance"]]
    with imp_path.open(encoding="utf-8") as fh:
        for row in list(csv.DictReader(fh))[:10]:
            imp_rows.append([row["feature"], num(float(row["importance"]), 4)])
    err_path = REPORTS / "AAPL_xgboost_error_analysis.csv"
    err_rows = [["Vol quintile", "N", "Accuracy", "Up rate", "Mean vol"]]
    with err_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            err_rows.append([
                row["vol_quintile"], row["n"], num(float(row["accuracy"])),
                num(float(row["up_rate"])), num(float(row["mean_vol"]), 5),
            ])
    story += [
        P("12. Interpretation: importance, SHAP and errors", H1),
        P("Tree feature importance shows which columns the model splits on most; SHAP goes further by attributing "
          "each individual prediction to each feature. Both are descriptive, not causal: a feature can be "
          "important without being tradable."),
        table(imp_rows, widths=[8 * cm, 5 * cm], font=8),
        P("Top XGBoost features are dominated by long-horizon trend and market-context variables - exactly the "
          "slow-moving features a model can latch onto when there is no short-term signal."),
    ]
    story += figure(FIGURES / "AAPL_xgboost_shap.png", "Figure 8: SHAP summary plot for the tuned XGBoost model on a 500-row test sample.")
    story += [
        P("<b>Error analysis by volatility quintile.</b> Rows are ranked by 20-day volatility and split into five "
          "groups. Accuracy and up-rate by bucket help identify regimes where the model fails."),
        table(err_rows, widths=[3 * cm, 2.2 * cm, 3 * cm, 3 * cm, 3.2 * cm], font=7.8),
        P("High-volatility periods are the hardest: labels become noisier, reversals are frequent, and trading "
          "costs consume any small edge. This is where a naive 70%+ backtest would usually be hiding a bug."),
    ]

    # ---- 13 Multi ticker --------------------------------------------------
    multi_rows = [["Ticker", "Acc", "AUC", "WF CV", "Strategy", "Buy & hold"]]
    for d in multi:
        multi_rows.append([
            d["ticker"], num(d["metrics"]["accuracy"]), num(d["metrics"]["roc_auc"]),
            num(d["walk_forward"]["mean_accuracy"]),
            pct(d["backtest"]["total_return"]), pct(d["buy_and_hold"]["total_return"]),
        ])
    story += [
        P("13. Multi-ticker robustness check", H1),
        P("Testing one stock or one period is not enough. The same XGBoost pipeline was applied to all eight "
          "tickers. Six models score below their always-up baseline on the test set. NVDA is the single apparent "
          "winner, with 53.33% accuracy and a strategy return of 208.4% versus 167.5% buy-and-hold - but its "
          "walk-forward CV accuracy was only 0.5013, essentially random. Choosing the best of eight after the "
          "fact is selection bias, not evidence of an edge."),
        table(multi_rows, widths=[2.8 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 3 * cm, 3 * cm], font=7.8),
    ]
    story += figure(FIGURES / "report_multiticker.png", "Figure 9: Strategy versus buy-and-hold total return across all eight tickers.")

    # ---- 14 App -----------------------------------------------------------
    story += [
        P("14. Streamlit application", H1),
        P("The dashboard is the presentation layer of the same pipeline. It discovers saved joblib artefacts, "
          "loads the requested model, downloads/caches prices, rebuilds features (including market context when "
          "the artefact needs it), shows the latest probability and signal, plots an interactive candlestick "
          "chart with SMA 20/50 and volume, runs the backtest on the selected date range, and offers SHAP and "
          "data-quality panels. A prominent disclaimer is always visible."),
        code("""
artifacts = list_artifacts()                 # discover models/*.joblib
art  = load_artifact(chosen_path)
probs = predict_proba(art, featured[features])
st.metric("P(up tomorrow)", f"{probs[-1]:.1%}")
result = backtest(featured, probs, threshold, cost, mode)
st.plotly_chart(equity_figure)
"""),
        P("Run locally with <font face='Courier'>streamlit run app/streamlit_app.py</font>. The Streamlit Cloud "
          "free tier can deploy the repository directly; raw data and models are regenerated by the CLI steps "
          "or uploaded as secrets/artefacts if the deployed app needs pretrained models."),
    ]

    # ---- 15 Upgrades ------------------------------------------------------
    story += [
        P("15. Optional upgrades and future work", H1),
        bullets([
            "<b>Sentiment:</b> score daily news headlines with FinBERT and add a sentiment feature. Risk: look-ahead bias if headline timestamps are not handled carefully.",
            "<b>Macro features:</b> interest rates, oil, dollar index and VIX term structure; test whether regime context helps.",
            "<b>Ensemble:</b> average probabilities from several models; in this project that would likely converge toward the always-up prediction.",
            "<b>Regression target:</b> predict next-day return magnitude, then convert to a position size rather than a binary signal.",
            "<b>LSTM:</b> a 30-60 day lookback sequence model, only after tabular baselines are exhausted.",
            "<b>Realistic execution:</b> model bid/ask spread, slippage, order queue and overnight gap risk.",
        ]),
    ]

    # ---- 16 How to run ----------------------------------------------------
    story += [
        P("16. How to run the project", H1),
        code("""
# macOS only: OpenMP for XGBoost / LightGBM
brew install libomp

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.cli download --tickers AAPL MSFT GOOGL AMZN META TSLA NVDA SPY '^VIX' --start 2010-01-01
python -m src.cli prepare --tickers AAPL MSFT GOOGL AMZN META TSLA NVDA SPY --market
python -m src.cli train --ticker AAPL --model xgboost --with-market --tune
python -m src.cli report --ticker AAPL --with-market
python -m src.cli explain --ticker AAPL --model xgboost --with-market
streamlit run app/streamlit_app.py
"""),
        P("The Makefile provides shortcuts: <font face='Courier'>make setup</font>, "
          "<font face='Courier'>make train</font>, <font face='Courier'>make report</font>, "
          "<font face='Courier'>make app</font>. Notebooks 01-03 reproduce the research path interactively."),
    ]

    # ---- 17 Limitations ---------------------------------------------------
    story += [
        P("17. Limitations, mistakes avoided and final verdict", H1),
        P("<b>Limitations.</b> Results come from one data vendor and a specific 2023-2026 regime. Costs are "
          "modelled but slippage, spread and market impact are not. Adjusted prices assume vendor correctness. "
          "Feature importance is associative, not causal. Tuning across multiple models and tickers introduces "
          "selection bias. No claim of profitability is made."),
        P("<b>Mistakes deliberately avoided.</b> No future information in features; no random train/test split; "
          "no tuning on the test set; no backtest without costs; no celebration of suspiciously high accuracy; "
          "no conclusions drawn from a single stock or a single period."),
        callout("<b>Final verdict:</b> The system is technically sound and fully reproducible, but the honest "
                "finding is that next-day direction is not predictable from this feature set after costs. "
                "The rigorous evaluation, not the accuracy, is the deliverable.", bg=colors.HexColor("#e8f5e9")),
    ]

    # ---- Appendix ---------------------------------------------------------
    story += [
        P("Appendix A: file and command reference", H1),
        table([
            ["File / command", "Purpose"],
            ["src/data.py", "Download, flatten, validate, clean and prepare OHLCV data"],
            ["src/features.py", "34 point-in-time features, labels, chronological split"],
            ["src/models.py", "Estimators, walk-forward CV, tuning, artefact save/load"],
            ["src/evaluate.py", "Baselines, comparison tables, error analysis"],
            ["src/backtest.py", "Cost-aware strategy, Sharpe/Sortino/drawdown, plots"],
            ["src/cli.py", "download / prepare / train / report / explain / app / all"],
            ["app/streamlit_app.py", "Interactive dashboard"],
            ["reports/*.csv, *.json", "All result tables and run metadata"],
            ["reports/figures/", "Charts embedded in this document"],
        ], widths=[5.6 * cm, 10.6 * cm], font=8, center_from=99),
        Spacer(1, 0.5 * cm),
        P("Generated automatically from the project artefacts by scripts/build_pdf.py.", CAPTION),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
