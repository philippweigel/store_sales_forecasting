"""Charts for the written case study.

Two forecast series carry identity through colour and are always direct-labelled
as well, so the charts survive greyscale printing and colour-vision deficiency.
The actual sales line is deliberately not a third colour: it is the reference
the forecasts are judged against, so it wears neutral ink.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from . import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"

ACTUAL = "#3d3d3a"
BASELINE = "#eb6834"
MODEL = "#2a78d6"


def _style_axes(ax: plt.Axes, *, title: str, subtitle: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)

    if subtitle:
        ax.set_title(subtitle, color=INK_SOFT, fontsize=10, loc="left", pad=8)
        ax.figure.suptitle(title, color=INK, fontsize=13, fontweight="600",
                           x=ax.get_position().x0, ha="left", y=0.98)
    else:
        ax.set_title(title, color=INK, fontsize=13, fontweight="600", loc="left", pad=10)


def _millions(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:,.0f}M"))


def _save(fig: plt.Figure, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=160, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def plot_sales_history(panel: pd.DataFrame) -> None:
    """Total weekly turnover, to establish the seasonal shape being forecast."""
    weekly = panel.groupby("Date", as_index=False)["Weekly_Sales"].sum()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(weekly["Date"], weekly["Weekly_Sales"], color=ACTUAL, linewidth=2)

    peaks = weekly.nlargest(3, "Weekly_Sales")
    ax.scatter(peaks["Date"], peaks["Weekly_Sales"], s=42, color=BASELINE, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    for _, row in peaks.iterrows():
        ax.annotate(row["Date"].strftime("%b %Y"),
                    (row["Date"], row["Weekly_Sales"]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=INK_SOFT)

    # Headroom so the peak labels never run into the subtitle.
    ax.set_ylim(top=weekly["Weekly_Sales"].max() * 1.12)

    _style_axes(ax, title="Total weekly sales across all 45 stores",
                subtitle="The year turns on a handful of weeks in November and "
                         "December; the rest hold a narrow band",
                ylabel="Weekly sales")
    _millions(ax)
    _save(fig, "01_sales_history.png")


def plot_seasonal_profile(panel: pd.DataFrame) -> None:
    """Average turnover by calendar week: the pattern a planner must anticipate."""
    df = panel.dropna(subset=["Weekly_Sales"]).copy()
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    profile = df.groupby("WeekOfYear", as_index=False)["Weekly_Sales"].mean()

    overall = profile["Weekly_Sales"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(profile["WeekOfYear"], profile["Weekly_Sales"], color=MODEL, width=0.7)
    ax.axhline(overall, color=INK_SOFT, linewidth=1.5, linestyle="--")
    ax.annotate("average week", (1, overall), textcoords="offset points",
                xytext=(2, 6), fontsize=9, color=INK_SOFT)

    _style_axes(ax, title="Average sales per store-department, by calendar week",
                subtitle="Weeks 47-52 run far above the year's baseline — the window "
                         "where ordering errors are most expensive",
                ylabel="Average weekly sales")
    ax.set_xlabel("Calendar week", color=INK_SOFT, fontsize=9)
    _save(fig, "02_seasonal_profile.png")


def plot_accuracy_comparison(summary: pd.DataFrame) -> None:
    """The headline: error under the current planning method versus the model."""
    indexed = summary.set_index("Model")
    labels = ["Seasonal naive", "XGBoost"]
    values = [indexed.loc[label, "WMAPE"] * 100 for label in labels]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, values, color=[BASELINE, MODEL], width=0.38)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=11, fontweight="600", color=INK)

    reduction = (values[0] - values[1]) / values[0]
    _style_axes(ax, title="Forecast error on the 12-week hold-out",
                subtitle=f"Weighted MAPE — lower is better. "
                         f"The model cuts error by {reduction:.0%}.",
                ylabel="WMAPE")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, max(values) * 1.35)
    _save(fig, "03_accuracy_comparison.png")


def plot_forecast_vs_actual(results: pd.DataFrame, store: int, dept: int) -> None:
    """One series in detail, so the numbers stop being abstract."""
    series = results[(results["Store"] == store) & (results["Dept"] == dept)]
    series = series.sort_values("Date")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(series["Date"], series["Actual"], color=ACTUAL, linewidth=2.5,
            label="Actual", zorder=3)
    ax.plot(series["Date"], series["Baseline_Forecast"], color=BASELINE,
            linewidth=2, linestyle="--", label="Seasonal naive")
    ax.plot(series["Date"], series["XGBoost_Forecast"], color=MODEL,
            linewidth=2, label="XGBoost")

    last = series.iloc[-1]
    for column, colour, label in (
        ("Actual", ACTUAL, "Actual"),
        ("Baseline_Forecast", BASELINE, "Seasonal naive"),
        ("XGBoost_Forecast", MODEL, "XGBoost"),
    ):
        ax.annotate(f" {label}", (last["Date"], last[column]), color=colour,
                    fontsize=9, va="center", fontweight="600")

    # Room on the right for the direct labels, which would otherwise be clipped.
    span = series["Date"].max() - series["Date"].min()
    ax.set_xlim(series["Date"].min(), series["Date"].max() + span * 0.22)

    _style_axes(ax, title=f"Store {store}, department {dept}: hold-out forecast",
                subtitle="Forecasts made once, 12 weeks ahead — never revised with "
                         "newer sales",
                ylabel="Weekly sales")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left",
              ncols=3)
    _save(fig, "04_forecast_vs_actual.png")


def plot_accuracy_by_store_type(by_type: pd.DataFrame) -> None:
    """Where the gain shows up, so the result is not a single averaged number."""
    df = by_type.sort_values("StoreType")
    x = range(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i - width / 2 - 0.01 for i in x], df["Seasonal naive_WMAPE"] * 100,
           width=width, color=BASELINE, label="Seasonal naive")
    ax.bar([i + width / 2 + 0.01 for i in x], df["XGBoost_WMAPE"] * 100,
           width=width, color=MODEL, label="XGBoost")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Type {t}" for t in df["StoreType"]])

    _style_axes(ax, title="Forecast error by store format",
                subtitle="The model wins in every format, not only on average",
                ylabel="WMAPE")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=2)
    _save(fig, "05_accuracy_by_store_type.png")


# Column names are for the code; this chart is read by people who will never
# open the code.
FEATURE_LABELS = {
    "SeasonalLevel": "Typical level of this series",
    "SalesRollMean4": "Average of last 4 known weeks",
    "SalesRollMean13": "Average of last 13 known weeks",
    "SalesRollMean52": "Average of last 52 known weeks",
    "SalesRollStd4": "Volatility, last 4 known weeks",
    "SalesRollStd13": "Volatility, last 13 known weeks",
    "SalesRollStd52": "Volatility, last 52 known weeks",
    "SalesLag12": "Sales 12 weeks ago",
    "SalesLag13": "Sales 13 weeks ago",
    "SalesLag14": "Sales 14 weeks ago",
    "SalesLag15": "Sales 15 weeks ago",
    "SalesLag52": "Sales same week last year",
    "SalesLag104": "Sales same week two years ago",
    "SeasonalExpectation": "Seasonal expectation for this week",
    "SeasonalIndex": "This week's seasonal index",
    "SalesTrend13v52": "Recent trend vs. yearly average",
    "SalesYoYDelta": "Year-on-year change",
    "SalesCV52": "Relative volatility",
    "Size": "Store size",
    "Type": "Store format",
    "DeptCat": "Department",
    "StoreCat": "Store",
    "MarkdownsTracked": "Promotions tracked yet",
    "MarkdownTotal": "Total promotion spend",
    "WeeksSinceStart": "Weeks since history begins",
    "WeekOfYear": "Calendar week",
    "IsPreChristmas": "Week before Christmas",
    "IsChristmas": "Christmas week",
    "IsPreThanksgiving": "Week before Thanksgiving",
    "IsThanksgiving": "Thanksgiving week",
    "CPI": "Consumer price index",
    "Unemployment": "Local unemployment",
    "Temperature": "Temperature",
    "Fuel_Price": "Fuel price",
}


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 12) -> None:
    """What the model actually leans on -- a transparency exhibit, not a ranking."""
    df = importance.head(top_n).iloc[::-1]
    labels = df["Feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    share = df["Gain"] / importance["Gain"].sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, share, color=MODEL, height=0.65)

    _style_axes(ax, title="What drives the forecast",
                subtitle="Share of total model gain — recent sales level and the "
                         "series' own seasonal shape do most of the work",
                ylabel="")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _save(fig, "06_feature_importance.png")


def build_all(panel: pd.DataFrame, artefacts: dict[str, pd.DataFrame]) -> None:
    results = artefacts["results"]
    plot_sales_history(panel)
    plot_seasonal_profile(panel)
    plot_accuracy_comparison(artefacts["summary"])
    plot_accuracy_by_store_type(artefacts["by_type"])
    plot_feature_importance(artefacts["importance"])

    # Pick a high-volume series so the illustrative chart is representative of
    # where the money actually is, rather than a cherry-picked easy one.
    busiest = (
        results.groupby(["Store", "Dept"])["Actual"].sum().idxmax()
    )
    plot_forecast_vs_actual(results, int(busiest[0]), int(busiest[1]))
