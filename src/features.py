"""Time-series feature construction for the weekly store-department panel.

Every feature here obeys one rule: a row for week *t* may only contain
information that was already available at the moment the order decision for
week *t* had to be made. With a 12-week planning horizon, that means the most
recent sales figure the planner can see is the one from 12 weeks earlier -- so
no lag shorter than the horizon exists in this feature set. Using last week's
sales to predict a quarter ahead is the most common way a forecasting demo
flatters itself; it is deliberately impossible here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The four retail events the source data flags as holiday weeks. Naming them
# individually matters because their sales effects differ by an order of
# magnitude -- Thanksgiving and Christmas drive the year, Labor Day barely
# registers -- and a single boolean flag would force the model to average them.
NAMED_HOLIDAY_WEEKS = {
    "SuperBowl": ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "LaborDay": ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "Thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "Christmas": ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}

LAG_MULTIPLIERS = (0, 1, 2, 3)
SEASONAL_LAGS = (52, 104)
ROLLING_WINDOWS = (4, 13, 52)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date = df["Date"]

    iso = date.dt.isocalendar()
    df["WeekOfYear"] = iso.week.astype(int)
    df["Month"] = date.dt.month
    df["Year"] = date.dt.year
    df["WeeksSinceStart"] = ((date - date.min()).dt.days // 7).astype(int)

    # Week-of-year is circular: week 52 sits next to week 1. Encoding it as a
    # raw integer would tell a tree that those two weeks are maximally far
    # apart, so the cyclical pair is provided alongside it.
    df["WeekSin"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52.0)
    df["WeekCos"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52.0)

    for name, dates in NAMED_HOLIDAY_WEEKS.items():
        stamps = pd.to_datetime(dates)
        df[f"Is{name}"] = date.isin(stamps).astype(int)
        # The build-up week matters more than the event week for ordering:
        # stock has to be on the shelf before the peak.
        df[f"IsPre{name}"] = date.isin(stamps - pd.Timedelta(weeks=1)).astype(int)

    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df


def add_lag_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add sales history features, none fresher than `horizon` weeks old.

    The panel is on a continuous weekly grid, so a positional shift within a
    store-department group is a true shift in calendar time even for series
    with gaps in their recorded history.
    """
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    sales = df.groupby(["Store", "Dept"], sort=False)["Weekly_Sales"]

    lags = sorted({horizon + m for m in LAG_MULTIPLIERS} | set(SEASONAL_LAGS))
    for lag in lags:
        df[f"SalesLag{lag}"] = sales.shift(lag)

    # Rolling statistics are computed on the already-shifted series, so the
    # window can never reach into the horizon it is trying to predict.
    shifted = sales.shift(horizon)
    grouped = shifted.groupby([df["Store"], df["Dept"]], sort=False)
    for window in ROLLING_WINDOWS:
        roll = grouped.rolling(window, min_periods=max(2, window // 4))
        df[f"SalesRollMean{window}"] = roll.mean().reset_index(level=[0, 1], drop=True)
        df[f"SalesRollStd{window}"] = roll.std().reset_index(level=[0, 1], drop=True)

    # Level differences describe the trend the planner would already have seen.
    df["SalesTrend13v52"] = df["SalesRollMean13"] - df["SalesRollMean52"]
    df["SalesYoYDelta"] = df[f"SalesLag{horizon}"] - df["SalesLag52"]

    # How volatile this series is relative to its own level: a department whose
    # sales swing wildly deserves a different amount of model confidence than a
    # steady one, and the ratio makes that comparable across store sizes.
    df["SalesCV52"] = df["SalesRollStd52"] / df["SalesRollMean52"].abs().replace(0, np.nan)
    return df


def _expanding_mean(values: pd.Series, keys: list[pd.Series], *,
                    exclude_current: bool, min_periods: int = 1) -> pd.Series:
    """Running mean over a group's own history, ignoring missing observations.

    Written with cumulative sums rather than a per-group lambda: at ~170k
    week-of-year groups the lambda form takes minutes, this takes under a
    second, and both compute the same quantity.
    """
    filled = values.fillna(0.0)
    observed = values.notna().astype("int64")

    grouped_sum = filled.groupby(keys, sort=False).cumsum()
    grouped_count = observed.groupby(keys, sort=False).cumsum()

    if exclude_current:
        grouped_sum = grouped_sum - filled
        grouped_count = grouped_count - observed

    return (grouped_sum / grouped_count).where(grouped_count >= min_periods)


def add_seasonal_profile(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Each series' own week-of-year signature, built from prior years only.

    The share of a department's normal turnover that lands in, say, week 47 is
    the single most useful thing to know about it. It is expressed as a ratio to
    that series' own running level, so a small store and a large one with the
    same Christmas pattern get the same index. The value for 2012 week 47 is
    built only from earlier observations of week 47.
    """
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    series_keys = [df["Store"], df["Dept"]]

    lagged_sales = df.groupby(series_keys, sort=False)["Weekly_Sales"].shift(horizon)
    level = _expanding_mean(lagged_sales, series_keys, exclude_current=False, min_periods=8)

    ratio = df["Weekly_Sales"] / level.replace(0, np.nan)
    df["SeasonalIndex"] = _expanding_mean(
        ratio, [df["Store"], df["Dept"], df["WeekOfYear"]], exclude_current=True
    )
    df["SeasonalLevel"] = level
    df["SeasonalExpectation"] = df["SeasonalIndex"] * level
    return df


def build_feature_frame(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = add_calendar_features(panel)
    df = add_lag_features(df, horizon)
    df = add_seasonal_profile(df, horizon)

    df["Type"] = df["Type"].astype("category")
    df["StoreCat"] = df["Store"].astype("category")
    df["DeptCat"] = df["Dept"].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs, stated explicitly so nothing leaks in by accident."""
    excluded = {
        "Weekly_Sales",
        "Date",
        "Store",
        "Dept",
        "HasSales",
        # Calendar year is dropped deliberately: a tree cannot extrapolate to a
        # year it never saw, so it would only help the model memorise 2010-2012.
        "Year",
    }
    return [
        c
        for c in df.columns
        if c not in excluded and not c.endswith(("_Forecast", "_Error", "_AbsError"))
    ]
