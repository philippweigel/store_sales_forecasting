"""Forecast accuracy metrics and segment breakdowns.

WMAPE rather than MAPE is the headline figure. Classic MAPE divides by the
actual value, and this data contains 73 zero-sales weeks and 1,285 negative
ones (returns exceeding sales), which makes that division either undefined or
explosive -- a single week of 5 EUR actual sales can move an unweighted MAPE by
several points. WMAPE (total absolute error divided by total actual volume)
answers the question a planner actually asks: across everything I ordered, what
share of the volume did I get wrong? MAPE is still reported, restricted to weeks
with meaningful sales, because clients expect to see it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this weekly turnover a percentage error stops being informative.
MAPE_FLOOR = 100.0


def wmape(actual: pd.Series, forecast: pd.Series) -> float:
    denominator = actual.abs().sum()
    if denominator == 0:
        return np.nan
    return float((actual - forecast).abs().sum() / denominator)


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    return float((actual - forecast).abs().mean())


def mape(actual: pd.Series, forecast: pd.Series, floor: float = MAPE_FLOOR) -> float:
    mask = actual.abs() >= floor
    if not mask.any():
        return np.nan
    return float(((actual[mask] - forecast[mask]).abs() / actual[mask].abs()).mean())


def bias(actual: pd.Series, forecast: pd.Series) -> float:
    """Average signed error, as a share of volume.

    Separating bias from error size matters commercially: an unbiased forecast
    that is noisy costs safety stock, while a consistently low forecast causes
    stockouts no amount of safety stock policy will fix.
    """
    denominator = actual.abs().sum()
    if denominator == 0:
        return np.nan
    return float((forecast - actual).sum() / denominator)


def score(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    return {
        "WMAPE": wmape(actual, forecast),
        "MAPE": mape(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "MAE": mae(actual, forecast),
        "Bias": bias(actual, forecast),
    }


def score_models(df: pd.DataFrame, actual_col: str,
                 forecast_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for label, column in forecast_cols.items():
        metrics = score(df[actual_col], df[column])
        rows.append({"Model": label, "Weeks": len(df), **metrics})
    return pd.DataFrame(rows)


def score_by_segment(df: pd.DataFrame, segment: str, actual_col: str,
                     forecast_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(segment, observed=True):
        row = {
            segment: value,
            "Weeks": len(group),
            "ActualVolume": float(group[actual_col].sum()),
        }
        for label, column in forecast_cols.items():
            row[f"{label}_WMAPE"] = wmape(group[actual_col], group[column])
            row[f"{label}_RMSE"] = rmse(group[actual_col], group[column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ActualVolume", ascending=False)


def improvement(baseline: float, candidate: float) -> float:
    """Relative reduction in an error metric, where lower is better."""
    if baseline in (0, np.nan) or pd.isna(baseline):
        return np.nan
    return float((baseline - candidate) / baseline)
