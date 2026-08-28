"""Proof that the model is never shown information from the future.

The argument a client will want is not "we were careful" but "here is the
check". This rebuilds the entire feature set on a copy of the data in which
every hold-out sales figure has been erased, and asserts the hold-out feature
values come out identical. If any feature drew on a value the planner could not
have known, erasing that value would change the result and this test would fail.

Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data, features, models


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return data.build_panel()


@pytest.fixture(scope="module")
def frame(panel: pd.DataFrame) -> pd.DataFrame:
    return features.build_feature_frame(panel, config.HOLDOUT_WEEKS)


def test_panel_is_a_continuous_weekly_grid(panel: pd.DataFrame):
    gaps = (
        panel.sort_values(["Store", "Dept", "Date"])
        .groupby(["Store", "Dept"])["Date"]
        .diff()
        .dropna()
    )
    assert (gaps == pd.Timedelta(weeks=1)).all()


def test_one_row_per_store_dept_week(panel: pd.DataFrame):
    assert not panel.duplicated(["Store", "Dept", "Date"]).any()


def test_no_sales_feature_is_fresher_than_the_horizon(frame: pd.DataFrame):
    lags = [
        int(c.removeprefix("SalesLag"))
        for c in features.feature_columns(frame)
        if c.startswith("SalesLag")
    ]
    assert lags, "expected lag features to exist"
    assert min(lags) >= config.HOLDOUT_WEEKS


def test_holdout_features_survive_erasing_the_future(panel: pd.DataFrame, frame: pd.DataFrame):
    split = models.make_time_split(frame)

    censored = panel.copy()
    censored.loc[censored["Date"] >= split.holdout_start, "Weekly_Sales"] = np.nan
    censored_frame = features.build_feature_frame(censored, config.HOLDOUT_WEEKS)

    holdout_mask = frame["Date"] >= split.holdout_start
    keys = ["Store", "Dept", "Date"]
    feature_cols = [c for c in features.feature_columns(frame) if c not in keys]

    actual = frame.loc[holdout_mask, keys + feature_cols].reset_index(drop=True)
    censored_result = censored_frame.loc[
        censored_frame["Date"] >= split.holdout_start, keys + feature_cols
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, censored_result)


def test_split_blocks_do_not_overlap_in_time(frame: pd.DataFrame):
    split = models.make_time_split(frame)

    assert frame.loc[split.train, "Date"].max() < split.validation_start
    assert frame.loc[split.validation, "Date"].max() < split.holdout_start
    assert frame.loc[split.holdout, "Date"].min() >= split.holdout_start
    assert not (split.train & split.holdout).any()


def test_baseline_matches_same_week_last_year(frame: pd.DataFrame):
    forecast = models.seasonal_naive_forecast(frame, config.HOLDOUT_WEEKS)
    has_seasonal = frame["SalesLag52"].notna()
    assert np.allclose(forecast[has_seasonal], frame.loc[has_seasonal, "SalesLag52"])
