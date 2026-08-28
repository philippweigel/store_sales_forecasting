"""Baseline and machine-learning forecasters, plus the time-based split.

Both models are asked the same question under the same rules: standing at the
end of the training history, forecast the next 12 weeks for every
store-department combination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config, features


@dataclass(frozen=True)
class Split:
    """Row masks for the three time blocks, plus the dates that separate them."""

    train: pd.Series
    validation: pd.Series
    holdout: pd.Series
    validation_start: pd.Timestamp
    holdout_start: pd.Timestamp

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Block": "Train", "Rows": int(self.train.sum())},
                {"Block": "Validation", "Rows": int(self.validation.sum())},
                {"Block": "Hold-out", "Rows": int(self.holdout.sum())},
            ]
        )


def make_time_split(df: pd.DataFrame,
                    holdout_weeks: int = config.HOLDOUT_WEEKS,
                    validation_weeks: int = config.VALIDATION_WEEKS) -> Split:
    """Cut the history into train / validation / hold-out by date.

    A random split would be meaningless here: the model would train on weeks
    that come after the weeks it is scored on. The validation block exists so
    that early stopping never observes the hold-out -- otherwise the reported
    accuracy quietly becomes an in-sample number.
    """
    last_date = df["Date"].max()
    holdout_start = last_date - pd.Timedelta(weeks=holdout_weeks - 1)
    validation_start = holdout_start - pd.Timedelta(weeks=validation_weeks)

    usable = df["Weekly_Sales"].notna()
    return Split(
        train=usable & (df["Date"] < validation_start),
        validation=usable & (df["Date"] >= validation_start) & (df["Date"] < holdout_start),
        holdout=usable & (df["Date"] >= holdout_start),
        validation_start=validation_start,
        holdout_start=holdout_start,
    )


def seasonal_naive_forecast(df: pd.DataFrame, horizon: int) -> pd.Series:
    """The planning method this project is meant to improve on.

    "Order what we sold in the same week last year" is what most retailers
    without a forecasting system actually do, and it is a genuinely strong
    benchmark in seasonal categories. Where no same-week-last-year figure
    exists, it falls back to the recent average the planner would have had --
    which is the other thing they actually do.
    """
    seasonal = df["SalesLag52"]
    recent_average = df["SalesRollMean13"]
    fallback = df[f"SalesLag{horizon}"]

    return seasonal.fillna(recent_average).fillna(fallback).fillna(0.0)


def build_model(**overrides) -> xgb.XGBRegressor:
    """One gradient-boosted model across all series.

    Fitting 3,331 separate models would mean 3,331 things to retrain, monitor
    and explain, and each would see only its own thin history. A single model
    with store and department as categorical inputs learns patterns that
    transfer -- a Christmas shape observed in one store informs the same
    department elsewhere -- and is what an operational deployment would look
    like.

    The objective is absolute rather than squared error: squared error lets the
    handful of 500k+ weeks dominate the fit, at the expense of the thousands of
    ordinary weeks that make up most ordering decisions.
    """
    params = {
        "objective": "reg:absoluteerror",
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.5,
        "enable_categorical": True,
        "tree_method": "hist",
        "early_stopping_rounds": 60,
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
    }
    params.update(overrides)
    return xgb.XGBRegressor(**params)


def fit_model(df: pd.DataFrame, split: Split,
              feature_cols: list[str] | None = None) -> tuple[xgb.XGBRegressor, list[str]]:
    feature_cols = feature_cols or features.feature_columns(df)

    model = build_model()
    model.fit(
        df.loc[split.train, feature_cols],
        df.loc[split.train, "Weekly_Sales"],
        eval_set=[(df.loc[split.validation, feature_cols],
                   df.loc[split.validation, "Weekly_Sales"])],
        verbose=False,
    )
    return model, feature_cols


def predict(model: xgb.XGBRegressor, df: pd.DataFrame,
            feature_cols: list[str]) -> np.ndarray:
    return model.predict(df[feature_cols])


def feature_importance(model: xgb.XGBRegressor, feature_cols: list[str]) -> pd.DataFrame:
    gains = model.get_booster().get_score(importance_type="gain")
    return (
        pd.DataFrame({"Feature": feature_cols})
        .assign(Gain=lambda d: d["Feature"].map(gains).fillna(0.0))
        .sort_values("Gain", ascending=False)
        .reset_index(drop=True)
    )
