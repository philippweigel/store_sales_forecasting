"""End-to-end run: raw files in, evaluated forecasts and Power BI extracts out.

    python -m src.run_pipeline
"""

from __future__ import annotations

import pandas as pd

from . import config, data, evaluate, features, models, plots

FORECAST_COLUMNS = {"Seasonal naive": "Baseline_Forecast", "XGBoost": "XGBoost_Forecast"}


def run() -> dict[str, pd.DataFrame]:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/6  Loading and profiling raw data")
    raw = data.load_raw()
    profile = data.profile_raw(raw)
    profile.to_csv(config.OUTPUTS_DIR / "data_quality_report.csv", index=False)

    print("2/6  Building the weekly panel")
    panel = data.build_panel(raw)
    data.save_panel(panel)

    print("3/6  Engineering time-series features")
    frame = features.build_feature_frame(panel, config.HOLDOUT_WEEKS)
    frame.to_parquet(config.PROCESSED_DIR / "features.parquet", index=False)

    print("4/6  Splitting by date and forecasting with the baseline")
    split = models.make_time_split(frame)
    frame["Baseline_Forecast"] = models.seasonal_naive_forecast(frame, config.HOLDOUT_WEEKS)

    print("5/6  Training the global XGBoost model")
    model, feature_cols = models.fit_model(frame, split)
    frame["XGBoost_Forecast"] = models.predict(model, frame, feature_cols)

    print("6/6  Scoring and exporting")
    results = _assemble_results(frame, split)
    summary = evaluate.score_models(results, "Actual", FORECAST_COLUMNS)
    importance = models.feature_importance(model, feature_cols)

    by_type = evaluate.score_by_segment(results, "StoreType", "Actual", FORECAST_COLUMNS)
    by_dept = evaluate.score_by_segment(results, "Dept", "Actual", FORECAST_COLUMNS)
    by_week = evaluate.score_by_segment(results, "Date", "Actual", FORECAST_COLUMNS)

    _write_outputs(results, summary, by_type, by_dept, by_week, importance, split)

    artefacts = {
        "results": results,
        "summary": summary,
        "by_type": by_type,
        "by_dept": by_dept,
        "by_week": by_week,
        "importance": importance,
    }
    plots.build_all(panel, artefacts)
    _print_headline(summary, split)

    return artefacts


def _assemble_results(frame: pd.DataFrame, split: models.Split) -> pd.DataFrame:
    """Shape the hold-out predictions into the flat table Power BI expects."""
    holdout = frame.loc[split.holdout].copy()

    results = pd.DataFrame(
        {
            "Store": holdout["Store"],
            "Dept": holdout["Dept"],
            "StoreType": holdout["Type"].astype(str),
            "StoreSize": holdout["Size"],
            "Date": holdout["Date"],
            "IsHolidayWeek": holdout["IsHoliday"].astype(bool),
            "Actual": holdout["Weekly_Sales"],
            "Baseline_Forecast": holdout["Baseline_Forecast"],
            "XGBoost_Forecast": holdout["XGBoost_Forecast"],
        }
    )

    for label, column in FORECAST_COLUMNS.items():
        prefix = "Baseline" if column.startswith("Baseline") else "XGBoost"
        results[f"{prefix}_Error"] = results[column] - results["Actual"]
        results[f"{prefix}_AbsError"] = results[f"{prefix}_Error"].abs()

    return results.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def _write_outputs(results, summary, by_type, by_dept, by_week, importance,
                   split: models.Split) -> None:
    out = config.OUTPUTS_DIR
    results.to_csv(out / "forecast_vs_actual.csv", index=False)
    summary.to_csv(out / "model_summary.csv", index=False)
    by_type.to_csv(out / "accuracy_by_store_type.csv", index=False)
    by_dept.to_csv(out / "accuracy_by_department.csv", index=False)
    by_week.to_csv(out / "accuracy_by_week.csv", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)

    pd.DataFrame(
        [
            {"Setting": "Forecast horizon (weeks)", "Value": config.HOLDOUT_WEEKS},
            {"Setting": "Hold-out starts", "Value": split.holdout_start.date()},
            {"Setting": "Validation starts", "Value": split.validation_start.date()},
            {"Setting": "Training rows", "Value": int(split.train.sum())},
            {"Setting": "Hold-out rows", "Value": int(split.holdout.sum())},
        ]
    ).to_csv(out / "run_configuration.csv", index=False)


def _print_headline(summary: pd.DataFrame, split: models.Split) -> None:
    indexed = summary.set_index("Model")
    gain = evaluate.improvement(
        indexed.loc["Seasonal naive", "WMAPE"], indexed.loc["XGBoost", "WMAPE"]
    )

    print(f"\nHold-out: {split.holdout_start.date()} onwards, "
          f"{int(split.holdout.sum()):,} store-department weeks")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nWMAPE improvement over the seasonal-naive baseline: {gain:.1%}")


if __name__ == "__main__":
    run()
