# Retail demand forecasting — weekly, store × department

A twelve-week-ahead demand forecast for 3,331 store-department series, evaluated
against the planning method it is meant to replace.

**Result: 18.0% lower forecast error (WMAPE) than a seasonal-naive baseline**, on a
hold-out window the model never saw, with the baseline's systematic under-forecast
bias effectively eliminated.

For the business-facing write-up, see **[BUSINESS_SUMMARY.md](BUSINESS_SUMMARY.md)**.

---

## The problem framing

The forecast horizon is **twelve weeks**, matching a realistic order lead time. This
is not a detail — it constrains the entire feature set:

> With a twelve-week lead time, the most recent sales figure available when the order
> is placed is twelve weeks old. So the model gets nothing fresher.

There is no `SalesLag1` in this project, and there cannot be. A model handed last
week's sales to predict a quarter ahead will post excellent numbers that collapse in
production. This is the most common flaw in forecasting demonstrations, and the
reason for the leakage test described below.

## Results

Measured on 35,563 store-department weeks from 2012-08-10 onward:

| Model | WMAPE | MAPE | RMSE | MAE | Bias |
|---|---|---|---|---|---|
| Seasonal naive (baseline) | 10.79% | 20.02% | 3,627 | 1,682 | −1.69% |
| **XGBoost (global)** | **8.85%** | **19.02%** | **2,872** | **1,379** | **+0.42%** |
| *Improvement* | *18.0%* | *5.0%* | *20.8%* | *18.0%* | — |

Consistency checks: the model wins in **all 3 store formats**, **61 of 80
departments**, and **12 of 12 hold-out weeks**.

### Why WMAPE rather than MAPE

The data contains 73 zero-sales weeks and 1,285 negative-sales weeks (returns
exceeding sales). Plain MAPE divides by each week's actual value, so those rows make
it either undefined or explosive — a single week of trivial turnover can shift it by
several points.

WMAPE (total absolute error ÷ total actual volume) answers what a planner actually
asks: *across everything I ordered, what share of the volume did I get wrong?* MAPE
is still reported, restricted to weeks above a turnover floor, because clients expect
to see it.

## Project structure

```
├── data/
│   ├── raw/                     # source CSVs (not committed)
│   └── processed/               # panel.parquet, features.parquet
├── notebooks/
│   ├── 01_data_exploration.ipynb        # data quality, seasonality, design decisions
│   └── 02_forecasting_and_results.ipynb # features, split, models, evaluation
├── src/
│   ├── config.py                # paths, horizon, constants
│   ├── data.py                  # loading, cleaning, weekly-grid panel
│   ├── features.py              # calendar, lag, rolling, seasonal features
│   ├── models.py                # time split, baseline, global XGBoost
│   ├── evaluate.py              # WMAPE / MAPE / RMSE / MAE / bias, segments
│   ├── plots.py                 # case-study charts
│   └── run_pipeline.py          # end-to-end run
├── tests/
│   └── test_no_leakage.py       # the leakage proof and panel invariants
└── outputs/
    ├── forecast_vs_actual.csv   # Power BI extract
    ├── model_summary.csv
    ├── accuracy_by_{store_type,department,week}.csv
    ├── feature_importance.csv
    ├── data_quality_report.csv
    ├── run_configuration.csv
    └── figures/
```

The notebooks are narrative; the logic lives in `src/` and is imported by both. There
is one implementation of each step, not one per notebook.

## Setup

Requires Python 3.11+. Place the source CSVs (`train.csv`, `features.csv`,
`stores.csv`) in `data/raw/`.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python -m src.run_pipeline      # full run: ~1 minute
python -m pytest tests -q       # leakage and data-integrity checks
```

The pipeline prints the headline comparison and writes every artefact under
`outputs/`.

## Key design decisions

**A continuous weekly calendar underneath everything.** Of the 3,331 series, 605 have
weeks missing *inside* their span. Shifting rows positionally on ragged data makes
"last year's sales" silently point at the wrong week, and nothing errors. Every series
is therefore reindexed onto an unbroken weekly grid first; missing weeks are held open
as blanks, never filled with invented zeros, and are excluded from both training and
scoring.

**Time-based three-way split.** Train → validation → hold-out, cut by date. The
validation block exists so early stopping never observes the hold-out; tuning against
the hold-out would quietly turn the reported accuracy into an in-sample number.

**A credible baseline.** Seasonal-naive — same week last year, falling back to a
recent average where no such week exists. This is what planners without a forecasting
system actually do, and in seasonal categories it is genuinely hard to beat. Beating a
deliberately weak baseline by a large margin would be a less honest result.

**One global model, not 3,331 local ones.** Per-series models would mean thousands of
artefacts to retrain and monitor, each seeing only its own thin history. A single
model with store and department as categorical features learns transferable patterns
and reflects how this would actually be deployed.

**Absolute-error objective.** `reg:absoluteerror` rather than squared error, so the
handful of 500k+ weeks cannot dominate the fit at the expense of the thousands of
ordinary weeks that make up most ordering decisions.

**Promotion columns handled as structurally absent.** `MarkDown1-5` are empty before
2011-11-11 because tracking had not started, not because values went missing. They are
zero-filled with a separate era flag, so the model cannot read the start of tracking
as a surge in promotional activity.

**Negative sales are kept.** 0.3% of weeks show returns exceeding sales. They are real
events; the metric choice accommodates them rather than the data being edited to suit
the metric.

## The leakage test

The claim "no future information is used" is worth nothing without a check. So
`tests/test_no_leakage.py` performs one:

1. Erase **every** hold-out sales figure from the panel.
2. Rebuild the entire feature set from that censored data.
3. Assert the hold-out feature values are identical to the originals.

If any feature drew on a value the planner could not have known, erasing it would
change the result and the test fails. The suite also asserts the weekly grid is
unbroken, the split blocks do not overlap in time, and no sales feature is fresher
than the horizon.

```bash
python -m pytest tests -q
```

## Power BI output

`outputs/forecast_vs_actual.csv` is a flat fact table — one row per store, department
and week:

| Column | Meaning |
|---|---|
| `Store`, `Dept`, `StoreType`, `StoreSize` | grain and store attributes |
| `Date`, `IsHolidayWeek` | week and calendar flag |
| `Actual` | recorded sales |
| `Baseline_Forecast`, `XGBoost_Forecast` | the two forecasts |
| `Baseline_Error`, `XGBoost_Error` | signed error (forecast − actual) |
| `Baseline_AbsError`, `XGBoost_AbsError` | absolute error |

Signed and absolute errors are pre-computed so WMAPE aggregates correctly in the BI
layer: a weighted error ratio cannot be averaged from row-level percentages, it has to
be summed as `SUM(AbsError) / SUM(Actual)` over whatever slice the user selects.

## Limitations

- **~2.7 years of history** — barely two annual cycles. A third year would likely
  help more than any modelling change.
- **The hold-out contains no major holiday.** It covers late summer 2012. Accuracy
  through the November–December peak is not measured here.
- **Department-level, not article-level.** Real ordering happens at article level,
  where series are thinner and noisier. The method carries over; the figures would not.
- **Demand only.** Turning forecasts into order proposals needs lead times, minimum
  order quantities and shelf capacity.

## Data

Public retail sales dataset: 45 stores, 81 departments, weekly observations from
2010-02-05 to 2012-10-26. Raw files are not committed — place them in `data/raw/`.
