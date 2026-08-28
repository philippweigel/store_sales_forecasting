"""Loading, validation and cleaning of the raw retail sales extract.

The end product is a tidy weekly panel: exactly one row per
(Store, Dept, Date) on a continuous weekly calendar, which is what every
downstream lag/rolling feature relies on being true.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw() -> dict[str, pd.DataFrame]:
    """Read the four source files as delivered.

    stores.csv arrives with carriage-return-only line endings, so it needs the
    python engine to parse into 45 rows rather than a single row.
    """
    train = pd.read_csv(config.RAW_DIR / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(config.RAW_DIR / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(config.RAW_DIR / "stores.csv", engine="python")
    return {"train": train, "features": features, "stores": stores}


def profile_raw(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise the raw inputs so data issues are stated, not discovered later."""
    train = raw["train"]
    series = train.groupby(["Store", "Dept"])
    lengths = series.size()
    n_weeks = train["Date"].nunique()

    span = series["Date"].agg(["min", "max", "size"])
    expected = ((span["max"] - span["min"]).dt.days // 7) + 1

    checks = {
        "rows": len(train),
        "stores": train["Store"].nunique(),
        "departments": train["Dept"].nunique(),
        "store_dept_series": len(lengths),
        "weeks_covered": n_weeks,
        "first_week": train["Date"].min().date(),
        "last_week": train["Date"].max().date(),
        "duplicate_keys": int(train.duplicated(["Store", "Dept", "Date"]).sum()),
        "series_with_full_history": int((lengths == n_weeks).sum()),
        "series_under_52_weeks": int((lengths < 52).sum()),
        "series_with_internal_gaps": int((span["size"] != expected).sum()),
        "missing_weeks_inside_spans": int((expected - span["size"]).sum()),
        "negative_sales_rows": int((train["Weekly_Sales"] < 0).sum()),
        "zero_sales_rows": int((train["Weekly_Sales"] == 0).sum()),
        "min_weekly_sales": round(float(train["Weekly_Sales"].min()), 2),
        "max_weekly_sales": round(float(train["Weekly_Sales"].max()), 2),
        "median_weekly_sales": round(float(train["Weekly_Sales"].median()), 2),
    }
    return pd.DataFrame({"check": checks.keys(), "value": checks.values()})


def _clean_features(features: pd.DataFrame) -> pd.DataFrame:
    """Resolve the promotion columns' structural missingness.

    MarkDown1-5 are not "missing" before 2011-11-11 -- promotions simply were
    not tracked yet. Filling those with 0 and flagging the tracked era keeps the
    model from reading the start of tracking as a change in promotion intensity.
    """
    features = features.copy()
    tracking_start = pd.Timestamp(config.MARKDOWN_TRACKING_START)

    features["MarkdownsTracked"] = (features["Date"] >= tracking_start).astype(int)
    features[config.MARKDOWN_COLS] = features[config.MARKDOWN_COLS].fillna(0.0)
    features["MarkdownTotal"] = features[config.MARKDOWN_COLS].sum(axis=1)

    # IsHoliday is carried by the sales table; keeping both copies would only
    # create a redundant merge conflict.
    return features.drop(columns=["IsHoliday"])


def build_panel(raw: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Merge the sources and place every series on a continuous weekly grid.

    Roughly 600 of the 3,331 series have weeks missing in the middle of their
    history. Shifting rows positionally on that ragged data would silently make
    "last week" mean "some earlier week", so each series is reindexed across its
    own first-to-last week. Weeks with no recorded sales stay as missing targets
    rather than being invented as zeros; they are excluded from training and
    scoring but still occupy their calendar slot so lags line up correctly.
    """
    raw = raw or load_raw()
    train, features, stores = raw["train"], raw["features"], raw["stores"]

    panel = train.merge(_clean_features(features), on=["Store", "Date"], how="left")
    panel = panel.merge(stores, on="Store", how="left")

    panel = _reindex_to_weekly_grid(panel)
    panel = panel.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    panel["HasSales"] = panel["Weekly_Sales"].notna()
    return panel


def _reindex_to_weekly_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Insert placeholder rows for weeks missing inside a series' own span."""
    spans = panel.groupby(["Store", "Dept"])["Date"].agg(["min", "max"])

    grid = pd.concat(
        [
            pd.DataFrame(
                {
                    "Store": store,
                    "Dept": dept,
                    "Date": pd.date_range(row["min"], row["max"], freq="7D"),
                }
            )
            for (store, dept), row in spans.iterrows()
        ],
        ignore_index=True,
    )

    filled = grid.merge(panel, on=["Store", "Dept", "Date"], how="left")

    # Store-level and calendar attributes are known for every week regardless of
    # whether sales were recorded, so they are refilled from their own sources.
    store_cols = ["Type", "Size"]
    filled = filled.drop(columns=store_cols).merge(
        panel[["Store", *store_cols]].drop_duplicates(), on="Store", how="left"
    )

    week_cols = [
        "Temperature",
        "Fuel_Price",
        *config.MARKDOWN_COLS,
        "MarkdownTotal",
        "MarkdownsTracked",
        "CPI",
        "Unemployment",
    ]
    week_level = panel[["Store", "Date", *week_cols]].drop_duplicates(["Store", "Date"])
    filled = filled.drop(columns=week_cols).merge(
        week_level, on=["Store", "Date"], how="left"
    )

    holidays = panel[["Date", "IsHoliday"]].drop_duplicates("Date")
    filled = filled.drop(columns=["IsHoliday"]).merge(holidays, on="Date", how="left")
    filled["IsHoliday"] = filled["IsHoliday"].fillna(False).astype(bool)

    return filled


def save_panel(panel: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PROCESSED_DIR / "panel.parquet", index=False)
