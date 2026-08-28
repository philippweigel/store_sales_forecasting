"""Central paths and modelling constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Length of the hold-out window, in weeks, cut from the end of the history.
# 12 weeks ~ one planning quarter, which is the horizon an ordering process
# actually needs to cover.
HOLDOUT_WEEKS = 12

# Weeks reserved before the hold-out for early-stopping/validation, so that
# hyperparameter choices never see the hold-out.
VALIDATION_WEEKS = 12

# Markdown/promotion tracking only begins on this date in the source data.
MARKDOWN_TRACKING_START = "2011-11-11"

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]

RANDOM_SEED = 42
