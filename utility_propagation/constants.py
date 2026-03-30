from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADULT_PATH = ROOT / "data" / "adult.csv"

TARGET_COL = "income"
NUMERIC_COLS = [
    "age", "fnlwgt", "educational-num",
    "capital-gain", "capital-loss", "hours-per-week",
]
CAT_ERROR_COLS = ["workclass", "occupation", "native-country"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
