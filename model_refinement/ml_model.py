"""
Configurable ML model (Gradient Boosting) for the model-refinement use case.
All inputs are part of the config: dataset path, n_rows, train/test split, and model params.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
TARGET_COL = "income"
NUMERIC_COLS = [
    "age", "fnlwgt", "educational-num",
    "capital-gain", "capital-loss", "hours-per-week",
]

# Metric name -> function (y_true, y_pred) -> float
METRIC_FNS: Dict[str, Callable[..., float]] = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
}


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target y from Adult-style DataFrame."""
    drop = [c for c in ["explanation", "confidence"] if c in df.columns]
    df = df.drop(columns=drop)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    y = (df[TARGET_COL].astype(str).str.strip() == ">50K").astype(int)
    X_df = df.drop(columns=[TARGET_COL])
    numeric = [c for c in NUMERIC_COLS if c in X_df.columns]
    cat_cols = [c for c in X_df.columns if c not in numeric]
    X_df = X_df.astype(str)
    X = pd.get_dummies(X_df, columns=cat_cols, drop_first=False, dtype=float)
    for c in numeric:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X, y


class TrainableModel:
    """
    ML model (Gradient Boosting) parameterized by a full config: dataset path,
    number of rows, train/test split, and all model hyperparameters.
    """

    def __init__(
        self,
        # Data config
        dataset_path: str = "data/adult.csv",
        n_rows: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        # Metrics to compute in test()
        metrics: Optional[List[str]] = None,
        # GradientBoostingClassifier params
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        max_features: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.n_rows = n_rows
        self.test_size = test_size
        self.random_state = random_state
        self.metrics = metrics or ["accuracy", "f1"]

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.max_features = max_features

        self._model: Optional[GradientBoostingClassifier] = None
        self._X_train: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series] = None
        self._feature_columns: Optional[List[str]] = None

    def _resolve_path(self) -> Path:
        p = Path(self.dataset_path)
        if not p.is_absolute():
            p = ROOT / p
        return p

    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        path = self._resolve_path()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        if self.n_rows is not None:
            df = df.head(self.n_rows)
        return _prepare_xy(df)

    def train(self) -> "TrainableModel":
        """Load data, train/test split, fit model. Stores test set for test()."""
        X, y = self._load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self._feature_columns = list(X.columns)
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        self._model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict for input X. Aligns columns to training feature order if needed."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if self._feature_columns is not None:
            for c in self._feature_columns:
                if c not in X.columns:
                    X = X.copy()
                    X[c] = 0
            X = X.reindex(columns=self._feature_columns, fill_value=0)
        return self._model.predict(X)

    def test(self) -> Dict[str, float]:
        """
        Run predict on the held-out test set and compute each configured metric.
        Returns dict of metric_name -> value.
        """
        if self._model is None or self._X_test is None or self._y_test is None:
            raise RuntimeError("Model not trained. Call train() first.")
        y_pred = self.predict(self._X_test)
        result = {}
        for name in self.metrics:
            fn = METRIC_FNS.get(name)
            if fn is None:
                continue
            result[name] = float(fn(self._y_test, y_pred))
        return result

    def train_metrics(self) -> Dict[str, float]:
        """Compute metrics on the training set. Call after train()."""
        if self._model is None or self._X_train is None or self._y_train is None:
            raise RuntimeError("Model not trained. Call train() first.")
        y_pred = self.predict(self._X_train)
        result = {}
        for name in self.metrics:
            fn = METRIC_FNS.get(name)
            if fn is None:
                continue
            result[name] = float(fn(self._y_train, y_pred))
        return result

    def get_train_test_metrics(self) -> Dict[str, Dict[str, float]]:
        """After train(), return { 'train': {...}, 'test': {...} }."""
        return {"train": self.train_metrics(), "test": self.test()}

    def get_config(self) -> Dict[str, Any]:
        """Return current config (dataset path, n_rows, and all model params) for logging/LLM."""
        return {
            "dataset_path": self.dataset_path,
            "n_rows": self.n_rows,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "metrics": self.metrics,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "subsample": self.subsample,
            "max_features": self.max_features,
        }

    def get_tool_result(self) -> Dict[str, Any]:
        """After train(), return test metrics plus feature importances for the analyzer tool."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        out = dict(self.test())
        names = self._feature_columns or []
        imp = self._model.feature_importances_.tolist()
        out["feature_importances"] = dict(zip(names, imp))
        return out

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrainableModel":
        """Build TrainableModel from a config dict (e.g. from CSV or LLM). Skips unknown keys."""
        kw: Dict[str, Any] = {}
        for key in [
            "dataset_path", "n_rows", "test_size", "random_state", "metrics",
            "n_estimators", "max_depth", "learning_rate", "min_samples_leaf",
            "min_samples_split", "subsample", "max_features",
        ]:
            if key not in config:
                continue
            v = config[key]
            if key == "n_rows" and v is not None:
                v = int(v) if isinstance(v, (int, float)) else None
            if key == "metrics" and isinstance(v, str):
                v = [s.strip() for s in v.strip("[]").replace("'", "").split(",")] if v else None
            kw[key] = v
        return cls(**kw)
