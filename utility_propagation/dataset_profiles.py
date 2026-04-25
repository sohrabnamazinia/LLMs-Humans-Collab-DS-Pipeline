"""Dataset profiles for utility-propagation grid (same stages, different tabular benchmarks)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PropagationDataset:
    """One tabular binary classification setting for run_grid / fit_propagation."""

    name: str
    csv_path: Path
    target_col: str
    target_encoding: Literal["adult_income", "binary_int"]
    drop_cols: tuple[str, ...]
    numeric_cols: tuple[str, ...]
    cat_error_cols: tuple[str, ...]
    feature_groups: Dict[str, Optional[List[str]]]
    # Optional: fewer collection-fraction levels for large/wide tabular runs (same factorial structure).
    n_row_fracs: Optional[Tuple[float, ...]] = None


# --- Adult (UCI-style) — original paper setting ---
_ADULT_NUMERIC = (
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
)
ADULT_PROFILE = PropagationDataset(
    name="adult",
    csv_path=_ROOT / "data" / "adult.csv",
    target_col="income",
    target_encoding="adult_income",
    drop_cols=(),
    numeric_cols=_ADULT_NUMERIC,
    cat_error_cols=("workclass", "occupation", "native-country"),
    feature_groups={
        "numeric_only": list(_ADULT_NUMERIC),
        "demographics": list(_ADULT_NUMERIC)
        + ["education", "marital-status", "race", "sex", "relationship"],
        "wide": None,
    },
)

# --- Bank marketing (tabular benchmark; n >> Adult train pool cap) ---
_BANK_NUMERIC = (
    "Age",
    "AnnualIncome",
    "NetWorth",
    "CreditScore",
    "CreditLimit",
    "AccountLengthYears",
    "TenureWithBank",
    "AccountBalance",
    "NumBankProducts",
    "InvestmentPortfolioValue",
    "TotalTransactions",
    "AvgTransactionValue",
    "NumOnlineTransactions",
    "NumMobileAppLogins",
    "BranchVisitFrequency",
    "WebsiteActivityScore",
    "LastContactDuration",
    "NumContactsInCampaign",
    "NumPrevCampaignContacts",
    "CallResponseScore",
    "DaysSinceLastContact",
    "PreviousYearDeposit",
    "MarketingScore",
    "ResponsePropensity",
)

BANK_PROFILE = PropagationDataset(
    name="bank",
    csv_path=_ROOT / "data" / "Bank_Marketing_Dataset.csv",
    target_col="TermDepositSubscribed",
    target_encoding="binary_int",
    drop_cols=("ClientID",),
    numeric_cols=_BANK_NUMERIC,
    cat_error_cols=("Gender", "MaritalStatus", "Region"),
    feature_groups={
        "numeric_only": ["Age", "CreditScore", "AnnualIncome", "AccountBalance", "CreditLimit"],
        "demographics": [
            "Age",
            "Gender",
            "MaritalStatus",
            "EducationLevel",
            "EmploymentStatus",
            "AnnualIncome",
        ],
        "wide": None,
    },
    # Slightly coarser than Adult so wide+GB over ~10k rows stays tractable for a second benchmark.
    n_row_fracs=(0.20, 0.40, 0.60, 0.80, 0.95),
)

PROFILES: dict[str, PropagationDataset] = {
    "adult": ADULT_PROFILE,
    "bank": BANK_PROFILE,
}


def get_profile(name: str) -> PropagationDataset:
    key = name.strip().lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {sorted(PROFILES)}")
    return PROFILES[key]
