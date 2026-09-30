"""Central configuration for the AI Credit Risk Prediction pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "cs-training.csv"
PROCESSED_DATA_PATH = DATA_DIR / "credit_risk_processed.csv"
SQLITE_DB_PATH = DATA_DIR / "credit_risk.db"

SQL_DIR = PROJECT_ROOT / "sql"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_credit_risk_model.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_comparison_metrics.json"

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
FIGURES_DIR = DASHBOARD_DIR / "figures"

# Ensure directories exist
for directory in [DATA_DIR, SQL_DIR, MODELS_DIR, DASHBOARD_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class RiskThresholds:
    """Configurable probability thresholds for Low, Medium, and High risk tiers."""

    low_max: float = 0.20     # PD < 0.20 -> Low Risk
    medium_max: float = 0.50  # 0.20 <= PD < 0.50 -> Medium Risk, PD >= 0.50 -> High Risk

    def classify(self, probability: float) -> str:
        """Convert predicted default probability into Low, Medium, or High risk tier."""
        if probability < self.low_max:
            return "Low"
        elif probability < self.medium_max:
            return "Medium"
        return "High"

    def to_credit_score(self, probability: float) -> int:
        """Map default probability [0, 1] to a FICO-like credit score [300, 850]."""
        p = max(min(float(probability), 0.999), 0.001)
        score = int(round(850 - 550 * (p ** 0.65)))
        return max(300, min(850, score))


@dataclass
class PipelineConfig:
    """Configuration parameters for data processing, feature engineering, and modeling."""

    random_state: int = 42
    test_size: float = 0.20
    imbalance_strategy: str = "smote"  # Options: "smote", "class_weight", "none"
    optuna_trials: int = 15
    cv_folds: int = 3
    sample_size: int = 15000
    thresholds: RiskThresholds = field(default_factory=RiskThresholds)

    raw_feature_columns: List[str] = field(
        default_factory=lambda: [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ]
    )

    engineered_feature_columns: List[str] = field(
        default_factory=lambda: [
            "credit_utilization",
            "age",
            "age_bucket_code",
            "debt_to_income",
            "MonthlyIncome",
            "payment_to_income",
            "income_per_dependent",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberRealEstateLoansOrLines",
            "real_estate_loan_ratio",
            "NumberOfDependents",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "total_delinquencies",
            "weighted_delinquency_score",
            "any_delinquency_flag",
            "severe_delinquency_flag",
            "high_utilization_flag",
            "high_dti_flag",
            "income_missing_flag",
            "dependents_missing_flag",
        ]
    )

    feature_display_names: Dict[str, str] = field(
        default_factory=lambda: {
            "credit_utilization": "Revolving Credit Utilization",
            "age": "Applicant Age (Years)",
            "age_bucket_code": "Age Segment Tier",
            "debt_to_income": "Debt-to-Income Ratio (DTI)",
            "MonthlyIncome": "Monthly Gross Income ($)",
            "payment_to_income": "Payment-to-Income Ratio (PTI)",
            "income_per_dependent": "Income per Household Member ($)",
            "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines & Loans",
            "NumberRealEstateLoansOrLines": "Real Estate / Mortgage Lines",
            "real_estate_loan_ratio": "Real Estate Share of Credit Lines",
            "NumberOfDependents": "Number of Dependents",
            "NumberOfTime30-59DaysPastDueNotWorse": "30-59 Days Past Due Count",
            "NumberOfTime60-89DaysPastDueNotWorse": "60-89 Days Past Due Count",
            "NumberOfTimes90DaysLate": "90+ Days Severe Delinquency Count",
            "total_delinquencies": "Total Historical Delinquencies",
            "weighted_delinquency_score": "Severity-Weighted Delinquency Index",
            "any_delinquency_flag": "Prior Delinquency Flag",
            "severe_delinquency_flag": "Severe 90+ Day Delinquency Flag",
            "high_utilization_flag": "High Credit Utilization (>80%) Flag",
            "high_dti_flag": "High Debt-to-Income (>43%) Flag",
            "income_missing_flag": "Unverified / Imputed Income Flag",
            "dependents_missing_flag": "Imputed Dependents Flag",
        }
    )


DEFAULT_CONFIG = PipelineConfig()
