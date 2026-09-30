"""Unit tests for data cleaning and domain feature engineering."""

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.feature_engineering import CreditFeatureEngineer


def test_feature_engineer_cleans_anomalies_and_creates_features():
    """Verify age=0, missing income/dependents, and bureau codes 96/98 are cleaned and ratios engineered."""
    raw_df = pd.DataFrame(
        [
            {
                "ApplicantID": 1,
                "SeriousDlqin2yrs": 0,
                "RevolvingUtilizationOfUnsecuredLines": 0.25,
                "age": 0,  # anomaly -> should be replaced with >=18 median
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.30,
                "MonthlyIncome": 6000.0,
                "NumberOfOpenCreditLinesAndLoans": 6,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2.0,
            },
            {
                "ApplicantID": 2,
                "SeriousDlqin2yrs": 1,
                "RevolvingUtilizationOfUnsecuredLines": 150.0,  # extreme outlier
                "age": 38,
                "NumberOfTime30-59DaysPastDueNotWorse": 98,  # special bureau code
                "DebtRatio": 1800.0,  # unverified income artifact
                "MonthlyIncome": np.nan,
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 2,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 1,
                "NumberOfDependents": np.nan,
            },
        ]
    )

    fe = CreditFeatureEngineer()
    transformed = fe.fit_transform(raw_df)

    # Check all required engineered columns exist
    for col in DEFAULT_CONFIG.engineered_feature_columns:
        assert col in transformed.columns, f"Missing engineered column: {col}"

    # Row 0: age anomaly fixed
    assert transformed.loc[0, "age"] >= 18
    assert transformed.loc[0, "any_delinquency_flag"] == 0
    assert transformed.loc[0, "severe_delinquency_flag"] == 0

    # Row 1: missing flags set, special code 98 cleaned, utilization & DTI capped
    assert transformed.loc[1, "income_missing_flag"] == 1
    assert transformed.loc[1, "dependents_missing_flag"] == 1
    assert transformed.loc[1, "MonthlyIncome"] > 0
    assert transformed.loc[1, "credit_utilization"] <= 2.0
    assert transformed.loc[1, "debt_to_income"] <= 2.5
    assert transformed.loc[1, "payment_to_income"] > 0
    assert transformed.loc[1, "NumberOfTime30-59DaysPastDueNotWorse"] == 0
    assert transformed.loc[1, "total_delinquencies"] == 3
    assert transformed.loc[1, "any_delinquency_flag"] == 1
    assert transformed.loc[1, "severe_delinquency_flag"] == 1
