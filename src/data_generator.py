"""Dataset loader and generator for the Kaggle 'Give Me Some Credit' schema."""

import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, RAW_DATA_PATH

logger = logging.getLogger(__name__)


def generate_give_me_some_credit_dataset(
    n_samples: int = 15000,
    random_state: int = 42,
    save_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Load 'cs-training.csv' if already present on disk, or synthesize a realistic
    'Give Me Some Credit' dataset matching the exact Kaggle schema, statistical
    distributions, missingness patterns (~19% MonthlyIncome, ~2.5% Dependents),
    and real-world outliers (e.g., age=0, utilization spikes, delinquency codes 96/98).
    """
    if save_path.exists():
        logger.info(f"Loading existing dataset from {save_path}")
        df = pd.read_csv(save_path)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "ApplicantID"})
        elif "ApplicantID" not in df.columns:
            df.insert(0, "ApplicantID", np.arange(1, len(df) + 1))
        return df

    logger.info(
        f"Synthesizing realistic 'Give Me Some Credit' dataset ({n_samples:,} rows) -> {save_path}"
    )
    rng = np.random.default_rng(random_state)

    applicant_id = np.arange(1, n_samples + 1)

    # 1. Age distribution (mean ~52, std ~14.7, clipped 21-95, plus 2 anomalies at 0)
    age = rng.normal(loc=51.5, scale=14.5, size=n_samples).round().astype(int)
    age = np.clip(age, 21, 95)

    # 2. Dependents (Poisson-like, higher for ages 30-55)
    dep_lambda = np.where((age >= 28) & (age <= 55), 1.15, 0.45)
    dependents = rng.poisson(lam=dep_lambda).astype(float)
    dependents = np.clip(dependents, 0, 8)

    # 3. Monthly Income (Log-normal correlated with age)
    age_income_factor = np.where(age < 30, 0.78, np.where(age < 60, 1.12, 0.92))
    monthly_income = (
        rng.lognormal(mean=8.55, sigma=0.62, size=n_samples) * age_income_factor
    ).round(0)
    monthly_income = np.clip(monthly_income, 800, 85000)

    # 4. Open Credit Lines and Real Estate Loans
    open_credit_lines = rng.negative_binomial(n=4, p=0.32, size=n_samples) + 1
    open_credit_lines = np.clip(open_credit_lines, 0, 36)

    re_prob = np.where((age >= 30) & (monthly_income >= 4200), 0.65, 0.25)
    real_estate_loans = rng.binomial(n=3, p=re_prob, size=n_samples)
    real_estate_loans = np.minimum(real_estate_loans, open_credit_lines)

    # 5. Revolving Utilization of Unsecured Lines (Beta distribution + over-limit tail)
    base_util = rng.beta(a=1.25, b=2.6, size=n_samples)
    over_limit_mask = rng.random(n_samples) < 0.045
    base_util[over_limit_mask] = rng.uniform(1.01, 1.65, size=over_limit_mask.sum())
    # Younger borrowers tend to have slightly higher utilization
    util = np.clip(base_util * np.where(age < 32, 1.18, 0.96), 0.0, 2.2)

    # 6. Debt Ratio (Monthly debt obligations / income)
    base_dti = (
        0.14
        + 0.16 * np.minimum(util, 1.2)
        + 0.07 * real_estate_loans
        + 0.012 * open_credit_lines
        + rng.gamma(shape=1.8, scale=0.09, size=n_samples)
    )
    debt_ratio = np.clip(base_dti, 0.01, 2.5)

    # 7. Past Due Delinquencies (correlated latent credit stress factor)
    stress_factor = (
        1.35 * (util > 0.75).astype(float)
        + 0.85 * (debt_ratio > 0.55).astype(float)
        + 0.65 * (monthly_income < 3200).astype(float)
        + 0.45 * (age < 30).astype(float)
        + rng.normal(0, 0.55, size=n_samples)
    )
    delinq_rate_30_59 = np.clip(0.08 * np.exp(0.75 * stress_factor), 0.01, 3.5)
    delinq_rate_60_89 = np.clip(0.03 * np.exp(0.82 * stress_factor), 0.005, 2.2)
    delinq_rate_90_plus = np.clip(0.035 * np.exp(0.92 * stress_factor), 0.005, 2.8)

    days_30_59 = rng.poisson(lam=delinq_rate_30_59)
    days_60_89 = rng.poisson(lam=delinq_rate_60_89)
    days_90_plus = rng.poisson(lam=delinq_rate_90_plus)

    # 8. True Default Probability (Log-odds model calibrated to ~6.7% - 11.5% default rate)
    logit = (
        -3.65
        + 1.65 * np.minimum(util, 1.5)
        + 0.85 * np.minimum(debt_ratio, 1.8)
        + 0.58 * np.minimum(days_30_59, 6)
        + 0.82 * np.minimum(days_60_89, 5)
        + 1.28 * np.minimum(days_90_plus, 6)
        - 0.018 * (age - 45)
        - 0.35 * np.log1p(monthly_income / 5000.0)
        + 0.14 * dependents
        - 0.12 * (real_estate_loans == 1).astype(float)
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    serious_dlq = rng.binomial(n=1, p=prob_default, size=n_samples)

    # 9. Inject authentic Kaggle 'Give Me Some Credit' noise/anomalies for cleaning pipeline
    # - Missing MonthlyIncome (~18%) and high DebtRatio artifact when income is unverified
    missing_inc_mask = rng.random(n_samples) < 0.18
    monthly_income_with_nan = monthly_income.copy().astype(float)
    monthly_income_with_nan[missing_inc_mask] = np.nan
    debt_ratio_with_outliers = debt_ratio.copy()
    extreme_dti_mask = missing_inc_mask & (rng.random(n_samples) < 0.35)
    debt_ratio_with_outliers[extreme_dti_mask] = rng.uniform(
        100.0, 3500.0, size=extreme_dti_mask.sum()
    ).round(0)

    # - Missing NumberOfDependents (~2.5%)
    missing_dep_mask = rng.random(n_samples) < 0.025
    dependents_with_nan = dependents.copy()
    dependents_with_nan[missing_dep_mask] = np.nan

    # - Extreme utilization outliers (e.g., 50 - 2500) on 0.2% of rows
    extreme_util_mask = rng.random(n_samples) < 0.002
    util_with_outliers = util.copy()
    util_with_outliers[extreme_util_mask] = rng.uniform(
        25.0, 1200.0, size=extreme_util_mask.sum()
    )

    # - Special delinquency codes 96 and 98 on 0.15% of rows
    special_code_mask = rng.random(n_samples) < 0.0015
    days_30_59[special_code_mask] = 98
    days_60_89[special_code_mask] = 98
    days_90_plus[special_code_mask] = 98

    # - Age = 0 anomaly on 2 rows
    age[0] = 0
    age[1] = 0

    df = pd.DataFrame(
        {
            "ApplicantID": applicant_id,
            "SeriousDlqin2yrs": serious_dlq,
            "RevolvingUtilizationOfUnsecuredLines": np.round(util_with_outliers, 6),
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": days_30_59,
            "DebtRatio": np.round(debt_ratio_with_outliers, 6),
            "MonthlyIncome": monthly_income_with_nan,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": days_90_plus,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": days_60_89,
            "NumberOfDependents": dependents_with_nan,
        }
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(
        f"Saved dataset to {save_path} with shape {df.shape} and default rate {df['SeriousDlqin2yrs'].mean():.2%}"
    )
    return df


def generate_auxiliary_tables(
    df_applicants: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate auxiliary tables (`loan_applications` and `regional_macro`) to support
    rich multi-table SQL JOINs, window functions, and regional cohort analyses.
    """
    rng = np.random.default_rng(random_state)
    n = len(df_applicants)

    regions = ["Northeast", "Midwest", "South", "West", "Pacific"]
    loan_purposes = [
        "Debt Consolidation",
        "Home Improvement",
        "Small Business",
        "Auto Refinance",
        "Medical Expenses",
        "Major Purchase",
    ]
    employment_statuses = ["Full-Time", "Self-Employed", "Part-Time", "Retired"]

    applicant_regions = rng.choice(regions, size=n, p=[0.20, 0.22, 0.30, 0.18, 0.10])
    purposes = rng.choice(
        loan_purposes, size=n, p=[0.42, 0.18, 0.10, 0.12, 0.08, 0.10]
    )

    ages = df_applicants["age"].values
    emp_status = np.where(
        ages >= 65,
        "Retired",
        rng.choice(employment_statuses[:3], size=n, p=[0.74, 0.16, 0.10]),
    )

    inc = df_applicants["MonthlyIncome"].fillna(5400.0).values
    requested_loan_amount = np.clip(
        (inc * rng.uniform(1.5, 5.5, size=n)).round(-2), 2000, 75000
    )
    loan_term_months = rng.choice([24, 36, 48, 60], size=n, p=[0.15, 0.45, 0.15, 0.25])
    interest_rate_apr = np.round(
        np.clip(
            7.5
            + 8.5 * np.minimum(df_applicants["RevolvingUtilizationOfUnsecuredLines"].values, 1.2)
            + 3.2 * (df_applicants["NumberOfTimes90DaysLate"].values > 0)
            + rng.normal(0, 1.5, size=n),
            5.49,
            28.99,
        ),
        2,
    )

    df_loans = pd.DataFrame(
        {
            "ApplicationID": np.arange(100001, 100001 + n),
            "ApplicantID": df_applicants["ApplicantID"].values,
            "Region": applicant_regions,
            "LoanPurpose": purposes,
            "EmploymentStatus": emp_status,
            "RequestedLoanAmount": requested_loan_amount,
            "LoanTermMonths": loan_term_months,
            "EstimatedAPR": interest_rate_apr,
        }
    )

    df_regions = pd.DataFrame(
        {
            "Region": regions,
            "RegionalUnemploymentRate": [4.1, 3.8, 4.4, 4.6, 4.9],
            "MedianHouseholdIncome": [78500, 68200, 64800, 76400, 84200],
            "CostOfLivingIndex": [114.5, 94.2, 92.8, 108.6, 126.4],
        }
    )

    return df_loans, df_regions
