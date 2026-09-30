"""Data cleaning and domain feature engineering for Credit Risk Prediction."""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG

AGE_BUCKET_LABELS = ["<25", "25-34", "35-49", "50-64", "65+"]
AGE_BUCKET_MAP = {label: idx for idx, label in enumerate(AGE_BUCKET_LABELS)}


def assign_age_bucket(age_series: pd.Series) -> pd.Series:
    """Categorize applicant age into standard credit risk age cohorts."""
    return pd.cut(
        age_series,
        bins=[0, 24, 34, 49, 64, 120],
        labels=AGE_BUCKET_LABELS,
        include_lowest=True,
    ).astype(str)


class CreditFeatureEngineer:
    """
    Stateful data cleaner and feature transformer that learns imputation medians
    from the training split and applies deterministic cleaning + feature engineering
    at training and real-time inference time.
    """

    def __init__(self) -> None:
        self.median_age_: float = 52.0
        self.median_income_: float = 5400.0
        self.income_by_age_bucket_: Dict[str, float] = {
            "<25": 3100.0,
            "25-34": 4600.0,
            "35-49": 5900.0,
            "50-64": 6100.0,
            "65+": 4800.0,
        }
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "CreditFeatureEngineer":
        """Learn imputation statistics from the training dataset only (no data leakage)."""
        valid_ages = df.loc[df["age"] >= 18, "age"]
        if len(valid_ages) > 0:
            self.median_age_ = float(valid_ages.median())

        valid_inc = df.loc[df["MonthlyIncome"].notna() & (df["MonthlyIncome"] > 0), "MonthlyIncome"]
        if len(valid_inc) > 0:
            self.median_income_ = float(valid_inc.median())

        temp_age = df["age"].where(df["age"] >= 18, self.median_age_)
        temp_buckets = assign_age_bucket(temp_age)
        grouped = (
            df.assign(_bucket=temp_buckets)
            .groupby("_bucket")["MonthlyIncome"]
            .median()
            .dropna()
            .to_dict()
        )
        for k, v in grouped.items():
            if np.isfinite(v) and v > 0:
                self.income_by_age_bucket_[str(k)] = float(v)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean anomalies, impute missing values, and engineer credit risk features."""
        out = df.copy()

        # 1. Clean Age anomalies (e.g., age == 0 in Give Me Some Credit)
        out["age"] = pd.to_numeric(out["age"], errors="coerce").fillna(self.median_age_)
        out["age"] = np.where(out["age"] < 18, self.median_age_, np.clip(out["age"], 18, 99)).astype(int)

        # 2. Age Buckets
        out["age_bucket"] = assign_age_bucket(out["age"])
        out["age_bucket_code"] = out["age_bucket"].map(AGE_BUCKET_MAP).fillna(2).astype(int)

        # 3. Missingness indicators & Imputation for MonthlyIncome and NumberOfDependents
        out["income_missing_flag"] = out["MonthlyIncome"].isna().astype(int)
        out["dependents_missing_flag"] = out["NumberOfDependents"].isna().astype(int)

        fallback_incomes = out["age_bucket"].map(self.income_by_age_bucket_).fillna(self.median_income_)
        out["MonthlyIncome"] = pd.to_numeric(out["MonthlyIncome"], errors="coerce")
        out["MonthlyIncome"] = np.where(
            out["MonthlyIncome"].isna() | (out["MonthlyIncome"] <= 0),
            fallback_incomes,
            out["MonthlyIncome"],
        )
        out["MonthlyIncome"] = np.clip(out["MonthlyIncome"], 500.0, 250000.0)

        out["NumberOfDependents"] = (
            pd.to_numeric(out["NumberOfDependents"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 15.0)
        )

        # 4. Clean Special Delinquency Codes (96, 98 in Give Me Some Credit)
        delinq_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
        for col in delinq_cols:
            vals = pd.to_numeric(out[col], errors="coerce").fillna(0)
            # Replace special bureau codes 96/98 with 0 and cap at 20
            vals = np.where(vals >= 90, 0, vals)
            out[col] = np.clip(vals, 0, 20).astype(int)

        # 5. Clean Revolving Credit Utilization
        raw_util = pd.to_numeric(
            out["RevolvingUtilizationOfUnsecuredLines"], errors="coerce"
        ).fillna(0.0)
        # In Give Me Some Credit, values > 10 are typically raw balances or errors
        out["credit_utilization"] = np.where(
            raw_util > 3.0, 1.15, np.clip(raw_util, 0.0, 2.0)
        )

        # 6. Clean DebtRatio & Engineer Debt-to-Income (DTI)
        raw_dti = pd.to_numeric(out["DebtRatio"], errors="coerce").fillna(0.25)
        # When income was missing, DebtRatio often stores absolute monthly debt payment
        normalized_dti = np.where(
            raw_dti > 5.0,
            raw_dti / np.maximum(out["MonthlyIncome"], 1000.0),
            raw_dti,
        )
        out["debt_to_income"] = np.clip(normalized_dti, 0.0, 2.5)

        # 7. Engineer Payment-to-Income (PTI)
        # Estimate total monthly minimum obligation across revolving + open credit lines relative to income
        open_lines = pd.to_numeric(
            out["NumberOfOpenCreditLinesAndLoans"], errors="coerce"
        ).fillna(4).clip(0, 50)
        re_lines = pd.to_numeric(
            out["NumberRealEstateLoansOrLines"], errors="coerce"
        ).fillna(0).clip(0, 20)
        out["NumberOfOpenCreditLinesAndLoans"] = open_lines.astype(int)
        out["NumberRealEstateLoansOrLines"] = re_lines.astype(int)

        estimated_monthly_payment = (
            out["debt_to_income"] * out["MonthlyIncome"] * 0.65
            + out["credit_utilization"] * 450.0
            + re_lines * 320.0
            + np.maximum(open_lines - re_lines, 0) * 45.0
        )
        out["payment_to_income"] = np.clip(
            estimated_monthly_payment / np.maximum(out["MonthlyIncome"], 500.0),
            0.0,
            2.5,
        )

        # 8. Household & Credit Line Structure Ratios
        out["income_per_dependent"] = out["MonthlyIncome"] / (1.0 + out["NumberOfDependents"])
        out["real_estate_loan_ratio"] = out["NumberRealEstateLoansOrLines"] / (
            1.0 + out["NumberOfOpenCreditLinesAndLoans"]
        )

        # 9. Delinquency Flags & Severity Score
        out["total_delinquencies"] = (
            out["NumberOfTime30-59DaysPastDueNotWorse"]
            + out["NumberOfTime60-89DaysPastDueNotWorse"]
            + out["NumberOfTimes90DaysLate"]
        )
        out["weighted_delinquency_score"] = (
            1.0 * out["NumberOfTime30-59DaysPastDueNotWorse"]
            + 2.0 * out["NumberOfTime60-89DaysPastDueNotWorse"]
            + 3.5 * out["NumberOfTimes90DaysLate"]
        )
        out["any_delinquency_flag"] = (out["total_delinquencies"] > 0).astype(int)
        out["severe_delinquency_flag"] = (out["NumberOfTimes90DaysLate"] > 0).astype(int)

        # 10. High Utilization & High DTI Flags
        out["high_utilization_flag"] = (out["credit_utilization"] > 0.80).astype(int)
        out["high_dti_flag"] = (out["debt_to_income"] > 0.43).astype(int)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit imputer statistics and transform DataFrame."""
        return self.fit(df).transform(df)
