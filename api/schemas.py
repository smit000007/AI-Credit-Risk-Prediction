"""Pydantic request and response schemas for the Credit Risk FastAPI service."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ApplicantRequest(BaseModel):
    """Input payload for a loan applicant credit risk evaluation."""

    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        default=0.32,
        ge=0.0,
        description="Total balance on credit cards and personal lines of credit divided by sum of credit limits.",
    )
    age: int = Field(
        default=42,
        ge=18,
        le=105,
        description="Applicant age in years.",
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        default=0,
        ge=0,
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Number of times borrower has been 30-59 days past due but no worse in the last 2 years.",
    )
    DebtRatio: float = Field(
        default=0.28,
        ge=0.0,
        description="Monthly debt payments, alimony, living costs divided by monthly gross income.",
    )
    MonthlyIncome: Optional[float] = Field(
        default=6500.0,
        ge=0.0,
        description="Monthly gross income in USD (nullable; imputed if missing).",
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        default=7,
        ge=0,
        description="Number of open loans (installment like car loan or mortgage) and lines of credit.",
    )
    NumberOfTimes90DaysLate: int = Field(
        default=0,
        ge=0,
        description="Number of times borrower has been 90 days or more past due.",
    )
    NumberRealEstateLoansOrLines: int = Field(
        default=1,
        ge=0,
        description="Number of mortgage and real estate loans including home equity lines of credit.",
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        default=0,
        ge=0,
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Number of times borrower has been 60-89 days past due but no worse in the last 2 years.",
    )
    NumberOfDependents: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        description="Number of dependents in family excluding themselves.",
    )
    low_risk_threshold: Optional[float] = Field(
        default=0.20,
        gt=0.0,
        lt=1.0,
        description="Upper probability threshold for Low risk tier.",
    )
    medium_risk_threshold: Optional[float] = Field(
        default=0.50,
        gt=0.0,
        lt=1.0,
        description="Upper probability threshold for Medium risk tier (above is High risk).",
    )


class ShapReasonSchema(BaseModel):
    """Single SHAP feature contribution explanation."""

    feature: str
    display_name: str
    feature_value: float
    formatted_value: str
    shap_value: float
    direction: str
    explanation: str


class PredictionResponse(BaseModel):
    """Credit risk prediction response with tier, credit score, engineered metrics, and SHAP reasons."""

    model_name: str
    default_probability: float
    default_probability_pct: float
    risk_tier: str
    credit_score: int
    recommended_decision: str
    thresholds_used: Dict[str, float]
    engineered_features: Dict[str, Union[float, int, str]]
    shap_reasons: List[ShapReasonSchema]
