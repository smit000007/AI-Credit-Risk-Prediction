"""FastAPI application serving Credit Risk predictions, risk tiers, and SHAP explanations."""

from functools import lru_cache
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

from api.schemas import ApplicantRequest, PredictionResponse
from src.config import BEST_MODEL_PATH
from src.explainability import explain_single_applicant
from src.model_training import predict_credit_risk

app = FastAPI(
    title="AI Credit Risk Prediction API",
    description=(
        "Production REST API that predicts 2-year loan default probability, "
        "classifies applicants into Low / Medium / High risk tiers, computes "
        "engineered credit ratios, and returns per-customer SHAP explanations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_model_bundle() -> Dict[str, Any]:
    """Load the persisted Joblib model artifact bundle."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {BEST_MODEL_PATH}. Run `python -m src.pipeline` first."
        )
    return joblib.load(BEST_MODEL_PATH)


def _applicant_to_dataframe(req: ApplicantRequest) -> pd.DataFrame:
    """Convert an ApplicantRequest payload into the raw DataFrame schema expected by the pipeline."""
    return pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": req.RevolvingUtilizationOfUnsecuredLines,
                "age": req.age,
                "NumberOfTime30-59DaysPastDueNotWorse": req.NumberOfTime30_59DaysPastDueNotWorse,
                "DebtRatio": req.DebtRatio,
                "MonthlyIncome": req.MonthlyIncome,
                "NumberOfOpenCreditLinesAndLoans": req.NumberOfOpenCreditLinesAndLoans,
                "NumberOfTimes90DaysLate": req.NumberOfTimes90DaysLate,
                "NumberRealEstateLoansOrLines": req.NumberRealEstateLoansOrLines,
                "NumberOfTime60-89DaysPastDueNotWorse": req.NumberOfTime60_89DaysPastDueNotWorse,
                "NumberOfDependents": req.NumberOfDependents,
            }
        ]
    )


DECISION_MAP = {
    "Low": "Auto-Approve (Standard Prime Terms)",
    "Medium": "Manual Underwriting Review / Risk-Adjusted Pricing",
    "High": "Decline Application / Require Secured Collateral",
}


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Service liveness and model readiness check."""
    bundle = load_model_bundle()
    return {
        "status": "healthy",
        "active_model": bundle["model_name"],
        "feature_count": len(bundle["feature_columns"]),
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """Return benchmark metrics across all models and global SHAP feature importance."""
    bundle = load_model_bundle()
    return {
        "best_model": bundle["model_name"],
        "default_thresholds": {
            "low_max": bundle["thresholds"].low_max,
            "medium_max": bundle["thresholds"].medium_max,
        },
        "metrics_comparison": bundle.get("metrics_table", []),
        "global_shap_importance": bundle.get("shap_global_importance", []),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_applicant(request: ApplicantRequest) -> PredictionResponse:
    """Score a single loan applicant and return risk tier, probability, and top SHAP reasons."""
    bundle = load_model_bundle()
    low_t = request.low_risk_threshold or bundle["thresholds"].low_max
    med_t = request.medium_risk_threshold or bundle["thresholds"].medium_max
    if low_t >= med_t:
        raise HTTPException(
            status_code=400,
            detail="low_risk_threshold must be strictly less than medium_risk_threshold.",
        )

    df_raw = _applicant_to_dataframe(request)
    scored_df = predict_credit_risk(
        df_raw,
        bundle,
        low_threshold=low_t,
        medium_threshold=med_t,
    )
    shap_reasons = explain_single_applicant(scored_df, bundle, top_k=6)
    row = scored_df.iloc[0]

    tier = str(row["risk_tier"])
    prob = float(row["predicted_pd"])

    return PredictionResponse(
        model_name=bundle["model_name"],
        default_probability=round(prob, 4),
        default_probability_pct=round(prob * 100.0, 2),
        risk_tier=tier,
        credit_score=int(row["credit_score"]),
        recommended_decision=DECISION_MAP.get(tier, "Manual Review"),
        thresholds_used={"low_max": low_t, "medium_max": med_t},
        engineered_features={
            "age_bucket": str(row["age_bucket"]),
            "credit_utilization": round(float(row["credit_utilization"]), 4),
            "debt_to_income": round(float(row["debt_to_income"]), 4),
            "payment_to_income": round(float(row["payment_to_income"]), 4),
            "income_per_dependent": round(float(row["income_per_dependent"]), 2),
            "total_delinquencies": int(row["total_delinquencies"]),
            "weighted_delinquency_score": round(float(row["weighted_delinquency_score"]), 2),
            "any_delinquency_flag": int(row["any_delinquency_flag"]),
            "severe_delinquency_flag": int(row["severe_delinquency_flag"]),
        },
        shap_reasons=shap_reasons,
    )


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(requests: List[ApplicantRequest]) -> List[PredictionResponse]:
    """Score a batch of loan applicants."""
    return [predict_applicant(r) for r in requests]
