"""Unit tests for credit risk metrics, risk tier thresholds, Joblib model bundle, and SHAP explanations."""

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, RiskThresholds
from src.explainability import explain_single_applicant
from src.model_training import evaluate_credit_model, predict_credit_risk


def test_risk_thresholds_and_credit_score():
    """Verify configurable Low / Medium / High risk tier mapping and credit score bounds."""
    thresholds = RiskThresholds(low_max=0.20, medium_max=0.50)
    assert thresholds.classify(0.05) == "Low"
    assert thresholds.classify(0.199) == "Low"
    assert thresholds.classify(0.20) == "Medium"
    assert thresholds.classify(0.49) == "Medium"
    assert thresholds.classify(0.50) == "High"
    assert thresholds.classify(0.88) == "High"

    score_low_pd = thresholds.to_credit_score(0.02)
    score_high_pd = thresholds.to_credit_score(0.75)
    assert 300 <= score_high_pd < score_low_pd <= 850


def test_evaluate_credit_model_metrics():
    """Verify ROC-AUC, Gini, KS, Precision, Recall, and F1 calculation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.80, 0.92])
    metrics = evaluate_credit_model(y_true, y_prob, decision_threshold=0.40)

    assert metrics["ROC_AUC"] == 1.0
    assert metrics["Gini"] == 1.0
    assert metrics["KS_Statistic"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["F1_Score"] == 1.0


def test_saved_joblib_model_prediction_and_shap():
    """Verify saved Joblib artifact scores Low vs High risk applicants and computes SHAP reasons."""
    assert BEST_MODEL_PATH.exists(), "Best model Joblib file must exist"
    bundle = joblib.load(BEST_MODEL_PATH)

    sample_df = pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.10,
                "age": 52,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.18,
                "MonthlyIncome": 9500.0,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 1.0,
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 1.15,
                "age": 24,
                "NumberOfTime30-59DaysPastDueNotWorse": 3,
                "DebtRatio": 0.85,
                "MonthlyIncome": 2100.0,
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 3,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 2,
                "NumberOfDependents": 3.0,
            },
        ]
    )

    scored = predict_credit_risk(sample_df, bundle)
    assert len(scored) == 2
    assert scored.loc[0, "predicted_pd"] < scored.loc[1, "predicted_pd"]
    assert scored.loc[0, "risk_tier"] == "Low"
    assert scored.loc[1, "risk_tier"] in ("Medium", "High")

    reasons = explain_single_applicant(scored.iloc[[1]], bundle, top_k=5)
    assert len(reasons) == 5
    assert "shap_value" in reasons[0]
    assert "explanation" in reasons[0]
