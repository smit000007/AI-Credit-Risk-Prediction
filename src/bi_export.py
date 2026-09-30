"""Power BI CSV export module for overview, segment analysis, model performance, and customer drill-down."""

import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from src.config import DASHBOARD_DIR
from src.explainability import compute_shap_matrix
from src.model_training import predict_credit_risk

logger = logging.getLogger(__name__)


def export_power_bi_datasets(
    df_full_with_loans: pd.DataFrame,
    artifact_bundle: Dict[str, Any],
    metrics_df: pd.DataFrame,
    shap_importance_df: pd.DataFrame,
    sql_results: Dict[str, pd.DataFrame],
    output_dir: Path = DASHBOARD_DIR,
) -> Dict[str, Path]:
    """
    Export customer-level predictions, segment summaries, model performance,
    SHAP feature importances, and SQL query outputs to CSV for Power BI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files: Dict[str, Path] = {}

    # 1. Score all applicants
    scored_df = predict_credit_risk(df_full_with_loans, artifact_bundle)

    # Compute top SHAP driver per customer using fast tree explainer
    feature_cols = artifact_bundle["feature_columns"]
    display_map = artifact_bundle.get("feature_display_names", {})
    lgb_or_xgb = artifact_bundle.get("all_models", {}).get("LightGBM") or artifact_bundle.get("all_models", {}).get("XGBoost")
    if lgb_or_xgb is not None:
        import shap
        fast_explainer = shap.TreeExplainer(lgb_or_xgb)
        shap_mat = fast_explainer.shap_values(scored_df[feature_cols])
        if isinstance(shap_mat, list) and len(shap_mat) > 1:
            shap_mat = shap_mat[1]
    else:
        shap_mat = compute_shap_matrix(scored_df[feature_cols], artifact_bundle)

    top_feat_idx = np.argmax(np.abs(shap_mat), axis=1)
    top_shap_names = [display_map.get(feature_cols[i], feature_cols[i]) for i in top_feat_idx]
    top_shap_vals = [float(shap_mat[row_idx, col_idx]) for row_idx, col_idx in enumerate(top_feat_idx)]

    scored_df["top_shap_driver"] = top_shap_names
    scored_df["top_shap_value"] = np.round(top_shap_vals, 4)

    # Underwriting recommendation & Expected Loss (PD * LGD * EAD, assuming LGD=65%)
    scored_df["recommended_decision"] = scored_df["risk_tier"].map(
        {
            "Low": "Auto-Approve (Standard Terms)",
            "Medium": "Manual Review / Risk-Adjusted Pricing",
            "High": "Decline / Require Collateral",
        }
    )
    loan_amt = scored_df.get("RequestedLoanAmount", pd.Series(15000.0, index=scored_df.index))
    scored_df["expected_loss_usd"] = np.round(scored_df["predicted_pd"] * 0.65 * loan_amt, 2)

    # Select clean Power BI customer drill-down columns
    drilldown_cols = [
        "ApplicantID",
        "Region",
        "LoanPurpose",
        "EmploymentStatus",
        "RequestedLoanAmount",
        "LoanTermMonths",
        "EstimatedAPR",
        "age",
        "age_bucket",
        "MonthlyIncome",
        "NumberOfDependents",
        "credit_utilization",
        "debt_to_income",
        "payment_to_income",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "total_delinquencies",
        "weighted_delinquency_score",
        "SeriousDlqin2yrs",
        "predicted_pd",
        "risk_tier",
        "credit_score",
        "recommended_decision",
        "top_shap_driver",
        "top_shap_value",
        "expected_loss_usd",
    ]
    existing_cols = [c for c in drilldown_cols if c in scored_df.columns]
    df_predictions = scored_df[existing_cols].copy()

    pred_path = output_dir / "predictions_export.csv"
    df_predictions.to_csv(pred_path, index=False)
    exported_files["predictions_export"] = pred_path

    # 2. Segment Summary CSV (by Region, Age Bucket, Loan Purpose, and Risk Tier)
    segment_summary = (
        df_predictions.groupby(["Region", "age_bucket", "LoanPurpose", "risk_tier"], observed=False)
        .agg(
            applicant_count=("ApplicantID", "count"),
            observed_defaults=("SeriousDlqin2yrs", "sum"),
            observed_default_rate_pct=("SeriousDlqin2yrs", lambda s: round(float(s.mean()) * 100.0, 2)),
            avg_predicted_pd_pct=("predicted_pd", lambda s: round(float(s.mean()) * 100.0, 2)),
            avg_credit_score=("credit_score", lambda s: round(float(s.mean()), 0)),
            avg_monthly_income=("MonthlyIncome", lambda s: round(float(s.mean()), 0)),
            avg_utilization_pct=("credit_utilization", lambda s: round(float(s.mean()) * 100.0, 2)),
            avg_dti_pct=("debt_to_income", lambda s: round(float(s.mean()) * 100.0, 2)),
            total_loan_exposure_usd=("RequestedLoanAmount", lambda s: round(float(s.sum()), 2)),
            total_expected_loss_usd=("expected_loss_usd", lambda s: round(float(s.sum()), 2)),
        )
        .reset_index()
    )
    segment_summary = segment_summary[segment_summary["applicant_count"] > 0].reset_index(drop=True)
    seg_path = output_dir / "segment_summary.csv"
    segment_summary.to_csv(seg_path, index=False)
    exported_files["segment_summary"] = seg_path

    # 3. Model Performance CSV
    perf_path = output_dir / "model_performance.csv"
    metrics_df.to_csv(perf_path, index=False)
    exported_files["model_performance"] = perf_path

    # 4. Global SHAP Importance CSV
    shap_path = output_dir / "shap_global_importance.csv"
    shap_importance_df.to_csv(shap_path, index=False)
    exported_files["shap_global_importance"] = shap_path

    # 5. Export SQL Query Tables for BI
    for q_name, q_df in sql_results.items():
        q_path = output_dir / f"sql_{q_name}.csv"
        q_df.to_csv(q_path, index=False)
        exported_files[f"sql_{q_name}"] = q_path

    logger.info(f"Exported {len(exported_files)} Power BI CSV datasets to {output_dir}")
    return exported_files
