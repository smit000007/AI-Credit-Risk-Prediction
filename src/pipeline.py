"""End-to-end orchestration pipeline for the AI Credit Risk Prediction system."""

import logging
from typing import Any, Dict

from src.bi_export import export_power_bi_datasets
from src.config import DEFAULT_CONFIG, PROCESSED_DATA_PATH, PipelineConfig
from src.data_generator import generate_give_me_some_credit_dataset
from src.database import (
    init_sqlite_database,
    load_applicants_from_db,
    run_sql_analysis,
)
from src.eda import generate_eda_charts
from src.explainability import generate_global_shap_summary
from src.model_training import train_and_evaluate_all_models
from src.preprocessing import prepare_train_test_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("credit_risk_pipeline")


def run_full_pipeline(config: PipelineConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Execute the complete Credit Risk data, SQL, EDA, ML, SHAP, and BI export pipeline."""
    logger.info("=== STEP 1: Data Acquisition (Give Me Some Credit Schema) ===")
    df_raw = generate_give_me_some_credit_dataset(
        n_samples=config.sample_size,
        random_state=config.random_state,
    )

    logger.info("=== STEP 2: SQLite Database Ingestion & SQL Analytics ===")
    db_path = init_sqlite_database(df_raw)
    sql_results = run_sql_analysis(db_path)
    df_full_joined = load_applicants_from_db(db_path)

    logger.info("=== STEP 3: Stratified Split, Data Cleaning, Feature Engineering & SMOTE ===")
    prep_data = prepare_train_test_data(df_raw, config=config)

    # Save cleaned + engineered dataset
    df_all_clean = prep_data.feature_engineer.transform(df_full_joined)
    df_all_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Saved processed feature dataset to {PROCESSED_DATA_PATH}")

    logger.info("=== STEP 4: Exploratory Data Analysis (EDA) Charts ===")
    eda_charts = generate_eda_charts(df_all_clean)

    logger.info("=== STEP 5: Model Training, Optuna Tuning & Evaluation ===")
    training_output = train_and_evaluate_all_models(prep_data, config=config)
    artifact_bundle = training_output["artifact_bundle"]
    metrics_df = training_output["metrics_df"]

    logger.info("=== STEP 6: SHAP Global & Local Explainability ===")
    shap_output = generate_global_shap_summary(prep_data.X_test, artifact_bundle)
    # Attach global SHAP table into the saved Joblib bundle for quick UI/API access
    artifact_bundle["shap_global_importance"] = shap_output["importance_df"].to_dict(
        orient="records"
    )
    import joblib
    from src.config import BEST_MODEL_PATH
    joblib.dump(artifact_bundle, BEST_MODEL_PATH)

    logger.info("=== STEP 7: Power BI CSV Exports (Predictions & Segment Summaries) ===")
    bi_files = export_power_bi_datasets(
        df_full_with_loans=df_full_joined,
        artifact_bundle=artifact_bundle,
        metrics_df=metrics_df,
        shap_importance_df=shap_output["importance_df"],
        sql_results=sql_results,
    )

    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("\n--- MODEL BENCHMARK COMPARISON ---")
    print(metrics_df.to_string(index=False))
    print("\n--- TOP 8 GLOBAL SHAP DRIVERS ---")
    print(shap_output["importance_df"].head(8).to_string(index=False))

    return {
        "db_path": db_path,
        "sql_results": sql_results,
        "metrics_df": metrics_df,
        "best_model_name": training_output["best_model_name"],
        "shap_importance_df": shap_output["importance_df"],
        "eda_charts": eda_charts,
        "bi_files": bi_files,
    }


if __name__ == "__main__":
    run_full_pipeline()
