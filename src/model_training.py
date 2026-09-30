"""Model training, Optuna hyperparameter tuning, credit risk metrics (ROC-AUC, Precision, Recall, F1, KS, Gini), and Joblib persistence."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_CONFIG,
    FIGURES_DIR,
    MODEL_METRICS_PATH,
    PipelineConfig,
)
from src.preprocessing import PreprocessedData

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic between default and non-default distributions."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    pos_probs = y_prob_arr[y_true_arr == 1]
    neg_probs = y_prob_arr[y_true_arr == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
    ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    return float(ks_stat)


def evaluate_credit_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float = 0.35,
) -> Dict[str, float]:
    """
    Calculate comprehensive credit risk evaluation metrics:
    ROC-AUC, Precision, Recall, F1, KS Statistic, and Gini Coefficient.
    """
    y_pred = (np.asarray(y_prob) >= decision_threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    gini = float(2.0 * roc_auc - 1.0)
    ks = compute_ks_statistic(y_true, y_prob)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "ROC_AUC": round(roc_auc, 4),
        "Gini": round(gini, 4),
        "KS_Statistic": round(ks, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
    }


def _tune_random_forest(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    config: PipelineConfig,
) -> RandomForestClassifier:
    """Tune Random Forest hyperparameters using Optuna stratified cross-validation."""
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 260, step=20),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_samples_split": trial.suggest_int("min_samples_split", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced_subsample",
            "random_state": config.random_state,
            "n_jobs": -1,
        }
        aucs = []
        for train_idx, val_idx in skf.split(X_tr, y_tr):
            clf = RandomForestClassifier(**params)
            clf.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
            preds = clf.predict_proba(X_tr.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_tr.iloc[val_idx], preds))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
    )
    study.optimize(objective, n_trials=max(8, config.optuna_trials // 2))
    best_params = study.best_params
    best_params.update(
        {
            "class_weight": "balanced_subsample",
            "random_state": config.random_state,
            "n_jobs": -1,
        }
    )
    model = RandomForestClassifier(**best_params)
    model.fit(X_tr, y_tr)
    return model


def _tune_xgboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    config: PipelineConfig,
) -> xgb.XGBClassifier:
    """Tune XGBoost hyperparameters using Optuna stratified cross-validation."""
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 320, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "eval_metric": "logloss",
            "random_state": config.random_state,
            "n_jobs": -1,
        }
        aucs = []
        for train_idx, val_idx in skf.split(X_tr, y_tr):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
            preds = clf.predict_proba(X_tr.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_tr.iloc[val_idx], preds))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
    )
    study.optimize(objective, n_trials=config.optuna_trials)
    best_params = study.best_params
    best_params.update(
        {
            "eval_metric": "logloss",
            "random_state": config.random_state,
            "n_jobs": -1,
        }
    )
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_tr, y_tr)
    return model


def _tune_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    config: PipelineConfig,
) -> lgb.LGBMClassifier:
    """Tune LightGBM hyperparameters using Optuna stratified cross-validation."""
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 320, step=20),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "random_state": config.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        aucs = []
        for train_idx, val_idx in skf.split(X_tr, y_tr):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
            preds = clf.predict_proba(X_tr.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_tr.iloc[val_idx], preds))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
    )
    study.optimize(objective, n_trials=config.optuna_trials)
    best_params = study.best_params
    best_params.update(
        {
            "random_state": config.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
    )
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)
    return model


def plot_model_comparison_charts(
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    metrics_df: pd.DataFrame,
    y_test: pd.Series,
    best_probs: np.ndarray,
    config: PipelineConfig = DEFAULT_CONFIG,
    output_dir: Path = FIGURES_DIR,
) -> Dict[str, Path]:
    """Generate ROC curves, metric comparison bars, and Risk Tier separation visuals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    # Chart 5: ROC Curves & Benchmark Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    palette = ["#64748b", "#10b981", "#2563eb", "#7c3aed"]
    for (name, (fpr, tpr, auc_val)), color in zip(roc_curves.items(), palette):
        axes[0].plot(fpr, tpr, linewidth=2.3, label=f"{name} (AUC = {auc_val:.4f})", color=color)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline (AUC = 0.5000)")
    axes[0].set_title("ROC Curves Across Tuned Credit Risk Models", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    x = np.arange(len(metrics_df))
    w = 0.25
    axes[1].bar(x - w, metrics_df["ROC_AUC"], width=w, label="ROC-AUC", color="#2563eb")
    axes[1].bar(x, metrics_df["Gini"], width=w, label="Gini", color="#7c3aed")
    axes[1].bar(x + w, metrics_df["KS_Statistic"], width=w, label="KS Statistic", color="#10b981")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_df["Model"], rotation=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Discrimination Metrics: ROC-AUC, Gini & KS", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    p5 = output_dir / "05_model_roc_and_ks_comparison.png"
    fig.savefig(p5, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved["model_comparison"] = p5

    # Chart 6: Risk Tier Separation (Low, Medium, High)
    tiers = [config.thresholds.classify(p) for p in best_probs]
    tier_df = pd.DataFrame({"Tier": tiers, "ActualDefault": y_test.values, "PD": best_probs})
    tier_order = ["Low", "Medium", "High"]
    tier_stats = (
        tier_df.groupby("Tier")
        .agg(
            Applicants=("ActualDefault", "count"),
            ActualDefaultRate=("ActualDefault", "mean"),
            MeanPredictedPD=("PD", "mean"),
        )
        .reindex(tier_order)
        .fillna(0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    tier_colors = ["#10b981", "#f59e0b", "#ef4444"]
    b1 = axes[0].bar(tier_order, tier_stats["Applicants"], color=tier_colors, edgecolor="black", width=0.5)
    axes[0].set_title("Test Portfolio Distribution by Predicted Risk Tier", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Borrowers")
    for bar in b1:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + max(tier_stats["Applicants"]) * 0.015,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(tier_stats["Applicants"]) * 1.18)

    b2 = axes[1].bar(
        tier_order,
        tier_stats["ActualDefaultRate"] * 100,
        color=tier_colors,
        edgecolor="black",
        width=0.5,
    )
    axes[1].set_title("Observed Default Rate (%) Across Risk Tiers", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Observed 2-Year Default Rate (%)")
    for bar in b2:
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.6,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[1].set_ylim(0, max(tier_stats["ActualDefaultRate"] * 100) * 1.22)

    plt.tight_layout()
    p6 = output_dir / "06_risk_tier_separation.png"
    fig.savefig(p6, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved["risk_tier_separation"] = p6

    return saved


def train_and_evaluate_all_models(
    data: PreprocessedData,
    config: PipelineConfig = DEFAULT_CONFIG,
    save_model_path: Path = BEST_MODEL_PATH,
    save_metrics_path: Path = MODEL_METRICS_PATH,
) -> Dict[str, Any]:
    """
    Train Logistic Regression (baseline), Random Forest, XGBoost, and LightGBM
    with Optuna tuning. Compare all models on ROC-AUC, Precision, Recall, F1, KS,
    and Gini, and persist the winning model artifact with Joblib.
    """
    logger.info("1/4 Training Baseline Logistic Regression...")
    lr_model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        random_state=config.random_state,
    )
    lr_model.fit(data.X_train_scaled, data.y_train_resampled)

    logger.info("2/4 Tuning & Training Random Forest with Optuna...")
    rf_model = _tune_random_forest(data.X_train_resampled, data.y_train_resampled, config)

    logger.info("3/4 Tuning & Training XGBoost with Optuna...")
    xgb_model = _tune_xgboost(data.X_train_resampled, data.y_train_resampled, config)

    logger.info("4/4 Tuning & Training LightGBM with Optuna...")
    lgb_model = _tune_lightgbm(data.X_train_resampled, data.y_train_resampled, config)

    models_dict = {
        "Logistic Regression": (lr_model, True),
        "Random Forest": (rf_model, False),
        "XGBoost": (xgb_model, False),
        "LightGBM": (lgb_model, False),
    }

    metrics_rows = []
    roc_curves = {}
    test_probabilities: Dict[str, np.ndarray] = {}

    for name, (mdl, use_scaled) in models_dict.items():
        X_eval = data.X_test_scaled if use_scaled else data.X_test
        # Calibrate probability back to true population prior when trained on SMOTE-resampled data
        raw_prob = mdl.predict_proba(X_eval)[:, 1]
        prior_true = float(data.y_train.mean())
        prior_smote = float(data.y_train_resampled.mean())
        # Bayes prior correction to keep probabilities well-calibrated to actual default rate
        eps = 1e-6
        p_clip = np.clip(raw_prob, eps, 1.0 - eps)
        odds_adj = (p_clip / (1.0 - p_clip)) * (
            (prior_true / (1.0 - prior_true)) / (prior_smote / (1.0 - prior_smote))
        )
        calibrated_prob = odds_adj / (1.0 + odds_adj)

        test_probabilities[name] = calibrated_prob
        m = evaluate_credit_model(data.y_test.values, calibrated_prob, decision_threshold=0.20)
        m["Model"] = name
        metrics_rows.append(m)

        fpr, tpr, _ = roc_curve(data.y_test.values, calibrated_prob)
        roc_curves[name] = (fpr, tpr, m["ROC_AUC"])

    metrics_df = pd.DataFrame(metrics_rows)[
        ["Model", "ROC_AUC", "Gini", "KS_Statistic", "Precision", "Recall", "F1_Score"]
    ].sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)

    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_model_obj, best_uses_scaled = models_dict[best_model_name]
    best_probs = test_probabilities[best_model_name]

    logger.info(
        f"Best Model: {best_model_name} (ROC-AUC={metrics_df.iloc[0]['ROC_AUC']:.4f}, "
        f"Gini={metrics_df.iloc[0]['Gini']:.4f}, KS={metrics_df.iloc[0]['KS_Statistic']:.4f})"
    )

    # Generate comparison and risk tier charts
    chart_paths = plot_model_comparison_charts(
        roc_curves=roc_curves,
        metrics_df=metrics_df,
        y_test=data.y_test,
        best_probs=best_probs,
        config=config,
    )

    # Save metrics JSON
    save_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics_path.write_text(
        json.dumps(metrics_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    # Build self-contained model bundle and save with Joblib
    background_sample = data.X_train.sample(
        n=min(250, len(data.X_train)), random_state=config.random_state
    )
    artifact_bundle = {
        "model": best_model_obj,
        "model_name": best_model_name,
        "uses_scaled_features": best_uses_scaled,
        "all_models": {k: v[0] for k, v in models_dict.items()},
        "feature_engineer": data.feature_engineer,
        "scaler": data.scaler,
        "feature_columns": config.engineered_feature_columns,
        "feature_display_names": config.feature_display_names,
        "thresholds": config.thresholds,
        "prior_true": float(data.y_train.mean()),
        "prior_smote": float(data.y_train_resampled.mean()),
        "metrics_table": metrics_df.to_dict(orient="records"),
        "background_X": background_sample,
    }
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_bundle, save_model_path)
    logger.info(f"Saved best model artifact bundle to {save_model_path}")

    return {
        "artifact_bundle": artifact_bundle,
        "metrics_df": metrics_df,
        "test_probabilities": test_probabilities,
        "best_model_name": best_model_name,
        "chart_paths": chart_paths,
    }


def predict_credit_risk(
    df_raw_applicants: pd.DataFrame,
    artifact_bundle: Dict[str, Any],
    low_threshold: float | None = None,
    medium_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Run end-to-end inference on raw applicant records:
    clean & engineer features -> predict calibrated default probability ->
    assign Low/Medium/High risk tier and FICO-equivalent score.
    """
    fe = artifact_bundle["feature_engineer"]
    scaler = artifact_bundle["scaler"]
    model = artifact_bundle["model"]
    uses_scaled = artifact_bundle["uses_scaled_features"]
    feature_cols = artifact_bundle["feature_columns"]
    thresholds = artifact_bundle["thresholds"]

    low_max = low_threshold if low_threshold is not None else thresholds.low_max
    med_max = medium_threshold if medium_threshold is not None else thresholds.medium_max

    df_engineered = fe.transform(df_raw_applicants)
    X = df_engineered[feature_cols]
    if uses_scaled:
        X_in = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
    else:
        X_in = X

    raw_prob = model.predict_proba(X_in)[:, 1]
    prior_true = artifact_bundle.get("prior_true", 0.08)
    prior_smote = artifact_bundle.get("prior_smote", 0.25)
    eps = 1e-6
    p_clip = np.clip(raw_prob, eps, 1.0 - eps)
    odds_adj = (p_clip / (1.0 - p_clip)) * (
        (prior_true / (1.0 - prior_true)) / (prior_smote / (1.0 - prior_smote))
    )
    calibrated_prob = odds_adj / (1.0 + odds_adj)

    df_out = df_engineered.copy()
    df_out["predicted_pd"] = np.round(calibrated_prob, 4)
    df_out["risk_tier"] = [
        "Low" if p < low_max else ("Medium" if p < med_max else "High")
        for p in calibrated_prob
    ]
    df_out["credit_score"] = [thresholds.to_credit_score(p) for p in calibrated_prob]
    return df_out
