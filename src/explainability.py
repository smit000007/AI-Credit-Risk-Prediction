"""SHAP global and per-customer explainability module for Credit Risk models."""

import logging
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def _extract_positive_class_shap(shap_vals: Any) -> np.ndarray:
    """Normalize SHAP output across LightGBM, XGBoost, Random Forest, and Linear models to 2D (n_samples, n_features)."""
    if isinstance(shap_vals, shap.Explanation):
        vals = shap_vals.values
    elif isinstance(shap_vals, list):
        vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    else:
        vals = np.asarray(shap_vals)

    if vals.ndim == 3:
        # Shape (n_samples, n_features, 2) -> select positive class index 1
        vals = vals[:, :, 1]
    return np.asarray(vals, dtype=float)


def get_shap_explainer(artifact_bundle: Dict[str, Any]) -> Any:
    """Create a SHAP explainer for the saved model."""
    model = artifact_bundle["model"]
    bg_X = artifact_bundle["background_X"]
    if artifact_bundle.get("uses_scaled_features", False):
        scaler = artifact_bundle["scaler"]
        bg_in = pd.DataFrame(
            scaler.transform(bg_X), columns=bg_X.columns, index=bg_X.index
        )
        return shap.LinearExplainer(model, bg_in)
    return shap.TreeExplainer(model)


def compute_shap_matrix(
    X_features: pd.DataFrame,
    artifact_bundle: Dict[str, Any],
) -> np.ndarray:
    """Compute 2D SHAP value matrix (n_samples, n_features) for positive default class."""
    explainer = get_shap_explainer(artifact_bundle)
    if artifact_bundle.get("uses_scaled_features", False):
        scaler = artifact_bundle["scaler"]
        X_in = pd.DataFrame(
            scaler.transform(X_features),
            columns=X_features.columns,
            index=X_features.index,
        )
        raw_shap = explainer.shap_values(X_in)
    else:
        X_in = X_features
        use_approx = len(X_in) > 100
        raw_shap = explainer.shap_values(
            X_in, approximate=use_approx, check_additivity=False
        )
    return _extract_positive_class_shap(raw_shap)


def generate_global_shap_summary(
    X_sample: pd.DataFrame,
    artifact_bundle: Dict[str, Any],
    output_dir: Path = FIGURES_DIR,
) -> Dict[str, Any]:
    """
    Compute global SHAP feature importance table and save global SHAP summary
    and sample customer waterfall plots to `dashboard/figures/`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_X = X_sample.iloc[: min(200, len(X_sample))].copy()
    shap_matrix = compute_shap_matrix(eval_X, artifact_bundle)
    feature_cols = list(eval_X.columns)
    display_map = artifact_bundle.get("feature_display_names", {})
    display_names = [display_map.get(c, c) for c in feature_cols]

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    total_imp = max(float(mean_abs.sum()), 1e-9)

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "display_name": display_names,
                "mean_abs_shap": np.round(mean_abs, 5),
                "relative_importance_pct": np.round(mean_abs / total_imp * 100.0, 2),
            }
        )
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Chart 7: Global SHAP Summary (Top 12 Features)
    top_df = importance_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.barh(
        top_df["display_name"],
        top_df["mean_abs_shap"],
        color="#2563eb",
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_title(
        f"Global SHAP Feature Importance ({artifact_bundle['model_name']})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP Value| (Impact on Default Log-Odds / Probability)")
    for bar, pct in zip(bars, top_df["relative_importance_pct"]):
        w = bar.get_width()
        ax.text(
            w + max(top_df["mean_abs_shap"]) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f} ({pct:.1f}%)",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlim(0, max(top_df["mean_abs_shap"]) * 1.22)
    plt.tight_layout()
    p7 = output_dir / "07_shap_global_summary.png"
    fig.savefig(p7, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Chart 8: Per-Customer Local SHAP Explanation Example
    sample_idx = int(np.argmax(shap_matrix.sum(axis=1)))
    single_row = eval_X.iloc[[sample_idx]]
    local_exp = explain_single_applicant(single_row, artifact_bundle, top_k=8)
    fig_local = plot_customer_shap_contributions(local_exp, title_suffix="(High-Risk Sample Applicant)")
    p8 = output_dir / "08_shap_customer_example.png"
    fig_local.savefig(p8, dpi=160, bbox_inches="tight")
    plt.close(fig_local)

    return {
        "importance_df": importance_df,
        "shap_matrix": shap_matrix,
        "global_plot_path": p7,
        "customer_plot_path": p8,
    }


def explain_single_applicant(
    X_single_engineered: pd.DataFrame,
    artifact_bundle: Dict[str, Any],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Generate per-customer SHAP feature attributions for a single engineered row.
    Returns top_k drivers sorted by absolute SHAP impact with human-readable reasons.
    """
    feature_cols = artifact_bundle["feature_columns"]
    display_map = artifact_bundle.get("feature_display_names", {})
    X_row = X_single_engineered[feature_cols].iloc[[0]]
    shap_row = compute_shap_matrix(X_row, artifact_bundle)[0]
    row_values = X_row.iloc[0].to_dict()

    order = np.argsort(np.abs(shap_row))[::-1][:top_k]
    reasons: List[Dict[str, Any]] = []

    for idx in order:
        feat = feature_cols[idx]
        disp = display_map.get(feat, feat)
        val = float(row_values[feat])
        s_val = float(shap_row[idx])
        direction = "increases_risk" if s_val >= 0 else "decreases_risk"
        impact_text = "increased default risk" if s_val >= 0 else "reduced default risk"

        if "utilization" in feat or "ratio" in feat or "to_income" in feat:
            val_str = f"{val * 100:.1f}%"
        elif "Income" in feat or "income_per" in feat:
            val_str = f"${val:,.0f}"
        else:
            val_str = f"{val:.0f}" if val.is_integer() else f"{val:.2f}"

        reasons.append(
            {
                "feature": feat,
                "display_name": disp,
                "feature_value": round(val, 4),
                "formatted_value": val_str,
                "shap_value": round(s_val, 4),
                "direction": direction,
                "explanation": f"{disp} ({val_str}) {impact_text} ({s_val:+.3f} SHAP)",
            }
        )

    return reasons


def plot_customer_shap_contributions(
    reasons: List[Dict[str, Any]],
    title_suffix: str = "",
) -> plt.Figure:
    """Create a horizontal bar chart showing local SHAP contributions for one customer."""
    df_r = pd.DataFrame(reasons).iloc[::-1]
    labels = [f"{r['display_name']} = {r['formatted_value']}" for _, r in df_r.iterrows()]
    vals = df_r["shap_value"].values
    colors = ["#ef4444" if v >= 0 else "#10b981" for v in vals]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.barh(labels, vals, color=colors, edgecolor="black", alpha=0.9)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title(f"Per-Customer SHAP Risk Drivers {title_suffix}".strip(), fontsize=12.5, fontweight="bold")
    ax.set_xlabel("SHAP Contribution (+ Increases Default Risk | - Decreases Default Risk)")

    max_span = max(np.max(np.abs(vals)), 0.1)
    ax.set_xlim(-max_span * 1.32, max_span * 1.32)
    for bar, v in zip(bars, vals):
        offset = max_span * 0.03 if v >= 0 else -max_span * 0.03
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}",
            va="center",
            ha=ha,
            fontweight="bold",
            fontsize=9.5,
        )
    plt.tight_layout()
    return fig
