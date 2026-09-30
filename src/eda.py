"""Exploratory Data Analysis (EDA) chart generator for Credit Risk Prediction."""

import logging
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Consistent dark/modern financial palette
PALETTE = {
    "Low": "#10b981",
    "Medium": "#f59e0b",
    "High": "#ef4444",
    "Non-Default": "#2563eb",
    "Default": "#dc2626",
}


def generate_eda_charts(
    df_clean: pd.DataFrame, output_dir: Path = FIGURES_DIR
) -> Dict[str, Path]:
    """Generate and save EDA charts to `dashboard/figures/`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    saved_paths: Dict[str, Path] = {}

    # 1. Target Distribution & Default Rate by Age Bucket
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    target_counts = df_clean["SeriousDlqin2yrs"].value_counts().sort_index()
    labels = ["Performing (0)", "Default / Serious Dlq (1)"]
    colors = ["#2563eb", "#dc2626"]
    bars = axes[0].bar(labels, target_counts.values, color=colors, width=0.52, edgecolor="black", alpha=0.9)
    axes[0].set_title("Loan Portfolio Target Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Applicants")
    for bar in bars:
        height = bar.get_height()
        pct = height / len(df_clean) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + len(df_clean) * 0.01,
            f"{int(height):,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    axes[0].set_ylim(0, max(target_counts.values) * 1.18)

    age_order = ["<25", "25-34", "35-49", "50-64", "65+"]
    age_stats = (
        df_clean.groupby("age_bucket", observed=False)["SeriousDlqin2yrs"]
        .agg(["count", "mean"])
        .reindex(age_order)
    )
    age_bars = axes[1].bar(
        age_stats.index.astype(str),
        age_stats["mean"] * 100,
        color=["#ef4444", "#f97316", "#eab308", "#3b82f6", "#10b981"],
        edgecolor="black",
        alpha=0.9,
    )
    axes[1].set_title("2-Year Default Rate (%) by Age Cohort", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Age Bucket (Years)")
    axes[1].set_ylabel("Default Rate (%)")
    for bar in age_bars:
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.35,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    axes[1].set_ylim(0, max(age_stats["mean"] * 100) * 1.22)

    plt.tight_layout()
    p1 = output_dir / "01_target_and_age_default_rate.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved_paths["target_and_age"] = p1

    # 2. Credit Utilization & Payment-to-Income Distributions by Default Status
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    df_plot = df_clean.copy()
    df_plot["Status"] = df_plot["SeriousDlqin2yrs"].map({0: "Performing", 1: "Defaulted"})

    sns.kdeplot(
        data=df_plot,
        x="credit_utilization",
        hue="Status",
        common_norm=False,
        fill=True,
        palette={"Performing": "#2563eb", "Defaulted": "#dc2626"},
        alpha=0.35,
        ax=axes[0],
        clip=(0, 1.5),
    )
    axes[0].axvline(0.80, color="#dc2626", linestyle="--", linewidth=1.5, label="80% High Util Threshold")
    axes[0].set_title("Revolving Credit Utilization Density by Outcome", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Revolving Credit Utilization Ratio")
    axes[0].legend()

    sns.boxplot(
        data=df_plot,
        x="Status",
        y="debt_to_income",
        hue="Status",
        palette={"Performing": "#3b82f6", "Defaulted": "#ef4444"},
        showfliers=False,
        width=0.45,
        ax=axes[1],
        legend=False,
    )
    axes[1].axhline(0.43, color="#b91c1c", linestyle="--", linewidth=1.5, label="43% Qualified Mortgage DTI Cap")
    axes[1].set_title("Debt-to-Income (DTI) Distribution by Outcome", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Loan Outcome")
    axes[1].set_ylabel("Debt-to-Income Ratio")
    axes[1].legend()

    plt.tight_layout()
    p2 = output_dir / "02_utilization_and_dti_distributions.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved_paths["utilization_and_dti"] = p2

    # 3. Delinquency History Impact on Default Probability
    fig, ax = plt.subplots(figsize=(10, 5.5))
    delinq_bins = pd.cut(
        df_clean["total_delinquencies"],
        bins=[-1, 0, 1, 2, 4, 100],
        labels=["0 (Clean)", "1 Past Due", "2 Past Due", "3-4 Past Due", "5+ Past Due"],
    )
    delinq_rates = (
        df_clean.groupby(delinq_bins, observed=False)["SeriousDlqin2yrs"]
        .agg(["count", "mean"])
    )
    bars = ax.bar(
        delinq_rates.index.astype(str),
        delinq_rates["mean"] * 100,
        color=["#10b981", "#f59e0b", "#f97316", "#ef4444", "#991b1b"],
        edgecolor="black",
        width=0.55,
    )
    ax.set_title("Default Rate (%) by Historical Delinquency Frequency", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Past-Due Delinquency Events (30-89 & 90+ Days)")
    ax.set_ylabel("Observed Default Rate (%)")
    for bar, (_, row) in zip(bars, delinq_rates.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}%\n(n={int(row['count']):,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )
    ax.set_ylim(0, max(delinq_rates["mean"] * 100) * 1.25)
    plt.tight_layout()
    p3 = output_dir / "03_delinquency_severity_impact.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved_paths["delinquency_impact"] = p3

    # 4. Feature Correlation Heatmap
    corr_cols = [
        "SeriousDlqin2yrs",
        "credit_utilization",
        "age",
        "debt_to_income",
        "MonthlyIncome",
        "payment_to_income",
        "income_per_dependent",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "total_delinquencies",
        "weighted_delinquency_score",
        "severe_delinquency_flag",
    ]
    fig, ax = plt.subplots(figsize=(11, 9))
    corr_matrix = df_clean[corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.9,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Engineered Credit Risk Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = output_dir / "04_feature_correlation_heatmap.png"
    fig.savefig(p4, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved_paths["correlation_heatmap"] = p4

    logger.info(f"Generated {len(saved_paths)} EDA charts in {output_dir}")
    return saved_paths
