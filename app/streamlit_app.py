"""Streamlit Web Application for AI Credit Risk Prediction & SHAP Explainability."""

from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import BEST_MODEL_PATH, DASHBOARD_DIR, FIGURES_DIR, SQLITE_DB_PATH
from src.database import parse_named_queries, run_sql_analysis
from src.explainability import explain_single_applicant
from src.model_training import predict_credit_risk

st.set_page_config(
    page_title="AI Credit Risk Prediction & SHAP Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for executive fintech look
st.markdown(
    """
    <style>
    .tier-card {
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    .tier-low { background: linear-gradient(135deg, #059669, #10b981); }
    .tier-medium { background: linear-gradient(135deg, #d97706, #f59e0b); }
    .tier-high { background: linear-gradient(135deg, #b91c1c, #ef4444); }
    .reason-pos {
        border-left: 4px solid #ef4444;
        background-color: rgba(239, 68, 68, 0.08);
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.45rem;
        border-radius: 0 8px 8px 0;
    }
    .reason-neg {
        border-left: 4px solid #10b981;
        background-color: rgba(16, 185, 129, 0.08);
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.45rem;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_bundle():
    """Load the trained Joblib model bundle."""
    if not BEST_MODEL_PATH.exists():
        from src.pipeline import run_full_pipeline
        run_full_pipeline()
    return joblib.load(BEST_MODEL_PATH)


bundle = get_model_bundle()

# Preset profiles for instant demonstration
PRESETS = {
    "Custom Applicant": None,
    "Prime Homeowner (Low Risk)": {
        "util": 0.14,
        "age": 48,
        "income": 9200.0,
        "dti": 0.22,
        "open_lines": 9,
        "re_loans": 2,
        "dependents": 1,
        "d30": 0,
        "d60": 0,
        "d90": 0,
    },
    "Stretched Mid-Career Borrower (Medium Risk)": {
        "util": 0.78,
        "age": 34,
        "income": 4400.0,
        "dti": 0.49,
        "open_lines": 11,
        "re_loans": 1,
        "dependents": 2,
        "d30": 1,
        "d60": 0,
        "d90": 0,
    },
    "Distressed Subprime Applicant (High Risk)": {
        "util": 1.12,
        "age": 26,
        "income": 2650.0,
        "dti": 0.74,
        "open_lines": 5,
        "re_loans": 0,
        "dependents": 3,
        "d30": 2,
        "d60": 1,
        "d90": 2,
    },
}

st.sidebar.title("🛡️ Credit Underwriting Controls")
selected_preset = st.sidebar.selectbox(
    "Load Sample Applicant Profile",
    options=list(PRESETS.keys()),
    index=1,
)
preset_vals = PRESETS[selected_preset] or PRESETS["Prime Homeowner (Low Risk)"]

st.sidebar.markdown("### 1. Applicant Financials & Credit")
revolving_util = st.sidebar.slider(
    "Revolving Credit Utilization (%)",
    min_value=0,
    max_value=150,
    value=int(round(preset_vals["util"] * 100)),
    step=1,
    help="Balance on unsecured credit lines divided by total credit limit.",
) / 100.0

age = st.sidebar.slider(
    "Applicant Age (Years)",
    min_value=18,
    max_value=90,
    value=int(preset_vals["age"]),
)

monthly_income = st.sidebar.number_input(
    "Monthly Gross Income ($)",
    min_value=500.0,
    max_value=100000.0,
    value=float(preset_vals["income"]),
    step=250.0,
)

debt_ratio = st.sidebar.slider(
    "Debt-to-Income Ratio (DTI %)",
    min_value=1,
    max_value=150,
    value=int(round(preset_vals["dti"] * 100)),
    step=1,
) / 100.0

open_lines = st.sidebar.number_input(
    "Open Credit Lines & Loans",
    min_value=0,
    max_value=40,
    value=int(preset_vals["open_lines"]),
)

re_loans = st.sidebar.number_input(
    "Real Estate / Mortgage Loans",
    min_value=0,
    max_value=15,
    value=int(preset_vals["re_loans"]),
)

dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=int(preset_vals["dependents"]),
)

st.sidebar.markdown("### 2. Past-Due Delinquency History (2 Yrs)")
days_30_59 = st.sidebar.number_input(
    "30-59 Days Past Due Count",
    min_value=0,
    max_value=15,
    value=int(preset_vals["d30"]),
)
days_60_89 = st.sidebar.number_input(
    "60-89 Days Past Due Count",
    min_value=0,
    max_value=15,
    value=int(preset_vals["d60"]),
)
days_90_plus = st.sidebar.number_input(
    "90+ Days Severe Delinquency Count",
    min_value=0,
    max_value=15,
    value=int(preset_vals["d90"]),
)

st.sidebar.markdown("### 3. Configurable Risk Tier Thresholds")
low_threshold = st.sidebar.slider(
    "Low → Medium PD Threshold",
    min_value=0.05,
    max_value=0.40,
    value=float(bundle["thresholds"].low_max),
    step=0.01,
)
medium_threshold = st.sidebar.slider(
    "Medium → High PD Threshold",
    min_value=max(low_threshold + 0.05, 0.25),
    max_value=0.85,
    value=max(float(bundle["thresholds"].medium_max), low_threshold + 0.10),
    step=0.01,
)

# Header
st.title("🛡️ AI Credit Risk Prediction & Explainable Underwriting Platform")
st.caption(
    f"Active Champion Model: **{bundle['model_name']}** | Dataset: **Give Me Some Credit (15,000 Applicants)** "
    f"| Tier Thresholds: **Low < {low_threshold:.0%}**, **Medium {low_threshold:.0%}–{medium_threshold:.0%}**, **High ≥ {medium_threshold:.0%}**"
)

tab_underwrite, tab_models, tab_sql, tab_bi = st.tabs(
    [
        "🔍 Live Applicant Scoring & SHAP",
        "📊 Model Benchmark & Global SHAP",
        "🗄️ SQLite Segment & Window Analytics",
        "📈 Power BI Exports & Customer Drill-Down",
    ]
)

with tab_underwrite:
    applicant_df = pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": revolving_util,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": days_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": monthly_income,
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": days_90_plus,
                "NumberRealEstateLoansOrLines": re_loans,
                "NumberOfTime60-89DaysPastDueNotWorse": days_60_89,
                "NumberOfDependents": float(dependents),
            }
        ]
    )

    scored_df = predict_credit_risk(
        applicant_df,
        bundle,
        low_threshold=low_threshold,
        medium_threshold=medium_threshold,
    )
    shap_reasons = explain_single_applicant(scored_df, bundle, top_k=8)
    row = scored_df.iloc[0]

    pd_prob = float(row["predicted_pd"])
    tier = str(row["risk_tier"])
    credit_score = int(row["credit_score"])

    decision_map = {
        "Low": "AUTO-APPROVE — Prime Interest Tier",
        "Medium": "MANUAL REVIEW — Verify Income & Cap Credit Limit",
        "High": "DECLINE / REQUIRE COLLATERAL — Elevated Default Risk",
    }
    tier_css = {"Low": "tier-low", "Medium": "tier-medium", "High": "tier-high"}[tier]

    col_banner, col_gauge = st.columns([1.15, 1.0])
    with col_banner:
        st.markdown(
            f"""
            <div class="tier-card {tier_css}">
                <h4 style="margin:0; font-weight:500; opacity:0.92;">PREDICTED RISK CLASSIFICATION</h4>
                <h1 style="margin:0.2rem 0; font-size:2.5rem;">{tier.upper()} RISK TIER</h1>
                <p style="margin:0; font-size:1.08rem; font-weight:600;">{decision_map[tier]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("2-Year Default Prob (PD)", f"{pd_prob * 100:.2f}%")
        m2.metric("Implied Credit Score", f"{credit_score}", delta="300–850 Scale")
        m3.metric("Age Cohort", f"{row['age_bucket']} yrs")

        st.markdown("#### Engineered Financial & Delinquency Ratios")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Credit Utilization", f"{row['credit_utilization'] * 100:.1f}%")
        e2.metric("Debt-to-Income (DTI)", f"{row['debt_to_income'] * 100:.1f}%")
        e3.metric("Payment-to-Income (PTI)", f"{row['payment_to_income'] * 100:.1f}%")
        e4.metric(
            "Weighted Delinquency",
            f"{row['weighted_delinquency_score']:.1f}",
            delta=f"{int(row['total_delinquencies'])} events",
            delta_color="inverse",
        )

    with col_gauge:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(pd_prob * 100, 2),
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Predicted 2-Year Default Probability (PD)", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1e293b"},
                    "steps": [
                        {"range": [0, low_threshold * 100], "color": "#10b981"},
                        {"range": [low_threshold * 100, medium_threshold * 100], "color": "#f59e0b"},
                        {"range": [medium_threshold * 100, 100], "color": "#ef4444"},
                    ],
                    "threshold": {
                        "line": {"color": "#111827", "width": 4},
                        "thickness": 0.8,
                        "value": pd_prob * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=310, margin=dict(l=25, r=25, t=45, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    col_shap_chart, col_shap_text = st.columns([1.25, 1.0])

    with col_shap_chart:
        st.subheader("Per-Customer SHAP Feature Contributions")
        df_shap = pd.DataFrame(shap_reasons).iloc[::-1]
        df_shap["label"] = df_shap["display_name"] + " (" + df_shap["formatted_value"] + ")"
        df_shap["Impact"] = np.where(
            df_shap["shap_value"] >= 0, "Increases Risk (+)", "Reduces Risk (-)"
        )
        fig_shap = px.bar(
            df_shap,
            x="shap_value",
            y="label",
            color="Impact",
            orientation="h",
            color_discrete_map={
                "Increases Risk (+)": "#ef4444",
                "Reduces Risk (-)": "#10b981",
            },
            text=df_shap["shap_value"].map(lambda v: f"{v:+.3f}"),
            labels={"shap_value": "SHAP Impact on Default Log-Odds", "label": "Feature (Applicant Value)"},
        )
        fig_shap.update_layout(height=390, margin=dict(l=10, r=20, t=20, b=20), legend_title_text="")
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_shap_text:
        st.subheader("Top Explainable Decision Reasons (SHAP)")
        for r in shap_reasons[:6]:
            css_cls = "reason-pos" if r["shap_value"] >= 0 else "reason-neg"
            icon = "🔺" if r["shap_value"] >= 0 else "🟢"
            st.markdown(
                f"""<div class="{css_cls}">{icon} <strong>{r['display_name']}</strong>: {r['explanation']}</div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("🧪 Interactive What-If Credit Utilization Sensitivity Curve")
    util_grid = np.linspace(0.02, 1.30, 25)
    sim_df = pd.concat([applicant_df] * len(util_grid), ignore_index=True)
    sim_df["RevolvingUtilizationOfUnsecuredLines"] = util_grid
    sim_scored = predict_credit_risk(
        sim_df, bundle, low_threshold=low_threshold, medium_threshold=medium_threshold
    )
    fig_sim = go.Figure()
    fig_sim.add_trace(
        go.Scatter(
            x=util_grid * 100,
            y=sim_scored["predicted_pd"] * 100,
            mode="lines+markers",
            name="Simulated Default Probability (%)",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig_sim.add_hline(
        y=low_threshold * 100,
        line_dash="dash",
        line_color="#10b981",
        annotation_text="Low/Medium Cutoff",
    )
    fig_sim.add_hline(
        y=medium_threshold * 100,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Medium/High Cutoff",
    )
    fig_sim.update_layout(
        height=320,
        xaxis_title="Simulated Revolving Credit Utilization (%)",
        yaxis_title="Predicted Default Probability (%)",
        margin=dict(l=20, r=20, t=25, b=20),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

with tab_models:
    st.subheader("Model Benchmark Comparison (Optuna Tuned)")
    metrics_df = pd.DataFrame(bundle["metrics_table"])
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["ROC_AUC", "Gini", "KS_Statistic", "Precision", "Recall", "F1_Score"],
            color="#d1fae5",
        ),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    roc_img = FIGURES_DIR / "05_model_roc_and_ks_comparison.png"
    tier_img = FIGURES_DIR / "06_risk_tier_separation.png"
    shap_img = FIGURES_DIR / "07_shap_global_summary.png"
    eda_img = FIGURES_DIR / "01_target_and_age_default_rate.png"

    if roc_img.exists():
        c1.image(str(roc_img), caption="ROC Curves & KS / Gini Benchmark", use_container_width=True)
    if tier_img.exists():
        c2.image(str(tier_img), caption="Risk Tier Volume & Observed Default Rate", use_container_width=True)

    c3, c4 = st.columns(2)
    if shap_img.exists():
        c3.image(str(shap_img), caption="Global SHAP Feature Importance", use_container_width=True)
    if eda_img.exists():
        c4.image(str(eda_img), caption="Portfolio Target & Age Cohort Default Rates", use_container_width=True)

with tab_sql:
    st.subheader("SQLite Credit Risk Analysis Queries (Segments, Window Functions & Joins)")
    sql_queries = parse_named_queries()
    sql_results = run_sql_analysis(SQLITE_DB_PATH)

    selected_q = st.selectbox(
        "Select SQL Analysis Query",
        options=list(sql_queries.keys()),
        format_func=lambda k: k.replace("_", " ").title(),
    )
    col_sql_code, col_sql_table = st.columns([1.0, 1.25])
    with col_sql_code:
        st.markdown("**SQL Query Definition (`sql/analysis_queries.sql`)**")
        st.code(sql_queries[selected_q], language="sql")
    with col_sql_table:
        st.markdown("**Query Output Table**")
        st.dataframe(sql_results[selected_q], use_container_width=True, height=360)

with tab_bi:
    st.subheader("Power BI Dashboard Exports & Customer Drill-Down Explorer")
    pred_csv = DASHBOARD_DIR / "predictions_export.csv"
    seg_csv = DASHBOARD_DIR / "segment_summary.csv"

    if pred_csv.exists() and seg_csv.exists():
        df_pred = pd.read_csv(pred_csv)
        df_seg = pd.read_csv(seg_csv)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Scored Borrowers", f"{len(df_pred):,}")
        b2.metric("Low Risk Share", f"{(df_pred['risk_tier'] == 'Low').mean() * 100:.1f}%")
        b3.metric("Medium Risk Share", f"{(df_pred['risk_tier'] == 'Medium').mean() * 100:.1f}%")
        b4.metric("High Risk Share", f"{(df_pred['risk_tier'] == 'High').mean() * 100:.1f}%")

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇️ Download Customer Predictions CSV (`predictions_export.csv`)",
            data=pred_csv.read_bytes(),
            file_name="predictions_export.csv",
            mime="text/csv",
        )
        dl2.download_button(
            "⬇️ Download Segment Summary CSV (`segment_summary.csv`)",
            data=seg_csv.read_bytes(),
            file_name="segment_summary.csv",
            mime="text/csv",
        )

        st.markdown("#### Customer Drill-Down Preview (Top 100 Records)")
        tier_filter = st.multiselect(
            "Filter by Risk Tier",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
        st.dataframe(
            df_pred[df_pred["risk_tier"].isin(tier_filter)].head(100),
            use_container_width=True,
        )
