-- ============================================================================
-- SQL Credit Risk Analysis Queries (SQLite)
-- Includes: Default Rate by Segment, Multi-Table JOINs, and Window Functions
-- ============================================================================

-- [QUERY: default_by_age_segment]
-- 1. Default rate, utilization, and income metrics segmented by Age Bucket
WITH age_cohorts AS (
    SELECT
        ApplicantID,
        SeriousDlqin2yrs,
        CASE
            WHEN age < 25 THEN '1. Under 25'
            WHEN age BETWEEN 25 AND 34 THEN '2. 25-34 (Early Career)'
            WHEN age BETWEEN 35 AND 49 THEN '3. 35-49 (Mid Career)'
            WHEN age BETWEEN 50 AND 64 THEN '4. 50-64 (Pre-Retirement)'
            ELSE '5. 65+ (Senior)'
        END AS age_segment,
         MIN(RevolvingUtilizationOfUnsecuredLines, 2.0) AS clean_utilization,
        COALESCE(MonthlyIncome, 5400.0) AS effective_income,
        CASE WHEN DebtRatio > 5.0 THEN 0.45 ELSE DebtRatio END AS clean_dti
    FROM applicants
    WHERE age >= 18
)
SELECT
    age_segment,
    COUNT(*) AS total_applicants,
    SUM(SeriousDlqin2yrs) AS total_defaults,
    ROUND(AVG(SeriousDlqin2yrs) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(clean_utilization) * 100.0, 2) AS avg_utilization_pct,
    ROUND(AVG(effective_income), 0) AS avg_monthly_income,
    ROUND(
        SUM(SeriousDlqin2yrs) * 100.0 / SUM(SUM(SeriousDlqin2yrs)) OVER (),
        2
    ) AS share_of_all_defaults_pct
FROM age_cohorts
GROUP BY age_segment
ORDER BY age_segment;

-- [QUERY: delinquency_and_utilization_matrix]
-- 2. Default rate by Delinquency Severity and Revolving Credit Utilization Tier
WITH risk_matrix AS (
    SELECT
        ApplicantID,
        SeriousDlqin2yrs,
        CASE
            WHEN NumberOfTimes90DaysLate > 0 AND NumberOfTimes90DaysLate < 90 THEN '3. Severe (90+ DPD)'
            WHEN ("NumberOfTime30-59DaysPastDueNotWorse" > 0 OR "NumberOfTime60-89DaysPastDueNotWorse" > 0)
                 AND "NumberOfTime30-59DaysPastDueNotWorse" < 90 THEN '2. Moderate (30-89 DPD)'
            ELSE '1. Clean (No Past Due)'
        END AS delinquency_status,
        CASE
            WHEN RevolvingUtilizationOfUnsecuredLines < 0.30 THEN 'Low (<30%)'
            WHEN RevolvingUtilizationOfUnsecuredLines < 0.70 THEN 'Medium (30-70%)'
            WHEN RevolvingUtilizationOfUnsecuredLines <= 1.00 THEN 'High (70-100%)'
            ELSE 'Over-Limit (>100%)'
        END AS utilization_band
    FROM applicants
)
SELECT
    delinquency_status,
    utilization_band,
    COUNT(*) AS applicant_count,
    SUM(SeriousDlqin2yrs) AS default_count,
    ROUND(AVG(SeriousDlqin2yrs) * 100.0, 2) AS default_rate_pct,
    RANK() OVER (ORDER BY AVG(SeriousDlqin2yrs) DESC) AS risk_rank
FROM risk_matrix
GROUP BY delinquency_status, utilization_band
ORDER BY delinquency_status, default_rate_pct DESC;

-- [QUERY: window_utilization_deciles]
-- 3. Window Functions: NTILE(10) Utilization Deciles with Cumulative Default Capture (KS/Lift style)
WITH clean_base AS (
    SELECT
        ApplicantID,
        SeriousDlqin2yrs,
        CASE
            WHEN RevolvingUtilizationOfUnsecuredLines > 2.5 THEN 1.5
            ELSE RevolvingUtilizationOfUnsecuredLines
        END AS util_capped,
        COALESCE(MonthlyIncome, 5400.0) AS monthly_income,
        NTILE(10) OVER (
            ORDER BY
                CASE
                    WHEN RevolvingUtilizationOfUnsecuredLines > 2.5 THEN 1.5
                    ELSE RevolvingUtilizationOfUnsecuredLines
                END DESC
        ) AS utilization_decile
    FROM applicants
    WHERE age >= 18
),
decile_summary AS (
    SELECT
        utilization_decile,
        COUNT(*) AS borrowers,
        SUM(SeriousDlqin2yrs) AS defaults,
        ROUND(MIN(util_capped) * 100.0, 1) AS min_util_pct,
        ROUND(MAX(util_capped) * 100.0, 1) AS max_util_pct,
        ROUND(AVG(SeriousDlqin2yrs) * 100.0, 2) AS decile_default_rate_pct
    FROM clean_base
    GROUP BY utilization_decile
)
SELECT
    utilization_decile,
    borrowers,
    defaults,
    min_util_pct,
    max_util_pct,
    decile_default_rate_pct,
    ROUND(
        SUM(defaults) OVER (ORDER BY utilization_decile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        * 100.0 / SUM(defaults) OVER (),
        2
    ) AS cumulative_defaults_captured_pct
FROM decile_summary
ORDER BY utilization_decile;

-- [QUERY: regional_loan_purpose_joins]
-- 4. Multi-Table JOINs + Partitioned Window Functions across Applicants, Loan Applications, and Regional Macro
WITH joined_loans AS (
    SELECT
        a.ApplicantID,
        a.SeriousDlqin2yrs,
        COALESCE(a.MonthlyIncome, r.MedianHouseholdIncome / 12.0) AS effective_monthly_income,
        l.Region,
        l.LoanPurpose,
        l.EmploymentStatus,
        l.RequestedLoanAmount,
        l.EstimatedAPR,
        r.RegionalUnemploymentRate,
        r.CostOfLivingIndex
    FROM applicants a
    INNER JOIN loan_applications l ON a.ApplicantID = l.ApplicantID
    INNER JOIN regional_macro r ON l.Region = r.Region
    WHERE a.age >= 18
),
purpose_region_stats AS (
    SELECT
        Region,
        LoanPurpose,
        RegionalUnemploymentRate,
        COUNT(*) AS application_volume,
        ROUND(AVG(RequestedLoanAmount), 0) AS avg_loan_amount,
        ROUND(AVG(EstimatedAPR), 2) AS avg_apr_pct,
        ROUND(AVG(SeriousDlqin2yrs) * 100.0, 2) AS segment_default_rate_pct
    FROM joined_loans
    GROUP BY Region, LoanPurpose, RegionalUnemploymentRate
)
SELECT
    Region,
    LoanPurpose,
    RegionalUnemploymentRate,
    application_volume,
    avg_loan_amount,
    avg_apr_pct,
    segment_default_rate_pct,
    ROUND(
        AVG(segment_default_rate_pct) OVER (PARTITION BY Region),
        2
    ) AS regional_avg_default_rate_pct,
    ROUND(
        segment_default_rate_pct - AVG(segment_default_rate_pct) OVER (PARTITION BY Region),
        2
    ) AS spread_vs_regional_avg_pct,
    DENSE_RANK() OVER (
        PARTITION BY Region ORDER BY segment_default_rate_pct DESC
    ) AS purpose_risk_rank_in_region
FROM purpose_region_stats
ORDER BY Region, purpose_risk_rank_in_region;
