"""Unit tests for SQLite database ingestion and SQL analysis queries."""

from src.config import SQLITE_DB_PATH
from src.database import load_applicants_from_db, parse_named_queries, run_sql_analysis


def test_sqlite_queries_execute_and_return_expected_columns():
    """Verify all 4 named SQL queries (segments, matrix, window deciles, joins) execute cleanly."""
    assert SQLITE_DB_PATH.exists(), "SQLite database should exist at data/credit_risk.db"

    queries = parse_named_queries()
    expected_query_names = {
        "default_by_age_segment",
        "delinquency_and_utilization_matrix",
        "window_utilization_deciles",
        "regional_loan_purpose_joins",
    }
    assert expected_query_names.issubset(set(queries.keys()))

    results = run_sql_analysis(SQLITE_DB_PATH)
    for name in expected_query_names:
        df_res = results[name]
        assert len(df_res) > 0, f"SQL query '{name}' returned 0 rows"

    # Window deciles must have exactly 10 deciles and cumulative capture reaching 100%
    deciles_df = results["window_utilization_deciles"]
    assert len(deciles_df) == 10
    assert abs(float(deciles_df["cumulative_defaults_captured_pct"].iloc[-1]) - 100.0) < 0.5

    # Joined applicant table has regional metadata
    joined_df = load_applicants_from_db(SQLITE_DB_PATH)
    assert "Region" in joined_df.columns
    assert "LoanPurpose" in joined_df.columns
    assert len(joined_df) >= 1000
