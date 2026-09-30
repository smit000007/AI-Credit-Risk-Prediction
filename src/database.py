"""SQLite database management and SQL analysis runner."""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict
import pandas as pd

from src.config import SQLITE_DB_PATH, SQL_DIR
from src.data_generator import generate_auxiliary_tables

logger = logging.getLogger(__name__)


def init_sqlite_database(
    df_applicants: pd.DataFrame,
    db_path: Path = SQLITE_DB_PATH,
    schema_path: Path = SQL_DIR / "schema.sql",
) -> Path:
    """
    Create and populate the SQLite database with `applicants`, `loan_applications`,
    and `regional_macro` tables.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    df_loans, df_regions = generate_auxiliary_tables(df_applicants)

    with sqlite3.connect(db_path) as conn:
        if schema_path.exists():
            schema_sql = schema_path.read_text(encoding="utf-8")
            conn.executescript(schema_sql)

        df_applicants.to_sql("applicants", conn, if_exists="replace", index=False)
        df_regions.to_sql("regional_macro", conn, if_exists="replace", index=False)
        df_loans.to_sql("loan_applications", conn, if_exists="replace", index=False)

    logger.info(
        f"Populated SQLite database at {db_path} with {len(df_applicants):,} applicants, "
        f"{len(df_loans):,} loan applications, and {len(df_regions)} regions."
    )
    return db_path


def parse_named_queries(queries_path: Path = SQL_DIR / "analysis_queries.sql") -> Dict[str, str]:
    """Parse named SQL queries tagged with `-- [QUERY: name]`."""
    raw_sql = queries_path.read_text(encoding="utf-8")
    pattern = r"--\s*\[QUERY:\s*([a-zA-Z0-9_]+)\]\s*\n(.*?)(?=(?:--\s*\[QUERY:)|\Z)"
    matches = re.findall(pattern, raw_sql, flags=re.DOTALL)
    queries = {}
    for name, sql_body in matches:
        clean_sql = sql_body.strip()
        if clean_sql:
            queries[name] = clean_sql
    return queries


def run_sql_analysis(
    db_path: Path = SQLITE_DB_PATH,
    queries_path: Path = SQL_DIR / "analysis_queries.sql",
) -> Dict[str, pd.DataFrame]:
    """Execute all SQL analysis queries against the SQLite database and return DataFrames."""
    queries = parse_named_queries(queries_path)
    results: Dict[str, pd.DataFrame] = {}

    with sqlite3.connect(db_path) as conn:
        for name, sql in queries.items():
            df_res = pd.read_sql_query(sql, conn)
            results[name] = df_res
            logger.info(f"Executed SQL query '{name}' -> {len(df_res)} rows")

    return results


def load_applicants_from_db(db_path: Path = SQLITE_DB_PATH) -> pd.DataFrame:
    """Read the full applicants table joined with loan application metadata from SQLite."""
    query = """
    SELECT
        a.*,
        l.Region,
        l.LoanPurpose,
        l.EmploymentStatus,
        l.RequestedLoanAmount,
        l.LoanTermMonths,
        l.EstimatedAPR
    FROM applicants a
    LEFT JOIN loan_applications l ON a.ApplicantID = l.ApplicantID
    ORDER BY a.ApplicantID
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
