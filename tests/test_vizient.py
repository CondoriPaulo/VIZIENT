"""
Tests for cleaners/vizient_cleaner.py

Fixtures are synthetic — no real patient or employee data.
Run with: pytest tests/ -v --tb=short
"""
import io
import logging
import os

import pandas as pd
import pytest

# Allow imports from project root when running pytest from VIZIENT/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cleaners.vizient_cleaner import clean_vizient, _parse_gpa, _fix_gpa_boundary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "vizient_sample.xlsx")

OUTPUT_COLS = {
    "nurse_id", "organization", "cohort_date", "education_school",
    "nursing_degree", "gpa", "employment_status", "is_terminated",
    "termination_date", "tenure_months",
}


def _fixture_bytes() -> bytes:
    with open(FIXTURE, "rb") as f:
        return f.read()


def _run() -> pd.DataFrame:
    return clean_vizient(_fixture_bytes())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tab_stripping():
    """No cell in the output should contain a tab character."""
    df = _run()
    for col in df.select_dtypes(include="object").columns:
        for val in df[col].dropna():
            assert "\t" not in str(val), f"Tab found in column '{col}': '{val}'"


def test_nurse_id_stripped():
    """Nurse IDs come in with trailing \\t; output should be clean."""
    df = _run()
    assert "N001" in df["nurse_id"].values


def test_gpa_boundary_fix():
    """
    Row N002 has GPA='Full Time' and employment_status='3.0 - 3.49' (bleed).
    After fix: gpa should parse to 3.25 and employment_status should be 'Full Time'.
    """
    df = _run()
    row = df[df["nurse_id"] == "N002"]
    assert row.iloc[0]["gpa"] == 3.25
    assert row.iloc[0]["employment_status"] == "Full Time"


def test_gpa_no_fix_needed():
    """Row N001 has correct GPA and status — should not be swapped."""
    df = _run()
    row = df[df["nurse_id"] == "N001"]
    assert row.iloc[0]["gpa"] == 3.75
    assert row.iloc[0]["employment_status"] == "Full Time"


def test_gpa_parse_3_5_and_above():
    assert _parse_gpa("3.5 and above") == 3.75


def test_gpa_parse_3_0_to_3_49():
    assert _parse_gpa("3.0 - 3.49") == 3.25


def test_gpa_parse_2_5_to_2_99():
    assert _parse_gpa("2.5 - 2.99") == 2.75


def test_gpa_parse_below_2_0():
    assert _parse_gpa("below 2.0") == 1.75


def test_gpa_parse_unknown_returns_none(caplog):
    """Unrecognized GPA value should return None and log a WARNING."""
    with caplog.at_level(logging.WARNING, logger="root"):
        result = _parse_gpa("honors_gpa")
    assert result is None
    assert "honors_gpa" in caplog.text


def test_gpa_parse_none():
    assert _parse_gpa(None) is None


def test_is_terminated_yes():
    """Termination='Yes' → is_terminated=1."""
    df = _run()
    row = df[df["nurse_id"] == "N003"]
    assert row.iloc[0]["is_terminated"] == 1


def test_is_terminated_no():
    """Termination='No' → is_terminated=0."""
    df = _run()
    row = df[df["nurse_id"] == "N001"]
    assert row.iloc[0]["is_terminated"] == 0


def test_termination_date_parsed():
    """Termination Date '11/30/2023' should parse to datetime.date(2023, 11, 30)."""
    import datetime
    df = _run()
    row = df[df["nurse_id"] == "N003"]
    assert row.iloc[0]["termination_date"] == datetime.date(2023, 11, 30)


def test_cohort_date_parsed():
    """Cohort Date '08/28/2023' should parse to datetime.date(2023, 8, 28)."""
    import datetime
    df = _run()
    row = df[df["nurse_id"] == "N001"]
    assert row.iloc[0]["cohort_date"] == datetime.date(2023, 8, 28)


def test_tenure_parsed_to_months():
    """Tenure stored as decimal years (1.5) should be converted to integer months (18)."""
    df = _run()
    row = df[df["nurse_id"] == "N001"]
    assert row.iloc[0]["tenure_months"] == 18  # 1.5 years * 12 = 18 months


def test_active_null_tenure_logged(caplog):
    """Active nurse (N004) with blank tenure should trigger a WARNING."""
    with caplog.at_level(logging.WARNING, logger="root"):
        df = _run()
    assert "active nurses have null tenure_months" in caplog.text


def test_output_columns_match_schema():
    """Output DataFrame has exactly the columns that match fact_nurse."""
    df = _run()
    assert set(df.columns) == OUTPUT_COLS


def test_no_nan_strings_in_output():
    """NaN values should be proper nulls, not the string 'nan'."""
    df = _run()
    rows = df.where(pd.notnull(df), None).to_dict(orient="records")
    for row in rows:
        for val in row.values():
            assert val != "nan", f"Found string 'nan' in output row: {row}"


def test_all_10_rows_returned():
    """All 10 rows in the fixture should be returned (no blank-row dropping for Vizient)."""
    df = _run()
    assert len(df) == 10
