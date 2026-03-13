import io
import logging
import re
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# GPA column boundary detection patterns
# ---------------------------------------------------------------------------

_GPA_PATTERN = re.compile(
    r"\d+\.\d+\s*(and above|[-\u2013]\s*\d+\.\d+)",
    re.IGNORECASE,
)

_EMPLOYMENT_STATUSES = {
    "active",
    "terminated",
    "resigned",
    "leave of absence",
    "prn",
    "part time",
    "full time",
    "unknown",
}

# GPA text range → decimal midpoint for SQL DECIMAL(3,2) storage
_GPA_MIDPOINTS = {
    "3.5 and above": 3.75,
    "3.0 - 3.49":    3.25,
    "2.5 - 2.99":    2.75,
    "2.0 - 2.49":    2.25,
    "below 2.0":     1.75,
}


# ---------------------------------------------------------------------------
# GPA boundary fix
# ---------------------------------------------------------------------------

def _fix_gpa_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect rows where GPA and Employment Status columns are swapped
    (GPA contains a status string, employment_status contains a GPA range)
    and swap them back.

    This is a row-level fix — only affected rows are modified.
    """
    gpa_col = "gpa"
    emp_col = "employment_status"

    def _is_gpa_like(val) -> bool:
        return bool(_GPA_PATTERN.search(str(val))) if pd.notna(val) else False

    def _is_status_like(val) -> bool:
        return str(val).strip().lower() in _EMPLOYMENT_STATUSES if pd.notna(val) else False

    needs_fix = (
        df[gpa_col].apply(_is_status_like) &
        df[emp_col].apply(_is_gpa_like)
    )
    fixed_count = int(needs_fix.sum())
    if fixed_count > 0:
        logging.warning("Vizient: boundary-swapped GPA/employment_status on %d rows", fixed_count)
        df.loc[needs_fix, [gpa_col, emp_col]] = (
            df.loc[needs_fix, [emp_col, gpa_col]].values
        )
    else:
        logging.info("Vizient: no GPA/employment_status boundary swap needed")
    return df


# ---------------------------------------------------------------------------
# GPA parsing
# ---------------------------------------------------------------------------

def _parse_gpa(val) -> Optional[float]:
    """Convert a GPA text range to a decimal midpoint. Returns None if unrecognized."""
    if pd.isna(val):
        return None
    key = str(val).strip().lower()
    for pattern, midpoint in _GPA_MIDPOINTS.items():
        if pattern in key:
            return midpoint
    # Try parsing as a raw float (some rows may have actual numeric GPAs)
    try:
        return float(key)
    except ValueError:
        logging.warning("Vizient: unrecognized GPA value: '%s'", val)
        return None


# ---------------------------------------------------------------------------
# Column rename map
# ---------------------------------------------------------------------------

_COLUMN_RENAME = {
    "Organization":             "organization",
    "Cohort Date":              "cohort_date",
    "Nurse ID":                 "nurse_id",
    "Education":                "education_school",
    "Nursing Degree Received":  "nursing_degree",
    "GPA":                      "gpa",
    "Employment Status":        "employment_status",
    "Termination":              "termination_raw",
    "Termination Date":         "termination_date",
    "Tenure":                   "tenure_months",
}

_OUTPUT_COLS = [
    "nurse_id",
    "organization",
    "cohort_date",
    "education_school",
    "nursing_degree",
    "gpa",
    "employment_status",
    "is_terminated",
    "termination_date",
    "tenure_months",
]


# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------

def clean_vizient(file_bytes: bytes) -> pd.DataFrame:
    """
    Accept raw Excel bytes. Return a clean DataFrame whose columns
    match fact_nurse exactly.

    Logic App POSTs the file as application/octet-stream; this function
    receives those raw bytes and returns the cleaned data. It never
    touches Azure SQL — that is the Logic App's job.
    """
    # Step 1: Read Excel — dtype=str so we control all type coercion
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Duplicate Vizent for Data Manip",
        dtype=str,
        engine="openpyxl",
    )

    # Step 2: Strip \t (and surrounding whitespace) from every cell and column name
    # Every cell in the Vizient export has a trailing \t appended
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()

    logging.info("Vizient: read %d rows", len(df))

    # Step 3: Rename columns to snake_case
    df = df.rename(columns=_COLUMN_RENAME)

    # Step 4: Fix GPA/employment_status column boundary bleed
    df = _fix_gpa_boundary(df)

    # Step 5: Parse GPA text ranges → decimal midpoints
    df["gpa"] = df["gpa"].apply(_parse_gpa)

    # Step 6: Parse dates
    df["cohort_date"]      = pd.to_datetime(df["cohort_date"],      format="%m/%d/%Y", errors="coerce").dt.date
    df["termination_date"] = pd.to_datetime(df["termination_date"], format="%m/%d/%Y", errors="coerce").dt.date

    # Step 7: Derive is_terminated (1 = Yes, 0 = No/blank)
    df["is_terminated"] = (
        df["termination_raw"].str.strip().str.lower() == "yes"
    ).astype(int)

    # Step 8: Tenure is stored as decimal years (e.g. 1.5 = 18 months).
    # Convert to integer months: multiply by 12 and round.
    df["tenure_months"] = (
        pd.to_numeric(df["tenure_months"].str.strip(), errors="coerce") * 12
    ).round().astype("Int64")
    active_null_tenure = df[(df["is_terminated"] == 0) & df["tenure_months"].isna()]
    if len(active_null_tenure) > 0:
        logging.warning(
            "Vizient: %d active nurses have null tenure_months", len(active_null_tenure)
        )

    # Step 9: Select and order output columns
    return df[_OUTPUT_COLS].copy()
