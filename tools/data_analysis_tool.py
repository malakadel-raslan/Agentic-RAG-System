"""
Tool #4: analyze_data
For questions that need statistics over tabular data (CSV files placed in
data/documents/). Deliberately constrained to a few safe operations rather
than arbitrary code execution.
"""
import os
import pandas as pd
from langchain_core.tools import tool
from config import DOCUMENTS_DIR

_VALID_OPS = {"describe", "mean", "sum", "max", "min", "count", "correlation"}


@tool
def analyze_data(csv_filename: str, operation: str, column: str = "") -> str:
    """
    Run basic statistical analysis on a CSV file from the knowledge base.
    - csv_filename: name of the CSV file (must exist in the documents folder)
    - operation: one of "describe", "mean", "sum", "max", "min", "count", "correlation"
    - column: (optional) specific column to analyze; leave blank to analyze all
      numeric columns, or required for "correlation" as "col_a,col_b".
    Use this instead of guessing numbers when the question involves data
    stored in a spreadsheet/CSV in the knowledge base.
    """
    path = os.path.join(DOCUMENTS_DIR, csv_filename)
    if not os.path.exists(path):
        return f"File not found: {csv_filename}. Check the exact filename in the knowledge base."

    if operation not in _VALID_OPS:
        return f"Unsupported operation '{operation}'. Choose from: {sorted(_VALID_OPS)}"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"Could not read CSV: {e}"

    try:
        if operation == "describe":
            return df.describe(include="all").to_string()
        if operation == "correlation":
            cols = [c.strip() for c in column.split(",")]
            if len(cols) != 2:
                return "For correlation, provide column as 'col_a,col_b'."
            return str(df[cols[0]].corr(df[cols[1]]))

        target = df[column] if column else df.select_dtypes(include="number")
        result = getattr(target, operation)()
        return str(result)
    except Exception as e:
        return f"Error during analysis: {e}"
