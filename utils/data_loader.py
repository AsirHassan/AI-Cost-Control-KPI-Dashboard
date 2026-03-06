from __future__ import annotations

from io import BytesIO
import re
import pandas as pd
import streamlit as st


ACTUAL_COLUMN_ALIASES = {
    "Posting Date": ["postingdate", "date", "transactiondate", "documentdate"],
    "Accounting Month": [
        "accountingmonth",
        "month",
        "fiscalmonth",
        "yearmonth",
        "period",
    ],
    "Department": ["department", "dept", "function"],
    "G/L Group Identifier": [
        "glgroupidentifier",
        "glgroupid",
        "glgroupnumber",
    ],
    "G/L Group Name": [
        "glgroupname",
        "groupname",
        "glname",
    ],
    "Amount": [
        "amount",
        "actual",
        "actualcost",
        "actualamount",
        "totalactual",
        "value",
    ],
    "Currency": ["currency", "currencycode", "curr"],
}

FORECAST_COLUMN_ALIASES = {
    "Accounting Month": ACTUAL_COLUMN_ALIASES["Accounting Month"],
    "Department": ACTUAL_COLUMN_ALIASES["Department"],
    "G/L Group Identifier": ACTUAL_COLUMN_ALIASES["G/L Group Identifier"],
    "G/L Group Name": ACTUAL_COLUMN_ALIASES["G/L Group Name"],
    "Currency": ACTUAL_COLUMN_ALIASES["Currency"],
    "Total Cost": [
        "totalcost",
        "total_cost",
        "forecast",
        "forecastcost",
        "forecastamount",
        "budget",
        "plannedcost",
        "cost",
    ],
}


def _normalize_col_token(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _rename_with_aliases(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    token_to_col = {_normalize_col_token(c): c for c in df.columns}
    rename_map: dict[str, str] = {}

    for target, alias_tokens in aliases.items():
        target_token = _normalize_col_token(target)
        candidates = [target_token] + [_normalize_col_token(a) for a in alias_tokens]

        matched_col = None
        for alias in candidates:
            matched_col = token_to_col.get(alias)
            if matched_col is not None:
                break

            # Fallback: allow suffix/prefix variants such as "totalcosteur"
            # for sufficiently specific aliases only.
            if len(alias) >= 5:
                prefix_matches = [col for tok, col in token_to_col.items() if tok.startswith(alias)]
                if len(prefix_matches) == 1:
                    matched_col = prefix_matches[0]
                    break

        if matched_col is not None and matched_col != target:
            rename_map[matched_col] = target

    return df.rename(columns=rename_map)


def _require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _normalize_key_text(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace({"nan": pd.NA, "None": pd.NA, "NaT": pd.NA})
    # Normalize spreadsheet numeric-like IDs (e.g., 1001.0 -> 1001)
    out = out.str.replace(r"\.0+$", "", regex=True)
    return out


def _parse_month_token(value: object) -> float:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return float("nan")

    # YYYY-MM or YYYY/MM
    m = re.match(r"^\d{4}[-/](\d{1,2})$", text)
    if m:
        month = int(m.group(1))
        return float(month) if 1 <= month <= 12 else float("nan")

    # YYYYMM
    m = re.match(r"^\d{6}$", text)
    if m:
        month = int(text[-2:])
        return float(month) if 1 <= month <= 12 else float("nan")

    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return float("nan")
    return float(dt.month)


def _coerce_accounting_month(series: pd.Series) -> pd.Series:
    # Accept both numeric month values (e.g., 5) and date-like values (e.g., 2024-05).
    numeric = pd.to_numeric(series, errors="coerce")
    normalized = numeric.copy()

    # Handle numeric YYYYMM (e.g., 202406) by mapping to month component.
    numeric_int = pd.to_numeric(series, errors="coerce").astype("Int64")
    yyyymm_mask = (
        numeric_int.notna()
        & (numeric_int >= 190001)
        & (numeric_int <= 210012)
        & ((numeric_int % 100).between(1, 12))
    )
    if yyyymm_mask.any():
        normalized.loc[yyyymm_mask] = (numeric_int.loc[yyyymm_mask] % 100).astype(float)

    unresolved = normalized.isna()
    if unresolved.any():
        normalized.loc[unresolved] = series.loc[unresolved].apply(_parse_month_token)

    if normalized.isna().any():
        bad_values = (
            series[normalized.isna()]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        sample = bad_values[:3]
        raise ValueError(
            "Unsupported values in 'Accounting Month'. "
            f"Examples: {sample}. Use month numbers (1-12) or date-like values such as 2024-05."
        )

    return normalized.astype(int)


def _normalize_frames(actual: pd.DataFrame, forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual.columns = actual.columns.str.strip()
    forecast.columns = forecast.columns.str.strip()

    actual = _rename_with_aliases(actual, ACTUAL_COLUMN_ALIASES)
    forecast = _rename_with_aliases(forecast, FORECAST_COLUMN_ALIASES)

    _require_columns(
        actual,
        [
            "Posting Date",
            "Accounting Month",
            "Department",
            "G/L Group Identifier",
            "G/L Group Name",
            "Amount",
            "Currency",
        ],
        "Actual_Data",
    )
    _require_columns(
        forecast,
        [
            "Accounting Month",
            "Department",
            "G/L Group Identifier",
            "G/L Group Name",
            "Total Cost",
            "Currency",
        ],
        "Forecast_Data",
    )

    key_cols = ["Department", "G/L Group Identifier", "G/L Group Name", "Currency"]
    for col in key_cols:
        actual[col] = _normalize_key_text(actual[col])
        forecast[col] = _normalize_key_text(forecast[col])

    actual["Posting Date"] = pd.to_datetime(actual["Posting Date"])
    actual["Accounting Month"] = _coerce_accounting_month(actual["Accounting Month"])
    forecast["Accounting Month"] = _coerce_accounting_month(forecast["Accounting Month"])

    actual["YearMonth"] = actual["Posting Date"].dt.to_period("M")
    return actual, forecast


@st.cache_data
def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = pd.read_excel(path, sheet_name="Actual_Data")
    forecast = pd.read_excel(path, sheet_name="Forecast_Data")
    return _normalize_frames(actual, forecast)


@st.cache_data
def load_data_from_bytes(file_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook = BytesIO(file_bytes)
    actual = pd.read_excel(workbook, sheet_name="Actual_Data")
    workbook.seek(0)
    forecast = pd.read_excel(workbook, sheet_name="Forecast_Data")
    return _normalize_frames(actual, forecast)
