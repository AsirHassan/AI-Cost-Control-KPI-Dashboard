from __future__ import annotations

from io import BytesIO
import pandas as pd
import streamlit as st


def _normalize_frames(actual: pd.DataFrame, forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual.columns = actual.columns.str.strip()
    forecast.columns = forecast.columns.str.strip()

    actual["Posting Date"] = pd.to_datetime(actual["Posting Date"])
    actual["Accounting Month"] = actual["Accounting Month"].astype(int)
    actual["Currency"] = actual["Currency"].astype(str)

    forecast["Accounting Month"] = forecast["Accounting Month"].astype(int)
    forecast["Currency"] = forecast["Currency"].astype(str)

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
