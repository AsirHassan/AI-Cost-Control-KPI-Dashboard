import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path):
    actual = pd.read_excel(path, sheet_name="Actual_Data")
    forecast = pd.read_excel(path, sheet_name="Forecast_Data")

    actual.columns = actual.columns.str.strip()
    forecast.columns = forecast.columns.str.strip()

    actual["Posting Date"] = pd.to_datetime(actual["Posting Date"])
    actual["Accounting Month"] = actual["Accounting Month"].astype(int)
    actual["Currency"] = actual["Currency"].astype(str)

    forecast["Accounting Month"] = forecast["Accounting Month"].astype(int)
    forecast["Currency"] = forecast["Currency"].astype(str)

    actual["YearMonth"] = actual["Posting Date"].dt.to_period("M")

    return actual, forecast
