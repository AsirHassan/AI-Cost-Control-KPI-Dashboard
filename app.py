from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st
import numpy as np


sys.path.append(os.path.dirname(__file__))

from utils.data_loader import load_data, load_data_from_bytes
from utils.aggregations import aggregate_monthly, top_gl_groups, aggregate_with_forecast
from utils.anomaly_detection import detect_anomalies
from utils.charts import (
    monthly_trend,
    top_gl_bar,
    anomaly_trend_simple,
    variance_waterfall,
    mom_change_bar,
)
from utils.ai_engine import generate_finance_insights, LLMConfig

# This is our Page config / header section
st.set_page_config(
    page_title="Cost Control Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
# 💼 Cost Control Dashboard  
**Expense transparency, forecasting & anomaly detection**  
*Expense transparency, forecasting & anomaly detection*
"""
)
st.divider()


# Data source (uploaded file or default sample)
DEFAULT_DATA_PATH = "data/finance_case_study_synthetic_data_TH-2.xlsx"
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"],
    help="Workbook must contain sheets: Actual_Data and Forecast_Data",
)

try:
    if uploaded_file is not None:
        actual_df, forecast_df = load_data_from_bytes(uploaded_file.getvalue())
        st.sidebar.caption(f"Using uploaded file: {uploaded_file.name}")
    else:
        actual_df, forecast_df = load_data(DEFAULT_DATA_PATH)
        st.sidebar.caption("Using default sample dataset")
except Exception as e:
    st.error(
        "Could not load the selected file. Ensure it has sheets named "
        "'Actual_Data' and 'Forecast_Data'."
    )
    st.error(f"Load error: {e}")
    st.stop()

required_actual_cols = {
    "Posting Date",
    "Accounting Month",
    "Department",
    "G/L Group Identifier",
    "G/L Group Name",
    "Amount",
    "Currency",
}
required_forecast_cols = {
    "Accounting Month",
    "Department",
    "G/L Group Identifier",
    "G/L Group Name",
    "Total Cost",
    "Currency",
}

missing_actual = sorted(required_actual_cols - set(actual_df.columns))
missing_forecast = sorted(required_forecast_cols - set(forecast_df.columns))

if missing_actual:
    st.error(f"Actual_Data is missing columns: {missing_actual}")
    st.stop()

if missing_forecast:
    st.error(f"Forecast_Data is missing columns: {missing_forecast}")
    st.stop()


# EUR-only validation (per your dataset)
if actual_df["Currency"].nunique() != 1 or actual_df["Currency"].iloc[0] != "EUR":
    st.error("Actual data is not EUR-only. Please provide an EUR-only dataset.")
    st.stop()

if forecast_df["Currency"].nunique() != 1 or forecast_df["Currency"].iloc[0] != "EUR":
    st.error("Forecast data is not EUR-only. Please provide an EUR-only dataset.")
    st.stop()



# This is the sidebar filters and navigation section
st.sidebar.divider()
st.sidebar.header("Filters")

department = st.sidebar.selectbox(
    "Department",
    sorted(actual_df["Department"].dropna().unique()),
)

# Tabs / Navigation section
st.sidebar.divider()
st.sidebar.subheader("Navigation")
st.sidebar.caption("Cost Control • Finance")


page = st.sidebar.radio(
    "Go to",
    [
        "📊 Cost Overview",
        "📈 Forecast & Dynamic Risk Signals",
        "🚨 Anomaly Review Workflow",
        "🧠 AI Generated Finance Insights",
    ],
    index=0,
)

month_min = int(actual_df["Accounting Month"].min())
month_max = int(actual_df["Accounting Month"].max())

month_range = st.sidebar.slider(
    "Accounting Month",
    min_value=month_min,
    max_value=month_max,
    value=(month_min, month_max),
)

gl_groups = st.sidebar.multiselect(
    "G/L Group",
    sorted(actual_df["G/L Group Name"].dropna().unique()),
    default=sorted(actual_df["G/L Group Name"].dropna().unique()),
)

filtered_actual = actual_df[
    (actual_df["Department"] == department)
    & (actual_df["G/L Group Name"].isin(gl_groups))
    & (actual_df["Accounting Month"].between(month_range[0], month_range[1]))
].copy()

filtered_forecast = forecast_df[
    (forecast_df["Department"] == department)
    & (forecast_df["G/L Group Name"].isin(gl_groups))
    & (forecast_df["Accounting Month"].between(month_range[0], month_range[1]))
].copy()


# TAB 1: Cost Overview & KPI (Task 1)
if page == "📊 Cost Overview":
    st.markdown("## KPI Dashboard — Cost Overview")

    if filtered_actual.empty:
        st.warning("No rows match the selected filters.")
    else:
        # 1) Build MONTHLY actual vs forecast (fast + consistent)
        actual_m = (
            filtered_actual.groupby("Accounting Month", as_index=False)
            .agg(Actual=("Amount", "sum"))
        )

        forecast_m = (
            filtered_forecast.groupby("Accounting Month", as_index=False)
            .agg(Forecast=("Total Cost", "sum"))
        )

        monthly = (
            pd.merge(actual_m, forecast_m, on="Accounting Month", how="outer")
            .fillna(0.0)
            .sort_values("Accounting Month")
            .reset_index(drop=True)
        )

        monthly["Actual"] = pd.to_numeric(monthly["Actual"], errors="coerce").fillna(0.0).round(2)
        monthly["Forecast"] = pd.to_numeric(monthly["Forecast"], errors="coerce").fillna(0.0).round(2)

        monthly["Variance"] = (monthly["Actual"] - monthly["Forecast"]).round(2)
        monthly["Variance_Pct"] = np.where(
            monthly["Forecast"] != 0,
            (monthly["Variance"] / monthly["Forecast"]).round(4),
            0.0,
        )

        # 2) KPI Values (Totals across selected months)
        total_actual = float(monthly["Actual"].sum())
        total_forecast = float(monthly["Forecast"].sum())
        total_variance = float((total_actual - total_forecast))
        total_var_pct = (total_variance / total_forecast) if total_forecast != 0 else 0.0

        # Avg monthly actual (based on months that exist in the selection)
        month_count = int(monthly["Accounting Month"].nunique())
        avg_monthly_actual = float(monthly["Actual"].mean()) if month_count > 0 else 0.0

        # Worst/Best month by variance (Overspend = biggest positive; Underspend = most negative)
        worst_row = monthly.loc[monthly["Variance"].idxmax()] if len(monthly) else None
        best_row = monthly.loc[monthly["Variance"].idxmin()] if len(monthly) else None

        # MoM change: use latest month available in the selection
        latest_month = int(monthly["Accounting Month"].max()) if len(monthly) else None
        prev_month = latest_month - 1 if latest_month and (latest_month - 1) in set(monthly["Accounting Month"]) else None

        latest_actual = float(monthly.loc[monthly["Accounting Month"] == latest_month, "Actual"].iloc[0]) if latest_month else 0.0
        prev_actual = float(monthly.loc[monthly["Accounting Month"] == prev_month, "Actual"].iloc[0]) if prev_month else None

        mom_change = (latest_actual - prev_actual) if prev_actual is not None else None
        mom_change_pct = ((mom_change / prev_actual) if (prev_actual not in (None, 0)) else None)

        # Risk months KPI: count months where absolute variance% >= threshold
        risk_threshold_pct = 0.10
        risk_months = int((np.abs(monthly["Variance_Pct"]) >= risk_threshold_pct).sum())

        # 3) Extra KPIs (Drivers / Coverage)
        gl_count = int(filtered_actual["G/L Group Name"].nunique())

        top_driver_series = (
            filtered_actual.groupby("G/L Group Name")["Amount"].sum()
            .sort_values(ascending=False)
        )
        top_driver_name = top_driver_series.index[0] if len(top_driver_series) else "—"
        top_driver_value = float(top_driver_series.iloc[0]) if len(top_driver_series) else 0.0

        # 4) KPI ROWS (optimized labels + no duplicate metrics)
        # Row 1: Totals
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Actual (EUR)", f"{total_actual:,.0f}")
        k2.metric("Total Forecast (EUR)", f"{total_forecast:,.0f}")
        k3.metric("Total Variance (EUR)", f"{total_variance:+,.0f}")
        k4.metric("Total Variance %", f"{total_var_pct:+.1%}")

        # Row 2: Monthly behavior / extremes
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Avg Monthly Actual (EUR)", f"{avg_monthly_actual:,.0f}")

        if worst_row is not None:
            k6.metric(
                "Worst Month (Overspend)",
                f"M{int(worst_row['Accounting Month'])}",
                delta=f"{float(worst_row['Variance']):+,.0f} EUR",
            )
        else:
            k6.metric("Worst Month (Overspend)", "—")

        if best_row is not None:
            k7.metric(
                "Best Month (Underspend)",
                f"M{int(best_row['Accounting Month'])}",
                delta=f"{float(best_row['Variance']):+,.0f} EUR",
            )
        else:
            k7.metric("Best Month (Underspend)", "—")

        if mom_change is None:
            k8.metric("MoM Change (Latest)", "No prior month")
        else:
            k8.metric(
                f"MoM Change (M{latest_month} vs M{prev_month})",
                f"{mom_change:+,.0f} EUR",
                delta=f"{mom_change_pct:+.1%}" if mom_change_pct is not None else None,
            )

        # Row 3: Coverage + drivers + risk
        k9, k10, k11, k12 = st.columns(4)
        k9.metric("Months Selected", f"{month_count}")
        k10.metric("Active G/L Groups", f"{gl_count}")
        k11.metric(f"Risk Months (|Var%| ≥ {risk_threshold_pct:.0%})", f"{risk_months}")
        k12.metric("Top Cost Driver (Actual)", top_driver_name, f"{top_driver_value:,.0f} EUR")

        st.divider()

        # 5) Existing charts + table (unchanged)
        monthly_agg = aggregate_monthly(filtered_actual)
        top_groups = top_gl_groups(filtered_actual)

        left, right = st.columns([2, 1])

        with left:
            st.plotly_chart(
                monthly_trend(monthly_agg, filtered_actual),
                use_container_width=True,
                key="chart_monthly_trend_tab1",
            )

        with right:
            st.plotly_chart(
                top_gl_bar(top_groups, value_col="Total_Actual", n=10),
                use_container_width=True,
                key="chart_top_gl_tab1",
            )

        st.subheader("Detailed Transactions")
        st.dataframe(filtered_actual, use_container_width=True)


# TAB 2: Forecast & Dynamic Risk Signals (Task 2)

elif page == "📈 Forecast & Dynamic Risk Signals":
    st.subheader("Forecast & Dynamic Risk Signals")

    st.info(
        """
The dashboard focuses on budget compliance (Actual vs Forecast) and flags exceptional months. Historical trends support the detection logic, while the charts remain straightforward.
"""
    )

    if filtered_actual.empty:
        st.warning("No rows match the selected filters.")
    else:
        merged = aggregate_with_forecast(filtered_actual, filtered_forecast)

        if merged.empty:
            st.warning("No merged actual/forecast data available for the selected filters.")
        else:
            groups = sorted(merged["G/L Group Name"].dropna().unique())

            selected_group = st.segmented_control(
                "Select G/L Group for Analysis",
                options=groups,
                default=groups[0] if groups else None,
                key="tab2_group",
            )


            group_merged = merged[merged["G/L Group Name"] == selected_group].copy()

            # Anomaly detection needs:
            # actual_df: Accounting Month, G/L Group Identifier, Total_Actual
            # forecast_df: Accounting Month, G/L Group Identifier, Total Cost
            actual_for_anom = group_merged[["Accounting Month", "G/L Group Identifier", "Total_Actual"]].copy()
            forecast_for_anom = group_merged[["Accounting Month", "G/L Group Identifier", "Total Cost"]].copy()

            anomaly_df = detect_anomalies(actual_for_anom, forecast_for_anom)

            if "Total Cost" not in anomaly_df.columns:
                anomaly_df = anomaly_df.merge(
                    forecast_for_anom,
                    on=["Accounting Month", "G/L Group Identifier"],
                    how="left",
                ).fillna({"Total Cost": 0.0})

            st.plotly_chart(
                anomaly_trend_simple(anomaly_df),
                use_container_width=True,
                key="chart_anomaly_simple_tab2",
            )

            st.plotly_chart(
                variance_waterfall(anomaly_df[["Accounting Month", "Total_Actual", "Total Cost"]]),
                use_container_width=True,
                key="chart_variance_waterfall_tab2"
            )

            st.subheader("Signals (Table)")
            show_cols = [
                "Accounting Month",
                "Total_Actual",
                "Total Cost",
                "Forecast_Variance",
                "Forecast_Variance_Pct",
                "MoM_Change",
                "MoM_Change_Pct",
                "Anomaly_Flag",
                "Anomaly_Type"
            ]
            show_cols = [c for c in show_cols if c in anomaly_df.columns]
            st.dataframe(
            anomaly_df[show_cols].style.format({
                "Total_Actual": "{:,.2f}",
                "Total Cost": "{:,.2f}",
                "Forecast_Variance": "{:,.2f}",
                "Forecast_Variance_Pct": "{:+.1%}",
                "MoM_Change": "{:+,.2f}",
                "MoM_Change_Pct": "{:+.1%}",
            }),
        use_container_width=True
        )


# TAB 3: Anomaly Review Workflow (Task 3)
elif page == "🚨 Anomaly Review Workflow":
    st.subheader("Anomaly Review Workflow")
    st.info(
        """
This tab shows the review workflow for detected cost exceptions. Anomalies are based on MoM movements in Actuals, supported by Forecast context."""
    )

    if filtered_actual.empty:
        st.warning("No rows match the selected filters.")
    else:
        merged = aggregate_with_forecast(filtered_actual, filtered_forecast)

        if merged.empty:
            st.warning("No merged actual/forecast data available for the selected filters.")
        else:
            groups = sorted(merged["G/L Group Name"].dropna().unique())

            selected_group = st.segmented_control(
                "Select G/L Group",
                options=groups,
                default=groups[0] if groups else None,
                key="tab3_group",
        )

            group_merged = merged[merged["G/L Group Name"] == selected_group].copy()

            actual_for_anom = group_merged[["Accounting Month", "G/L Group Identifier", "Total_Actual"]].copy()
            forecast_for_anom = group_merged[["Accounting Month", "G/L Group Identifier", "Total Cost"]].copy()

            anomaly_df = detect_anomalies(actual_for_anom, forecast_for_anom)

            if "Total Cost" not in anomaly_df.columns:
                anomaly_df = (
                    anomaly_df.merge(
                        forecast_for_anom,
                        on=["Accounting Month", "G/L Group Identifier"],
                        how="left",
                    )
                    .fillna({"Total Cost": 0.0})
                )

            st.plotly_chart(
                mom_change_bar(anomaly_df, title="MoM Change (Actual) — Review Focus"),
                use_container_width=True,
                key="chart_mom_change_tab3",
            )

            anomalies_only = anomaly_df[anomaly_df.get("Anomaly_Flag", "") == "Anomaly"].copy()

            if anomalies_only.empty:
                st.success("No anomalies detected for the selected G/L Group.")
                st.session_state["summary_df"] = pd.DataFrame()
            else:
                if "anomaly_workflow" not in st.session_state:
                    st.session_state.anomaly_workflow = {}

                records = []

                for _, row in anomalies_only.iterrows():
                    month = int(row["Accounting Month"])
                    key = f"{month}_{selected_group}"

                    if key not in st.session_state.anomaly_workflow:
                        st.session_state.anomaly_workflow[key] = {
                            "status": "New",
                            "reason": "Timing / Accrual",
                            "notes": "",
                        }

                    actual_val = float(row.get("Total_Actual", 0.0))
                    forecast_val = float(row.get("Total Cost", 0.0))

                    variance = actual_val - forecast_val
                    variance_pct = (variance / forecast_val) if forecast_val != 0 else None

                    mom_change = row.get("MoM_Change", np.nan)
                    mom_change_pct = row.get("MoM_Change_Pct", np.nan)
                    mom_signal = row.get("MoM_Signal", "")

                    mom_impact = float(abs(mom_change)) if pd.notna(mom_change) else 0.0

                    header = f"📅 Month {month}"
                    if pd.notna(mom_change):
                        header += f" — MoM: {mom_change:+,.0f} EUR"
                    if variance_pct is not None:
                        header += f" | Variance: {variance:+,.0f} EUR ({variance_pct:+.1%})"
                    else:
                        header += f" | Variance: {variance:+,.0f} EUR"

                    with st.expander(header):
                        status = st.selectbox(
                            "Status",
                            ["New", "Reviewed", "Explained", "Action Required"],
                            index=["New", "Reviewed", "Explained", "Action Required"].index(
                                st.session_state.anomaly_workflow[key]["status"]
                            ),
                            key=f"status_{key}",
                        )

                        reason_options = [
                            "Timing / Accrual",
                            "One-off Expense",
                            "Volume Related",
                            "Price Increase",
                            "Reclassification",
                            "Data Issue",
                            "Under Investigation",
                            "Other",
                        ]
                        reason = st.selectbox(
                            "Reason (Variance Driver)",
                            reason_options,
                            index=reason_options.index(st.session_state.anomaly_workflow[key]["reason"]),
                            key=f"reason_{key}",
                        )

                        notes = st.text_area(
                            "Finance Notes",
                            value=st.session_state.anomaly_workflow[key]["notes"],
                            placeholder="e.g. One-off invoice, timing issue, accrual correction",
                            key=f"notes_{key}",
                        )

                        st.session_state.anomaly_workflow[key].update(
                            {"status": status, "reason": reason, "notes": notes}
                        )

                    records.append(
                        {
                            "Month": month,
                            "G/L Group": selected_group,
                            "Actual (EUR)": round(actual_val, 2),
                            "Forecast (EUR)": round(forecast_val, 2),
                            "Variance (EUR)": round(variance, 2),
                            "Variance %": round(variance_pct, 4) if variance_pct is not None else None,
                            "MoM": mom_signal,
                            "Anomaly Type": row.get("Anomaly_Type", ""),
                            "Status": st.session_state.anomaly_workflow[key]["status"],
                            "Reason": st.session_state.anomaly_workflow[key]["reason"],
                            "Notes": st.session_state.anomaly_workflow[key]["notes"],
                            "_MoM_Impact_EUR": mom_impact,
                        }
                    )

                st.divider()
                st.subheader("Anomaly Summary")

                summary_df = pd.DataFrame(records)

                if summary_df.empty:
                    st.info("No anomalies to summarize for the selected G/L Group.")
                    st.session_state["summary_df"] = pd.DataFrame()
                else:
                    if "_MoM_Impact_EUR" not in summary_df.columns:
                        if "MoM Change (EUR)" in summary_df.columns:
                            summary_df["_MoM_Impact_EUR"] = summary_df["MoM Change (EUR)"].abs().fillna(0.0)
                        else:
                            summary_df["_MoM_Impact_EUR"] = 0.0

                    summary_df = summary_df.sort_values(
                        ["_MoM_Impact_EUR", "Month"], ascending=[False, True]
                    )

                    preferred_cols = [
                        "Month",
                        "G/L Group",
                        "Actual (EUR)",
                        "Forecast (EUR)",
                        "Variance (EUR)",
                        "Variance %",
                        "MoM",
                        "Anomaly Type",
                        "Status",
                        "Reason",
                        "Notes",
                    ]
                    summary_df = summary_df[[c for c in preferred_cols if c in summary_df.columns]].copy()

                    if "Variance %" in summary_df.columns:
                        summary_df["Variance %"] = pd.to_numeric(summary_df["Variance %"], errors="coerce")

                    fmt_map = {}
                    for c in ["Actual (EUR)", "Forecast (EUR)", "Variance (EUR)"]:
                        if c in summary_df.columns:
                            fmt_map[c] = "{:,.2f}"
                    if "Variance %" in summary_df.columns:
                        fmt_map["Variance %"] = "{:+.1%}"

                    styled = summary_df.style.format(fmt_map, na_rep="—")

                    st.session_state["summary_df"] = summary_df
                    st.dataframe(styled, use_container_width=True)


# TAB 4: AI Finance Insights (Task 4)

elif page == "🧠 AI Generated Finance Insights":
    st.subheader("AI-Generated Finance Insights")

    st.warning(
        "AI-generated insights are based on aggregated analytics and anomaly signals. "
        "They support decision-making and do not replace professional judgment."
    )

    api_key = st.text_input("Enter API Key", type="password")

    if filtered_actual.empty or filtered_forecast.empty:
        st.info("Select filters that return data to enable AI insights.")
    else:
        def _dedupe_report(text: str) -> str:
            if not text:
                return text
            t = text.strip()

            markers = ["### 🚩 Top 3 Cost Risks", "🚩 Top 3 Cost Risks"]
            for m in markers:
                first = t.find(m)
                if first != -1:
                    second = t.find(m, first + 1)
                    if second != -1:
                        return t[:second].strip()

            n = len(t)
            if n % 2 == 0:
                half = n // 2
                if t[:half].strip() == t[half:].strip():
                    return t[:half].strip()

            return t

        anom_actual = (
            filtered_actual.groupby(
                ["Accounting Month", "G/L Group Identifier", "G/L Group Name", "Department"],
                as_index=False,
            )
            .agg(Total_Actual=("Amount", "sum"))
        )

        anom_forecast = (
            filtered_forecast.groupby(
                ["Accounting Month", "G/L Group Identifier", "G/L Group Name", "Department"],
                as_index=False,
            )
            .agg(**{"Total Cost": ("Total Cost", "sum")})
        )

        actual_for_anom = anom_actual[["Accounting Month", "G/L Group Identifier", "Total_Actual"]].copy()
        forecast_for_anom = anom_forecast[["Accounting Month", "G/L Group Identifier", "Total Cost"]].copy()

        anom_out = detect_anomalies(actual_for_anom, forecast_for_anom)

        labels = anom_actual[["Accounting Month", "G/L Group Identifier", "G/L Group Name", "Department"]].copy()
        anom_out = anom_out.merge(labels, on=["Accounting Month", "G/L Group Identifier"], how="left")

        if "MoM_Change" in anom_out.columns:
            anom_out["MoM_Impact_EUR"] = anom_out["MoM_Change"].abs()
        else:
            anom_out["MoM_Impact_EUR"] = 0.0

        if "MoM_Signal" not in anom_out.columns:
            if "MoM_Change" in anom_out.columns:
                def _mom_sig(v):
                    if pd.isna(v):
                        return "⚪ No prior data"
                    if v > 0:
                        return f"🔴 +{v:,.0f} EUR"
                    if v < 0:
                        return f"🟢 {v:,.0f} EUR"
                    return "⚪ 0 EUR"
                anom_out["MoM_Signal"] = anom_out["MoM_Change"].apply(_mom_sig)
            else:
                anom_out["MoM_Signal"] = "⚪ Not enough data"

        if "Total Cost" in anom_out.columns:
            anom_out["Forecast_Variance"] = anom_out["Total_Actual"] - anom_out["Total Cost"]
            anom_out["Forecast_Variance_Pct"] = np.where(
                anom_out["Total Cost"] != 0,
                anom_out["Forecast_Variance"] / anom_out["Total Cost"],
                np.nan,
            )
        else:
            anom_out["Forecast_Variance"] = np.nan
            anom_out["Forecast_Variance_Pct"] = np.nan

        material_budget = (anom_out["Forecast_Variance_Pct"].abs() >= 0.10)
        anom_out["Likely_Driver"] = np.where(
            material_budget,
            "Likely one-off invoice / volume step-change (material budget variance)",
            "Likely timing/accrual or reclassification (variance mostly phasing)",
        )

        top_mom = (
            anom_out.sort_values("MoM_Impact_EUR", ascending=False)
            .head(15)
            .copy()
        )

        forecast_actual_agg = (
            filtered_actual.groupby(
                ["Accounting Month", "G/L Group Name", "G/L Group Identifier", "Department"],
                as_index=False,
            )
            .agg(Total_Actual=("Amount", "sum"))
        )

        forecast_compare_df = (
            filtered_forecast.merge(
                forecast_actual_agg,
                on=["Accounting Month", "G/L Group Name", "G/L Group Identifier", "Department"],
                how="left",
            )
            .assign(
                Total_Actual=lambda x: x["Total_Actual"].fillna(0.0),
                Deviation=lambda x: x["Total_Actual"] - x["Total Cost"],
                Deviation_Pct=lambda x: np.where(
                    x["Total Cost"] != 0,
                    (x["Total_Actual"] - x["Total Cost"]) / x["Total Cost"],
                    np.nan,
                ),
            )
        )

        top_budget = (
            forecast_compare_df.assign(AbsDev=lambda x: x["Deviation"].abs())
            .sort_values("AbsDev", ascending=False)
            .head(10)
            .copy()
        )

        mom_cols = [
            "Accounting Month",
            "G/L Group Name",
            "Department",
            "MoM_Signal",
            "Total_Actual",
            "Total Cost",
            "Forecast_Variance",
            "Anomaly_Type",
            "Likely_Driver",
        ]
        mom_cols = [c for c in mom_cols if c in top_mom.columns]
        mom_block = top_mom[mom_cols].to_string(index=False)

        bud_cols = [
            "Accounting Month",
            "G/L Group Name",
            "Department",
            "Total Cost",
            "Total_Actual",
            "Deviation",
            "Deviation_Pct",
        ]
        bud_cols = [c for c in bud_cols if c in top_budget.columns]
        budget_block = top_budget[bud_cols].to_string(index=False)

        anomaly_summary = (
            "Top MoM Movers (review signals):\n"
            + mom_block
            + "\n\nTop Budget Deviations (context only):\n"
            + budget_block
        )

        forecast_summary = (
            forecast_compare_df[
                ["Accounting Month", "G/L Group Name", "Department", "Total Cost", "Total_Actual", "Deviation", "Deviation_Pct"]
            ]
            .sort_values(["Accounting Month", "G/L Group Name"])
            .head(40)
            .to_string(index=False)
        )
        
        if not api_key:
            st.info("Enter an API key to generate the AI finance summary.")
        else:
            if st.button("Generate AI Finance Summary"):
                cfg = LLMConfig(api_key=api_key)
                with st.spinner("🤖 Generating finance insights..."):
                    insights = generate_finance_insights(
                        anomaly_summary=anomaly_summary,
                        forecast_summary=forecast_summary,
                        config=cfg,
                    )

                insights = _dedupe_report(insights)

                st.divider()
                st.markdown(insights)
