from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def _validate_cols(df: pd.DataFrame, required: list[str], df_name: str = "df") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def monthly_trend(monthly_df: pd.DataFrame, raw_df: pd.DataFrame) -> go.Figure:

    _validate_cols(monthly_df, ["Accounting Month", "Total_Actual"], "monthly_df")
    _validate_cols(raw_df, ["Accounting Month", "G/L Group Name", "Amount"], "raw_df")

    plot_monthly = monthly_df.copy()
    plot_monthly["Total_Actual"] = _to_num(plot_monthly["Total_Actual"]).fillna(0.0)

    mean_cost = float(plot_monthly["Total_Actual"].mean())
    std_cost = float(plot_monthly["Total_Actual"].std(ddof=1)) if len(plot_monthly) > 1 else 0.0
    threshold = mean_cost + std_cost

    peak_df = plot_monthly[plot_monthly["Total_Actual"] > threshold].copy()

    top_drivers = []
    for month in peak_df["Accounting Month"].tolist():
        s = (
            raw_df[raw_df["Accounting Month"] == month]
            .groupby("G/L Group Name")["Amount"]
            .sum()
            .sort_values(ascending=False)
        )
        driver = s.index[0] if len(s) else "N/A"
        top_drivers.append(f"Main driver: {driver}")

    if len(peak_df):
        peak_df["Top_Driver"] = top_drivers

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_monthly["Accounting Month"],
            y=plot_monthly["Total_Actual"],
            mode="lines+markers",
            name="Monthly Cost",
            hovertemplate="Month: %{x}<br>Total: %{y:,.0f} EUR<extra></extra>",
        )
    )

    if len(peak_df):
        fig.add_trace(
            go.Scatter(
                x=peak_df["Accounting Month"],
                y=peak_df["Total_Actual"],
                mode="markers+text",
                marker=dict(size=12, color="red"),
                name="Peak Months",
                text=peak_df["Top_Driver"],
                textposition="top center",
                textfont=dict(size=12, color="black"),
                hovertemplate="<b>%{text}</b><br>Month: %{x}<br>Cost: %{y:,.0f} EUR<extra></extra>",
                cliponaxis=False,
            )
        )

    fig.update_layout(
        title="Monthly Total Expenses",
        xaxis_title="Accounting Month",
        yaxis_title="Total Cost (EUR)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def top_gl_bar(df: pd.DataFrame, value_col: str = "Total_Actual", n: int = 10) -> go.Figure:
    _validate_cols(df, ["G/L Group Name", value_col], "df")

    plot_df = df.copy()
    plot_df[value_col] = _to_num(plot_df[value_col]).fillna(0.0)
    plot_df = plot_df.sort_values(value_col, ascending=False).head(int(n))

    fig = px.bar(
        plot_df,
        x=value_col,
        y="G/L Group Name",
        orientation="h",
        title=f"Top {n} Cost Driving G/L Groups",
        labels={value_col: "Cost (EUR)", "G/L Group Name": "G/L Group"},
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def anomaly_trend_simple(df: pd.DataFrame) -> go.Figure:
    _validate_cols(df, ["Accounting Month", "Total_Actual", "Total Cost"], "df")

    plot_df = df.copy()
    plot_df["Total_Actual"] = _to_num(plot_df["Total_Actual"]).fillna(0.0)
    plot_df["Total Cost"] = _to_num(plot_df["Total Cost"]).fillna(0.0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["Accounting Month"],
            y=plot_df["Total_Actual"],
            mode="lines+markers",
            name="Actual",
            hovertemplate="Month: %{x}<br>Actual: %{y:,.0f} EUR<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Accounting Month"],
            y=plot_df["Total Cost"],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash"),
            hovertemplate="Month: %{x}<br>Forecast: %{y:,.0f} EUR<extra></extra>",
        )
    )

    if "Anomaly_Flag" in plot_df.columns:
        anomalies = plot_df[plot_df["Anomaly_Flag"] == "Anomaly"]
        if len(anomalies):
            fig.add_trace(
                go.Scatter(
                    x=anomalies["Accounting Month"],
                    y=anomalies["Total_Actual"],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(size=10, color="red"),
                    hovertemplate="Month: %{x}<br>Anomaly: %{y:,.0f} EUR<extra></extra>",
                )
            )

    fig.update_layout(
        title="Actual vs Forecast (Anomalies Highlighted)",
        xaxis_title="Accounting Month",
        yaxis_title="Cost (EUR)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def variance_waterfall(df: pd.DataFrame) -> go.Figure:
    _validate_cols(df, ["Accounting Month", "Total_Actual", "Total Cost"], "df")

    plot_df = df.copy().sort_values("Accounting Month")
    plot_df["Total_Actual"] = _to_num(plot_df["Total_Actual"]).fillna(0.0)
    plot_df["Total Cost"] = _to_num(plot_df["Total Cost"]).fillna(0.0)

    plot_df["Variance"] = (plot_df["Total_Actual"] - plot_df["Total Cost"]).round(2)
    plot_df["Cum_Variance"] = plot_df["Variance"].cumsum().round(2)

    x = plot_df["Accounting Month"].astype(int).tolist()
    y = plot_df["Variance"].tolist()

    customdata = list(zip(plot_df["Variance"], plot_df["Cum_Variance"]))
    text = [f"{v:+,.0f} EUR" for v in plot_df["Variance"]]

    fig = go.Figure(
        go.Waterfall(
            x=x,
            y=y,
            measure=["relative"] * len(x),
            text=text,
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                "Month: %{x}"
                "<br>Monthly variance: %{customdata[0]:+,.2f} EUR"
                "<br>Cumulative variance: %{customdata[1]:+,.2f} EUR"
                "<extra></extra>"
            ),
            connector={"line": {"dash": "dot"}},
            name="Variance",
        )
    )

    fig.update_layout(
        title="Monthly Budget Variance Bridge (Actual − Forecast)",
        xaxis_title="Accounting Month",
        yaxis_title="Cumulative Variance (EUR)",
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig



def mom_change_bar(df, title="Month-over-Month Change (Actual)"):
    d = df.copy().sort_values("Accounting Month")
    d["MoM_Change"] = d["MoM_Change"].fillna(0.0)

    bar_colors = np.where(d["MoM_Change"] >= 0, "red", "green")

    hover = []
    for _, r in d.iterrows():
        m = int(r["Accounting Month"])
        mom = float(r.get("MoM_Change", 0.0))
        mom_pct = r.get("MoM_Change_Pct", np.nan)
        fv = r.get("Forecast_Variance", np.nan)
        fv_pct = r.get("Forecast_Variance_Pct", np.nan)

        h = f"<b>Month {m}</b><br>"
        h += f"MoM Change: {mom:,.0f} EUR<br>"
        if pd.notna(mom_pct):
            h += f"MoM %: {mom_pct:+.1%}<br>"
        if pd.notna(fv):
            h += f"Budget Variance: {float(fv):+,.0f} EUR<br>"
        if pd.notna(fv_pct):
            h += f"Budget Var %: {float(fv_pct):+,.1%}<br>"
        hover.append(h + "<extra></extra>")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=d["Accounting Month"],
        y=d["MoM_Change"],
        marker=dict(color=bar_colors),
        hovertemplate=hover,
        name="MoM Change"
    ))

    if "Anomaly_Flag" in d.columns:
        anom = d[d["Anomaly_Flag"] == "Anomaly"]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom["Accounting Month"],
                y=anom["MoM_Change"],
                mode="markers",
                marker=dict(size=12, color="white", line=dict(width=2, color="red")),
                name="Anomalies"
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Accounting Month",
        yaxis_title="MoM Change (EUR)",
        hovermode="x unified"
    )
    return fig