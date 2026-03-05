# utils/anomaly_detection.py
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_anomalies(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    mom_pct_threshold: float = 0.25,   # 25% => "Watch"
    mom_anomaly_pct: float = 0.50,     # 50% => "Anomaly"
    mom_abs_threshold: float = 10000,  # 10k EUR (only matters with >=25% change)
) -> pd.DataFrame:

    req_a = {"Accounting Month", "G/L Group Identifier", "Total_Actual"}
    req_f = {"Accounting Month", "G/L Group Identifier", "Total Cost"}

    if not req_a.issubset(actual_df.columns):
        raise KeyError(f"actual_df missing: {sorted(req_a - set(actual_df.columns))}")
    if not req_f.issubset(forecast_df.columns):
        raise KeyError(f"forecast_df missing: {sorted(req_f - set(forecast_df.columns))}")

    df = actual_df.copy()
    fc = forecast_df.copy()

    df["Total_Actual"] = pd.to_numeric(df["Total_Actual"], errors="coerce").fillna(0.0)
    fc["Total Cost"] = pd.to_numeric(fc["Total Cost"], errors="coerce")

    out = pd.merge(
        df,
        fc,
        on=["Accounting Month", "G/L Group Identifier"],
        how="left",
    )
    out["Total Cost"] = out["Total Cost"].fillna(0.0)

    out = out.sort_values(["G/L Group Identifier", "Accounting Month"]).reset_index(drop=True)

    # 1) MoM metrics (THE core of anomaly) 
    out["Prev_Month_Actual"] = out.groupby("G/L Group Identifier")["Total_Actual"].shift(1)
    out["MoM_Change"] = out["Total_Actual"] - out["Prev_Month_Actual"]

    out["MoM_Change_Pct"] = np.where(
        out["Prev_Month_Actual"] > 0,
        out["MoM_Change"] / out["Prev_Month_Actual"],
        np.nan
    )

    # 2) Forecast context columns (still useful for finance view)
    out["Forecast_Variance"] = out["Total_Actual"] - out["Total Cost"]
    out["Forecast_Variance_Pct"] = np.where(
        out["Total Cost"] != 0,
        out["Forecast_Variance"] / out["Total Cost"],
        np.nan,
    )

    # 3) Anomaly flags based on MoM change
    out["Anomaly_Flag"] = "Normal"
    out["Anomaly_Type"] = "None"

    pct = out["MoM_Change_Pct"].abs()
    eur = out["MoM_Change"].abs()

    # Watch (25%+) 
    watch = pct >= mom_pct_threshold

    # Anomaly (50%+) OR (>=10k AND >=25%) 
    anomaly = (pct >= mom_anomaly_pct) | ((eur >= mom_abs_threshold) & (pct >= mom_pct_threshold))

    out.loc[watch, ["Anomaly_Flag", "Anomaly_Type"]] = ["Watch", "MoM Change (≥25%)"]
    out.loc[anomaly, ["Anomaly_Flag", "Anomaly_Type"]] = ["Anomaly", "Spike" ]  # temporary

    # Spike/Drop for anomalies
    spike = anomaly & (out["MoM_Change"] > 0)
    drop  = anomaly & (out["MoM_Change"] < 0)
    out.loc[spike, "Anomaly_Type"] = "Spike"
    out.loc[drop,  "Anomaly_Type"] = "Drop"

    # First month (no prior data) -> keep Normal
    no_prior = out["Prev_Month_Actual"].isna()
    out.loc[no_prior, ["Anomaly_Flag", "Anomaly_Type"]] = ["Normal", "None"]

    # 4) UX-friendly MoM signal (emoji + formatted)
    def _format_mom_signal(row):
        if pd.isna(row["MoM_Change"]) or pd.isna(row["MoM_Change_Pct"]):
            return "⚪ No prior data"

        change_eur = float(row["MoM_Change"])
        change_pct = float(row["MoM_Change_Pct"])

        if change_eur > 0:
            return f"🔴 +{change_eur:,.0f} EUR (+{change_pct:.0%})"
        if change_eur < 0:
            return f"🟢 {change_eur:,.0f} EUR ({change_pct:.0%})"
        return "⚪ 0 EUR (0%)"

    out["MoM_Signal"] = out.apply(_format_mom_signal, axis=1)
    out["MoM_Impact_EUR"] = out["MoM_Change"].abs().fillna(0.0)

    # 5) Rounding (avoid float artifacts)
    money_cols = ["Total_Actual", "Total Cost", "Forecast_Variance", "Prev_Month_Actual", "MoM_Change", "MoM_Impact_EUR"]
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].round(2)

    pct_cols = ["Forecast_Variance_Pct", "MoM_Change_Pct"]
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].round(4)

    return out