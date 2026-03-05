from __future__ import annotations
import pandas as pd


# 1. We did Monthly aggregation to detect MoM anomalies
def aggregate_monthly(actual_df: pd.DataFrame) -> pd.DataFrame:
    required = {"Accounting Month", "Amount"}
    if not required.issubset(actual_df.columns):
        raise KeyError(f"Missing required columns: {required - set(actual_df.columns)}")

    monthly = (
        actual_df
        .groupby("Accounting Month", as_index=False)
        .agg(Total_Actual=("Amount", "sum"))
        .sort_values("Accounting Month")
    )

    return monthly


# 2. Top G/L Groups aggregation for ranking
def top_gl_groups(actual_df: pd.DataFrame) -> pd.DataFrame:
    required = {"G/L Group Name", "Amount"}
    if not required.issubset(actual_df.columns):
        raise KeyError(f"Missing required columns: {required - set(actual_df.columns)}")

    top_groups = (
        actual_df
        .groupby("G/L Group Name", as_index=False)
        .agg(Total_Actual=("Amount", "sum"))
        .sort_values("Total_Actual", ascending=False)
    )

    return top_groups


# 3. Actual vs Forecast aggregation with finance metrics for merged view
def aggregate_with_forecast(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "Accounting Month",
        "Department",
        "G/L Group Identifier",
        "G/L Group Name",
        "Currency"
    ]

    actual_agg = (
        actual_df
        .groupby(keys, as_index=False)
        .agg(Total_Actual=("Amount", "sum"))
    )

    merged = pd.merge(
        actual_agg,
        forecast_df,
        on=keys,
        how="outer"
    )

    merged["Has_Actual"] = merged["Total_Actual"].notna()

    merged["Total_Actual"] = merged["Total_Actual"].fillna(0.0)
    merged["Total Cost"] = merged["Total Cost"].fillna(0.0)

    merged["Variance"] = merged["Total_Actual"] - merged["Total Cost"]
    merged["Variance_Pct"] = merged.apply(
        lambda r: r["Variance"] / r["Total Cost"] if r["Total Cost"] != 0 else None,
        axis=1
    )

    return merged