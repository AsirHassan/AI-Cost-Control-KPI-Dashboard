# AI Cost Control KPI Dashboard

Interactive finance dashboard for cost transparency, forecast comparison, dynamic anomaly signaling, and AI-assisted executive commentary.

## What This Project Delivers

- Consolidated monthly cost visibility by department and G/L group.
- Forecast vs actual variance monitoring with KPI-level summaries.
- Dynamic month-over-month anomaly detection (spikes and drops).
- Lightweight anomaly review workflow for finance users.
- AI-generated short-form insights from aggregated finance signals.

## Application Walkthrough

Live Demo: https://costcontroldashboard.streamlit.app/

---

### 1) Cost Overview

- Total Actual, Forecast, Variance, and Variance % KPIs.
- Monthly trend chart and top cost-driver chart.
- Detailed filtered transaction table.

### 2) Forecast and Dynamic Risk Signals

- Actual vs Forecast signal chart.
- Variance waterfall for cumulative deviation tracking.
- Table with MoM changes, forecast variance, and anomaly flags.

### 3) Anomaly Review Workflow

- Review-only panel focused on anomaly rows.
- Status lifecycle: `New`, `Reviewed`, `Explained`, `Action Required`.
- Finance notes and reason tagging for audit-friendly review.

### 4) AI Generated Finance Insights

- Produces concise executive commentary from anomaly and budget context.
- Designed as decision support, not autonomous forecasting.
- Accepts user API key at runtime and does not hardcode credentials.

## Architecture

```text
app.py
utils/
  aggregations.py       -> monthly rollups and forecast merge helpers
  anomaly_detection.py  -> MoM-based anomaly logic and risk flags
  charts.py             -> Plotly visualizations
  data_loader.py        -> Excel load, schema normalization, validation
  ai_engine.py          -> LLM prompt assembly and API client call
```

## Data Input Contract

The app supports two Excel sheets:

- `Actual_Data`
- `Forecast_Data`

Canonical expected columns:

- `Actual_Data`: `Posting Date`, `Accounting Month`, `Department`, `G/L Group Identifier`, `G/L Group Name`, `Amount`, `Currency`
- `Forecast_Data`: `Accounting Month`, `Department`, `G/L Group Identifier`, `G/L Group Name`, `Total Cost`, `Currency`

The loader is resilient to common variants and aliases, for example:

- Forecast value aliases: `Forecast`, `Total_Cost`, `Budget`, `PlannedCost`, `Total Cost (EUR)` (prefix normalized).
- Accounting month formats: `1..12`, `2024-05`, `2024/05`, `202405`, standard parseable dates.
- Key normalization: trims spaces and normalizes numeric-like IDs such as `1001.0 -> 1001`.

If required fields are still missing, the app raises a readable validation error with available columns.

## Data Source Behavior

- Default mode: app loads `data/finance_case_study_synthetic_data_TH-2.xlsx`.
- Custom mode: user can upload `.xlsx` from the sidebar (`Data Source` panel).
- Uploaded file overrides default data for the active session.

## AI Integration Notes

- AI client is OpenAI-compatible and currently configured with:
  - Default base URL: `https://api.groq.com/openai/v1`
  - Default model: `llama-3.3-70b-versatile`
- If `openai` is unavailable, the app fails gracefully with a clear message in the AI output section.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Cloud)

- Repo main file: `app.py`
- Dependency source: `requirements.txt`
- After push:
  - Reboot app from Streamlit Cloud if code changes do not appear.
  - Clear cache if stale session state causes inconsistent behavior.

## Troubleshooting

### Import errors on cloud (`from utils...`)

- Ensure latest repo includes `utils/__init__.py`.
- Ensure app path insertion uses project-root priority (`sys.path.insert(0, ...)`).

### `Column(s) ['Total Cost'] do not exist`

- Usually caused by alternate forecast column naming.
- Loader now maps common aliases automatically.
- Confirm you uploaded `Forecast_Data` sheet (exact name).

### `invalid literal for int() with base 10: '2024-05'`

- Caused by strict month parsing in older version.
- Current loader supports date-like accounting month formats.

### AI tab has no action button

- Button appears when filtered actual data exists.
- If forecast data is empty, app still runs AI with limited budget context and shows a warning.

## Repository Notes

- Included datasets are synthetic and for demonstration/testing workflows.
- `data/finance_case_study_synthetic_data_TH-2_new.xlsx` is a compatible alternative sample with changed values and unchanged schema.
