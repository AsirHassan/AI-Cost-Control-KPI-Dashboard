# Finance Cost Control Dashboard – Case Study

## Overview

This project was developed as part of a **Data Analyst & Finance Intern case study**.  
The objective is to demonstrate how data analytics, forecasting, and automation can support **cost transparency, anomaly detection, and finance decision-making** in a plant controlling environment.

The application is built using **Python and Streamlit**, with a strong focus on usability, explainability, and practical finance workflows rather than technical complexity.

All data used in this project is **synthetic** and provided as part of the case study.

---

## Application Scope

The dashboard supports the following finance use cases:

- Transparent overview of plant expenses
- Monthly cost trend analysis
- Identification of abnormal spending patterns
- Structured review and documentation of cost anomalies
- AI-generated summaries to support finance interpretation

The solution is designed to reflect how such a tool could realistically be used by plant controlling or finance teams.

---

## Key Features

### 1. Cost Transparency (Task 1)
- Tabular view of all expense records
- Interactive filters (Department, G/L Group, Accounting Month)
- Monthly total expense trend
- Top cost-driving G/L groups
- Summary KPIs for quick interpretation

### 2. Forecasting & Dynamic Anomaly Detection (Task 2)
- Rolling historical averages used as expected cost benchmarks
- Automatic detection of unusual cost spikes and drops
- Visualization of actual vs. expected cost behavior
- Dynamic detection based on historical patterns rather than static thresholds

### 3. Automated Cost Anomaly Workflow (Task 3)
- Dedicated anomaly review panel
- Status tracking (New, Reviewed, Explained, Action Required)
- Finance notes for documentation and auditability
- Exception-based workflow to reduce month-end workload

### 4. AI-Generated Finance Insights (Task 4)
- Short AI-generated summaries based on detected anomalies and forecasts
- Designed to support, not replace, finance judgment
- AI does not generate financial figures or forecasts
- Governance and transparency are explicitly addressed in the UI

---

## Assumptions

- Costs are analyzed at **G/L Group level**, which is appropriate for plant controlling use cases
- Historical monthly behavior represents expected cost patterns unless disrupted by operational or accounting events
- Forecasting logic is intentionally simple and explainable to ensure finance trust and adoption

---

## Technology Stack

- Python
- Streamlit
- Pandas & NumPy
- Plotly
- Scikit-learn (supporting anomaly detection logic)
- External LLM API for AI-generated summaries

---

## How to Run the Application

1. (Optional) Create and activate a virtual environment  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:

    streamlit run app.py
4. Open the local URL provided in the terminal