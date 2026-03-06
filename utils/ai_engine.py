from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency guard
    OpenAI = None

@dataclass
class LLMConfig:
    api_key: str
    base_url: Optional[str] = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 240


def generate_finance_insights(
    anomaly_summary: str,
    forecast_summary: str,
    config: LLMConfig,
) -> str:
    """
    Generates short, finance-oriented executive insights using an LLM.

    IMPORTANT (your app logic):
    - "Anomaly" is driven by large Month-over-Month (MoM) change in ACTUAL costs.
    - Forecast is shown for context (budget compliance), not to define anomalies.

    Inputs should be compact, aggregated summaries (not raw transactional dumps).
    """

    anomaly_summary = (anomaly_summary or "").strip()
    forecast_summary = (forecast_summary or "").strip()

    if not anomaly_summary and not forecast_summary:
        return "No insights generated: anomaly and forecast summaries are empty."

    if OpenAI is None:
        return "AI insights unavailable: missing dependency 'openai'. Install from requirements."

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    system_msg = (
        "You are a Senior Plant Finance Controller at an automotive manufacturing plant. "
        "Write brief, action-oriented insights for executive stakeholders. "
        "Be factual, avoid speculation, and use clear finance language."
    )

    user_msg = f"""
You will write an executive summary using ONLY the inputs below.

DEFINITIONS (must follow):
- Anomaly = large Month-over-Month (MoM) change in ACTUAL costs (spike/drop).
- Forecast is context: mention budget variance only if it is material.

FORMAT (exact headings + bullet style):
You MUST include the leading "### " in each heading and a leading "- " for every bullet line.
If multiple risks are listed, vary the likely drivers across bullets (choose from: one-off invoice, phasing/timing, accruals, reclassification, volume step-change). Do not repeat the same driver phrase for all bullets.

### 🚩 Top 3 Cost Risks
- [G/L Group]: [One sentence. Start with: "Largest/Materal/Large ..." and include the EUR value in brackets.]
  Use a finance-style interpretation such as: one-off invoice, phasing/timing, accruals, reclassification, volume step-change.
  Use "suggests/likely/indicates" (do not claim certainty).

### ⚠️ Immediate Attention Required
- [Month]: [+/-X,XXX.00 EUR] - [One sentence action: "validate invoices, accruals, reclassification, scope/volume change"]

### 📉 Estimated Budget Impact
[One sentence: "Volatility may distort monthly steering and forecast accuracy even if full-year spend remains near plan."]

RULES:
1) Do NOT repeat the same number or sentence.
2) Do NOT use double asterisks (**) inside sentences; only use them in headings if needed.
3) Use currency formatting like 1,234.56 EUR (two decimals).
4) Keep the entire response under 130 words.
5) If a required detail is missing, say "Not enough data" (do not invent).
6) If using Forecast Summary to fill Top 3, select different G/L groups than already used, and include the deviation amount in EUR.
7) "Immediate Attention Required" MUST be the single largest MoM spike/drop by EUR from the anomalies input. Use the MoM amount and percent if provided.
8) Do NOT call forecast variance a "MoM spike". Forecast is context only.
9) Budget impact should reflect overall pattern: if variances are mostly small but volatility is high, say impact is on monthly steering/accuracy rather than overspend.
10) In "Top 3 Cost Risks", placeholder G/L groups are allowed ONLY when labeled "Not enough data" (no claims).
11) When explaining spikes/drops, prefer likely finance drivers such as timing/accruals, one-off invoices, volume changes, or reclassification — but do not claim certainty.
12) If you must use "Not enough data" in Top 3, do it at most twice and make each bullet specific to a different named G/L group (e.g., IT services, Maintenance & repairs).
13) For Top 3, each bullet must be ONE sentence starting with "Largest", "Material", or "Large" + include the MoM EUR value in brackets.
14) For Immediate Attention, write "step-change" and a clear finance action (validate invoices/accruals/reclassification/scope).
15) Always format EUR values with commas and two decimals, and include the currency: 35,152.00 EUR.



INPUTS:
Anomalies Summary (MoM-based):
{anomaly_summary}

Forecast Summary (context):
{forecast_summary}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        msg = resp.choices[0].message if resp.choices else None
        content = ""
        if msg is not None:
            content = getattr(msg, "content", None) or msg.get("content", "")

        content = (content or "").strip()
        return content if content else "No content generated."

    except Exception as e:
        return f"Error generating insights: {e}"
