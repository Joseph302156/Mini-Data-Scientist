from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy.orm import Session

from .eda import compute_eda_for_dataset
from .models import Dataset, ModelRun
from .schemas import InsightFinding, StructuredInsights


_SYSTEM_PROMPT = """\
You are a friendly data analyst helping non-technical business users understand their data.
Analyze the provided dataset information and return a JSON response with EXACTLY this structure:

{
  "headline": "One compelling sentence (max 15 words) summarizing the most important finding",
  "summary": "2-3 sentences in plain English explaining what the data shows and what it means for the business",
  "key_findings": [
    {
      "title": "Short title (5-8 words)",
      "detail": "1-2 sentences in plain English",
      "type": "trend | correlation | highlight | warning | model"
    }
  ],
  "data_quality_note": "One sentence about data completeness and reliability",
  "recommendation": "1-2 sentences on what action to take based on this data"
}

IMPORTANT RULES:
- Return ONLY valid JSON — no markdown, no code fences, no extra text
- Write like you are explaining to a smart non-technical manager
- NEVER use jargon: no R², RMSE, p-value, standard deviation, correlation coefficient,
  regression, classification, overfitting, hyperparameter, etc.
- Instead use plain language: "X% accurate", "strongly linked to", "tends to increase when",
  "most important factor", "within ±X on average"
- Round all numbers to be easy to understand (not "1234.56789", just "1,235")
- Include 4–6 key findings — make them actionable and meaningful
- Use "type" values: "trend" for patterns over time, "correlation" for linked variables,
  "highlight" for standout facts, "warning" for data issues, "model" for model performance
"""


def _build_context_payload(
    dataset: Dataset,
    eda: Dict[str, Any],
    model_run: Optional[ModelRun],
) -> str:
    payload: Dict[str, Any] = {
        "filename": dataset.filename,
        "n_rows": dataset.n_rows,
        "n_cols": dataset.n_cols,
    }

    # Add lightweight profile summary (avoid dumping the whole thing)
    if dataset.profile_json is not None:
        profile = dataset.profile_json
        col_types: Dict[str, int] = {}
        missing_cols: List[str] = []
        for col in profile.get("columns", []):
            ctype = col.get("inferred_type", "unknown")
            col_types[ctype] = col_types.get(ctype, 0) + 1
            if col.get("missing_ratio", 0) > 0.05:
                missing_cols.append(col["name"])
        payload["column_type_counts"] = col_types
        if missing_cols:
            payload["columns_with_notable_missing"] = missing_cols

    # EDA: numeric summary + top correlations only (keep prompt compact)
    payload["numeric_summary"] = eda.get("numeric_summary", {})
    top_corrs = sorted(
        eda.get("correlations", []),
        key=lambda c: abs(c.get("value", 0)),
        reverse=True,
    )[:10]
    payload["top_correlations"] = top_corrs

    if model_run is not None:
        fi = model_run.feature_importances_json or {}
        top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        payload["model"] = {
            "target_column": model_run.target_column,
            "task_type": model_run.task_type,
            "metrics": model_run.metrics_json,
            "top_feature_importances": dict(top_features),
        }

    return json.dumps(payload, indent=2)


def _parse_json_response(text: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def _fallback_insights(dataset: Dataset) -> StructuredInsights:
    return StructuredInsights(
        headline="Dataset loaded — configure OpenAI to unlock AI insights",
        summary=(
            f"Your dataset '{dataset.filename}' contains "
            f"{dataset.n_rows:,} records across {dataset.n_cols} fields. "
            "Set OPENAI_API_KEY to generate natural-language insights automatically."
        ),
        key_findings=[],
        data_quality_note="OpenAI API key is not configured — insights are unavailable.",
        recommendation="Add OPENAI_API_KEY to your environment and reload the page.",
    )


def generate_structured_insights(
    db: Session,
    dataset_id: str,
    model_id: Optional[str] = None,
    use_stage: str = "cleaned",
) -> StructuredInsights:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if ds is None:
        raise ValueError("Dataset not found")

    eda_result = compute_eda_for_dataset(db, dataset_id, use_stage=use_stage)
    eda_dict = eda_result.model_dump()

    model_run: Optional[ModelRun] = None
    if model_id is not None:
        model_run = db.query(ModelRun).filter(ModelRun.id == model_id).one_or_none()

    provider = os.getenv("INSIGHTS_PROVIDER", "openai")
    if provider != "openai":
        return _fallback_insights(ds)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_insights(ds)

    context = _build_context_payload(ds, eda_dict, model_run)
    user_message = f"Here is the dataset context for analysis:\n\n{context}"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("INSIGHTS_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        raw_text = resp.json()["choices"][0]["message"]["content"]

    try:
        parsed = _parse_json_response(raw_text)
    except (json.JSONDecodeError, KeyError):
        # Return a graceful fallback if parsing fails
        return StructuredInsights(
            headline="Insights generated — review the findings below",
            summary=raw_text[:500],
            key_findings=[],
            data_quality_note="",
            recommendation="",
        )

    findings = [
        InsightFinding(
            title=f.get("title", ""),
            detail=f.get("detail", ""),
            type=f.get("type", "highlight"),
        )
        for f in parsed.get("key_findings", [])
    ]

    return StructuredInsights(
        headline=parsed.get("headline", ""),
        summary=parsed.get("summary", ""),
        key_findings=findings,
        data_quality_note=parsed.get("data_quality_note", ""),
        recommendation=parsed.get("recommendation", ""),
    )


# ── Legacy plain-text insights (kept for backward compatibility) ──────────────

def generate_automated_insights(
    db: Session,
    dataset_id: str,
    model_id: Optional[str] = None,
    use_stage: str = "cleaned",
) -> Dict[str, Any]:
    structured = generate_structured_insights(db, dataset_id, model_id=model_id, use_stage=use_stage)
    text_parts = [structured.headline, "", structured.summary, ""]
    for f in structured.key_findings:
        text_parts.append(f"• {f.title}: {f.detail}")
    if structured.recommendation:
        text_parts += ["", f"Recommendation: {structured.recommendation}"]

    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "stage": use_stage,
        "insights_text": "\n".join(text_parts),
    }
