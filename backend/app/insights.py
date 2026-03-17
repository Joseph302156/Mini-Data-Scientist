from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx
from sqlalchemy.orm import Session

from .eda import compute_eda_for_dataset
from .models import Dataset, ModelRun


def _build_insights_prompt(
    dataset: Dataset,
    eda: Dict[str, Any],
    model_run: Optional[ModelRun],
) -> str:
    parts = []
    parts.append("You are a senior data analyst AI.")
    parts.append(
        "You will receive structured JSON describing a tabular dataset, its cleaning, EDA, and optional model results."
    )
    parts.append(
        "Your job is to write clear, concise business-style insights without repeating raw numbers excessively."
    )
    parts.append(
        "Focus on trends over time, relationships between variables, and which segments or features drive outcomes."
    )
    parts.append("Keep the tone professional but accessible to non-technical stakeholders.")
    parts.append("")

    payload: Dict[str, Any] = {"dataset_id": str(dataset.id), "filename": dataset.filename, "eda": eda}

    if dataset.profile_json is not None:
        payload["profile"] = dataset.profile_json
    if dataset.cleaning_json is not None:
        payload["cleaning"] = dataset.cleaning_json
    if dataset.features_json is not None:
        payload["features"] = dataset.features_json
    if model_run is not None:
        payload["model"] = {
            "id": str(model_run.id),
            "target_column": model_run.target_column,
            "task_type": model_run.task_type,
            "model_type": model_run.model_type,
            "metrics": model_run.metrics_json,
            "feature_importances": model_run.feature_importances_json,
        }

    parts.append("Here is the structured JSON context:")
    parts.append(json.dumps(payload, indent=2))
    parts.append("")
    parts.append(
        "Now, provide 3–7 bullet-point insights that highlight key patterns, correlations, trends, and drivers."
    )
    parts.append(
        "If a model is present, comment on which features are most important and how well the model performs."
    )

    return "\n".join(parts)


def generate_automated_insights(
    db: Session,
    dataset_id: str,
    model_id: Optional[str] = None,
    use_stage: str = "cleaned",
) -> Dict[str, Any]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if ds is None:
        raise ValueError("Dataset not found")

    eda_result = compute_eda_for_dataset(db, dataset_id, use_stage=use_stage)
    eda_dict = eda_result.model_dump()

    model_run: Optional[ModelRun] = None
    if model_id is not None:
        model_run = db.query(ModelRun).filter(ModelRun.id == model_id).one_or_none()

    prompt = _build_insights_prompt(ds, eda_dict, model_run)

    provider = os.getenv("INSIGHTS_PROVIDER", "openai")
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        # Speculative minimal call; adjust model name as needed by user env
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": os.getenv("INSIGHTS_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
    else:
        # Fallback: no external LLM, return a simple placeholder summary
        text = (
            "Automated insights provider is not configured. "
            "Set INSIGHTS_PROVIDER=openai and OPENAI_API_KEY to enable rich insights."
        )

    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "stage": use_stage,
        "eda": eda_dict,
        "insights_text": text,
    }

