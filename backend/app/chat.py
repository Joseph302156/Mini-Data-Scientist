from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx
from sqlalchemy.orm import Session

from .eda import compute_eda_for_dataset
from .models import Dataset, ModelRun
from .schemas import ChatRequest, ChatResponse


def _build_chat_prompt(
    dataset: Dataset,
    eda: Dict[str, Any],
    model_run: Optional[ModelRun],
    question: str,
) -> str:
    parts = []
    parts.append("You are an AI data analyst embedded in an analytics product.")
    parts.append(
        "You will receive structured JSON for a dataset, its cleaning, feature engineering, "
        "EDA summaries, and optional model results."
    )
    parts.append("The user will then ask a question about the data or model.")
    parts.append(
        "Answer directly and concretely, referencing the provided context. "
        "Prefer qualitative explanations over raw dumps of numbers."
    )
    parts.append("")

    payload: Dict[str, Any] = {
        "dataset_id": str(dataset.id),
        "filename": dataset.filename,
        "eda": eda,
    }
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

    parts.append("Context JSON:")
    parts.append(json.dumps(payload, indent=2))
    parts.append("")
    parts.append(f"User question: {question}")
    parts.append("")
    parts.append("Now answer the question in a few concise paragraphs or bullet points.")

    return "\n".join(parts)


def answer_chat_question(db: Session, req: ChatRequest) -> ChatResponse:
    ds = db.query(Dataset).filter(Dataset.id == req.dataset_id).one_or_none()
    if ds is None:
        raise ValueError("Dataset not found")

    eda_result = compute_eda_for_dataset(db, req.dataset_id, use_stage="cleaned")
    eda_dict = eda_result.model_dump()

    model_run: Optional[ModelRun] = None
    if req.model_id is not None:
        model_run = db.query(ModelRun).filter(ModelRun.id == req.model_id).one_or_none()

    prompt = _build_chat_prompt(ds, eda_dict, model_run, req.question)

    provider = os.getenv("INSIGHTS_PROVIDER", "openai")
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": os.getenv("INSIGHTS_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
    else:
        answer = (
            "Chat provider is not configured. "
            "Set INSIGHTS_PROVIDER=openai and OPENAI_API_KEY to enable the AI chat interface."
        )

    return ChatResponse(
        dataset_id=req.dataset_id,
        model_id=req.model_id,
        question=req.question,
        answer=answer,
    )

