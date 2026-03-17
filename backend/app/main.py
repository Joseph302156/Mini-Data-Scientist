from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from .pipeline import ingest_and_process
from .schemas import UploadResponse, DatasetStatus


app = FastAPI(title="Mini Data Scientist Backend")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/datasets/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported for now.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    dataset_id, profile, cleaning, features = ingest_and_process(df)

    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        status=DatasetStatus.FEATURED,
        profile=profile,
        cleaning=cleaning,
        features=features,
    )

