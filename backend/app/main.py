from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .pipeline import get_dataset_state, ingest_and_process, list_dataset_states
from .schemas import (
    DataPreview,
    DatasetInfo,
    DatasetStatus,
    UploadResponse,
)


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


@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    infos: list[DatasetInfo] = []
    for state in list_dataset_states():
        n_rows = int(len(state.raw_df))
        n_cols = int(state.raw_df.shape[1])
        status = DatasetStatus.RAW
        if state.features_df is not None:
            status = DatasetStatus.FEATURED
        elif state.cleaned_df is not None:
            status = DatasetStatus.CLEANED
        elif state.profile is not None:
            status = DatasetStatus.PROFILED

        infos.append(
            DatasetInfo(
                dataset_id=state.dataset_id,
                n_rows=n_rows,
                n_cols=n_cols,
                status=status,
            )
        )
    return infos


@app.get("/api/datasets/{dataset_id}/profile")
async def get_profile(dataset_id: str):
    try:
        state = get_dataset_state(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None

    if state.profile is None:
        raise HTTPException(status_code=404, detail="Profile not available for this dataset")
    return state.profile


@app.get("/api/datasets/{dataset_id}/cleaning")
async def get_cleaning_result(dataset_id: str):
    try:
        state = get_dataset_state(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None

    if state.cleaning_result is None:
        raise HTTPException(status_code=404, detail="Cleaning result not available for this dataset")
    return state.cleaning_result


@app.get("/api/datasets/{dataset_id}/features")
async def get_feature_result(dataset_id: str):
    try:
        state = get_dataset_state(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None

    if state.feature_result is None:
        raise HTTPException(status_code=404, detail="Feature result not available for this dataset")
    return state.feature_result


@app.get("/api/datasets/{dataset_id}/preview", response_model=DataPreview)
async def get_preview(
    dataset_id: str,
    stage: str = Query("raw", pattern="^(raw|cleaned|features)$"),
    limit: int = Query(5, ge=1, le=100),
) -> DataPreview:
    try:
        state = get_dataset_state(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None

    if stage == "raw":
        df = state.raw_df
    elif stage == "cleaned":
        if state.cleaned_df is None:
            raise HTTPException(status_code=404, detail="Cleaned data not available for this dataset")
        df = state.cleaned_df
    else:
        if state.features_df is None:
            raise HTTPException(status_code=404, detail="Feature data not available for this dataset")
        df = state.features_df

    head = df.head(limit)
    preview_records = head.to_dict(orient="records")

    return DataPreview(
        dataset_id=dataset_id,
        stage=stage,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        preview=preview_records,
    )


