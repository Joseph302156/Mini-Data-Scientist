from io import BytesIO

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from .chat import answer_chat_question
from .db import Base, engine, get_db
from .eda import compute_eda_for_dataset
from .insights import generate_automated_insights
from .modeling import (
    list_models_for_dataset,
    predict_with_model,
    predict_with_model_raw,
    suggest_targets_for_dataset,
    train_model_for_dataset,
)
from .models import Dataset, DatasetVersion
from .pipeline import get_dataset_state, ingest_and_process, list_dataset_states
from .schemas import (
    ChatRequest,
    ChatResponse,
    DataPreview,
    DatasetInfo,
    DatasetStatus,
    EdaResult,
    PredictRequest,
    PredictResponse,
    TargetListResponse,
    TrainModelRequest,
    TrainedModelSummary,
    UploadResponse,
)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Mini Data Scientist Backend")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported for now.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    dataset_id, profile, cleaning, features = ingest_and_process(db, file.filename, df)

    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        status=DatasetStatus.FEATURED,
        profile=profile,
        cleaning=cleaning,
        features=features,
    )


@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets(db: Session = Depends(get_db)) -> list[DatasetInfo]:
    rows = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    infos: list[DatasetInfo] = []
    for row in rows:
        infos.append(
            DatasetInfo(
                dataset_id=str(row.id),
                n_rows=row.n_rows or 0,
                n_cols=row.n_cols or 0,
                status=DatasetStatus(row.status),
            )
        )
    return infos


def _load_dataset_json_or_404(db: Session, dataset_id: str) -> Dataset:
    obj = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if obj is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return obj


@app.get("/api/datasets/{dataset_id}/profile")
async def get_profile(dataset_id: str, db: Session = Depends(get_db)):
    obj = _load_dataset_json_or_404(db, dataset_id)
    if obj.profile_json is None:
        raise HTTPException(status_code=404, detail="Profile not available for this dataset")
    return obj.profile_json


@app.get("/api/datasets/{dataset_id}/cleaning")
async def get_cleaning_result(dataset_id: str, db: Session = Depends(get_db)):
    obj = _load_dataset_json_or_404(db, dataset_id)
    if obj.cleaning_json is None:
        raise HTTPException(status_code=404, detail="Cleaning result not available for this dataset")
    return obj.cleaning_json


@app.get("/api/datasets/{dataset_id}/features")
async def get_feature_result(dataset_id: str, db: Session = Depends(get_db)):
    obj = _load_dataset_json_or_404(db, dataset_id)
    if obj.features_json is None:
        raise HTTPException(status_code=404, detail="Feature result not available for this dataset")
    return obj.features_json


@app.post("/api/datasets/{dataset_id}/models", response_model=TrainedModelSummary)
async def train_model(
    dataset_id: str,
    req: TrainModelRequest,
    db: Session = Depends(get_db),
) -> TrainedModelSummary:
    # simple existence check
    obj = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if obj is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        summary = train_model_for_dataset(db, dataset_id, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to train model: {exc}") from exc

    return summary


@app.get("/api/datasets/{dataset_id}/models", response_model=list[TrainedModelSummary])
async def list_models(
    dataset_id: str,
    db: Session = Depends(get_db),
) -> list[TrainedModelSummary]:
    obj = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if obj is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return list_models_for_dataset(db, dataset_id)


@app.get("/api/datasets/{dataset_id}/targets", response_model=TargetListResponse)
async def suggest_targets(
    dataset_id: str,
    db: Session = Depends(get_db),
) -> TargetListResponse:
    obj = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if obj is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        return suggest_targets_for_dataset(db, dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/predict", response_model=PredictResponse)
async def predict(
    model_id: str,
    req: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    try:
        return predict_with_model(db, model_id, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate predictions: {exc}") from exc


@app.post("/api/models/{model_id}/predict_raw", response_model=PredictResponse)
async def predict_raw(
    model_id: str,
    req: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    try:
        return predict_with_model_raw(db, model_id, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate predictions from raw data: {exc}") from exc


@app.get("/api/datasets/{dataset_id}/preview", response_model=DataPreview)
async def get_preview(
    dataset_id: str,
    stage: str = Query("raw", pattern="^(raw|cleaned|features)$"),
    limit: int = Query(5, ge=1, le=100),
) -> DataPreview:
    # first try in-memory cache
    try:
        state = get_dataset_state(dataset_id)
    except KeyError:
        state = None

    df = None
    if state is not None:
        if stage == "raw":
            df = state.raw_df
        elif stage == "cleaned":
            df = state.cleaned_df
        else:
            df = state.features_df

    # fall back to persisted parquet if needed
    if df is None:
        version = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.stage == stage)
            .one_or_none()
        )
        if version is None:
            raise HTTPException(status_code=404, detail=f"{stage.capitalize()} data not available for this dataset")
        try:
            df = pd.read_parquet(version.storage_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to load stored data: {exc}") from exc

    head = df.head(limit)
    preview_records = head.to_dict(orient="records")

    return DataPreview(
        dataset_id=dataset_id,
        stage=stage,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        preview=preview_records,
    )


@app.get("/api/datasets/{dataset_id}/eda", response_model=EdaResult)
async def get_eda(
    dataset_id: str,
    stage: str = Query("cleaned", pattern="^(raw|cleaned|features)$"),
    db: Session = Depends(get_db),
) -> EdaResult:
    try:
        result = compute_eda_for_dataset(db, dataset_id, use_stage=stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/api/datasets/{dataset_id}/insights")
async def get_insights(
    dataset_id: str,
    model_id: str | None = None,
    stage: str = Query("cleaned", pattern="^(raw|cleaned|features)$"),
    db: Session = Depends(get_db),
):
    try:
        result = generate_automated_insights(db, dataset_id, model_id=model_id, use_stage=stage)
    except RuntimeError as exc:
        # LLM not configured; still return EDA but with a clear message
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    try:
        return answer_chat_question(db, req)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


