from io import BytesIO

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from .chat import answer_chat_question
from .db import Base, engine, get_db
from .eda import compute_eda_for_dataset
from .insights import generate_automated_insights, generate_structured_insights
from .modeling import (
    list_models_for_dataset,
    predict_with_model,
    predict_with_model_raw,
    suggest_targets_for_dataset,
    train_model_for_dataset,
)
from .models import Dataset, DatasetVersion, ModelRun
from .pipeline import delete_dataset_state, get_dataset_state, ingest_and_process, list_dataset_states
from .schemas import (
    ChatRequest,
    ChatResponse,
    DataPreview,
    DatasetInfo,
    DatasetOverview,
    DatasetReport,
    DatasetStatus,
    EdaResult,
    Metric,
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
                filename=row.filename,
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
    db: Session = Depends(get_db),
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


@app.get("/api/datasets/{dataset_id}/report", response_model=DatasetReport)
async def get_report(
    dataset_id: str,
    model_id: str | None = None,
    db: Session = Depends(get_db),
) -> DatasetReport:
    ds = _load_dataset_json_or_404(db, dataset_id)

    # EDA
    try:
        eda_result = compute_eda_for_dataset(db, dataset_id, use_stage="cleaned")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Build dataset overview from profile JSON
    profile = ds.profile_json or {}
    numeric_cols = 0
    categorical_cols = 0
    datetime_cols = 0
    total_missing = 0
    total_cells = max((ds.n_rows or 0) * (ds.n_cols or 1), 1)
    for col in profile.get("columns", []):
        t = col.get("inferred_type", "")
        if t == "numeric":
            numeric_cols += 1
        elif t == "categorical":
            categorical_cols += 1
        elif t == "datetime":
            datetime_cols += 1
        total_missing += col.get("missing_count", 0)

    overview = DatasetOverview(
        n_rows=ds.n_rows or 0,
        n_cols=ds.n_cols or 0,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        missing_values_pct=round(total_missing / total_cells * 100, 1),
    )

    # Resolve model: explicit model_id or latest trained model
    resolved_model_run: ModelRun | None = None
    if model_id:
        resolved_model_run = db.query(ModelRun).filter(ModelRun.id == model_id).one_or_none()
    else:
        resolved_model_run = (
            db.query(ModelRun)
            .filter(ModelRun.dataset_id == dataset_id)
            .order_by(ModelRun.created_at.desc())
            .first()
        )

    model_summary: TrainedModelSummary | None = None
    if resolved_model_run is not None:
        fi = resolved_model_run.feature_importances_json or {}
        metrics_raw = resolved_model_run.metrics_json or {}
        metrics = [Metric(name=k, value=float(v)) for k, v in metrics_raw.items()]
        model_summary = TrainedModelSummary(
            model_id=str(resolved_model_run.id),
            dataset_id=dataset_id,
            target_column=resolved_model_run.target_column,
            task_type=resolved_model_run.task_type,
            model_type=resolved_model_run.model_type,
            created_at=resolved_model_run.created_at.isoformat(),
            metrics=metrics,
            feature_importances=fi,
        )

    # Structured AI insights
    insights = generate_structured_insights(
        db,
        dataset_id,
        model_id=str(resolved_model_run.id) if resolved_model_run else None,
        use_stage="cleaned",
    )

    return DatasetReport(
        dataset_id=dataset_id,
        filename=ds.filename,
        overview=overview,
        eda=eda_result,
        insights=insights,
        model=model_summary,
    )


@app.delete("/api/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
) -> None:
    # remove in-memory cache
    delete_dataset_state(dataset_id)

    obj = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if obj is None:
        # idempotent delete
        return

    # delete associated models
    db.query(ModelRun).filter(ModelRun.dataset_id == dataset_id).delete()
    # delete dataset (dataset_versions are ON DELETE CASCADE)
    db.delete(obj)
    db.commit()


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


