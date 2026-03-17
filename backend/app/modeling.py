from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from .models import DatasetVersion, ModelRun
from .schemas import (
    Metric,
    ModelType,
    PredictRequest,
    PredictResponse,
    TaskType,
    TrainModelRequest,
    TrainedModelSummary,
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_model(
    task_type: TaskType,
    model_type: ModelType,
    random_state: int,
):
    if task_type == TaskType.REGRESSION:
        if model_type == ModelType.LINEAR:
            return LinearRegression()
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )
    else:
        if model_type == ModelType.LINEAR:
            return LogisticRegression(max_iter=1000, n_jobs=-1)
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )
    raise ValueError(f"Unsupported model configuration: {task_type} / {model_type}")


def _compute_metrics(
    task_type: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Metric]:
    if task_type == TaskType.REGRESSION:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        r2 = float(r2_score(y_true, y_pred))
        return [Metric(name="rmse", value=rmse), Metric(name="r2", value=r2)]
    acc = float(accuracy_score(y_true, y_pred))
    return [Metric(name="accuracy", value=acc)]


def _extract_feature_importances(
    model,
    feature_names: List[str],
) -> Dict[str, float]:
    importances: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        vals = getattr(model, "feature_importances_")
        for name, val in zip(feature_names, vals):
            importances[name] = float(val)
    elif hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        for name, val in zip(feature_names, coefs):
            importances[name] = float(abs(val))
    return importances


def train_model_for_dataset(
    db: Session,
    dataset_id: str,
    req: TrainModelRequest,
) -> TrainedModelSummary:
    # load features parquet path
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.stage == "features")
        .one_or_none()
    )
    if version is None:
        raise ValueError("Features not available for this dataset")

    df = pd.read_parquet(version.storage_path)

    if req.target_column not in df.columns:
        raise ValueError(f"Target column '{req.target_column}' not found in features dataset")

    y = df[req.target_column]
    X = df.drop(columns=[req.target_column])

    feature_names = list(X.columns)
    X_values = X.to_numpy()
    y_values = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_values,
        y_values,
        test_size=req.test_size,
        random_state=req.random_state,
        stratify=y_values if req.task_type == TaskType.CLASSIFICATION else None,
    )

    model = _build_model(req.task_type, req.model_type, req.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _compute_metrics(req.task_type, y_test, y_pred)
    importances = _extract_feature_importances(model, feature_names)

    model_id = str(uuid4())
    model_path = MODELS_DIR / f"{model_id}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "task_type": req.task_type.value,
            "model_type": req.model_type.value,
            "target_column": req.target_column,
        },
        model_path,
    )

    db_obj = ModelRun(
        id=model_id,
        dataset_id=dataset_id,
        target_column=req.target_column,
        task_type=req.task_type.value,
        model_type=req.model_type.value,
        model_path=str(model_path),
        metrics_json=[m.model_dump() for m in metrics],
        feature_importances_json=importances,
    )
    db.add(db_obj)
    db.commit()

    return TrainedModelSummary(
        model_id=model_id,
        dataset_id=dataset_id,
        target_column=req.target_column,
        task_type=req.task_type,
        model_type=req.model_type,
        created_at=db_obj.created_at.isoformat(),
        metrics=metrics,
        feature_importances=importances,
    )


def list_models_for_dataset(db: Session, dataset_id: str) -> List[TrainedModelSummary]:
    runs = (
        db.query(ModelRun)
        .filter(ModelRun.dataset_id == dataset_id)
        .order_by(ModelRun.created_at.desc())
        .all()
    )
    results: List[TrainedModelSummary] = []
    for r in runs:
        metrics = [Metric(**m) for m in r.metrics_json]
        results.append(
            TrainedModelSummary(
                model_id=str(r.id),
                dataset_id=dataset_id,
                target_column=r.target_column,
                task_type=TaskType(r.task_type),
                model_type=ModelType(r.model_type),
                created_at=r.created_at.isoformat(),
                metrics=metrics,
                feature_importances=r.feature_importances_json or {},
            )
        )
    return results


def predict_with_model(
    db: Session,
    model_id: str,
    req: PredictRequest,
) -> PredictResponse:
    run = db.query(ModelRun).filter(ModelRun.id == model_id).one_or_none()
    if run is None:
        raise ValueError("Model not found")

    payload = joblib.load(run.model_path)
    model = payload["model"]
    feature_names: List[str] = payload["feature_names"]
    task_type = TaskType(payload["task_type"])
    model_type = ModelType(payload["model_type"])
    target_column: str = payload["target_column"]

    if not req.records:
        raise ValueError("No records provided for prediction")

    df = pd.DataFrame(req.records)

    # align columns with training feature order
    for name in feature_names:
        if name not in df.columns:
            df[name] = 0
    df = df[feature_names]

    X = df.to_numpy()

    y_pred = model.predict(X)
    predictions: List[Any] = [np.asscalar(p) if np.ndim(p) == 0 else p for p in y_pred]  # type: ignore[arg-type]

    probabilities: List[Dict[str, float]] | None = None
    if task_type == TaskType.CLASSIFICATION and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_")
        probabilities = []
        for row in proba:
            probabilities.append(
                {str(cls): float(p) for cls, p in zip(classes, row)},
            )

    return PredictResponse(
        model_id=str(run.id),
        dataset_id=str(run.dataset_id),
        target_column=target_column,
        task_type=task_type,
        model_type=model_type,
        predictions=predictions,
        probabilities=probabilities,
    )


