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

from .models import Dataset, DatasetVersion, ModelRun
from .schemas import (
    FeatureResult,
    FeatureStat,
    Metric,
    ModelType,
    PredictRequest,
    PredictResponse,
    TargetCandidate,
    TargetListResponse,
    TaskType,
    TrainModelRequest,
    TrainedModelSummary,
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


_TIME_COMPONENT_SUFFIXES = ("_year", "_month", "_dayofweek")


def _build_model_explanation(
    task_type: TaskType,
    model_type: ModelType,
    target_column: str,
    metrics: List[Metric],
    top_feature_names: List[str],
) -> str:
    top_feats = ", ".join(top_feature_names[:3]) if top_feature_names else "the most important features"
    if task_type == TaskType.REGRESSION:
        rmse = next((m.value for m in metrics if m.name == "rmse"), None)
        r2 = next((m.value for m in metrics if m.name == "r2"), None)
        rmse_txt = f"typical error of about {rmse:.3g} in the target’s units" if rmse is not None else "typical prediction error"
        r2_txt = f"it captures roughly {r2:.3g} of the target’s variation (R²)" if r2 is not None else "it learns patterns in the data"
        return (
            f"This model predicts `{target_column}` as a number. "
            f"It is built to generalize from your data, giving you a prediction for new rows using the patterns in {top_feats}. "
            f"On the held-out test set, it has a {rmse_txt} and {r2_txt}."
        )
    acc = next((m.value for m in metrics if m.name == "accuracy"), None)
    acc_txt = f"accuracy of about {acc:.3g}" if acc is not None else "accuracy on the test set"
    return (
        f"This model predicts `{target_column}` as a label (classification). "
        f"It decides which class a new row most likely belongs to using the strongest signals in {top_feats}. "
        f"On the held-out test set, it achieves {acc_txt}."
    )


def _is_time_component_feature(feature_name: str) -> bool:
    return any(feature_name.endswith(sfx) for sfx in _TIME_COMPONENT_SUFFIXES)


def _load_cleaned_df_and_feature_result(
    db: Session,
    dataset_id: str,
) -> tuple[pd.DataFrame, FeatureResult]:
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.stage == "cleaned")
        .one_or_none()
    )
    if version is None:
        raise ValueError("Cleaned data not available for this dataset")

    cleaned_df = pd.read_parquet(version.storage_path)

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if ds is None or ds.features_json is None:
        raise ValueError("Feature metadata not available for this dataset")

    feature_result = FeatureResult(**ds.features_json)
    return cleaned_df, feature_result


def _compute_feature_stats_for_names(
    cleaned_df: pd.DataFrame,
    feature_result: FeatureResult,
    feature_names: List[str],
) -> List[FeatureStat]:
    meta_by_name: Dict[str, Any] = {m.name: m for m in feature_result.feature_summary}
    out: List[FeatureStat] = []

    for fname in feature_names:
        meta = meta_by_name.get(fname)
        if meta is None:
            continue

        # Time-derived features: omit totals.
        if _is_time_component_feature(fname):
            # meta.source is the original datetime column
            dt = pd.to_datetime(cleaned_df.get(meta.source), errors="coerce")
            if fname.endswith("_year"):
                series = dt.dt.year
            elif fname.endswith("_month"):
                series = dt.dt.month
            else:
                series = dt.dt.dayofweek

            series = pd.to_numeric(series, errors="coerce").dropna()
            mean = float(series.mean()) if not series.empty else None
            out.append(
                FeatureStat(
                    feature_name=fname,
                    feature_type="time_component",
                    mean=mean,
                    total=None,
                    count_true=None,
                    prevalence=None,
                    note="Time-derived feature (totals omitted; totals are not meaningful for calendar parts).",
                )
            )
            continue

        if meta.type == "binary":
            # One-hot: feature name is `${source}_${category}`
            source = meta.source
            prefix = f"{source}_"
            category = fname[len(prefix) :] if fname.startswith(prefix) else fname
            series_raw = cleaned_df.get(source)
            if series_raw is None:
                continue
            series_str = series_raw.astype(str)
            count_true = int((series_str == category).sum())
            total_rows = int(len(series_str))
            prevalence = (count_true / total_rows) if total_rows > 0 else None
            out.append(
                FeatureStat(
                    feature_name=fname,
                    feature_type="binary",
                    mean=prevalence,
                    total=float(count_true),
                    count_true=count_true,
                    prevalence=prevalence,
                    note=f"Category `{category}` occurrences (one-hot encoded).",
                )
            )
            continue

        # Numeric features (including log1p-transformed numeric).
        if meta.type == "numeric":
            raw_series: pd.Series

            if fname.endswith("_log1p"):
                raw_col = fname[: -len("_log1p")]
                raw_obj = cleaned_df.get(raw_col)
                if raw_obj is None:
                    continue
                raw_series = pd.to_numeric(raw_obj, errors="coerce")
                raw_series = raw_series.fillna(0.0)
                transformed = np.log1p(np.clip(raw_series.to_numpy(), a_min=0, a_max=None))
                mean = float(np.mean(transformed)) if transformed.size else None
                total = float(np.sum(transformed)) if transformed.size else None
                out.append(
                    FeatureStat(
                        feature_name=fname,
                        feature_type="numeric",
                        mean=mean,
                        total=total,
                        note="Numeric feature (log1p-transformed). Totals are in the transformed space.",
                    )
                )
                continue

            # Regular numeric: totals are meaningful in the cleaned numeric space.
            raw_col = fname
            raw_obj = cleaned_df.get(raw_col)
            if raw_obj is None:
                continue
            raw_series = pd.to_numeric(raw_obj, errors="coerce")
            raw_series = raw_series.fillna(0.0)
            mean = float(raw_series.mean()) if raw_series.notna().any() else None
            total = float(raw_series.sum()) if raw_series.notna().any() else None
            out.append(
                FeatureStat(
                    feature_name=fname,
                    feature_type="numeric",
                    mean=mean,
                    total=total,
                    note="Numeric feature totals are meaningful in the cleaned value space.",
                )
            )
            continue

    return out


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

    # Enrich the UI with natural-language explanation + intuitive stats
    top_feature_names = sorted(importances.keys(), key=lambda k: importances[k], reverse=True)[:6]
    top_feature_stats: List[FeatureStat] = []
    model_explanation: str | None = None
    try:
        cleaned_df, feature_result = _load_cleaned_df_and_feature_result(db, dataset_id)
        top_feature_stats = _compute_feature_stats_for_names(
            cleaned_df=cleaned_df,
            feature_result=feature_result,
            feature_names=top_feature_names,
        )
        model_explanation = _build_model_explanation(
            task_type=req.task_type,
            model_type=req.model_type,
            target_column=req.target_column,
            metrics=metrics,
            top_feature_names=top_feature_names,
        )
    except Exception:
        # Keep training resilient; UI can still show metrics even if stats enrichment fails.
        top_feature_stats = []
        model_explanation = None

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
        model_explanation=model_explanation,
        top_feature_stats=top_feature_stats if top_feature_stats else None,
    )


def list_models_for_dataset(db: Session, dataset_id: str) -> List[TrainedModelSummary]:
    runs = (
        db.query(ModelRun)
        .filter(ModelRun.dataset_id == dataset_id)
        .order_by(ModelRun.created_at.desc())
        .all()
    )
    cleaned_df: pd.DataFrame | None = None
    feature_result: FeatureResult | None = None
    results: List[TrainedModelSummary] = []
    for r in runs:
        metrics = [Metric(**m) for m in r.metrics_json]
        top_feature_names = sorted(
            (r.feature_importances_json or {}).keys(),
            key=lambda k: (r.feature_importances_json or {}).get(k, 0.0),
            reverse=True,
        )[:6]

        top_feature_stats: List[FeatureStat] | None = None
        model_explanation: str | None = None
        try:
            if cleaned_df is None or feature_result is None:
                cleaned_df, feature_result = _load_cleaned_df_and_feature_result(db, dataset_id)
            if feature_result is not None and cleaned_df is not None:
                top_feature_stats = _compute_feature_stats_for_names(
                    cleaned_df=cleaned_df,
                    feature_result=feature_result,
                    feature_names=top_feature_names,
                )
                model_explanation = _build_model_explanation(
                    task_type=TaskType(r.task_type),
                    model_type=ModelType(r.model_type),
                    target_column=r.target_column,
                    metrics=metrics,
                    top_feature_names=top_feature_names,
                )
        except Exception:
            top_feature_stats = None
            model_explanation = None

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
                model_explanation=model_explanation,
                top_feature_stats=top_feature_stats,
            )
        )
    return results


def suggest_targets_for_dataset(db: Session, dataset_id: str) -> TargetListResponse:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if ds is None or ds.features_json is None:
        raise ValueError("Feature metadata not available for this dataset")

    feature_result = FeatureResult(**ds.features_json)

    candidates: List[TargetCandidate] = []
    for f in feature_result.feature_summary:
        # Heuristic: consider original (non-engineered) columns as targets
        if f.source != f.name:
            continue
        if f.role != "feature":
            continue
        # if numeric -> regression; else -> classification
        suggested = TaskType.REGRESSION if f.type == "numeric" else TaskType.CLASSIFICATION
        candidates.append(TargetCandidate(column=f.name, suggested_task=suggested))

    return TargetListResponse(dataset_id=dataset_id, candidates=candidates)


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
    # NumPy removed `np.asscalar`; use `.item()` for scalar conversion.
    predictions: List[Any] = [p.item() if np.ndim(p) == 0 else p for p in y_pred]  # type: ignore[arg-type]

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


def _load_feature_stats(df_features: pd.DataFrame, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name in feature_names:
        if name not in df_features.columns:
            continue
        col = df_features[name]
        if not np.issubdtype(col.dtype, np.number):
            continue
        mean = float(col.mean())
        std = float(col.std() or 1.0)
        stats[name] = {"mean": mean, "std": std}
    return stats


def predict_with_model_raw(
    db: Session,
    model_id: str,
    req: PredictRequest,
) -> PredictResponse:
    run = db.query(ModelRun).filter(ModelRun.id == model_id).one_or_none()
    if run is None:
        raise ValueError("Model not found")

    ds = db.query(Dataset).filter(Dataset.id == run.dataset_id).one_or_none()
    if ds is None or ds.features_json is None:
        raise ValueError("Feature metadata not available for this model's dataset")

    feature_result = FeatureResult(**ds.features_json)

    payload = joblib.load(run.model_path)
    model = payload["model"]
    feature_names: List[str] = payload["feature_names"]
    task_type = TaskType(payload["task_type"])
    model_type = ModelType(payload["model_type"])
    target_column: str = payload["target_column"]

    if not req.records:
        raise ValueError("No records provided for prediction")

    # load training features to compute scaling stats
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == run.dataset_id, DatasetVersion.stage == "features")
        .one_or_none()
    )
    if version is None:
        raise ValueError("Features not available for this dataset")
    df_train_features = pd.read_parquet(version.storage_path)
    feature_stats = _load_feature_stats(df_train_features, feature_names)

    # build feature matrix from raw records according to our feature naming conventions
    rows: List[List[float]] = []
    for record in req.records:
        row_vals: List[float] = []
        for fname in feature_names:
            # find metadata entry for this feature
            meta = next((f for f in feature_result.feature_summary if f.name == fname), None)
            if meta is None:
                row_vals.append(0.0)
                continue

            source = meta.source
            val: float = 0.0

            if meta.type == "numeric":
                # numeric feature, possibly log1p-standardized
                # For log1p-transformed numeric features, the raw input comes from the base column
                # (e.g. `revenue_log1p` expects `revenue` in the input record).
                if fname.endswith("_log1p"):
                    raw_col = fname[: -len("_log1p")]
                    raw_val = record.get(raw_col, 0.0)
                else:
                    src_name = source
                    raw_val = record.get(src_name, 0.0)
                try:
                    x = float(raw_val)
                except (TypeError, ValueError):
                    x = 0.0

                if fname.endswith("_log1p"):
                    x = float(np.log1p(max(x, 0.0)))

                stats = feature_stats.get(fname)
                if stats:
                    mean = stats["mean"]
                    std = stats["std"]
                    x = (x - mean) / std
                val = x

            elif meta.type == "binary":
                # one-hot encoded categorical: feature name is f"{source}_{category_value}"
                if source not in record:
                    val = 0.0
                else:
                    cat_val = str(record.get(source))
                    # extract category value by stripping "source_" prefix
                    prefix = f"{source}_"
                    category_value = fname[len(prefix) :] if fname.startswith(prefix) else fname
                    val = 1.0 if cat_val == category_value else 0.0

            else:
                # datetime-derived numeric features (e.g., col_year) or others
                if source in record:
                    try:
                        dt = pd.to_datetime(record[source], errors="coerce")
                    except Exception:
                        dt = pd.NaT
                    if fname.endswith("_year"):
                        val = float(dt.year) if not pd.isna(dt) else 0.0
                    elif fname.endswith("_month"):
                        val = float(dt.month) if not pd.isna(dt) else 0.0
                    elif fname.endswith("_dayofweek"):
                        val = float(dt.dayofweek) if not pd.isna(dt) else 0.0
                    else:
                        val = 0.0
                else:
                    val = 0.0

            row_vals.append(val)

        rows.append(row_vals)

    X = np.asarray(rows)
    y_pred = model.predict(X)
    # NumPy removed `np.asscalar`; use `.item()` for scalar conversion.
    predictions: List[Any] = [p.item() if np.ndim(p) == 0 else p for p in y_pred]  # type: ignore[arg-type]

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



