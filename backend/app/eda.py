from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .models import Dataset, DatasetVersion
from .schemas import EdaCorrelation, EdaGroupedStat, EdaHistogram, EdaLineSeries, EdaResult


def _numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty:
            continue
        desc = s.describe(percentiles=[0.25, 0.5, 0.75])
        summary[col] = {
            "count": float(desc.get("count", 0.0)),
            "mean": float(desc.get("mean", 0.0)),
            "std": float(desc.get("std", 0.0)),
            "min": float(desc.get("min", 0.0)),
            "p25": float(desc.get("25%", 0.0)),
            "p50": float(desc.get("50%", 0.0)),
            "p75": float(desc.get("75%", 0.0)),
            "max": float(desc.get("max", 0.0)),
        }
    return summary


def _histograms(df: pd.DataFrame, bins: int = 20) -> List[EdaHistogram]:
    histograms: List[EdaHistogram] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty:
            continue
        counts, bin_edges = np.histogram(s, bins=bins)
        histograms.append(
            EdaHistogram(
                column=col,
                bin_edges=[float(x) for x in bin_edges],
                counts=[int(c) for c in counts],
            )
        )
    return histograms


def _correlations(df: pd.DataFrame, max_columns: int = 30) -> List[EdaCorrelation]:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    numeric_cols = numeric_cols[:max_columns]
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr()
    result: List[EdaCorrelation] = []
    for i, col_i in enumerate(numeric_cols):
        for j, col_j in enumerate(numeric_cols):
            if j <= i:
                continue
            val = corr.loc[col_i, col_j]
            if pd.isna(val):
                continue
            result.append(
                EdaCorrelation(
                    column_x=col_i,
                    column_y=col_j,
                    value=float(val),
                )
            )
    return result


def _time_trends(
    df: pd.DataFrame,
    datetime_column: Optional[str],
    value_columns: List[str],
    freq: str = "D",
) -> List[EdaLineSeries]:
    if datetime_column is None or datetime_column not in df.columns:
        return []

    s_dt = pd.to_datetime(df[datetime_column], errors="coerce")
    mask = s_dt.notna()
    if not mask.any():
        return []

    df_t = df.loc[mask].copy()
    df_t[datetime_column] = s_dt[mask]
    df_t = df_t.set_index(datetime_column)

    series_list: List[EdaLineSeries] = []
    for col in value_columns:
        if col not in df_t.columns:
            continue
        s = pd.to_numeric(df_t[col], errors="coerce")
        agg = s.resample(freq).mean().dropna()
        if agg.empty:
            continue
        series_list.append(
            EdaLineSeries(
                name=col,
                points=[{"x": ts.isoformat(), "y": float(v)} for ts, v in agg.items()],
            )
        )
    return series_list


def _grouped_stats(
    df: pd.DataFrame,
    group_by: str,
    value_columns: List[str],
    top_k: int = 10,
) -> List[EdaGroupedStat]:
    if group_by not in df.columns:
        return []

    vc = df[group_by].value_counts(dropna=True).head(top_k)
    top_values = list(vc.index)
    df_g = df[df[group_by].isin(top_values)]

    grouped: List[EdaGroupedStat] = []
    for val in top_values:
        subset = df_g[df_g[group_by] == val]
        for col in value_columns:
            if col not in subset.columns:
                continue
            s = pd.to_numeric(subset[col], errors="coerce").dropna()
            if s.empty:
                continue
            grouped.append(
                EdaGroupedStat(
                    group_column=group_by,
                    group_value=str(val),
                    value_column=col,
                    mean=float(s.mean()),
                    sum=float(s.sum()),
                    count=int(s.count()),
                )
            )
    return grouped


def compute_eda_for_dataset(
    db: Session,
    dataset_id: str,
    use_stage: str = "cleaned",
    datetime_hint: Optional[str] = None,
    group_by_hint: Optional[str] = None,
) -> EdaResult:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if ds is None:
        raise ValueError("Dataset not found")

    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.stage == use_stage)
        .one_or_none()
    )
    if version is None:
        raise ValueError(f"No {use_stage} data available for this dataset")

    df = pd.read_parquet(version.storage_path)

    numeric_summary = _numeric_summary(df)
    histograms = _histograms(df)
    correlations = _correlations(df)

    datetime_col = datetime_hint
    if datetime_col is None:
        # heuristic: first datetime-like column
        for col in df.columns:
            s = pd.to_datetime(df[col], errors="ignore")
            if np.issubdtype(s.dtype, np.datetime64):
                datetime_col = col
                break

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    value_cols = numeric_cols[:5]
    trends = _time_trends(df, datetime_col, value_cols)

    group_by = group_by_hint
    if group_by is None:
        # heuristic: first non-numeric with low cardinality
        for col in df.columns:
            if col in numeric_cols:
                continue
            n_uniq = df[col].nunique(dropna=True)
            if 1 < n_uniq <= 20:
                group_by = col
                break

    grouped = _grouped_stats(df, group_by, value_cols) if group_by else []

    return EdaResult(
        dataset_id=dataset_id,
        stage=use_stage,
        numeric_summary=numeric_summary,
        histograms=histograms,
        correlations=correlations,
        trends=trends,
        grouped_stats=grouped,
        datetime_column=datetime_col,
        group_by_column=group_by,
    )

