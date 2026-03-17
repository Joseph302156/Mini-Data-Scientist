from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from .schemas import (
    CleaningColumnSummary,
    CleaningResult,
    ColumnFlags,
    ColumnProfile,
    ColumnType,
    DatasetProfile,
    FeatureResult,
    FeatureSummary,
)


@dataclass
class DatasetState:
    dataset_id: str
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame | None = None
    features_df: pd.DataFrame | None = None
    profile: DatasetProfile | None = None
    cleaning_result: CleaningResult | None = None
    feature_result: FeatureResult | None = None


DATASETS: Dict[str, DatasetState] = {}


def _infer_column_type(series: pd.Series) -> ColumnType:
    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERIC

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    # try to parse datetime heuristically
    sample = series.dropna().astype(str).head(50)
    if not sample.empty:
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() > 0.8:
            return ColumnType.DATETIME

    # low cardinality non-numeric -> categorical
    n_unique = series.nunique(dropna=True)
    if 0 < n_unique <= 50:
        return ColumnType.CATEGORICAL

    return ColumnType.TEXT


def profile_dataset(dataset_id: str, df: pd.DataFrame) -> DatasetProfile:
    columns: list[ColumnProfile] = []
    row_count = int(len(df))

    for col in df.columns:
        s = df[col]
        original_dtype = str(s.dtype)
        inferred_type = _infer_column_type(s)
        missing_count = int(s.isna().sum())
        missing_ratio = float(missing_count / row_count) if row_count else 0.0
        n_unique = int(s.nunique(dropna=True))

        flags = ColumnFlags()
        flags.high_missing = missing_ratio > 0.3
        flags.high_cardinality = n_unique > 100

        numeric_stats = None
        outlier_thresholds = None
        top_values = None

        if inferred_type == ColumnType.NUMERIC:
            s_num = pd.to_numeric(s, errors="coerce")
            desc = s_num.describe(percentiles=[0.25, 0.5, 0.75])
            mean = float(desc.get("mean", np.nan))
            std = float(desc.get("std", np.nan))
            p25 = float(desc.get("25%", np.nan))
            p50 = float(desc.get("50%", np.nan))
            p75 = float(desc.get("75%", np.nan))
            min_val = float(desc.get("min", np.nan))
            max_val = float(desc.get("max", np.nan))
            skew = float(s_num.skew()) if s_num.notna().any() else 0.0
            kurt = float(s_num.kurtosis()) if s_num.notna().any() else 0.0

            flags.skewed = abs(skew) > 1.0

            iqr = p75 - p25
            iqr_low = p25 - 1.5 * iqr
            iqr_high = p75 + 1.5 * iqr
            z_low, z_high = -3.0, 3.0

            numeric_stats = {
                "mean": mean,
                "std": std,
                "min": min_val,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "max": max_val,
                "skewness": skew,
                "kurtosis": kurt,
            }
            outlier_thresholds = {
                "iqr_low": float(iqr_low),
                "iqr_high": float(iqr_high),
                "z_low": float(z_low),
                "z_high": float(z_high),
            }

        if inferred_type == ColumnType.CATEGORICAL:
            vc = s.value_counts(dropna=True).head(10)
            top_values = [
                {"value": str(idx), "count": int(count)} for idx, count in vc.items()
            ]

        col_profile = ColumnProfile(
            name=col,
            inferred_type=inferred_type,
            original_dtype=original_dtype,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            n_unique=n_unique,
            flags=flags,
            numeric_stats=numeric_stats,
            categorical_stats=None,
            datetime_stats=None,
            outlier_thresholds=outlier_thresholds,
            top_values=top_values,
        )
        columns.append(col_profile)

    profile = DatasetProfile(
        dataset_id=dataset_id,
        row_count=row_count,
        column_count=len(df.columns),
        columns=columns,
    )
    return profile


def _auto_clean(df: pd.DataFrame, profile: DatasetProfile) -> Tuple[pd.DataFrame, CleaningResult]:
    df_clean = df.copy()
    n_rows_before = int(len(df_clean))
    n_cols_before = int(df_clean.shape[1])

    per_column_summary: dict[str, CleaningColumnSummary] = {}
    dropped_columns: list[str] = []

    # drop obvious junk: columns with > 90% missing or single unique value
    for col_profile in profile.columns:
        if col_profile.missing_ratio > 0.9 or col_profile.n_unique <= 1:
            dropped_columns.append(col_profile.name)

    if dropped_columns:
        df_clean = df_clean.drop(columns=dropped_columns, errors="ignore")

    # simple duplicate drop
    df_clean = df_clean.drop_duplicates()

    # per-column cleaning
    for col_profile in profile.columns:
        col = col_profile.name
        if col not in df_clean.columns:
            continue

        s = df_clean[col]
        missing_before = int(s.isna().sum())
        missing_after = missing_before
        outliers_clipped = 0
        rows_removed_as_outliers = 0
        strategy: dict[str, str] = {}

        if col_profile.inferred_type == ColumnType.NUMERIC:
            s_num = pd.to_numeric(s, errors="coerce")
            median = float(s_num.median()) if s_num.notna().any() else 0.0
            s_num = s_num.fillna(median)
            missing_after = int(s_num.isna().sum())
            strategy["missing"] = "median"

            if col_profile.outlier_thresholds:
                low = col_profile.outlier_thresholds.iqr_low
                high = col_profile.outlier_thresholds.iqr_high
                before_clip = s_num.copy()
                s_num = s_num.clip(lower=low, upper=high)
                outliers_clipped = int((before_clip != s_num).sum())
                strategy["outliers"] = "clip_iqr"

            df_clean[col] = s_num

        elif col_profile.inferred_type == ColumnType.CATEGORICAL:
            mode = s.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "__MISSING__"
            s_cat = s.fillna(fill_value)
            missing_after = int(s_cat.isna().sum())
            strategy["missing"] = "mode"
            df_clean[col] = s_cat

        elif col_profile.inferred_type == ColumnType.DATETIME:
            s_dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            df_clean[col] = s_dt
            # drop rows where datetime could not be parsed
            mask_valid = df_clean[col].notna()
            invalid_rows = int((~mask_valid).sum())
            if invalid_rows:
                df_clean = df_clean.loc[mask_valid].copy()
                rows_removed_as_outliers += invalid_rows
            missing_after = int(df_clean[col].isna().sum())
            strategy["invalid_datetime_rows"] = "drop"

        else:
            # text: simple trim + leave missing as-is
            df_clean[col] = s.astype(str).str.strip()
            strategy["text_normalization"] = "strip"

        per_column_summary[col] = CleaningColumnSummary(
            missing_before=missing_before,
            missing_after=missing_after,
            outliers_clipped=outliers_clipped,
            rows_removed_as_outliers=rows_removed_as_outliers,
            strategy=strategy,
        )

    n_rows_after = int(len(df_clean))
    n_cols_after = int(df_clean.shape[1])

    result = CleaningResult(
        dataset_id=profile.dataset_id,
        n_rows_before=n_rows_before,
        n_rows_after=n_rows_after,
        n_cols_before=n_cols_before,
        n_cols_after=n_cols_after,
        dropped_columns=dropped_columns,
        dropped_rows=n_rows_before - n_rows_after,
        per_column_summary=per_column_summary,
    )
    return df_clean, result


def _auto_feature_engineer(
    dataset_id: str,
    df_clean: pd.DataFrame,
    profile: DatasetProfile,
) -> Tuple[pd.DataFrame, FeatureResult]:
    df_features = pd.DataFrame(index=df_clean.index)
    feature_summaries: list[FeatureSummary] = []

    # numeric features: optional log for skewed positive, then standardize
    for col_profile in profile.columns:
        col = col_profile.name
        if col not in df_clean.columns:
            continue

        if col_profile.inferred_type == ColumnType.NUMERIC:
            s = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)

            # log1p if skewed and non-negative
            if col_profile.flags.skewed and s.min() >= 0:
                transformed = np.log1p(s)
                feature_name = f"{col}_log1p"
                source = f"{col} (log1p)"
            else:
                transformed = s
                feature_name = col
                source = col

            mean = transformed.mean()
            std = transformed.std() or 1.0
            standardized = (transformed - mean) / std

            df_features[feature_name] = standardized
            feature_summaries.append(
                FeatureSummary(
                    name=feature_name,
                    source=source,
                    type="numeric",
                    role="feature",
                )
            )

    # categorical: one-hot encode
    for col_profile in profile.columns:
        col = col_profile.name
        if col not in df_clean.columns:
            continue
        if col_profile.inferred_type != ColumnType.CATEGORICAL:
            continue

        dummies = pd.get_dummies(df_clean[col].astype("category"), prefix=col)
        df_features = pd.concat([df_features, dummies], axis=1)
        for dummy_col in dummies.columns:
            feature_summaries.append(
                FeatureSummary(
                    name=dummy_col,
                    source=col,
                    type="binary",
                    role="feature",
                )
            )

    # datetime: basic time components
    for col_profile in profile.columns:
        col = col_profile.name
        if col not in df_clean.columns:
            continue
        if col_profile.inferred_type != ColumnType.DATETIME:
            continue

        s_dt = pd.to_datetime(df_clean[col], errors="coerce")
        df_features[f"{col}_year"] = s_dt.dt.year
        df_features[f"{col}_month"] = s_dt.dt.month
        df_features[f"{col}_dayofweek"] = s_dt.dt.dayofweek

        feature_summaries.extend(
            [
                FeatureSummary(
                    name=f"{col}_year",
                    source=col,
                    type="numeric",
                    role="feature",
                ),
                FeatureSummary(
                    name=f"{col}_month",
                    source=col,
                    type="numeric",
                    role="feature",
                ),
                FeatureSummary(
                    name=f"{col}_dayofweek",
                    source=col,
                    type="numeric",
                    role="feature",
                ),
            ]
        )

    result = FeatureResult(
        dataset_id=dataset_id,
        n_rows=int(len(df_features)),
        n_raw_features=int(df_clean.shape[1]),
        n_final_features=int(df_features.shape[1]),
        target_column=None,
        feature_summary=feature_summaries,
    )
    return df_features, result


def ingest_and_process(df: pd.DataFrame) -> tuple[str, DatasetProfile, CleaningResult, FeatureResult]:
    dataset_id = str(uuid4())

    profile = profile_dataset(dataset_id, df)
    cleaned_df, cleaning_result = _auto_clean(df, profile)
    features_df, feature_result = _auto_feature_engineer(dataset_id, cleaned_df, profile)

    DATASETS[dataset_id] = DatasetState(
        dataset_id=dataset_id,
        raw_df=df,
        cleaned_df=cleaned_df,
        features_df=features_df,
        profile=profile,
        cleaning_result=cleaning_result,
        feature_result=feature_result,
    )

    return dataset_id, profile, cleaning_result, feature_result


def get_dataset_state(dataset_id: str) -> DatasetState:
    state = DATASETS.get(dataset_id)
    if state is None:
        raise KeyError(f"Unknown dataset_id: {dataset_id}")
    return state


def list_dataset_states() -> list[DatasetState]:
    return list(DATASETS.values())


