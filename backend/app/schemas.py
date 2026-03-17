from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetStatus(str, Enum):
    RAW = "raw"
    PROFILED = "profiled"
    CLEANED = "cleaned"
    FEATURED = "featured"


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"


class TopValue(BaseModel):
    value: Any
    count: int


class NumericStats(BaseModel):
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    max: float
    skewness: float
    kurtosis: float


class OutlierThresholds(BaseModel):
    iqr_low: float
    iqr_high: float
    z_low: float
    z_high: float


class ColumnFlags(BaseModel):
    high_missing: bool = False
    high_cardinality: bool = False
    skewed: bool = False
    many_outliers: bool = False


class ColumnProfile(BaseModel):
    name: str
    inferred_type: ColumnType
    original_dtype: str
    missing_count: int
    missing_ratio: float
    n_unique: int
    flags: ColumnFlags
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[Dict[str, Any]] = None
    datetime_stats: Optional[Dict[str, Any]] = None
    outlier_thresholds: Optional[OutlierThresholds] = None
    top_values: Optional[List[TopValue]] = None


class DatasetProfile(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]


class CleaningColumnSummary(BaseModel):
    missing_before: int
    missing_after: int
    outliers_clipped: int = 0
    rows_removed_as_outliers: int = 0
    strategy: Dict[str, Any] = Field(default_factory=dict)


class CleaningResult(BaseModel):
    dataset_id: str
    n_rows_before: int
    n_rows_after: int
    n_cols_before: int
    n_cols_after: int
    dropped_columns: List[str] = Field(default_factory=list)
    dropped_rows: int = 0
    per_column_summary: Dict[str, CleaningColumnSummary] = Field(default_factory=dict)


class FeatureSummary(BaseModel):
    name: str
    source: str
    type: str
    role: str


class FeatureResult(BaseModel):
    dataset_id: str
    n_rows: int
    n_raw_features: int
    n_final_features: int
    target_column: Optional[str]
    feature_summary: List[FeatureSummary]


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    status: DatasetStatus
    profile: DatasetProfile
    cleaning: CleaningResult
    features: FeatureResult

