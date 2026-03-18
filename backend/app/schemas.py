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


class DatasetInfo(BaseModel):
    dataset_id: str
    n_rows: int
    n_cols: int
    status: DatasetStatus


class DataPreview(BaseModel):
    dataset_id: str
    stage: str  # "raw" | "cleaned" | "features"
    n_rows: int
    n_cols: int
    preview: List[Dict[str, Any]]


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ModelType(str, Enum):
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"


class TrainModelRequest(BaseModel):
    target_column: str
    task_type: TaskType
    model_type: ModelType = ModelType.RANDOM_FOREST
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42


class Metric(BaseModel):
    name: str
    value: float


class TrainedModelSummary(BaseModel):
    model_id: str
    dataset_id: str
    target_column: str
    task_type: TaskType
    model_type: ModelType
    created_at: str
    metrics: List[Metric]
    feature_importances: Dict[str, float]
    model_explanation: Optional[str] = None
    top_feature_stats: Optional[List["FeatureStat"]] = None


class FeatureStat(BaseModel):
    feature_name: str
    feature_type: str  # "numeric" | "binary" | "time_component"
    mean: Optional[float] = None
    total: Optional[float] = None  # sum / total occurrences where meaningful
    count_true: Optional[int] = None  # for one-hot/binary
    prevalence: Optional[float] = None  # fraction true for one-hot/binary
    note: Optional[str] = None


class EdaHistogram(BaseModel):
    column: str
    bin_edges: List[float]
    counts: List[int]


class EdaCorrelation(BaseModel):
    column_x: str
    column_y: str
    value: float


class EdaLineSeries(BaseModel):
    name: str
    points: List[Dict[str, Any]]  # [{"x": iso_datetime or index, "y": float}]


class EdaGroupedStat(BaseModel):
    group_column: str
    group_value: str
    value_column: str
    mean: float
    sum: float
    count: int


class EdaResult(BaseModel):
    dataset_id: str
    stage: str
    numeric_summary: Dict[str, Dict[str, float]]
    histograms: List[EdaHistogram]
    correlations: List[EdaCorrelation]
    trends: List[EdaLineSeries]
    grouped_stats: List[EdaGroupedStat]
    datetime_column: Optional[str] = None
    group_by_column: Optional[str] = None


class TargetCandidate(BaseModel):
    column: str
    suggested_task: TaskType  # regression if numeric, classification otherwise


class TargetListResponse(BaseModel):
    dataset_id: str
    candidates: List[TargetCandidate]


class ChatRequest(BaseModel):
    dataset_id: str
    model_id: Optional[str] = None
    question: str


class ChatResponse(BaseModel):
    dataset_id: str
    model_id: Optional[str] = None
    question: str
    answer: str


class PredictRequest(BaseModel):
    # List of raw feature dicts; keys should align with training features
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    model_id: str
    dataset_id: str
    target_column: str
    task_type: TaskType
    model_type: ModelType
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None


# Resolve forward references for optional fields.
TrainedModelSummary.model_rebuild()




