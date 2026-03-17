from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .db import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    n_rows = Column(Integer, nullable=True)
    n_cols = Column(Integer, nullable=True)
    status = Column(String, nullable=False, default="featured")

    profile_json = Column(JSONB, nullable=True)
    cleaning_json = Column(JSONB, nullable=True)
    features_json = Column(JSONB, nullable=True)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    stage = Column(String, nullable=False)  # raw | cleaned | features
    storage_path = Column(String, nullable=False)  # parquet file path
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    target_column = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # regression | classification
    model_type = Column(String, nullable=False)  # linear | random_forest
    model_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metrics_json = Column(JSONB, nullable=False)
    feature_importances_json = Column(JSONB, nullable=True)



