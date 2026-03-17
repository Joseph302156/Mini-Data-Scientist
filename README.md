# Mini Data Scientist

Mini Data Scientist is a full-stack analytics playground that behaves like a lightweight “AI data scientist”:

- Upload a CSV and get automatic profiling, cleaning, feature engineering, and EDA.
- Train simple regression/classification models with metrics and feature importances.
- See a modern dashboard with tables, charts, AI-generated insights, and an AI analyst chat.

Tech stack: **FastAPI**, **pandas/scikit-learn**, **Postgres + parquet** on the backend, and **React + Vite + Tailwind** on the frontend, with **OpenAI** for insights and chat.

---

## Features (backend)

### 1. Dataset ingestion

- `POST /api/datasets/upload`
  - Accepts CSV upload.
  - Reads into pandas, infers schema.
  - Immediately runs profiling, cleaning, and feature engineering.
  - Persists:
    - `datasets` row (Postgres) with profile/cleaning/feature metadata as JSON.
    - Three parquet files on disk:
      - `{dataset_id}_raw.parquet`
      - `{dataset_id}_cleaned.parquet`
      - `{dataset_id}_features.parquet`
    - `dataset_versions` rows for each stage.

### 2. Profiling, cleaning, feature engineering

Implemented in `backend/app/pipeline.py`:

- **Profiling**
  - Per-column type inference: numeric / categorical / datetime / text.
  - Missingness (count, ratio).
  - Cardinality.
  - Numeric stats: mean, std, min, max, quartiles, skewness, kurtosis.
  - Basic outlier thresholds (IQR / z-score).
  - Top values for categoricals.

- **Cleaning**
  - Drops:
    - Columns with > 90% missing or only 1 unique value.
    - Duplicate rows.
  - For each column:
    - Numeric: median imputation + IQR clipping.
    - Categorical: mode imputation.
    - Datetime: parse and drop rows with invalid timestamps.
    - Text: `str.strip()` normalization.
  - Returns structured `CleaningResult` (rows/cols before/after, per-column strategies).

- **Feature engineering**
  - Numeric:
    - Optional `log1p` for skewed non-negative columns.
    - Standardization (mean 0, std 1) for all numeric features.
  - Categorical:
    - One-hot encoding (pandas `get_dummies`).
  - Datetime:
    - Extracts `year`, `month`, `dayofweek`.
  - Returns structured `FeatureResult` describing each engineered feature.

### 3. Dataset APIs

- `GET /api/datasets`
  - List datasets: id, row/col counts, status.
- `GET /api/datasets/{id}/profile`
- `GET /api/datasets/{id}/cleaning`
- `GET /api/datasets/{id}/features`
  - Fetch stored JSON profile/cleaning/feature metadata.
- `GET /api/datasets/{id}/preview?stage=raw|cleaned|features&limit=20`
  - Preview table rows at each stage.
  - Uses in-memory cache if present, falls back to parquet otherwise.

### 4. EDA APIs

See `backend/app/eda.py`.

- `GET /api/datasets/{id}/eda?stage=cleaned`
  - Computes and returns:
    - `numeric_summary` (per-numeric-column stats).
    - `histograms` (bin edges + counts).
    - `correlations` (pairwise Pearson for numeric columns).
    - `trends` (basic resampled time series if a datetime column exists).
    - `grouped_stats` (mean/sum/count for numeric columns by top-k categories).

### 5. Modeling APIs

See `backend/app/modeling.py`.

- `GET /api/datasets/{id}/targets`
  - Suggests target columns and tasks based on feature metadata (numeric → regression, else classification).

- `POST /api/datasets/{id}/models`
  - Train a model on the features dataset:
    - Task: regression or classification.
    - Model: linear or random forest.
  - Splits into train/test.
  - Computes metrics:
    - Regression: RMSE, R².
    - Classification: accuracy.
  - Extracts feature importances (tree importances or abs(coef)).
  - Persists model via `joblib` and a `model_runs` row.

- `GET /api/datasets/{id}/models`
  - List all models for a dataset, with metrics + feature importance summaries.

- `POST /api/models/{model_id}/predict`
  - Accepts records as **feature-space** dicts (keys = feature names).
  - Aligns to training feature order.
  - Returns predictions (and probabilities for classifiers).

- `POST /api/models/{model_id}/predict_raw`
  - Accepts records in **original schema** (raw columns).
  - Rebuilds the feature vector using `FeatureResult` + training stats:
    - Numeric transforms (log1p, standardization).
    - One-hot encoding for categoricals.
    - Datetime-derived features.
  - Runs predictions as above.

### 6. Automated insights API

See `backend/app/insights.py`.

- `GET /api/datasets/{id}/insights?model_id=...&stage=cleaned`
  - Computes EDA (`/eda`) and bundles it with:
    - Profile, cleaning, features metadata.
    - Optional model metrics + feature importances.
  - Calls OpenAI chat completions (configurable) to generate bullet-point insights.
  - Returns:
    - Full EDA payload.
    - `insights_text` (LLM output).

### 7. AI chat interface API

See `backend/app/chat.py`.

- `POST /api/chat`
  - Body:
    - `dataset_id` (required).
    - `model_id` (optional).
    - `question` (natural-language question).
  - Builds a prompt with:
    - EDA results.
    - Profile/cleaning/features JSON.
    - Optional model metrics/importances.
  - Asks OpenAI to answer as an “AI data analyst”.
  - Returns `ChatResponse` with the answer.

---

## Features (frontend)

Located in `frontend/`, built with React + Vite + Tailwind.

### Layout

- Dark, minimal layout inspired by Cursor / OpenAI / Notion:
  - Left sidebar: datasets + upload.
  - Main content: tables + charts.
  - Right column: models, insights, AI chat.

### Main UI regions

- **Dataset sidebar**
  - Upload CSV (maps to `POST /api/datasets/upload`).
  - List all datasets (`GET /api/datasets`).
  - Select dataset to drive the rest of the dashboard.

- **Data preview panel**
  - Toggle between `raw | cleaned | features` views.
  - Renders small table from `/api/datasets/{id}/preview`.

- **EDA charts panel**
  - Uses `/api/datasets/{id}/eda`.
  - Shows:
    - Simple histograms for top numeric columns.
    - List of strongest correlations.

- **Model panel**
  - Fetches target suggestions (`/api/datasets/{id}/targets`).
  - Shows existing models (`/api/datasets/{id}/models`).
  - Allows training new models with:
    - Target column.
    - Task type (regression / classification).
    - Model type (random forest / linear).
  - Highlights metrics and top important features.

- **Insights panel**
  - Calls `/api/datasets/{id}/insights?model_id=...`.
  - Displays LLM-generated explanations and observations.

- **AI chat panel**
  - Simple chat UI.
  - Sends `POST /api/chat` with the current dataset/model + user question.
  - Displays AI answers as chat bubbles.

---

## Running the project

### 1. Prerequisites

- Python 3.10+ (for FastAPI, pandas, scikit-learn, etc.)
- Node.js 18+ (for Vite/React frontend)
- Postgres running locally (or update `DATABASE_URL` accordingly)

### 2. Backend setup

From the project root:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install fastapi uvicorn[standard] sqlalchemy psycopg2-binary pandas numpy scikit-learn httpx python-dotenv
```

Configure env vars in `backend/.env` (example):

```bash
INSIGHTS_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key-here
INSIGHTS_MODEL=gpt-4o-mini
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/minids
```

Start Postgres and create the `minids` database if needed.

Run the backend (from `backend/`):

```bash
env $(cat .env | xargs) uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

FastAPI will be available at `http://localhost:8000`.

### 3. Frontend setup

From the project root:

```bash
cd frontend
npm install
```

The frontend dev server is started automatically by the helper script below.

### 4. One-command dev run

From the project root:

```bash
chmod +x dev.sh   # first time only
./dev.sh
```

- Starts the backend with `uvicorn` using `backend/.env`.
- Starts the frontend Vite dev server.
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

### 5. Basic workflow

1. Open the frontend in your browser (`http://localhost:5173`).
2. Use the sidebar to upload a CSV.
3. Explore:
   - Dataset preview (raw / cleaned / features).
   - EDA charts.
4. Choose a target and train a model.
5. Read AI-generated insights.
6. Ask questions in the chat panel about the dataset and model.

---

## Notes and future work

- Add dataset naming / descriptions (not just UUIDs).
- More configurable cleaning / feature pipelines (user-tunable presets).
- Per-row explanation (“why this prediction?”) using local feature attributions.
- Auth, multi-user workspaces, and persistence hardening for production. 

