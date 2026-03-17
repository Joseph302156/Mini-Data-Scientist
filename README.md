# Mini Data Scientist

Mini Data Scientist is a full-stack analytics playground that behaves like a lightweight “AI data scientist”:

- Upload a CSV and get automatic profiling, cleaning, feature engineering, and EDA.
- Train simple regression/classification models with metrics and feature importances.
- See a modern dashboard with tables, charts, AI-generated insights, and an AI analyst chat.

Tech stack: **FastAPI**, **pandas/scikit-learn**, **Postgres + parquet** on the backend, and **React + Vite + Tailwind** on the frontend, with **OpenAI** for insights and chat.

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

