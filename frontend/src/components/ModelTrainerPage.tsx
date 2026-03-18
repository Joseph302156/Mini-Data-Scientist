import { useEffect, useMemo, useState } from "react";
import axios from "axios";

type TargetCandidate = {
  column: string;
  suggested_task: "regression" | "classification";
};

type Metric = { name: string; value: number };

type FeatureSummaryItem = {
  name: string;
  source: string;
  type: "numeric" | "binary";
  role: string;
};

type FeatureResult = {
  dataset_id: string;
  feature_summary: FeatureSummaryItem[];
};

type ColumnProfile = {
  name: string;
  inferred_type: "numeric" | "categorical" | "datetime" | "text";
  top_values?: Array<{ value: unknown; count: number }>;
};

type DatasetProfile = {
  columns: ColumnProfile[];
};

type TrainedModelSummary = {
  model_id: string;
  dataset_id: string;
  target_column: string;
  task_type: "regression" | "classification";
  model_type: "linear" | "random_forest";
  created_at: string;
  metrics: Metric[];
  feature_importances: Record<string, number>;
  model_explanation?: string | null;
};

type PredictResponse = {
  model_id: string;
  dataset_id: string;
  target_column: string;
  task_type: "regression" | "classification";
  model_type: "linear" | "random_forest";
  predictions: any[];
  probabilities?: Array<Record<string, number>> | null;
};

type Props = {
  datasetId: string | null;
  selectedModelId: string | null;
  onSelectModel: (id: string | null) => void;
  onBack: () => void;
  trainerSeed?: {
    target: string;
    taskType: "regression" | "classification";
    modelType: "linear" | "random_forest";
  } | null;
};

function normalizeImportanceEntries(importances: Record<string, number>) {
  return Object.entries(importances || {})
    .slice()
    .sort((a, b) => b[1] - a[1]);
}

function isTimeComponentFeature(featureName: string) {
  return (
    featureName.endsWith("_year") ||
    featureName.endsWith("_month") ||
    featureName.endsWith("_dayofweek")
  );
}

export function ModelTrainerPage({
  datasetId,
  selectedModelId,
  onSelectModel,
  onBack,
  trainerSeed,
}: Props) {
  const [targets, setTargets] = useState<TargetCandidate[]>([]);
  const [target, setTarget] = useState<string>("");
  const [taskType, setTaskType] = useState<"regression" | "classification">("regression");
  const [modelType, setModelType] = useState<"linear" | "random_forest">("random_forest");
  const [models, setModels] = useState<TrainedModelSummary[]>([]);
  const [training, setTraining] = useState(false);

  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [features, setFeatures] = useState<FeatureResult | null>(null);

  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [predicting, setPredicting] = useState(false);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) {
      setTargets([]);
      setModels([]);
      setProfile(null);
      setFeatures(null);
      return;
    }

    async function load() {
      const [tRes, mRes, pRes, fRes] = await Promise.all([
        axios.get(`/api/datasets/${datasetId}/targets`),
        axios.get(`/api/datasets/${datasetId}/models`),
        axios.get(`/api/datasets/${datasetId}/profile`),
        axios.get(`/api/datasets/${datasetId}/features`),
      ]);

      setTargets(tRes.data.candidates);
      setModels(mRes.data);
      setProfile(pRes.data);
      setFeatures(fRes.data);

      if (trainerSeed) {
        setTarget(trainerSeed.target || "");
        setTaskType(trainerSeed.taskType);
        setModelType(trainerSeed.modelType);
      } else if (!target && tRes.data.candidates.length > 0) {
        const first = tRes.data.candidates[0];
        setTarget(first.column);
        setTaskType(first.suggested_task);
      }
    }

    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  const selectedModel = useMemo(() => {
    if (!selectedModelId) return null;
    return models.find((m) => m.model_id === selectedModelId) || null;
  }, [models, selectedModelId]);

  const requiredInputKeys = useMemo(() => {
    if (!selectedModel || !features) return [];

    const metaByName = new Map<string, FeatureSummaryItem>();
    for (const fs of features.feature_summary) metaByName.set(fs.name, fs);

    const keys = new Set<string>();
    for (const featureName of Object.keys(selectedModel.feature_importances || {})) {
      const meta = metaByName.get(featureName);
      if (!meta) continue;

      // Raw prediction expects:
      // - numeric log1p features: input key is the base column (strip `_log1p`)
      // - binary: input key is the categorical source column
      // - datetime components: input key is the datetime source column
      if (meta.type === "numeric" && featureName.endsWith("_log1p")) {
        const base = featureName.replace(/_log1p$/, "");
        keys.add(base);
      } else if (isTimeComponentFeature(featureName)) {
        keys.add(meta.source);
      } else {
        keys.add(meta.source);
      }
    }

    // Convert set -> stable list
    return Array.from(keys.values()).sort();
  }, [features, selectedModel]);

  useEffect(() => {
    // Reset prediction form when model changes
    if (!selectedModelId || requiredInputKeys.length === 0) {
      setInputs({});
      setPredictResult(null);
      return;
    }
    const next: Record<string, string> = {};
    for (const k of requiredInputKeys) next[k] = inputs[k] ?? "";
    setInputs(next);
    setPredictResult(null);
    setPredictError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId, requiredInputKeys.join("|")]);

  async function train() {
    if (!datasetId || !target) return;
    setTraining(true);
    try {
      const res = await axios.post(`/api/datasets/${datasetId}/models`, {
        target_column: target,
        task_type: taskType,
        model_type: modelType,
      });
      const trained = res.data as TrainedModelSummary;

      setModels((prev) => [trained, ...prev]);
      onSelectModel(trained.model_id);
    } finally {
      setTraining(false);
    }
  }

  function getColumnType(colName: string) {
    const c = profile?.columns.find((x) => x.name === colName);
    return c?.inferred_type || "text";
  }

  function buildRawRecord() {
    const record: Record<string, any> = {};
    for (const [k, v] of Object.entries(inputs)) {
      const t = getColumnType(k);
      if (t === "numeric") {
        const num = v === "" ? 0 : Number(v);
        record[k] = Number.isFinite(num) ? num : 0;
      } else {
        record[k] = v;
      }
    }
    return record;
  }

  async function predict() {
    if (!selectedModelId) return;
    setPredicting(true);
    setPredictError(null);
    try {
      const rec = buildRawRecord();
      const res = await axios.post<PredictResponse>(`/api/models/${selectedModelId}/predict_raw`, {
        records: [rec],
      });
      setPredictResult(res.data);
    } catch (e: any) {
      const detail =
        e?.response?.data?.detail ||
        e?.response?.data?.message ||
        e?.message ||
        "Prediction request failed.";
      setPredictError(String(detail));
      // eslint-disable-next-line no-console
      console.error(e);
    } finally {
      setPredicting(false);
    }
  }

  const topImportance = useMemo(() => {
    if (!selectedModel) return [];
    return normalizeImportanceEntries(selectedModel.feature_importances).slice(0, 10);
  }, [selectedModel]);

  const maxImportance = useMemo(() => {
    const vals = topImportance.map((x) => x[1]);
    return Math.max(...vals, 1e-9);
  }, [topImportance]);

  if (!datasetId) {
    return (
      <div className="p-6 text-xs text-neutral-500">
        Upload/select a dataset to train models and make predictions.
      </div>
    );
  }

  return (
    <div className="p-4 lg:p-6 h-full flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold text-neutral-200">Model & Predict</div>
          <div className="text-[11px] text-neutral-500">Train a model, visualize signals, score a future row.</div>
        </div>
        <button
          type="button"
          onClick={onBack}
          className="px-3 py-1.5 rounded-md border border-neutral-700 hover:bg-surface-800 text-[11px] transition text-neutral-200"
        >
          Back to dashboard
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 flex-1 overflow-hidden">
        <div className="xl:col-span-1 space-y-3 overflow-y-auto pr-1">
          <div className="rounded-xl border border-neutral-800 bg-surface-800/70 backdrop-blur-md p-4 text-xs space-y-3">
            <div className="text-xs font-medium text-neutral-200">Train model</div>

            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="w-full bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[12px] outline-none focus:border-accent-400"
            >
              <option value="">Target column</option>
              {targets.map((t) => (
                <option key={t.column} value={t.column}>
                  {t.column} ({t.suggested_task})
                </option>
              ))}
            </select>

            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value as typeof taskType)}
              className="w-full bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[12px] outline-none focus:border-accent-400"
            >
              <option value="regression">Regression</option>
              <option value="classification">Classification</option>
            </select>

            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as typeof modelType)}
              className="w-full bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[12px] outline-none focus:border-accent-400"
            >
              <option value="random_forest">Random forest</option>
              <option value="linear">Linear</option>
            </select>

            <button
              type="button"
              onClick={train}
              disabled={training || !target}
              className="w-full px-3 py-2 rounded-md bg-accent-500 text-white text-[12px] hover:bg-accent-400 disabled:bg-neutral-700 disabled:text-neutral-400 transition"
            >
              {training ? "Training…" : "Train & save model"}
            </button>

            <div className="pt-2 border-t border-neutral-800">
              <div className="text-[11px] text-neutral-400 mb-2">Existing models</div>
              <div className="space-y-2">
                {models.length === 0 && <div className="text-[11px] text-neutral-500">No models yet.</div>}
                {models.map((m) => (
                  <button
                    key={m.model_id}
                    type="button"
                    onClick={() => onSelectModel(m.model_id)}
                    className={`w-full text-left rounded-md border p-2 transition ${
                      selectedModelId === m.model_id
                        ? "border-accent-500/60 bg-accent-500/15"
                        : "border-neutral-800 bg-surface-900/40 hover:border-neutral-600"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="text-[12px] text-neutral-100 font-medium truncate">
                        {m.target_column} · {m.task_type}
                      </div>
                      <div className="text-[10px] uppercase tracking-wide text-neutral-500">{m.model_type}</div>
                    </div>
                    <div className="mt-1 text-[11px] text-neutral-400 space-x-2">
                      {m.metrics.map((me) => (
                        <span key={me.name}>
                          {me.name}:{me.value.toFixed(me.name === "accuracy" ? 3 : 2)}{" "}
                        </span>
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="xl:col-span-1 space-y-3 overflow-y-auto pr-1">
          <div className="rounded-xl border border-neutral-800 bg-surface-800/70 backdrop-blur-md p-4 text-xs space-y-3">
            <div className="text-xs font-medium text-neutral-200">Model signals</div>
            {!selectedModel && (
              <div className="text-[11px] text-neutral-500">Select a trained model to see what it learned.</div>
            )}
            {selectedModel && (
              <>
                <div className="text-[12px] text-neutral-200 font-medium">
                  Predicting `{selectedModel.target_column}` ({selectedModel.task_type})
                </div>

                {selectedModel.model_explanation && (
                  <div className="text-[11px] text-neutral-300 leading-relaxed">
                    {selectedModel.model_explanation}
                  </div>
                )}

                <div className="pt-2 border-t border-neutral-800">
                  <div className="text-[11px] text-neutral-400 mb-2">Top feature importances</div>
                  {topImportance.length === 0 && <div className="text-[11px] text-neutral-500">No feature importances.</div>}
                  {topImportance.map(([name, val]) => {
                    const pct = (val / maxImportance) * 100;
                    return (
                      <div key={name} className="mb-2">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="text-neutral-200 truncate pr-2">{name}</span>
                          <span className="text-neutral-500">{val.toFixed(4)}</span>
                        </div>
                        <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
                          <div className="h-full bg-accent-500/80" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>

        <div className="xl:col-span-1 space-y-3 overflow-y-auto">
          <div className="rounded-xl border border-neutral-800 bg-surface-800/70 backdrop-blur-md p-4 text-xs space-y-3">
            <div className="text-xs font-medium text-neutral-200">Make a prediction</div>
            {!selectedModel && (
              <div className="text-[11px] text-neutral-500">Train/select a model first, then score a new row.</div>
            )}
            {selectedModel && (
              <>
                <div className="text-[11px] text-neutral-400">
                  Provide values for the underlying dataset columns used by this model. Missing inputs default to 0/empty.
                </div>

                <div className="space-y-2">
                  {requiredInputKeys.length === 0 && (
                    <div className="text-[11px] text-neutral-500">No input fields detected for this model.</div>
                  )}
                  {requiredInputKeys.map((k) => {
                    const col = profile?.columns.find((c) => c.name === k);
                    const t = col?.inferred_type || "text";
                    const topVals = col?.top_values || [];

                    return (
                      <div key={k} className="space-y-1">
                        <div className="flex items-center justify-between">
                          <div className="text-[11px] text-neutral-200 font-medium">{k}</div>
                          <div className="text-[10px] uppercase tracking-wide text-neutral-500">{t}</div>
                        </div>

                        {t === "numeric" ? (
                          <input
                            type="number"
                            value={inputs[k] ?? ""}
                            onChange={(e) => setInputs((prev) => ({ ...prev, [k]: e.target.value }))}
                            className="w-full bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[12px] outline-none focus:border-accent-400"
                            placeholder="e.g. 12.34"
                          />
                        ) : (
                          <input
                            type="text"
                            value={inputs[k] ?? ""}
                            onChange={(e) => setInputs((prev) => ({ ...prev, [k]: e.target.value }))}
                            className="w-full bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[12px] outline-none focus:border-accent-400"
                            placeholder={t === "datetime" ? "YYYY-MM-DD" : "e.g. Enterprise"}
                          />
                        )}

                        {t === "categorical" && topVals.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {topVals.slice(0, 6).map((tv) => (
                              <button
                                key={String(tv.value)}
                                type="button"
                                className="text-[10px] px-2 py-0.5 rounded-md border border-neutral-800 bg-surface-900 hover:border-neutral-600 text-neutral-300 transition"
                                onClick={() => setInputs((prev) => ({ ...prev, [k]: String(tv.value) }))}
                              >
                                {String(tv.value)}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                <button
                  type="button"
                  disabled={predicting || !selectedModelId}
                  onClick={predict}
                  className="w-full px-3 py-2 rounded-md bg-accent-500 text-white text-[12px] hover:bg-accent-400 disabled:bg-neutral-700 disabled:text-neutral-400 transition"
                >
                  {predicting ? "Predicting…" : "Predict"}
                </button>

                {predictResult && (
                  <div className="rounded-md border border-neutral-800 bg-surface-900/40 p-3 space-y-2">
                    <div className="text-[11px] text-neutral-400">Prediction</div>
                    <div className="text-[14px] font-semibold text-neutral-100">
                      {predictResult.task_type === "regression"
                        ? Number(predictResult.predictions[0]).toFixed(3)
                        : String(predictResult.predictions[0])}
                    </div>
                    {predictResult.task_type === "classification" && predictResult.probabilities?.[0] && (
                      <div className="pt-1">
                        <div className="text-[11px] text-neutral-400 mb-1">Top probabilities</div>
                        <div className="space-y-1">
                          {Object.entries(predictResult.probabilities[0])
                            .slice()
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 5)
                            .map(([cls, p]) => (
                              <div key={cls} className="flex items-center justify-between text-[11px]">
                                <span className="text-neutral-200">{cls}</span>
                                <span className="text-neutral-500">{(p * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {predictError && (
                  <div className="rounded-md border border-red-900/50 bg-red-900/20 p-3 text-[11px] text-red-200">
                    {predictError}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

