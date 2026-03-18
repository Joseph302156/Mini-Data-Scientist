import { useEffect, useState } from "react";
import axios from "axios";

type TargetCandidate = {
  column: string;
  suggested_task: "regression" | "classification";
};

type TargetListResponse = {
  dataset_id: string;
  candidates: TargetCandidate[];
};

type Metric = { name: string; value: number };

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
  top_feature_stats?: FeatureStat[] | null;
};

type FeatureStat = {
  feature_name: string;
  feature_type: "numeric" | "binary" | "time_component";
  mean: number | null;
  total: number | null;
  count_true: number | null;
  prevalence: number | null;
  note: string | null;
};

type Props = {
  datasetId: string | null;
  selectedModelId: string | null;
  onSelectModel: (id: string | null) => void;
  onOpenTrainer?: (seed: {
    target: string;
    taskType: "regression" | "classification";
    modelType: "linear" | "random_forest";
  }) => void;
};

export function ModelPanel({ datasetId, selectedModelId, onSelectModel, onOpenTrainer }: Props) {
  const [targets, setTargets] = useState<TargetCandidate[]>([]);
  const [target, setTarget] = useState<string>("");
  const [taskType, setTaskType] = useState<"regression" | "classification">("regression");
  const [modelType, setModelType] = useState<"linear" | "random_forest">("random_forest");
  const [models, setModels] = useState<TrainedModelSummary[]>([]);
  const [training, setTraining] = useState(false);

  useEffect(() => {
    if (!datasetId) {
      setTargets([]);
      setModels([]);
      return;
    }
    async function load() {
      const [tRes, mRes] = await Promise.all([
        axios.get<TargetListResponse>(`/api/datasets/${datasetId}/targets`),
        axios.get<TrainedModelSummary[]>(`/api/datasets/${datasetId}/models`),
      ]);
      setTargets(tRes.data.candidates);
      if (!target && tRes.data.candidates.length > 0) {
        const first = tRes.data.candidates[0];
        setTarget(first.column);
        setTaskType(first.suggested_task);
      }
      setModels(mRes.data);
    }
    load();
  }, [datasetId]);

  async function train() {
    // Training happens in the full-page Model & Predict workspace to give it room
    // for visualization + prediction UI.
    if (onOpenTrainer) {
      onOpenTrainer({ target, taskType, modelType });
      return;
    }
    if (!datasetId || !target) return;
    setTraining(true);
    try {
      const res = await axios.post<TrainedModelSummary>(`/api/datasets/${datasetId}/models`, {
        target_column: target,
        task_type: taskType,
        model_type: modelType,
      });
      const m = res.data;
      setModels((prev) => [m, ...prev]);
      onSelectModel(m.model_id);
    } finally {
      setTraining(false);
    }
  }

  return (
    <div className="rounded-xl border border-neutral-800 bg-surface-800/70 backdrop-blur-md p-4 text-xs space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-neutral-200">Models</div>
        {datasetId && (
          <button
            type="button"
            onClick={() => (onOpenTrainer ? onOpenTrainer() : undefined)}
            className="px-2 py-1 rounded-md border border-neutral-700 text-[11px] text-neutral-300 hover:border-neutral-600 hover:bg-surface-700 transition"
          >
            Model & Predict
          </button>
        )}
      </div>
      {!datasetId && (
        <div className="text-[11px] text-neutral-500">
          Select or upload a dataset to configure and train models.
        </div>
      )}
      {datasetId && (
        <>
          <div className="flex flex-wrap gap-2 items-center">
            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[11px] outline-none focus:border-accent-400"
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
              className="bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[11px] outline-none focus:border-accent-400"
            >
              <option value="regression">Regression</option>
              <option value="classification">Classification</option>
            </select>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as typeof modelType)}
              className="bg-surface-900 border border-neutral-700 rounded-md px-2 py-1 text-[11px] outline-none focus:border-accent-400"
            >
              <option value="random_forest">Random forest</option>
              <option value="linear">Linear</option>
            </select>
            <button
              type="button"
              onClick={train}
              disabled={training || !target}
              className="px-3 py-1.5 rounded-md bg-accent-500 text-white text-[11px] hover:bg-accent-400 disabled:bg-neutral-700 disabled:text-neutral-400 transition"
            >
              {training ? "Training…" : "Train model"}
            </button>
          </div>
          <div className="border-t border-neutral-800 pt-2 mt-1 max-h-56 overflow-y-auto space-y-2">
            {models.length === 0 && (
              <div className="text-[11px] text-neutral-500">No models yet for this dataset.</div>
            )}
            {models.map((m) => (
              <button
                key={m.model_id}
                type="button"
                onClick={() => onSelectModel(m.model_id)}
                className={`w-full text-left rounded-md px-2 py-1.5 border text-[11px] mb-1 ${
                  selectedModelId === m.model_id
                    ? "border-accent-500/60 bg-accent-500/15"
                    : "border-neutral-800 bg-surface-900 hover:border-neutral-600"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-neutral-100">
                    {m.target_column} · {m.task_type}
                  </span>
                  <span className="uppercase tracking-wide text-[10px] text-neutral-500">
                    {m.model_type}
                  </span>
                </div>
                <div className="mt-1 flex flex-wrap gap-2 text-[10px] text-neutral-400">
                  {m.metrics.map((metric) => (
                    <span key={metric.name}>
                      {metric.name}:{" "}
                      <span className="text-neutral-200">
                        {metric.value.toFixed(metric.name === "accuracy" ? 3 : 2)}
                      </span>
                    </span>
                  ))}
                </div>
                {Object.keys(m.feature_importances).length > 0 && (
                  <div className="mt-1 text-[10px] text-neutral-500 truncate">
                    Top features:{" "}
                    {Object.entries(m.feature_importances)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .map(([name]) => name)
                      .join(", ")}
                  </div>
                )}

                {m.model_explanation && (
                  <div className="mt-2 text-[11px] text-neutral-300 leading-relaxed max-h-14 overflow-hidden">
                    {m.model_explanation}
                  </div>
                )}

                {m.top_feature_stats && m.top_feature_stats.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {m.top_feature_stats.slice(0, 4).map((fs) => {
                      const avgTxt =
                        typeof fs.mean === "number" && Number.isFinite(fs.mean)
                          ? fs.mean.toFixed(2)
                          : "—";

                      if (fs.feature_type === "time_component") {
                        return (
                          <div key={fs.feature_name} className="text-[10px] text-neutral-500 truncate">
                            {fs.feature_name}: avg {avgTxt}
                          </div>
                        );
                      }

                      if (fs.feature_type === "binary") {
                        const countTxt =
                          typeof fs.count_true === "number" && Number.isFinite(fs.count_true)
                            ? fs.count_true.toLocaleString()
                            : "—";
                        const prevTxt =
                          typeof fs.prevalence === "number" && Number.isFinite(fs.prevalence)
                            ? `${(fs.prevalence * 100).toFixed(1)}%`
                            : "—";
                        return (
                          <div key={fs.feature_name} className="text-[10px] text-neutral-500 truncate">
                            {fs.feature_name}: {prevTxt} true ({countTxt} rows)
                          </div>
                        );
                      }

                      const totalTxt =
                        typeof fs.total === "number" && Number.isFinite(fs.total)
                          ? fs.total.toLocaleString(undefined, { maximumFractionDigits: 2 })
                          : "—";

                      return (
                        <div key={fs.feature_name} className="text-[10px] text-neutral-500 truncate">
                          {fs.feature_name}: avg {avgTxt} · total {totalTxt}
                        </div>
                      );
                    })}
                  </div>
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

