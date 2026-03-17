import { useEffect, useState } from "react";
import axios from "axios";

type InsightsResponse = {
  dataset_id: string;
  model_id?: string | null;
  stage: string;
  insights_text: string;
};

type Props = {
  datasetId: string | null;
  modelId: string | null;
};

export function InsightsPanel({ datasetId, modelId }: Props) {
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!datasetId) {
      setData(null);
      return;
    }
    async function load() {
      setLoading(true);
      try {
        const res = await axios.get<InsightsResponse>(`/api/datasets/${datasetId}/insights`, {
          params: { model_id: modelId ?? undefined, stage: "cleaned" },
        });
        setData(res.data);
      } catch (e) {
        setData(null);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [datasetId, modelId]);

  return (
    <div className="rounded-xl border border-neutral-800 bg-surface-800/70 backdrop-blur-md p-4 text-xs space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-neutral-200">AI-generated insights</div>
        {loading && <div className="text-[11px] text-neutral-500">Thinking…</div>}
      </div>
      {!datasetId && (
        <div className="text-[11px] text-neutral-500">
          Select or upload a dataset to generate automatic insights.
        </div>
      )}
      {datasetId && !loading && !data && (
        <div className="text-[11px] text-neutral-500">
          Insights are not available yet. Check your OpenAI configuration.
        </div>
      )}
      {data && (
        <div className="prose prose-invert prose-sm max-w-none">
          {data.insights_text.split("\n").map((line, idx) => (
            <p key={idx} className="mb-1">
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

