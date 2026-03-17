import { useEffect, useState } from "react";
import axios from "axios";

type DataPreview = {
  dataset_id: string;
  stage: string;
  n_rows: number;
  n_cols: number;
  preview: Record<string, unknown>[];
};

type Props = {
  datasetId: string | null;
};

const stages: { key: "raw" | "cleaned" | "features"; label: string }[] = [
  { key: "raw", label: "Raw" },
  { key: "cleaned", label: "Cleaned" },
  { key: "features", label: "Features" },
];

export function DataPreviewPanel({ datasetId }: Props) {
  const [stage, setStage] = useState<"raw" | "cleaned" | "features">("cleaned");
  const [data, setData] = useState<DataPreview | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!datasetId) {
      setData(null);
      return;
    }
    async function load() {
      setLoading(true);
      try {
        const res = await axios.get<DataPreview>(`/api/datasets/${datasetId}/preview`, {
          params: { stage, limit: 20 },
        });
        setData(res.data);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [datasetId, stage]);

  if (!datasetId) {
    return (
      <div className="rounded-xl border border-neutral-800 bg-surface-800/50 p-4 text-xs text-neutral-400">
        Select or upload a dataset to see a preview.
      </div>
    );
  }

  const columns =
    data && data.preview.length > 0 ? Object.keys(data.preview[0] as Record<string, unknown>) : [];

  return (
    <div className="rounded-xl border border-neutral-800 bg-surface-800/70 backdrop-blur-md overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-neutral-800">
        <div className="text-xs font-medium text-neutral-200">Dataset preview</div>
        <div className="flex items-center gap-1 text-[11px] text-neutral-500">
          {stages.map((s) => (
            <button
              key={s.key}
              type="button"
              onClick={() => setStage(s.key)}
              className={`px-2 py-1 rounded-md ${
                stage === s.key
                  ? "bg-accent-500/20 text-accent-200"
                  : "hover:bg-surface-700 text-neutral-400"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
      <div className="max-h-80 overflow-auto text-xs">
        {loading && (
          <div className="p-4 text-neutral-500 text-[11px]">Loading {stage} data…</div>
        )}
        {!loading && data && data.preview.length === 0 && (
          <div className="p-4 text-neutral-500 text-[11px]">No rows to display.</div>
        )}
        {!loading && data && data.preview.length > 0 && (
          <table className="min-w-full border-separate border-spacing-0">
            <thead className="bg-surface-900 sticky top-0 z-10">
              <tr>
                {columns.map((c) => (
                  <th
                    key={c}
                    className="px-3 py-2 text-left text-[11px] font-medium text-neutral-400 border-b border-neutral-800 bg-surface-900"
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.preview.map((row, i) => (
                <tr key={i} className="hover:bg-surface-700/60">
                  {columns.map((c) => (
                    <td
                      key={c}
                      className="px-3 py-1.5 border-b border-neutral-900 text-[11px] text-neutral-200"
                    >
                      {String((row as Record<string, unknown>)[c] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

