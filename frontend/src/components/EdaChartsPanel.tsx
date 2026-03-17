import { useEffect, useState } from "react";
import axios from "axios";

type EdaHistogram = {
  column: string;
  bin_edges: number[];
  counts: number[];
};

type EdaCorrelation = {
  column_x: string;
  column_y: string;
  value: number;
};

type EdaResult = {
  dataset_id: string;
  stage: string;
  numeric_summary: Record<string, Record<string, number>>;
  histograms: EdaHistogram[];
  correlations: EdaCorrelation[];
};

type Props = {
  datasetId: string | null;
};

export function EdaChartsPanel({ datasetId }: Props) {
  const [eda, setEda] = useState<EdaResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!datasetId) {
      setEda(null);
      return;
    }
    async function load() {
      setLoading(true);
      try {
        const res = await axios.get<EdaResult>(`/api/datasets/${datasetId}/eda`, {
          params: { stage: "cleaned" },
        });
        setEda(res.data);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [datasetId]);

  if (!datasetId) {
    return (
      <div className="rounded-xl border border-neutral-800 bg-surface-800/70 p-4 text-xs text-neutral-500">
        Select or upload a dataset to see EDA charts.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-neutral-800 bg-surface-800/70 p-4 text-xs space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-neutral-200">Exploratory analysis</div>
        {loading && <div className="text-[11px] text-neutral-500">Computing…</div>}
      </div>
      {!loading && eda && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Histograms */}
          <div className="space-y-2">
            <div className="text-[11px] text-neutral-400">Top numeric histograms</div>
            <div className="space-y-2">
              {eda.histograms.slice(0, 3).map((h) => {
                const maxCount = Math.max(...h.counts, 1);
                return (
                  <div key={h.column}>
                    <div className="text-[11px] text-neutral-300 mb-1">{h.column}</div>
                    <div className="flex items-end gap-[1px] h-16 bg-surface-900 rounded-md overflow-hidden">
                      {h.counts.map((c, idx) => (
                        <div
                          // eslint-disable-next-line react/no-array-index-key
                          key={idx}
                          className="flex-1 bg-accent-500/70"
                          style={{ height: `${(c / maxCount) * 100 || 2}%` }}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}
              {eda.histograms.length === 0 && (
                <div className="text-[11px] text-neutral-500">No numeric columns to plot.</div>
              )}
            </div>
          </div>

          {/* Correlation heatmap summary */}
          <div className="space-y-2">
            <div className="text-[11px] text-neutral-400">Strong correlations</div>
            <div className="rounded-md bg-surface-900 border border-neutral-800 p-2 max-h-40 overflow-y-auto">
              {eda.correlations
                .slice()
                .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
                .slice(0, 10)
                .map((c) => (
                  <div key={`${c.column_x}-${c.column_y}`} className="flex justify-between text-[11px]">
                    <span className="text-neutral-200">
                      {c.column_x} ↔ {c.column_y}
                    </span>
                    <span className="text-neutral-400">{c.value.toFixed(2)}</span>
                  </div>
                ))}
              {eda.correlations.length === 0 && (
                <div className="text-[11px] text-neutral-500">Not enough numeric columns to compute correlations.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

