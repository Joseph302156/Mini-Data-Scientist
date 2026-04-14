import { useEffect, useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

type InsightFinding = {
  title: string;
  detail: string;
  type: "trend" | "correlation" | "highlight" | "warning" | "model";
};

type StructuredInsights = {
  headline: string;
  summary: string;
  key_findings: InsightFinding[];
  data_quality_note: string;
  recommendation: string;
};

type DatasetOverview = {
  n_rows: number;
  n_cols: number;
  numeric_cols: number;
  categorical_cols: number;
  datetime_cols: number;
  missing_values_pct: number;
};

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

type ModelSummary = {
  model_id: string;
  target_column: string;
  task_type: string;
  model_type: string;
  metrics: { name: string; value: number }[];
  feature_importances: Record<string, number>;
};

type DatasetReport = {
  dataset_id: string;
  filename: string;
  overview: DatasetOverview;
  eda: {
    histograms: EdaHistogram[];
    correlations: EdaCorrelation[];
    numeric_summary: Record<string, Record<string, number>>;
  };
  insights: StructuredInsights;
  model: ModelSummary | null;
};

type Props = {
  datasetId: string | null;
  modelId: string | null;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtNum(n: number): string {
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n % 1 === 0 ? n.toString() : n.toFixed(2);
}

function cleanFeatureName(name: string): string {
  return name
    .replace(/__log$/, "")
    .replace(/__std$/, "")
    .replace(/__year$/, " (year)")
    .replace(/__month$/, " (month)")
    .replace(/__day$/, " (day)")
    .replace(/__dayofweek$/, " (weekday)")
    .replace(/_/g, " ");
}

// ── Colour helpers ────────────────────────────────────────────────────────────

const FINDING_STYLES: Record<
  string,
  { border: string; badge: string; icon: string }
> = {
  trend: {
    border: "border-blue-500/40",
    badge: "bg-blue-500/20 text-blue-300",
    icon: "↗",
  },
  correlation: {
    border: "border-purple-500/40",
    badge: "bg-purple-500/20 text-purple-300",
    icon: "⟷",
  },
  highlight: {
    border: "border-emerald-500/40",
    badge: "bg-emerald-500/20 text-emerald-300",
    icon: "✦",
  },
  warning: {
    border: "border-amber-500/40",
    badge: "bg-amber-500/20 text-amber-300",
    icon: "⚠",
  },
  model: {
    border: "border-indigo-500/40",
    badge: "bg-indigo-500/20 text-indigo-300",
    icon: "◈",
  },
};

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="rounded-2xl bg-surface-800 border border-neutral-800 p-5 flex flex-col gap-1">
      <div className="text-xs text-neutral-500 uppercase tracking-wide">{label}</div>
      <div className="text-3xl font-bold text-neutral-50">
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      {sub && <div className="text-xs text-neutral-500">{sub}</div>}
    </div>
  );
}

function FindingCard({ finding }: { finding: InsightFinding }) {
  const style = FINDING_STYLES[finding.type] ?? FINDING_STYLES.highlight;
  return (
    <div
      className={`rounded-2xl bg-surface-800 border ${style.border} p-5 flex flex-col gap-3`}
    >
      <div className="flex items-center gap-2">
        <span
          className={`inline-flex items-center justify-center w-7 h-7 rounded-lg text-sm font-medium ${style.badge}`}
        >
          {style.icon}
        </span>
        <span
          className={`text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${style.badge}`}
        >
          {finding.type}
        </span>
      </div>
      <div className="text-sm font-semibold text-neutral-100 leading-snug">
        {finding.title}
      </div>
      <div className="text-sm text-neutral-400 leading-relaxed">{finding.detail}</div>
    </div>
  );
}

const CHART_COLORS = ["#4F46E5", "#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD"];

function DistributionChart({ histogram }: { histogram: EdaHistogram }) {
  const data = histogram.bin_edges.slice(0, -1).map((edge, i) => ({
    range: fmtNum(edge),
    count: histogram.counts[i],
  }));

  return (
    <div className="rounded-2xl bg-surface-800 border border-neutral-800 p-5">
      <div className="text-sm font-semibold text-neutral-200 mb-4 capitalize">
        {cleanFeatureName(histogram.column)}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} margin={{ top: 0, right: 0, left: -24, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" vertical={false} />
          <XAxis
            dataKey="range"
            tick={{ fontSize: 9, fill: "#6b7280" }}
            angle={-40}
            textAnchor="end"
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
          <Tooltip
            contentStyle={{
              background: "#0B0B12",
              border: "1px solid #374151",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#d1d5db" }}
            itemStyle={{ color: "#a5b4fc" }}
          />
          <Bar dataKey="count" fill="#4F46E5" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CorrelationChart({
  correlations,
}: {
  correlations: EdaCorrelation[];
}) {
  const data = correlations
    .slice()
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8)
    .map((c) => ({
      pair: `${cleanFeatureName(c.column_x)} & ${cleanFeatureName(c.column_y)}`,
      value: parseFloat(c.value.toFixed(2)),
    }));

  if (data.length === 0) return null;

  return (
    <div className="rounded-2xl bg-surface-800 border border-neutral-800 p-5">
      <div className="text-sm font-semibold text-neutral-200 mb-1">
        How Variables Move Together
      </div>
      <div className="text-xs text-neutral-500 mb-4">
        Positive = tend to increase together · Negative = one goes up when the other goes down
      </div>
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 40)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 24, left: 8, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" horizontal={false} />
          <XAxis
            type="number"
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <YAxis
            type="category"
            dataKey="pair"
            tick={{ fontSize: 11, fill: "#d1d5db" }}
            width={200}
          />
          <Tooltip
            contentStyle={{
              background: "#0B0B12",
              border: "1px solid #374151",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#d1d5db" }}
            itemStyle={{ color: "#a5b4fc" }}
            formatter={(v: number) => [v.toFixed(2), "Relationship strength"]}
          />
          <Bar dataKey="value" radius={[0, 3, 3, 0]}>
            {data.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.value >= 0 ? "#10B981" : "#EF4444"}
                fillOpacity={0.7 + 0.3 * Math.abs(entry.value)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function FeatureImportanceChart({ model }: { model: ModelSummary }) {
  const data = Object.entries(model.feature_importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, value], i) => ({
      feature: cleanFeatureName(name),
      importance: parseFloat(value.toFixed(4)),
      color: CHART_COLORS[i % CHART_COLORS.length],
    }));

  if (data.length === 0) return null;

  // Format accuracy for display
  const accuracyMetric = model.metrics.find(
    (m) => m.name === "accuracy" || m.name === "r2"
  );
  const accuracyLabel =
    accuracyMetric
      ? model.task_type === "regression"
        ? `${Math.round(accuracyMetric.value * 100)}% variance explained`
        : `${Math.round(accuracyMetric.value * 100)}% accurate`
      : null;

  return (
    <div className="rounded-2xl bg-surface-800 border border-indigo-500/30 p-5">
      <div className="flex items-start justify-between mb-1">
        <div>
          <div className="text-sm font-semibold text-neutral-200">
            What Drives{" "}
            <span className="text-indigo-400 capitalize">
              {cleanFeatureName(model.target_column)}
            </span>
          </div>
          <div className="text-xs text-neutral-500 mt-0.5">
            Ranked by how much each factor influences the outcome
          </div>
        </div>
        {accuracyLabel && (
          <div className="text-xs bg-indigo-500/20 text-indigo-300 px-2.5 py-1 rounded-full font-medium shrink-0 ml-4">
            {accuracyLabel}
          </div>
        )}
      </div>
      <div className="mt-4">
        <ResponsiveContainer width="100%" height={Math.max(250, data.length * 36)}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 0, right: 24, left: 8, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10, fill: "#6b7280" }} />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fontSize: 11, fill: "#d1d5db" }}
              width={180}
            />
            <Tooltip
              contentStyle={{
                background: "#0B0B12",
                border: "1px solid #374151",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "#d1d5db" }}
              itemStyle={{ color: "#a5b4fc" }}
              formatter={(v: number) => [v.toFixed(4), "Influence score"]}
            />
            <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
              {data.map((entry, idx) => (
                <Cell key={idx} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function InlineChatSection({
  datasetId,
  modelId,
}: {
  datasetId: string;
  modelId: string | null;
}) {
  const [messages, setMessages] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function send() {
    if (!datasetId || !input.trim() || loading) return;
    const question = input.trim();
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await axios.post("/api/chat", {
        dataset_id: datasetId,
        model_id: modelId,
        question,
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.data.answer as string },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "There was an error reaching the AI service.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-2xl bg-surface-800 border border-neutral-800 overflow-hidden">
      <div className="px-5 py-4 border-b border-neutral-800 flex items-center justify-between">
        <div className="text-sm font-semibold text-neutral-200">Ask Anything</div>
        <div className="text-xs text-neutral-500">
          Ask follow-up questions about your data
        </div>
      </div>
      <div className="px-5 py-4 space-y-3 max-h-72 overflow-y-auto">
        {messages.length === 0 && (
          <div className="text-sm text-neutral-500">
            Try: "What are the main trends?", "Which customers are most
            valuable?", "What should I focus on?"
          </div>
        )}
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
              m.role === "user"
                ? "ml-auto bg-accent-500 text-white"
                : "mr-auto bg-surface-700 border border-neutral-700 text-neutral-200"
            }`}
          >
            {m.content}
          </div>
        ))}
        {loading && (
          <div className="mr-auto bg-surface-700 border border-neutral-700 text-neutral-400 rounded-xl px-4 py-2.5 text-sm">
            Thinking…
          </div>
        )}
      </div>
      <div className="border-t border-neutral-800 px-5 py-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask a question about your data…"
          disabled={loading}
          className="flex-1 bg-surface-700 border border-neutral-700 rounded-xl px-4 py-2 text-sm outline-none focus:border-accent-400 placeholder:text-neutral-600"
        />
        <button
          type="button"
          onClick={send}
          disabled={loading || !input.trim()}
          className="px-4 py-2 rounded-xl bg-accent-500 text-white text-sm font-medium hover:bg-accent-400 disabled:bg-neutral-700 disabled:text-neutral-500 transition"
        >
          Send
        </button>
      </div>
    </div>
  );
}

// ── Main InsightsPage component ───────────────────────────────────────────────

export function InsightsPage({ datasetId, modelId }: Props) {
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) {
      setReport(null);
      return;
    }
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const params: Record<string, string> = {};
        if (modelId) params.model_id = modelId;
        const res = await axios.get<DatasetReport>(
          `/api/datasets/${datasetId}/report`,
          { params }
        );
        setReport(res.data);
      } catch (e: unknown) {
        const msg =
          axios.isAxiosError(e) && e.response?.data?.detail
            ? e.response.data.detail
            : "Failed to load insights report.";
        setError(msg);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [datasetId, modelId]);

  if (!datasetId) {
    return (
      <div className="flex items-center justify-center h-full text-neutral-500 text-sm">
        Select or upload a dataset to generate an insights report.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="w-10 h-10 rounded-full border-2 border-accent-500 border-t-transparent animate-spin" />
        <div className="text-neutral-400 text-sm">
          Analysing your data and generating insights…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-400 text-sm">
        {error}
      </div>
    );
  }

  if (!report) return null;

  const { overview, insights, eda, model } = report;

  // Pick top 6 histograms that look most interesting (highest variance)
  const topHistograms = eda.histograms
    .filter((h) => h.counts.some((c) => c > 0))
    .slice(0, 6);

  return (
    <div className="space-y-8 pb-16">
      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <div className="rounded-2xl bg-gradient-to-br from-indigo-900/40 to-surface-800 border border-indigo-500/20 p-7">
        <div className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-3">
          {report.filename}
        </div>
        <h2 className="text-2xl md:text-3xl font-bold text-neutral-50 leading-snug mb-4">
          {insights.headline || "Insights Report"}
        </h2>
        <p className="text-base text-neutral-300 leading-relaxed max-w-3xl">
          {insights.summary}
        </p>
        {insights.data_quality_note && (
          <div className="mt-4 inline-flex items-center gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-full px-3 py-1.5">
            <span>⚠</span> {insights.data_quality_note}
          </div>
        )}
      </div>

      {/* ── Stats row ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Total Records"
          value={overview.n_rows.toLocaleString()}
          sub="rows in the dataset"
        />
        <StatCard
          label="Data Fields"
          value={overview.n_cols}
          sub={`${overview.numeric_cols} numeric, ${overview.categorical_cols} categorical`}
        />
        <StatCard
          label="Data Completeness"
          value={`${(100 - overview.missing_values_pct).toFixed(1)}%`}
          sub={
            overview.missing_values_pct > 0
              ? `${overview.missing_values_pct}% values filled in`
              : "No missing values"
          }
        />
        <StatCard
          label={model ? "Prediction Target" : "Fields Analysed"}
          value={
            model
              ? cleanFeatureName(model.target_column)
              : overview.numeric_cols
          }
          sub={
            model
              ? `${model.model_type === "random_forest" ? "Random forest" : "Linear"} model`
              : "numeric fields"
          }
        />
      </div>

      {/* ── Key Findings ──────────────────────────────────────────────────── */}
      {insights.key_findings.length > 0 && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <h3 className="text-base font-semibold text-neutral-100">
              Key Findings
            </h3>
            <div className="h-px flex-1 bg-neutral-800" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {insights.key_findings.map((f, i) => (
              <FindingCard key={i} finding={f} />
            ))}
          </div>
        </div>
      )}

      {/* ── Feature Importance ────────────────────────────────────────────── */}
      {model && Object.keys(model.feature_importances).length > 0 && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <h3 className="text-base font-semibold text-neutral-100">
              What Matters Most
            </h3>
            <div className="h-px flex-1 bg-neutral-800" />
          </div>
          <FeatureImportanceChart model={model} />
        </div>
      )}

      {/* ── Distributions ─────────────────────────────────────────────────── */}
      {topHistograms.length > 0 && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <h3 className="text-base font-semibold text-neutral-100">
              Data Distributions
            </h3>
            <div className="h-px flex-1 bg-neutral-800" />
            <span className="text-xs text-neutral-500">
              How values are spread across each field
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {topHistograms.map((h) => (
              <DistributionChart key={h.column} histogram={h} />
            ))}
          </div>
        </div>
      )}

      {/* ── Correlations ──────────────────────────────────────────────────── */}
      {eda.correlations.length > 1 && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <h3 className="text-base font-semibold text-neutral-100">
              Relationships Between Fields
            </h3>
            <div className="h-px flex-1 bg-neutral-800" />
          </div>
          <CorrelationChart correlations={eda.correlations} />
        </div>
      )}

      {/* ── Recommendation ────────────────────────────────────────────────── */}
      {insights.recommendation && (
        <div className="rounded-2xl bg-emerald-900/20 border border-emerald-500/20 p-6">
          <div className="flex items-start gap-3">
            <span className="text-emerald-400 text-lg mt-0.5">→</span>
            <div>
              <div className="text-xs text-emerald-400 font-semibold uppercase tracking-wider mb-1">
                Recommended Action
              </div>
              <p className="text-sm text-neutral-200 leading-relaxed">
                {insights.recommendation}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── No model prompt ───────────────────────────────────────────────── */}
      {!model && (
        <div className="rounded-2xl border border-dashed border-neutral-700 p-6 text-center">
          <div className="text-sm text-neutral-400 mb-1">
            Train a model to unlock feature importance insights
          </div>
          <div className="text-xs text-neutral-600">
            Use the Overview tab to select a target column and train a model
          </div>
        </div>
      )}

      {/* ── AI Chat ───────────────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <h3 className="text-base font-semibold text-neutral-100">
            Ask a Question
          </h3>
          <div className="h-px flex-1 bg-neutral-800" />
        </div>
        <InlineChatSection datasetId={datasetId} modelId={modelId} />
      </div>
    </div>
  );
}
