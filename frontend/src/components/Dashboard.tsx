import { useEffect, useState } from "react";
import axios from "axios";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

type InsightFinding = { title: string; detail: string; type: string };

type StructuredInsights = {
  headline: string;
  summary: string;
  key_findings: InsightFinding[];
  data_quality_note: string;
  recommendation: string;
};

type DatasetOverview = {
  n_rows: number; n_cols: number;
  numeric_cols: number; categorical_cols: number;
  datetime_cols: number; missing_values_pct: number;
};

type EdaHistogram = { column: string; bin_edges: number[]; counts: number[] };
type EdaCorrelation = { column_x: string; column_y: string; value: number };

type ModelSummary = {
  model_id: string; target_column: string;
  task_type: string; model_type: string;
  metrics: { name: string; value: number }[];
  feature_importances: Record<string, number>;
};

type DatasetReport = {
  dataset_id: string; filename: string;
  overview: DatasetOverview;
  eda: {
    histograms: EdaHistogram[];
    correlations: EdaCorrelation[];
    numeric_summary: Record<string, Record<string, number>>;
  };
  insights: StructuredInsights;
  model: ModelSummary | null;
};

type PreviewRow = Record<string, unknown>;

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtNum(n: number): string {
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (Math.abs(n) >= 10_000) return (n / 1_000).toFixed(0) + "K";
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n % 1 === 0 ? n.toFixed(0) : n.toFixed(2);
}

function cleanName(name: string): string {
  return name
    .replace(/__log$/, "").replace(/__std$/, "")
    .replace(/__year$/, " (year)").replace(/__month$/, " (month)")
    .replace(/__day$/, " (day)").replace(/__dayofweek$/, " (weekday)")
    .replace(/_/g, " ");
}

const FINDING_STYLES: Record<string, { border: string; pillBg: string; pillText: string; icon: string }> = {
  trend:       { border: "#60A5FA", pillBg: "#EFF6FF", pillText: "#2563EB", icon: "↗" },
  correlation: { border: "#A78BFA", pillBg: "#F5F3FF", pillText: "#7C3AED", icon: "⟷" },
  highlight:   { border: "#34D399", pillBg: "#ECFDF5", pillText: "#059669", icon: "✦" },
  warning:     { border: "#FBBF24", pillBg: "#FFFBEB", pillText: "#B45309", icon: "⚠" },
  model:       { border: "#818CF8", pillBg: "#EEF2FF", pillText: "#4338CA", icon: "◈" },
};
const defaultStyle = FINDING_STYLES.highlight;

const CHART_PALETTE = ["#6366F1", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"];

const tooltipStyle = {
  background: "#fff", border: "1px solid #E2E8F0",
  borderRadius: 10, fontSize: 12,
  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
};

// ── Section header ────────────────────────────────────────────────────────────

function SectionHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="flex items-center gap-3 mb-5">
      <h2 className="text-base font-semibold text-slate-800 whitespace-nowrap">{title}</h2>
      <div className="h-px flex-1 bg-slate-200" />
      {sub && <span className="text-xs text-slate-400 whitespace-nowrap">{sub}</span>}
    </div>
  );
}

// ── Distribution chart ────────────────────────────────────────────────────────

function DistributionChart({ histogram, color }: { histogram: EdaHistogram; color: string }) {
  const data = histogram.bin_edges.slice(0, -1).map((edge, i) => ({
    range: fmtNum(edge),
    count: histogram.counts[i],
  }));
  const stats = { min: histogram.bin_edges[0], max: histogram.bin_edges[histogram.bin_edges.length - 1] };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
      <div className="text-sm font-semibold text-slate-800 mb-1 capitalize">{cleanName(histogram.column)}</div>
      <div className="text-xs text-slate-400 mb-4">
        {fmtNum(stats.min)} – {fmtNum(stats.max)}
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <BarChart data={data} margin={{ top: 0, right: 0, left: -28, bottom: 14 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis dataKey="range" tick={{ fontSize: 9, fill: "#94A3B8" }} angle={-35} textAnchor="end" interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 9, fill: "#94A3B8" }} />
          <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#475569" }} itemStyle={{ color: "#6366F1" }} />
          <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Correlation chart ─────────────────────────────────────────────────────────

function CorrelationChart({ correlations }: { correlations: EdaCorrelation[] }) {
  const data = correlations
    .slice().sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8)
    .map((c) => ({
      pair: `${cleanName(c.column_x)} & ${cleanName(c.column_y)}`,
      value: parseFloat(c.value.toFixed(2)),
    }));

  if (!data.length) return null;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
      <div className="text-sm font-semibold text-slate-800 mb-1">How Fields Relate to Each Other</div>
      <div className="text-xs text-slate-400 mb-4">Green = rise together · Red = one up, one down</div>
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 40)}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" horizontal={false} />
          <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 10, fill: "#94A3B8" }} tickFormatter={(v) => v.toFixed(1)} />
          <YAxis type="category" dataKey="pair" tick={{ fontSize: 11, fill: "#475569" }} width={190} />
          <Tooltip
            contentStyle={tooltipStyle}
            labelStyle={{ color: "#475569" }}
            itemStyle={{ color: "#6366F1" }}
            formatter={(v: number) => [v.toFixed(2), "Strength"]}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.value >= 0 ? "#10B981" : "#EF4444"} fillOpacity={0.6 + 0.4 * Math.abs(entry.value)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Feature importance chart ──────────────────────────────────────────────────

function FeatureImportanceChart({ model }: { model: ModelSummary }) {
  const data = Object.entries(model.feature_importances)
    .sort((a, b) => b[1] - a[1]).slice(0, 10)
    .map(([name, value], i) => ({
      feature: cleanName(name),
      importance: parseFloat(value.toFixed(4)),
      color: CHART_PALETTE[i % CHART_PALETTE.length],
    }));

  const accuracyMetric = model.metrics.find((m) => m.name === "accuracy" || m.name === "r2");
  const accuracyLabel = accuracyMetric
    ? model.task_type === "regression"
      ? `${Math.round(accuracyMetric.value * 100)}% variance explained`
      : `${Math.round(accuracyMetric.value * 100)}% accurate`
    : null;

  if (!data.length) return null;

  return (
    <div className="bg-white rounded-2xl border border-indigo-100 shadow-sm p-6">
      <div className="flex items-start justify-between mb-5">
        <div>
          <div className="text-base font-semibold text-slate-800">
            What Drives{" "}
            <span className="text-indigo-600 capitalize">{cleanName(model.target_column)}</span>?
          </div>
          <div className="text-xs text-slate-400 mt-0.5">Ranked by influence on the outcome</div>
        </div>
        {accuracyLabel && (
          <span className="text-xs font-semibold px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 shrink-0 ml-4">
            {accuracyLabel}
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={Math.max(260, data.length * 36)}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: "#94A3B8" }} />
          <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: "#475569" }} width={180} />
          <Tooltip
            contentStyle={tooltipStyle}
            labelStyle={{ color: "#475569" }}
            formatter={(v: number) => [v.toFixed(4), "Influence"]}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {data.map((entry, i) => <Cell key={i} fill={entry.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Chat box ──────────────────────────────────────────────────────────────────

function ChatBox({ datasetId, modelId }: { datasetId: string; modelId: string | null }) {
  const [messages, setMessages] = useState<{ role: "user" | "ai"; content: string }[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const SUGGESTIONS = [
    "What's unusual in this data?",
    "Which segment is most valuable?",
    "What drives high prices?",
    "Summarise for my team",
  ];

  async function send(question?: string) {
    const q = (question ?? input).trim();
    if (!q || loading) return;
    setMessages((p) => [...p, { role: "user", content: q }]);
    setInput("");
    setLoading(true);
    try {
      const res = await axios.post("/api/chat", { dataset_id: datasetId, model_id: modelId, question: q });
      setMessages((p) => [...p, { role: "ai", content: res.data.answer as string }]);
    } catch {
      setMessages((p) => [...p, { role: "ai", content: "Sorry, there was an error reaching the AI service." }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
        <span className="text-sm font-semibold text-slate-800">AI Analyst</span>
        <span className="flex items-center gap-1.5 text-xs text-emerald-600 font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse inline-block" />
          Ready
        </span>
      </div>

      {/* Messages */}
      <div className="px-6 py-5 space-y-3 min-h-[100px]">
        {messages.length === 0 && (
          <p className="text-sm text-slate-400">
            Ask anything about your data — patterns, trends, recommendations, or summaries.
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-indigo-500 text-white rounded-br-sm"
                  : "bg-slate-50 border border-slate-200 text-slate-700 rounded-bl-sm"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-50 border border-slate-200 text-slate-400 rounded-2xl rounded-bl-sm px-4 py-2.5 text-sm">
              Thinking…
            </div>
          </div>
        )}
      </div>

      {/* Suggestions */}
      {messages.length === 0 && (
        <div className="flex flex-wrap gap-2 px-6 pb-4">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="text-xs px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200 text-slate-500 hover:border-indigo-300 hover:text-indigo-600 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="border-t border-slate-100 px-4 py-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); send(); } }}
          placeholder="Ask a question about your data…"
          disabled={loading}
          className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-sm outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 placeholder:text-slate-400 text-slate-800 transition"
        />
        <button
          onClick={() => send()}
          disabled={loading || !input.trim()}
          className="px-5 py-2.5 rounded-xl bg-indigo-500 text-white text-sm font-semibold hover:bg-indigo-600 disabled:bg-slate-200 disabled:text-slate-400 transition"
        >
          Send →
        </button>
      </div>
    </div>
  );
}

// ── Data table section ────────────────────────────────────────────────────────

function DataTableSection({ datasetId }: { datasetId: string }) {
  const [stage, setStage] = useState<"raw" | "cleaned" | "features">("cleaned");
  const [preview, setPreview] = useState<{ n_rows: number; preview: PreviewRow[] } | null>(null);

  useEffect(() => {
    axios
      .get(`/api/datasets/${datasetId}/preview`, { params: { stage, limit: 6 } })
      .then((r) => setPreview(r.data))
      .catch(() => setPreview(null));
  }, [datasetId, stage]);

  const cols = preview?.preview?.[0] ? Object.keys(preview.preview[0]).slice(0, 7) : [];

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
        <span className="text-sm font-semibold text-slate-800">
          {preview ? `Showing ${preview.preview.length} of ${preview.n_rows.toLocaleString()} rows` : "Your Data"}
        </span>
        <div className="flex gap-1">
          {(["raw", "cleaned", "features"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStage(s)}
              className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-all capitalize ${
                stage === s
                  ? "bg-indigo-50 text-indigo-600 border border-indigo-200"
                  : "text-slate-400 border border-transparent hover:text-slate-600"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        {cols.length > 0 ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-100">
                {cols.map((col) => (
                  <th key={col} className="text-left px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview!.preview.map((row, i) => (
                <tr key={i} className="border-b border-slate-50 hover:bg-slate-50 transition-colors">
                  {cols.map((col) => {
                    const val = row[col];
                    const isNum = typeof val === "number";
                    const display = val == null ? "—" : String(val);
                    return (
                      <td key={col} className="px-5 py-3 text-slate-600 max-w-[180px] truncate">
                        {isNum ? (
                          <span className="font-medium text-slate-800">{display}</span>
                        ) : display.length > 12 ? (
                          <span title={display}>{display.slice(0, 18)}…</span>
                        ) : (
                          <span className="inline-block bg-indigo-50 text-indigo-700 text-xs px-2 py-0.5 rounded-full border border-indigo-100">
                            {display}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="px-6 py-8 text-sm text-slate-400 text-center">No preview available for this stage.</p>
        )}
      </div>
    </div>
  );
}

// ── Dashboard (main export) ───────────────────────────────────────────────────

type Props = {
  datasetId: string;
  modelId: string | null;
  onBack: () => void;
  onOpenTrainer: (seed?: { target: string; taskType: "regression" | "classification"; modelType: "linear" | "random_forest" } | null) => void;
  onModelTrained: (id: string) => void;
};

export function Dashboard({ datasetId, modelId, onBack, onOpenTrainer }: Props) {
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setReport(null);
    setLoading(true);
    setError(null);
    const params: Record<string, string> = {};
    if (modelId) params.model_id = modelId;
    axios
      .get<DatasetReport>(`/api/datasets/${datasetId}/report`, { params })
      .then((r) => setReport(r.data))
      .catch((e) => {
        const msg = axios.isAxiosError(e) && e.response?.data?.detail
          ? e.response.data.detail
          : "Failed to load report.";
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, [datasetId, modelId]);

  const { overview, insights, eda, model } = report ?? {};

  return (
    <div className="min-h-screen bg-slate-50">
      {/* ── Sticky navbar ── */}
      <nav
        className="sticky top-0 z-50 flex items-center justify-between px-8 h-14 border-b border-slate-200"
        style={{ background: "rgba(255,255,255,0.92)", backdropFilter: "blur(12px)" }}
      >
        {/* Left */}
        <div className="flex items-center gap-3">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center text-white text-sm font-bold"
            style={{ background: "linear-gradient(135deg,#6366f1,#8b5cf6)" }}
          >
            ◈
          </div>
          <span className="font-semibold text-slate-800 text-sm">Mini DS</span>
          <span className="text-slate-300">/</span>
          <button onClick={onBack} className="text-slate-400 text-sm hover:text-slate-600 transition">
            Datasets
          </button>
          <span className="text-slate-300">›</span>
          <span className="text-sm font-medium text-slate-700 truncate max-w-[200px]">
            {report?.filename ?? "Loading…"}
          </span>
        </div>

        {/* Right */}
        <div className="flex items-center gap-2">
          <button
            onClick={onBack}
            className="text-sm px-4 py-1.5 rounded-lg border border-slate-200 text-slate-500 hover:border-slate-300 hover:text-slate-700 transition"
          >
            ← Upload new
          </button>
          <button
            onClick={() => onOpenTrainer(null)}
            className="text-sm px-4 py-1.5 rounded-lg bg-indigo-500 text-white font-semibold hover:bg-indigo-600 transition shadow-sm"
          >
            Train Model
          </button>
        </div>
      </nav>

      {/* ── Loading ── */}
      {loading && (
        <div className="flex flex-col items-center justify-center min-h-[70vh] gap-5">
          <div className="w-12 h-12 rounded-full border-[3px] border-indigo-500 border-t-transparent animate-spin" />
          <div className="text-center">
            <p className="text-slate-700 font-medium">Analysing your data…</p>
            <p className="text-slate-400 text-sm mt-1">Generating AI insights — this takes about 10 seconds</p>
          </div>
        </div>
      )}

      {/* ── Error ── */}
      {error && !loading && (
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="text-center max-w-sm">
            <div className="text-3xl mb-4">⚠</div>
            <p className="text-slate-700 font-medium mb-2">Something went wrong</p>
            <p className="text-slate-400 text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* ── Main content ── */}
      {report && overview && insights && eda && (
        <main className="max-w-5xl mx-auto px-8 py-10 pb-24 space-y-12">

          {/* ── KPI Hero strip ── */}
          <div
            className="rounded-2xl p-8 border"
            style={{
              background: "linear-gradient(135deg, rgba(99,102,241,0.07) 0%, rgba(139,92,246,0.04) 50%, rgba(6,182,212,0.03) 100%)",
              borderColor: "rgba(99,102,241,0.15)",
            }}
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                { label: "Total Records", value: overview.n_rows.toLocaleString(), sub: "rows in dataset", color: "#0F172A" },
                {
                  label: "Data Fields", value: overview.n_cols.toString(),
                  sub: `${overview.numeric_cols} numeric · ${overview.categorical_cols} categories`, color: "#0F172A",
                },
                {
                  label: "Completeness",
                  value: `${(100 - overview.missing_values_pct).toFixed(1)}%`,
                  sub: overview.missing_values_pct > 0 ? `${overview.missing_values_pct}% values auto-filled` : "No missing values",
                  color: "#059669",
                },
                {
                  label: model ? "Model Accuracy" : "Fields Analysed",
                  value: model
                    ? (() => {
                        const m = model.metrics.find((x) => x.name === "accuracy" || x.name === "r2");
                        return m ? `${Math.round(m.value * 100)}%` : "—";
                      })()
                    : overview.numeric_cols.toString(),
                  sub: model ? `predicting ${cleanName(model.target_column)}` : "numeric fields",
                  color: "#4338CA",
                },
              ].map((kpi) => (
                <div key={kpi.label} className="pl-6 border-l border-white/60 first:pl-0 first:border-l-0">
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">{kpi.label}</div>
                  <div className="text-4xl font-extrabold tracking-tight leading-none mb-2" style={{ color: kpi.color }}>
                    {kpi.value}
                  </div>
                  <div className="text-xs text-slate-500">{kpi.sub}</div>
                </div>
              ))}
            </div>
          </div>

          {/* ── AI Story ── */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-8">
            <div className="inline-flex items-center gap-2 bg-indigo-50 border border-indigo-100 text-indigo-600 text-xs font-semibold uppercase tracking-wider px-3 py-1 rounded-full mb-5">
              <span>✦</span> AI Analysis
            </div>
            <h1 className="text-xl font-bold text-slate-900 leading-snug mb-4 max-w-2xl">
              {insights.headline || "Insights Report"}
            </h1>
            <p className="text-base text-slate-600 leading-relaxed max-w-3xl">{insights.summary}</p>
            {insights.data_quality_note && (
              <div className="mt-5 inline-flex items-center gap-2 text-xs bg-amber-50 border border-amber-200 text-amber-700 px-3 py-1.5 rounded-full">
                <span>⚠</span> {insights.data_quality_note}
              </div>
            )}
          </div>

          {/* ── Key Findings ── */}
          {insights.key_findings.length > 0 && (
            <section>
              <SectionHeader title="Key Findings" />
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {insights.key_findings.map((f, i) => {
                  const style = FINDING_STYLES[f.type] ?? defaultStyle;
                  return (
                    <div
                      key={i}
                      className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm hover:-translate-y-0.5 transition-transform"
                      style={{ borderTop: `3px solid ${style.border}` }}
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <span
                          className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide px-2.5 py-1 rounded-full border"
                          style={{ background: style.pillBg, color: style.pillText, borderColor: style.border + "44" }}
                        >
                          <span>{style.icon}</span>
                          {f.type}
                        </span>
                      </div>
                      <div className="text-sm font-semibold text-slate-800 mb-2 leading-snug">{f.title}</div>
                      <div className="text-sm text-slate-500 leading-relaxed">{f.detail}</div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* ── Model Insights ── */}
          {model && Object.keys(model.feature_importances).length > 0 && (
            <section>
              <SectionHeader title="What Matters Most" sub="Factors that influence the prediction target" />
              <FeatureImportanceChart model={model} />
            </section>
          )}

          {/* ── Data Distributions ── */}
          {eda.histograms.length > 0 && (
            <section>
              <SectionHeader title="Data Distributions" sub="How values are spread across key fields" />
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {eda.histograms
                  .filter((h) => h.counts.some((c) => c > 0))
                  .slice(0, 6)
                  .map((h, i) => (
                    <DistributionChart key={h.column} histogram={h} color={CHART_PALETTE[i % CHART_PALETTE.length]} />
                  ))}
              </div>
            </section>
          )}

          {/* ── Relationships ── */}
          {eda.correlations.length > 1 && (
            <section>
              <SectionHeader title="Relationships Between Fields" />
              <CorrelationChart correlations={eda.correlations} />
            </section>
          )}

          {/* ── Data table ── */}
          <section>
            <SectionHeader title="Your Data" />
            <DataTableSection datasetId={datasetId} />
          </section>

          {/* ── No model prompt ── */}
          {!model && (
            <div
              className="rounded-2xl border-2 border-dashed border-slate-200 p-8 text-center cursor-pointer hover:border-indigo-300 hover:bg-indigo-50/40 transition-all"
              onClick={() => onOpenTrainer(null)}
            >
              <div className="text-2xl mb-2">◈</div>
              <div className="text-sm font-semibold text-slate-600 mb-1">Train a model to unlock feature importance</div>
              <div className="text-xs text-slate-400">Click to open the model trainer →</div>
            </div>
          )}

          {/* ── Recommendation ── */}
          {insights.recommendation && (
            <div className="rounded-2xl bg-emerald-50 border border-emerald-200 p-6 flex items-start gap-4">
              <div className="w-9 h-9 rounded-xl bg-emerald-100 flex items-center justify-center text-emerald-600 text-lg shrink-0">
                →
              </div>
              <div>
                <div className="text-xs font-bold uppercase tracking-widest text-emerald-600 mb-2">
                  Recommended Action
                </div>
                <p className="text-sm text-slate-700 leading-relaxed">{insights.recommendation}</p>
              </div>
            </div>
          )}

          {/* ── Chat ── */}
          <section>
            <SectionHeader title="Ask Anything" />
            <ChatBox datasetId={datasetId} modelId={modelId} />
          </section>

        </main>
      )}
    </div>
  );
}
