import { useState } from "react";
import { Layout } from "./components/Layout";
import { DatasetSidebar } from "./components/DatasetSidebar";
import { DataPreviewPanel } from "./components/DataPreviewPanel";
import { InsightsPanel } from "./components/InsightsPanel";
import { ChatPanel } from "./components/ChatPanel";
import { ModelPanel } from "./components/ModelPanel";
import { EdaChartsPanel } from "./components/EdaChartsPanel";
import { ModelTrainerPage } from "./components/ModelTrainerPage";
import { InsightsPage } from "./components/InsightsPage";

type View = "overview" | "insights" | "trainer";

const TABS: { id: View; label: string; desc: string }[] = [
  { id: "overview", label: "Overview", desc: "Raw data, stats & models" },
  { id: "insights", label: "Insights Report", desc: "AI-powered analysis" },
];

export default function App() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [view, setView] = useState<View>("insights");
  const [trainerSeed, setTrainerSeed] = useState<{
    target: string;
    taskType: "regression" | "classification";
    modelType: "linear" | "random_forest";
  } | null>(null);

  function handleDatasetSelect(id: string) {
    setDatasetId(id);
    setModelId(null);
    setTrainerSeed(null);
    if (view === "trainer") setView("overview");
  }

  function sidebar() {
    return (
      <DatasetSidebar
        selectedId={datasetId}
        onSelect={handleDatasetSelect}
        onUploadComplete={(id) => {
          setDatasetId(id);
          setModelId(null);
          setTrainerSeed(null);
          setView("insights");
        }}
      />
    );
  }

  return (
    <Layout sidebar={sidebar()}>
      {/* ── Tab bar (hidden when in trainer view) ──────────────────── */}
      {view !== "trainer" && (
        <div className="flex items-center gap-1 mb-5 border-b border-neutral-800 pb-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setView(tab.id)}
              className={`relative px-4 py-2.5 text-sm font-medium transition rounded-t-lg ${
                view === tab.id
                  ? "text-neutral-50 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-accent-500 after:rounded-t"
                  : "text-neutral-500 hover:text-neutral-300"
              }`}
            >
              {tab.label}
              {tab.id === "insights" && (
                <span className="ml-1.5 text-[10px] bg-accent-500/20 text-accent-400 px-1.5 py-0.5 rounded-full font-semibold uppercase tracking-wide">
                  AI
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* ── Insights Report ─────────────────────────────────────────── */}
      {view === "insights" && (
        <div className="h-full overflow-y-auto pr-1">
          <InsightsPage datasetId={datasetId} modelId={modelId} />
        </div>
      )}

      {/* ── Overview (original dashboard) ───────────────────────────── */}
      {view === "overview" && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 h-full">
          <div className="space-y-4 xl:col-span-2">
            <DataPreviewPanel datasetId={datasetId} />
            <EdaChartsPanel datasetId={datasetId} />
          </div>
          <div className="space-y-4 flex flex-col">
            <ModelPanel
              datasetId={datasetId}
              selectedModelId={modelId}
              onSelectModel={(id) => setModelId(id)}
              onOpenTrainer={(seed) => {
                setTrainerSeed(seed ?? null);
                setView("trainer");
              }}
            />
            <InsightsPanel datasetId={datasetId} modelId={modelId} />
            <div className="h-72">
              <ChatPanel datasetId={datasetId} modelId={modelId} />
            </div>
          </div>
        </div>
      )}

      {/* ── Model Trainer ────────────────────────────────────────────── */}
      {view === "trainer" && (
        <ModelTrainerPage
          datasetId={datasetId}
          selectedModelId={modelId}
          onSelectModel={(id) => setModelId(id)}
          onBack={() => setView("overview")}
          trainerSeed={trainerSeed}
        />
      )}
    </Layout>
  );
}
