import { useState } from "react";
import { Layout } from "./components/Layout";
import { DatasetSidebar } from "./components/DatasetSidebar";
import { DataPreviewPanel } from "./components/DataPreviewPanel";
import { InsightsPanel } from "./components/InsightsPanel";
import { ChatPanel } from "./components/ChatPanel";
import { ModelPanel } from "./components/ModelPanel";
import { EdaChartsPanel } from "./components/EdaChartsPanel";
import { ModelTrainerPage } from "./components/ModelTrainerPage";

export default function App() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [view, setView] = useState<"dashboard" | "trainer">("dashboard");
  const [trainerSeed, setTrainerSeed] = useState<{
    target: string;
    taskType: "regression" | "classification";
    modelType: "linear" | "random_forest";
  } | null>(null);

  function sidebar() {
    return (
      <DatasetSidebar
        selectedId={datasetId}
        onSelect={(id) => {
          setDatasetId(id);
          setModelId(null);
          setTrainerSeed(null);
          setView("dashboard");
        }}
        onUploadComplete={(id) => {
          setDatasetId(id);
          setModelId(null);
          setTrainerSeed(null);
          setView("dashboard");
        }}
      />
    );
  }

  return (
    <Layout sidebar={sidebar()}>
      {view === "dashboard" && (
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
                setTrainerSeed(seed);
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
      {view === "trainer" && (
        <ModelTrainerPage
          datasetId={datasetId}
          selectedModelId={modelId}
          onSelectModel={(id) => setModelId(id)}
          onBack={() => setView("dashboard")}
          trainerSeed={trainerSeed}
        />
      )}
    </Layout>
  );
}

