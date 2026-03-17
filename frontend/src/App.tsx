import { useState } from "react";
import { Layout } from "./components/Layout";
import { DatasetSidebar } from "./components/DatasetSidebar";
import { DataPreviewPanel } from "./components/DataPreviewPanel";
import { InsightsPanel } from "./components/InsightsPanel";
import { ChatPanel } from "./components/ChatPanel";
import { ModelPanel } from "./components/ModelPanel";
import { EdaChartsPanel } from "./components/EdaChartsPanel";

export default function App() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);

  return (
    <Layout
      sidebar={
        <DatasetSidebar
          selectedId={datasetId}
          onSelect={(id) => {
            setDatasetId(id);
          }}
          onUploadComplete={(id) => {
            setDatasetId(id);
            setModelId(null);
          }}
        />
      }
    >
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
          />
          <InsightsPanel datasetId={datasetId} modelId={modelId} />
          <div className="flex-1 min-h-[220px]">
            <ChatPanel datasetId={datasetId} modelId={modelId} />
          </div>
        </div>
      </div>
    </Layout>
  );
}

