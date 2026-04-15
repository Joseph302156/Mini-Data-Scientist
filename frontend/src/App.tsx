import { useState } from "react";
import { UploadScreen } from "./components/UploadScreen";
import { Dashboard } from "./components/Dashboard";
import { ModelTrainerPage } from "./components/ModelTrainerPage";

type Screen = "upload" | "dashboard" | "trainer";

type TrainerSeed = {
  target: string;
  taskType: "regression" | "classification";
  modelType: "linear" | "random_forest";
} | null;

export default function App() {
  const [screen, setScreen] = useState<Screen>("upload");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [trainerSeed, setTrainerSeed] = useState<TrainerSeed>(null);

  function goToDashboard(id: string) {
    setDatasetId(id);
    setModelId(null);
    setScreen("dashboard");
  }

  function goToTrainer(seed?: TrainerSeed) {
    setTrainerSeed(seed ?? null);
    setScreen("trainer");
  }

  function goToUpload() {
    setScreen("upload");
  }

  if (screen === "upload") {
    return (
      <UploadScreen
        onSelect={goToDashboard}
        onUploadComplete={goToDashboard}
      />
    );
  }

  if (screen === "dashboard" && datasetId) {
    return (
      <Dashboard
        datasetId={datasetId}
        modelId={modelId}
        onBack={goToUpload}
        onOpenTrainer={goToTrainer}
        onModelTrained={(id) => setModelId(id)}
      />
    );
  }

  if (screen === "trainer" && datasetId) {
    return (
      <ModelTrainerPage
        datasetId={datasetId}
        selectedModelId={modelId}
        onSelectModel={(id) => {
          setModelId(id);
          setScreen("dashboard");
        }}
        onBack={() => setScreen("dashboard")}
        trainerSeed={trainerSeed}
      />
    );
  }

  // Fallback — shouldn't normally be reached
  return (
    <UploadScreen
      onSelect={goToDashboard}
      onUploadComplete={goToDashboard}
    />
  );
}
