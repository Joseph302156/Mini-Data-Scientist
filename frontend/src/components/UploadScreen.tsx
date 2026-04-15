import { useEffect, useRef, useState } from "react";
import axios from "axios";

type DatasetInfo = {
  dataset_id: string;
  filename: string;
  n_rows: number;
  n_cols: number;
  status: string;
};

type Props = {
  onSelect: (id: string) => void;
  onUploadComplete: (id: string) => void;
};

export function UploadScreen({ onSelect, onUploadComplete }: Props) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      const res = await axios.get<DatasetInfo[]>("/api/datasets");
      setDatasets(res.data);
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".csv")) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post("/api/datasets/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      await refresh();
      onUploadComplete(res.data.dataset_id as string);
    } catch {
      alert("Upload failed. Make sure the file is a valid CSV.");
    } finally {
      setUploading(false);
    }
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  async function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    if (!window.confirm("Delete this dataset and all its models?")) return;
    setDeletingId(id);
    try {
      await axios.delete(`/api/datasets/${id}`);
      await refresh();
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-6 py-16"
      style={{
        background:
          "radial-gradient(ellipse 90% 55% at 50% -5%, rgba(99,102,241,0.1) 0%, transparent 65%), #F8FAFC",
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center text-white text-lg font-bold shadow-md"
          style={{ background: "linear-gradient(135deg,#6366f1,#8b5cf6)" }}
        >
          ◈
        </div>
        <span className="text-2xl font-bold text-slate-900 tracking-tight">
          Mini Data Scientist
        </span>
      </div>
      <p className="text-slate-500 text-base mb-12 text-center">
        Upload any CSV — get instant AI-powered analysis, charts &amp; insights
      </p>

      {/* Drop zone */}
      <div
        onClick={() => !uploading && fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className="w-full max-w-lg rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 bg-white text-center px-10 py-14 select-none"
        style={{
          borderColor: dragOver ? "#6366f1" : "#CBD5E1",
          boxShadow: dragOver
            ? "0 0 0 4px rgba(99,102,241,0.12), 0 4px 24px rgba(99,102,241,0.1)"
            : "0 1px 3px rgba(0,0,0,0.06)",
          transform: dragOver ? "scale(1.01)" : "scale(1)",
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={onInputChange}
          disabled={uploading}
        />
        <div
          className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center text-2xl"
          style={{ background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)" }}
        >
          {uploading ? (
            <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          ) : (
            "⬆"
          )}
        </div>
        <p className="text-base font-semibold text-slate-800 mb-1">
          {uploading ? "Uploading…" : "Drop your CSV file here"}
        </p>
        <p className="text-sm text-slate-400 mb-4">
          {uploading ? "Processing your data…" : "or click to browse from your computer"}
        </p>
        <span className="inline-block text-xs font-semibold uppercase tracking-wider px-3 py-1 rounded-full bg-indigo-50 text-indigo-600 border border-indigo-100">
          CSV files only
        </span>
      </div>

      {/* Recent datasets */}
      {datasets.length > 0 && (
        <div className="w-full max-w-lg mt-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
            Recent datasets
          </p>
          <div className="grid grid-cols-3 gap-3">
            {datasets.map((d) => (
              <div
                key={d.dataset_id}
                onClick={() => onSelect(d.dataset_id)}
                className="group relative bg-white rounded-xl border border-slate-200 p-4 cursor-pointer transition-all duration-150 hover:border-indigo-300 hover:shadow-md"
              >
                <div className="text-lg mb-2">📄</div>
                <div className="text-xs font-semibold text-slate-800 truncate mb-1">
                  {d.filename}
                </div>
                <div className="text-xs text-slate-400">
                  {d.n_rows.toLocaleString()} rows · {d.n_cols} cols
                </div>
                {/* Delete button */}
                <button
                  onClick={(e) => handleDelete(e, d.dataset_id)}
                  disabled={deletingId === d.dataset_id}
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 w-5 h-5 flex items-center justify-center rounded-full text-slate-400 hover:text-red-500 hover:bg-red-50 transition-all text-xs"
                  title="Delete"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
