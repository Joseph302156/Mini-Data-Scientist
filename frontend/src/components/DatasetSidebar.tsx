import { useEffect, useState } from "react";
import axios from "axios";

type DatasetInfo = {
  dataset_id: string;
  filename: string;
  n_rows: number;
  n_cols: number;
  status: string;
};

type Props = {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUploadComplete: (id: string) => void;
};

export function DatasetSidebar({ selectedId, onSelect, onUploadComplete }: Props) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function refresh() {
    const res = await axios.get<DatasetInfo[]>("/api/datasets");
    setDatasets(res.data);
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post("/api/datasets/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const id = res.data.dataset_id as string;
      await refresh();
      onUploadComplete(id);
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this dataset and all related models?")) return;
    setDeletingId(id);
    try {
      await axios.delete(`/api/datasets/${id}`);
      await refresh();
      if (selectedId === id) {
        onSelect("");
      }
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-neutral-800">
        <label className="group flex items-center gap-2 rounded-lg border border-dashed border-neutral-700 px-3 py-2 text-xs text-neutral-400 hover:border-accent-400 hover:text-accent-300 cursor-pointer transition">
          <input
            type="file"
            accept=".csv"
            className="hidden"
            onChange={handleUpload}
            disabled={isUploading}
          />
          <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-neutral-900 text-accent-400 text-xs">
            ⬆
          </span>
          {isUploading ? "Uploading…" : "Upload CSV"}
        </label>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {datasets.map((d) => (
          <div key={d.dataset_id} className="group flex items-center gap-1">
            <button
              type="button"
              onClick={() => onSelect(d.dataset_id)}
              className={`flex-1 text-left px-3 py-2 rounded-md text-xs transition ${
                selectedId === d.dataset_id
                  ? "bg-accent-500/20 text-neutral-50 border border-accent-500/40"
                  : "hover:bg-surface-700 text-neutral-300 border border-transparent"
              }`}
            >
              <div className="flex items-center justify-between gap-1">
                <span className="truncate text-neutral-100">{d.filename}</span>
                <span className="text-[10px] uppercase tracking-wide text-neutral-500 shrink-0">
                  {d.status}
                </span>
              </div>
              <div className="mt-0.5 text-[11px] text-neutral-500">
                {d.n_rows.toLocaleString()} rows · {d.n_cols} cols
              </div>
            </button>
            <button
              type="button"
              onClick={() => handleDelete(d.dataset_id)}
              disabled={deletingId === d.dataset_id}
              className="opacity-0 group-hover:opacity-100 text-neutral-500 hover:text-red-400 text-xs px-1 transition"
              title="Delete dataset"
            >
              ✕
            </button>
          </div>
        ))}
        {datasets.length === 0 && (
          <div className="text-[11px] text-neutral-500 px-2 pt-4">
            No datasets yet. Upload a CSV to get started.
          </div>
        )}
      </div>
    </div>
  );
}

