import { useState } from "react";
import axios from "axios";

type Props = {
  datasetId: string | null;
  modelId: string | null;
};

type Message = {
  role: "user" | "assistant";
  content: string;
};

export function ChatPanel({ datasetId, modelId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
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
      const answer = res.data.answer as string;
      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "There was an error talking to the AI chat service.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-xl border border-neutral-800 bg-surface-800/70 backdrop-blur-md flex flex-col h-full">
      <div className="px-4 py-2 border-b border-neutral-800 flex items-center justify-between">
        <div className="text-xs font-medium text-neutral-200">AI Analyst</div>
        <div className="text-[11px] text-neutral-500">
          {datasetId ? "Ask about this dataset or model." : "Select a dataset to start chatting."}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2 text-xs">
        {messages.length === 0 && (
          <div className="text-[11px] text-neutral-500 mt-2">
            Try: “Summarize this dataset”, “What factors matter most?”, or “Explain the model’s
            behavior.”
          </div>
        )}
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`max-w-[90%] rounded-lg px-3 py-2 ${
              m.role === "user"
                ? "ml-auto bg-accent-500/80 text-neutral-50"
                : "mr-auto bg-surface-900 border border-neutral-700 text-neutral-100"
            }`}
          >
            {m.content.split("\n").map((line, i) => (
              <p key={i} className="mb-0.5">
                {line}
              </p>
            ))}
          </div>
        ))}
      </div>
      <div className="border-t border-neutral-800 px-3 py-2 flex items-center gap-2">
        <input
          type="text"
          placeholder="Ask a question about your data…"
          className="flex-1 bg-surface-900 border border-neutral-700 rounded-md px-3 py-1.5 text-xs outline-none focus:border-accent-400"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              send();
            }
          }}
          disabled={!datasetId || loading}
        />
        <button
          type="button"
          onClick={send}
          disabled={!datasetId || loading || !input.trim()}
          className="px-3 py-1.5 rounded-md text-xs bg-accent-500 text-white hover:bg-accent-400 disabled:bg-neutral-700 disabled:text-neutral-400 transition"
        >
          {loading ? "Sending…" : "Send"}
        </button>
      </div>
    </div>
  );
}

