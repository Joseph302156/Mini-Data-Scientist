import type { ReactNode } from "react";

type Props = {
  sidebar: ReactNode;
  children: ReactNode;
};

export function Layout({ sidebar, children }: Props) {
  return (
    <div className="flex h-screen bg-surface-900 text-neutral-100">
      <aside className="w-72 border-r border-neutral-800 bg-surface-800/80 backdrop-blur-md">
        <div className="h-16 flex items-center px-4 border-b border-neutral-800">
          <div className="h-8 w-8 rounded-lg bg-accent-500/20 flex items-center justify-center text-accent-400 font-semibold">
            MD
          </div>
          <div className="ml-3">
            <div className="text-sm font-semibold tracking-tight">Mini Data Scientist</div>
            <div className="text-xs text-neutral-400">AI analytics workspace</div>
          </div>
        </div>
        <div className="h-[calc(100%-4rem)] overflow-y-auto">{sidebar}</div>
      </aside>
      <main className="flex-1 flex flex-col">
        <header className="h-14 border-b border-neutral-800 flex items-center px-6 justify-between bg-surface-900/80 backdrop-blur-md">
          <div className="text-sm text-neutral-400">
            <span className="text-neutral-200 font-medium">Workspace</span> / Datasets
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-neutral-500">Powered by FastAPI + OpenAI</span>
          </div>
        </header>
        <section className="flex-1 overflow-hidden flex">
          <div className="flex-1 overflow-y-auto p-4 lg:p-6 space-y-4">{children}</div>
        </section>
      </main>
    </div>
  );
}

