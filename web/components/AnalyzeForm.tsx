"use client";

import { useState } from "react";
import { Analysis, analyze } from "@/lib/api";
import ResultView from "./ResultView";

const OPTIONAL = [
  { name: "offer_price", label: "Offer price", type: "number", step: "0.01" },
  { name: "shares_offered", label: "Shares offered", type: "number", step: "1" },
  { name: "range_low", label: "Filing range low", type: "number", step: "0.01" },
  { name: "range_high", label: "Filing range high", type: "number", step: "0.01" },
  { name: "ipo_date", label: "Expected IPO date", type: "date" },
  { name: "founding_year", label: "Founding year", type: "number", step: "1" },
];

export default function AnalyzeForm() {
  const [mode, setMode] = useState<"url" | "file">("url");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Analysis | null>(null);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    // drop empty optional fields so the API does not try to parse ""
    for (const [k, v] of Array.from(form.entries())) {
      if (v === "" || (v instanceof File && v.size === 0)) form.delete(k);
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await analyze(form));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const input =
    "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3 " +
    "focus:outline-none focus:ring-2 focus:ring-accent";

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className="space-y-4 rounded-lg border border-line bg-surface-2 p-5">
        <div className="flex gap-2 text-sm" role="tablist">
          {(["url", "file"] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              onClick={() => setMode(m)}
              className={`rounded-md px-3 py-1.5 ${
                mode === m ? "bg-surface font-medium text-ink shadow-sm" : "text-ink-2 hover:text-ink"
              }`}
            >
              {m === "url" ? "EDGAR link" : "Upload a file"}
            </button>
          ))}
        </div>

        {mode === "url" ? (
          <label className="block space-y-1">
            <span className="text-sm text-ink-2">Link to a final prospectus (424B4) on SEC EDGAR</span>
            <input
              name="url"
              type="url"
              required
              placeholder="https://www.sec.gov/Archives/edgar/data/…/…424b4.htm"
              className={input}
            />
          </label>
        ) : (
          <label className="block space-y-1">
            <span className="text-sm text-ink-2">Prospectus as PDF, HTML or TXT (US or Nigerian), up to 40 MB</span>
            <input
              name="file"
              type="file"
              required
              accept=".pdf,.htm,.html,.txt"
              className={`${input} file:mr-3 file:rounded file:border-0 file:bg-surface-2 file:px-3 file:py-1 file:text-ink`}
            />
          </label>
        )}

        <details className="text-sm">
          <summary className="cursor-pointer text-ink-2 hover:text-ink">
            Corrections (optional): fill in anything the bot cannot read
          </summary>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            {OPTIONAL.map((f) => (
              <label key={f.name} className="space-y-1">
                <span className="text-xs text-ink-2">{f.label}</span>
                <input name={f.name} type={f.type} step={f.step} min={f.type === "number" ? 0 : undefined} className={input} />
              </label>
            ))}
          </div>
        </details>

        <button
          type="submit"
          disabled={busy}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
        >
          {busy ? "Reading the prospectus… (up to a minute)" : "Analyze"}
        </button>
        {error && (
          <p role="alert" className="text-sm" style={{ color: "var(--against)" }}>
            {error}
          </p>
        )}
      </form>

      {result && <ResultView a={result} />}
    </div>
  );
}
