"use client";

import { useEffect, useRef, useState } from "react";
import { Analysis, Company, Step, analyzeStream, searchCompanies } from "@/lib/api";
import ResultView from "./ResultView";
import StepList, { STEP_ORDER } from "./StepList";
import { GLOSSARY, Icon, StatusBadge, Term } from "./ui";

type Mode = "search" | "file" | "link";
type Input = { url?: string; file?: File; label: string };
type Run = { phase: "uploading" | "running" | "done" | "error"; upload: number; steps: Step[]; error?: string };

const CORRECTIONS = [
  { name: "offer_price", label: "Offer price ($)", type: "number", step: "0.01" },
  { name: "range_low", label: "First price range: low ($)", type: "number", step: "0.01" },
  { name: "range_high", label: "First price range: high ($)", type: "number", step: "0.01" },
  { name: "shares_offered", label: "Shares offered", type: "number", step: "1" },
  { name: "ipo_date", label: "IPO date", type: "date" },
  { name: "founding_year", label: "Year founded", type: "number", step: "1" },
];
const SUGGESTIONS = ["Cerebras", "Jersey Mike's", "Reddit", "Airbnb"];
const MAX_MB = 40;

const fieldClass =
  "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3 " +
  "focus:outline-none focus:ring-2 focus:ring-accent";

function useCompanySearch(query: string) {
  const [state, setState] = useState<{ loading: boolean; results: Company[] | null; error?: string }>({
    loading: false,
    results: null,
  });
  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) {
      setState({ loading: false, results: null });
      return;
    }
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true, error: undefined }));
    // wait for a pause in typing before asking EDGAR
    const timer = setTimeout(() => {
      searchCompanies(q, ctrl.signal)
        .then((results) => setState({ loading: false, results }))
        .catch((e) => {
          if (e.name !== "AbortError") setState({ loading: false, results: null, error: e.message });
        });
    }, 350);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [query]);
  return state;
}

function CompanyResult({ c, onAnalyze }: { c: Company; onAnalyze: () => void }) {
  return (
    <li className="flex flex-col gap-3 py-3 sm:flex-row sm:items-center">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium text-ink">{c.name}</span>
          {c.tickers.length > 0 && <span className="text-sm text-ink-3">{c.tickers.slice(0, 2).join(", ")}</span>}
          <StatusBadge status={c.status} label={c.status_label} />
        </div>
        <p className="mt-1 text-sm text-ink-2">{c.description}</p>
      </div>
      {c.analyzable ? (
        <button type="button" onClick={onAnalyze}
          className="shrink-0 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90">
          {c.status === "registration" ? "Predict" : "Analyze"}
        </button>
      ) : (
        <span className="shrink-0 text-xs text-ink-3">Nothing to analyze</span>
      )}
    </li>
  );
}

function ProgressBar({ run, hasFile }: { run: Run; hasFile: boolean }) {
  const finished = run.steps.filter((s) => s.state !== "running").length;
  const uploadShare = hasFile ? 0.15 : 0;
  const fraction = run.phase === "done" ? 1 : uploadShare * run.upload + (1 - uploadShare) * (finished / STEP_ORDER.length);
  const running = run.steps.find((s) => s.state === "running");
  const label =
    run.phase === "uploading" ? `Uploading the file… ${Math.round(run.upload * 100)}%`
      : run.phase === "error" ? "Stopped"
        : running ? `${running.title}…` : run.phase === "done" ? "Done" : "Starting…";
  return (
    <div>
      <div className="mb-2 flex justify-between text-sm">
        <span className="text-ink-2" aria-live="polite">{label}</span>
        <span className="tabular-nums text-ink-3">{Math.round(fraction * 100)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full" style={{ background: "var(--neutral)" }}
        role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}
        aria-label="Analysis progress">
        <div className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${fraction * 100}%`, background: run.phase === "error" ? "var(--against)" : "var(--accent)" }} />
      </div>
    </div>
  );
}

export default function Analyzer() {
  const [mode, setMode] = useState<Mode>("search");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [url, setUrl] = useState("");
  const [correctionsOpen, setCorrectionsOpen] = useState(false);
  const [input, setInput] = useState<Input | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [result, setResult] = useState<Analysis | null>(null);
  const corrections = useRef<HTMLDivElement>(null);
  const abort = useRef<(() => void) | null>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const search = useCompanySearch(mode === "search" ? query : "");

  useEffect(() => () => abort.current?.(), []);

  function pickFile(f: File | undefined) {
    setFileError(null);
    if (!f) return;
    if (!/\.(pdf|html?|txt)$/i.test(f.name)) return setFileError("Choose a PDF, HTML or TXT file.");
    if (f.size > MAX_MB * 1024 * 1024) return setFileError(`That file is over ${MAX_MB} MB.`);
    setFile(f);
  }

  function start(next: Input) {
    abort.current?.();
    const form = new FormData();
    if (next.url) form.append("url", next.url);
    if (next.file) form.append("file", next.file);
    corrections.current?.querySelectorAll("input").forEach((el) => {
      if (el.value) form.append(el.name, el.value);
    });
    setInput(next);
    setResult(null);
    setRun({ phase: next.file ? "uploading" : "running", upload: 0, steps: [] });
    requestAnimationFrame(() => progressRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));

    const upsert = (steps: Step[], s: Step) =>
      steps.some((x) => x.id === s.id) ? steps.map((x) => (x.id === s.id ? s : x)) : [...steps, s];
    const fail = (message: string) =>
      setRun((r) => r && {
        ...r,
        phase: "error",
        error: message,
        steps: r.steps.some((s) => s.state === "running")
          ? r.steps.map((s) => (s.state === "running" ? { ...s, state: "error", summary: message } : s))
          : r.steps,
      });

    const job = analyzeStream(form, {
      upload: (f) => setRun((r) => r && { ...r, upload: f, phase: f >= 1 ? "running" : "uploading" }),
      event: (e) => {
        if (e.event === "step") {
          const { event: _ignored, ...step } = e;
          void _ignored;
          setRun((r) => r && { ...r, phase: "running", steps: upsert(r.steps, step) });
        } else if (e.event === "result") {
          setRun((r) => r && { ...r, phase: "done" });
          setResult(e.result);
        } else {
          fail(e.message);
        }
      },
    });
    abort.current = job.abort;
    job.done.catch((err: Error) => {
      if (err.name !== "AbortError") fail(err.message);
    });
  }

  function reset() {
    abort.current?.();
    setRun(null);
    setResult(null);
    setInput(null);
  }

  const busy = run?.phase === "uploading" || run?.phase === "running";
  const tabs: { id: Mode; label: string }[] = [
    { id: "search", label: "Search a company" },
    { id: "file", label: "Upload a prospectus" },
    { id: "link", label: "Paste an EDGAR link" },
  ];

  return (
    <div className="space-y-6">
      {!run && (
        <section className="space-y-4 rounded-lg border border-line bg-surface-2 p-5">
          <div className="flex flex-wrap gap-1 text-sm" role="tablist" aria-label="How to find the prospectus">
            {tabs.map((t) => (
              <button key={t.id} type="button" role="tab" aria-selected={mode === t.id} onClick={() => setMode(t.id)}
                className={`rounded-md px-3 py-1.5 ${mode === t.id ? "bg-surface font-medium text-ink shadow-sm" : "text-ink-2 hover:text-ink"}`}>
                {t.label}
              </button>
            ))}
          </div>

          {mode === "search" && (
            <div className="space-y-3">
              <label className="block">
                <span className="sr-only">Company name or ticker</span>
                <div className="relative">
                  <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-3">
                    <Icon name="search" />
                  </span>
                  <input autoFocus value={query} onChange={(e) => setQuery(e.target.value)} type="search"
                    placeholder="Company name or ticker, like Cerebras or CBRS" className={`${fieldClass} pl-9 text-base`} />
                </div>
              </label>
              <p className="text-xs text-ink-3">
                A live search of SEC EDGAR: any company that files with the SEC, including IPOs after the training
                data and ones that haven&apos;t priced yet.
              </p>
              {query.trim().length < 2 && (
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="text-ink-3">Try</span>
                  {SUGGESTIONS.map((s) => (
                    <button key={s} type="button" onClick={() => setQuery(s)}
                      className="rounded-full border border-line px-3 py-1 text-ink-2 hover:bg-surface hover:text-ink">
                      {s}
                    </button>
                  ))}
                </div>
              )}
              {search.loading && (
                <p className="flex items-center gap-2 text-sm text-ink-2">
                  <Icon name="spinner" /> Searching SEC EDGAR and checking each company&apos;s IPO status…
                </p>
              )}
              {search.error && <p role="alert" className="text-sm" style={{ color: "var(--against)" }}>{search.error}</p>}
              {!search.loading && search.results && (
                search.results.length ? (
                  <ul className="divide-y divide-line rounded-md border border-line bg-surface px-4">
                    {search.results.map((c) => (
                      <CompanyResult key={c.cik} c={c}
                        onAnalyze={() => start({ url: c.document, label: c.name })} />
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-ink-2">No companies on EDGAR match “{query.trim()}”. Try the full name or the ticker.</p>
                )
              )}
            </div>
          )}

          {mode === "file" && (
            <div className="space-y-3">
              <label
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}
                className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center ${dragging ? "border-accent bg-surface" : "border-line"}`}>
                <span className="text-ink-3"><Icon name="upload" className="h-6 w-6" /></span>
                {file ? (
                  <span className="text-sm text-ink">
                    <span className="font-medium">{file.name}</span>{" "}
                    <span className="text-ink-3">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
                  </span>
                ) : (
                  <span className="text-sm text-ink-2">Drop a prospectus here, or <span className="text-accent-ink underline">choose a file</span></span>
                )}
                <span className="text-xs text-ink-3">PDF, HTML or TXT, up to {MAX_MB} MB. US or Nigerian.</span>
                <input type="file" accept=".pdf,.htm,.html,.txt" className="sr-only"
                  onChange={(e) => pickFile(e.target.files?.[0])} />
              </label>
              {fileError && <p role="alert" className="text-sm" style={{ color: "var(--against)" }}>{fileError}</p>}
              <button type="button" disabled={!file} onClick={() => file && start({ file, label: file.name })}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
                Analyze this file
              </button>
            </div>
          )}

          {mode === "link" && (
            <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); if (url.trim()) start({ url: url.trim(), label: "EDGAR document" }); }}>
              <label className="block space-y-1">
                <span className="text-sm text-ink-2">Link to a <Term tip={GLOSSARY.prospectus}>prospectus</Term> on SEC EDGAR (a final 424B4, or an S-1 for an upcoming IPO)</span>
                <input value={url} onChange={(e) => setUrl(e.target.value)} type="url" required
                  placeholder="https://www.sec.gov/Archives/edgar/data/…" className={fieldClass} />
              </label>
              <button type="submit" className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white">Analyze</button>
            </form>
          )}
        </section>
      )}

      <details open={correctionsOpen} onToggle={(e) => setCorrectionsOpen((e.target as HTMLDetailsElement).open)}
        className={`rounded-lg border border-line p-4 text-sm ${run && !correctionsOpen ? "hidden" : ""}`}>
        <summary className="cursor-pointer text-ink-2 hover:text-ink">
          Add details the bot can&apos;t find (optional)
        </summary>
        <p className="mt-2 text-xs text-ink-3">
          Anything you enter here replaces what the bot reads. The most useful is the{" "}
          <Term tip={GLOSSARY.range}>first price range</Term>, if the bot couldn&apos;t find it.
        </p>
        <div ref={corrections} className="mt-3 grid gap-3 sm:grid-cols-3">
          {CORRECTIONS.map((f) => (
            <label key={f.name} className="space-y-1">
              <span className="text-xs text-ink-2">{f.label}</span>
              <input name={f.name} type={f.type} step={f.step} min={f.type === "number" ? 0 : undefined} className={fieldClass} />
            </label>
          ))}
        </div>
        {run && input && !busy && (
          <button type="button" onClick={() => start(input)}
            className="mt-3 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white">
            Run again with these details
          </button>
        )}
      </details>

      {run && (
        <section ref={progressRef} className="scroll-mt-4 space-y-4 rounded-lg border border-line p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="font-semibold">{run.phase === "done" ? "Analysis complete" : `Analyzing ${input?.label ?? ""}`}</h2>
            <button type="button" onClick={reset} className="text-sm text-ink-2 underline-offset-2 hover:text-ink hover:underline">
              {busy ? "Cancel" : "Analyze another"}
            </button>
          </div>
          {run.phase !== "done" && <ProgressBar run={run} hasFile={Boolean(input?.file)} />}
          {run.phase !== "done" && <StepList steps={run.steps} live />}
          {run.phase === "error" && (
            <div className="flex flex-wrap gap-3">
              <button type="button" onClick={() => setCorrectionsOpen(true)}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white">Add the missing details</button>
              {input && <button type="button" onClick={() => start(input)} className="rounded-md border border-line px-4 py-2 text-sm">Try again</button>}
            </div>
          )}
        </section>
      )}

      {result && <ResultView a={result} onImprove={() => setCorrectionsOpen(true)} />}
    </div>
  );
}
