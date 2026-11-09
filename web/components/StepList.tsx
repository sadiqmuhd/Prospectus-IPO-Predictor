"use client";

import { useState } from "react";
import type { Step } from "@/lib/api";
import { Icon } from "./ui";

/** The analysis steps in order, with the titles shown before each one has run. */
export const STEP_ORDER: { id: string; title: string }[] = [
  { id: "document", title: "Read the prospectus" },
  { id: "deal", title: "Find the deal terms" },
  { id: "range", title: "Check earlier filings for the price range" },
  { id: "underwriters", title: "Identify the banks selling the shares" },
  { id: "tone", title: "Read the Risk Factors" },
  { id: "market", title: "Check market conditions" },
  { id: "model", title: "Run the models" },
];

const STATE_ICON = {
  pending: { name: "circle", color: "var(--text-3)" },
  running: { name: "spinner", color: "var(--accent)" },
  done: { name: "check", color: "var(--success-ink)" },
  warning: { name: "warning", color: "var(--warning-ink)" },
  error: { name: "error", color: "var(--against)" },
} as const;

function hasEvidence(s: Step) {
  return Boolean(s.details?.length || s.quote || s.rows?.length || s.link);
}

function Evidence({ s }: { s: Step }) {
  return (
    <div className="mt-3 space-y-3 text-sm">
      {s.details && s.details.length > 0 && (
        <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-[minmax(0,12rem)_1fr]">
          {s.details.map((d) => (
            <div key={d.label} className="contents">
              <dt className="text-ink-3">{d.label}</dt>
              <dd className="text-ink">{d.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {s.quote && (
        <figure className="border-l-2 border-line pl-3">
          <figcaption className="mb-1 text-xs text-ink-3">From the document</figcaption>
          <blockquote className="whitespace-pre-line text-ink-2">“{s.quote}”</blockquote>
        </figure>
      )}
      {s.rows && s.rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-ink-3">
              <tr>
                {Object.keys(s.rows[0]).filter((k) => k !== "link").map((k) => (
                  <th key={k} className="py-1 pr-4 font-normal">{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {s.rows.map((r, i) => (
                <tr key={i} className="border-t border-line">
                  {Object.entries(r).filter(([k]) => k !== "link").map(([k, v], j) => (
                    <td key={k} className="py-1.5 pr-4 tabular-nums">
                      {j === 0 && r.link ? (
                        <a href={r.link} target="_blank" rel="noreferrer" className="text-accent-ink underline-offset-2 hover:underline">{v}</a>
                      ) : v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {s.link && (
        <a href={s.link} target="_blank" rel="noreferrer"
          className="inline-flex items-center gap-1 text-accent-ink underline-offset-2 hover:underline">
          <Icon name="link" className="h-3.5 w-3.5" /> Open the document on SEC EDGAR
        </a>
      )}
    </div>
  );
}

function StepItem({ s, defaultOpen }: { s: Step; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const icon = STATE_ICON[s.state];
  const expandable = (s.state === "done" || s.state === "warning") && hasEvidence(s);
  const head = (
    <>
      <span className="mt-0.5 shrink-0" style={{ color: icon.color }}>
        <Icon name={icon.name} />
      </span>
      <span className="min-w-0 flex-1">
        <span className={`block font-medium ${s.state === "pending" ? "text-ink-3" : "text-ink"}`}>{s.title}</span>
        {s.summary && <span className="mt-0.5 block text-sm text-ink-2">{s.summary}</span>}
      </span>
      {expandable && (
        <span className={`mt-0.5 shrink-0 text-ink-3 transition-transform ${open ? "rotate-180" : ""}`}>
          <Icon name="chevron" />
        </span>
      )}
    </>
  );
  return (
    <li className="py-3">
      {expandable ? (
        <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
          className="flex w-full items-start gap-3 rounded text-left focus-visible:outline-2 focus-visible:outline-accent">
          {head}
        </button>
      ) : (
        <div className="flex items-start gap-3" aria-live={s.state === "running" ? "polite" : undefined}>{head}</div>
      )}
      {expandable && open && <div className="pl-7"><Evidence s={s} /></div>}
    </li>
  );
}

export default function StepList({ steps, live = false }: { steps: Step[]; live?: boolean }) {
  const byId = new Map(steps.map((s) => [s.id, s]));
  // while running, show every step so people see what is still to come
  const shown: Step[] = live
    ? STEP_ORDER.map(({ id, title }) => byId.get(id) ?? { id, title, state: "pending" })
    : steps;
  return (
    <ol className="divide-y divide-line">
      {shown.map((s) => (
        <StepItem key={s.id} s={s} defaultOpen={false} />
      ))}
    </ol>
  );
}
