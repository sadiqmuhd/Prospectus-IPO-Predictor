"use client";

import { Analysis, BUCKETS, BUCKET_LABELS, Reason, Verdict, pct, signedPct } from "@/lib/api";
import StepList from "./StepList";
import { GLOSSARY, Icon, Term } from "./ui";

const BUCKET_COLORS = ["var(--b1)", "var(--b2)", "var(--b3)", "var(--b4)"];
const TONE_STYLE = {
  positive: { bg: "var(--success-bg)", ink: "var(--success-ink)" },
  neutral: { bg: "var(--surface-2)", ink: "var(--text-2)" },
  negative: { bg: "var(--danger-bg)", ink: "var(--danger-ink)" },
};

/** Saved examples predate the verdict field; derive the same wording from the probability. */
function verdictFor(p: number): Verdict {
  const in_ten = Math.round(p * 10);
  const [label, tone]: [string, Verdict["tone"]] =
    p >= 0.75 ? ["Likely to pop", "positive"] : p >= 0.6 ? ["Leaning towards a pop", "positive"]
      : p >= 0.45 ? ["Could go either way", "neutral"] : p >= 0.3 ? ["Leaning towards a fall", "negative"]
        : ["Likely to close below the offer price", "negative"];
  return { label, tone, in_ten, sentence: `About ${in_ten} in 10 chance its shares close above the offer price on the first day.` };
}

function Card({ title, children, className = "" }: { title: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-5 ${className}`}>
      <h2 className="mb-3 text-sm font-semibold text-ink">{title}</h2>
      {children}
    </section>
  );
}

function Headline({ a }: { a: Analysis }) {
  const p = a.predictions.pop!;
  const v = a.verdict ?? verdictFor(p.probability);
  const tone = TONE_STYLE[v.tone];
  const top = p.reasons[0];
  return (
    <section className="rounded-lg border border-line bg-surface p-6">
      <span className="inline-block rounded-full px-3 py-1 text-sm font-medium" style={{ background: tone.bg, color: tone.ink }}>
        {v.label}
      </span>
      <div className="mt-4 flex flex-wrap items-end gap-x-6 gap-y-2">
        <p className="text-5xl font-semibold tabular-nums tracking-tight">
          {v.in_ten} <span className="text-2xl font-normal text-ink-2">in 10</span>
        </p>
        <p className="max-w-md pb-1 text-ink-2">
          chance its shares close above the <Term tip={GLOSSARY.offer}>offer price</Term>
          {a.deal.offer_price != null && <> of ${a.deal.offer_price.toFixed(2)}{a.deal.expected_price && " (expected)"}</>}{" "}
          on the first day of trading.
        </p>
      </div>
      {top && (
        <p className="mt-4 text-sm text-ink">
          <span className="text-ink-2">Biggest factor: </span>
          {top.text ?? top.label}
          {top.contribution >= 0 ? ", which raises the chance." : ", which lowers the chance."}
        </p>
      )}
      {a.actual && (
        <p className="mt-3 text-sm text-ink-2">
          What actually happened: <span className="font-medium text-ink">{signedPct(a.actual.first_day_return)}</span> on day one ({BUCKET_LABELS[a.actual.bucket]}).
        </p>
      )}
    </section>
  );
}

function Buckets({ probs, predicted, actual }: { probs: Record<string, number>; predicted: string; actual?: string }) {
  return (
    <div className="space-y-2.5">
      {BUCKETS.map((b, i) => (
        <div key={b} className="grid grid-cols-[minmax(0,10.5rem)_1fr_2.75rem] items-center gap-3 text-sm">
          <span className={b === predicted ? "font-medium text-ink" : "text-ink-2"}>
            {BUCKET_LABELS[b]}
            {b === actual && <span className="ml-1 text-xs text-ink-3">(actual)</span>}
          </span>
          <div className="h-3 w-full rounded" style={{ background: "var(--neutral)" }} title={`${BUCKET_LABELS[b]}: ${pct(probs[b], 1)}`}>
            <div className="h-full rounded" style={{ width: `${probs[b] * 100}%`, background: BUCKET_COLORS[i], minWidth: 2 }} />
          </div>
          <span className="text-right tabular-nums text-ink-2">{pct(probs[b])}</span>
        </div>
      ))}
    </div>
  );
}

function Reasons({ reasons }: { reasons: Reason[] }) {
  const max = Math.max(...reasons.map((r) => Math.abs(r.contribution)), 1e-9);
  return (
    <ul className="space-y-3">
      {reasons.map((r) => {
        const up = r.contribution >= 0;
        const color = up ? "var(--toward)" : "var(--against)";
        return (
          <li key={r.feature} className="flex items-start gap-3 text-sm">
            <span className="mt-0.5 shrink-0" style={{ color }} aria-label={up ? "raises the chance" : "lowers the chance"}>
              <Icon name={up ? "up" : "down"} />
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-ink">{r.text ?? r.label}</p>
              <div className="mt-1 h-1.5 rounded-full" style={{ width: `${Math.max(4, (Math.abs(r.contribution) / max) * 100)}%`, background: color }}
                title={`SHAP ${r.contribution.toFixed(3)}`} />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export default function ResultView({ a, onImprove }: { a: Analysis; onImprove?: () => void }) {
  const p = a.predictions;
  const missingRange = !a.deal.price_range;
  return (
    <div className="space-y-5">
      {a.company && (
        <div>
          <h1 className="text-2xl font-semibold">{a.company}</h1>
          <p className="text-sm text-ink-2">IPO on {a.ipo_date}</p>
        </div>
      )}

      {p.pop && <Headline a={a} />}

      {a.notes.length > 0 && (
        <div className="space-y-2">
          {a.notes.map((n) => (
            <div key={n} className="flex flex-wrap items-center gap-3 rounded-md px-4 py-3 text-sm"
              style={{ background: "var(--warning-bg)", color: "var(--warning-ink)" }}>
              <span className="flex-1">{n}</span>
              {onImprove && missingRange && n.includes("price range") && (
                <button type="button" onClick={onImprove} className="rounded-md border px-3 py-1 text-xs font-medium"
                  style={{ borderColor: "currentColor" }}>
                  Add the range
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="grid gap-5 md:grid-cols-2">
        <Card title="Why the bot thinks so">
          {p.pop && <Reasons reasons={p.pop.reasons} />}
          <p className="mt-4 text-xs text-ink-3">
            The factors that moved this prediction most, largest first. Blue raises the chance of a{" "}
            <Term tip={GLOSSARY.pop}>pop</Term>; red lowers it.
          </p>
        </Card>
        <Card title={<>How big a first-day move? <Term tip={GLOSSARY.bucket}>(what these mean)</Term></>}>
          {p.bucket && <Buckets probs={p.bucket.probabilities} predicted={p.bucket.predicted} actual={a.actual?.bucket} />}
          {p.pop10 && (
            <p className="mt-4 text-sm text-ink-2">
              Chance of rising more than 10%: <span className="font-medium tabular-nums text-ink">{pct(p.pop10.probability)}</span>
            </p>
          )}
        </Card>
      </div>

      {a.steps && a.steps.length > 0 && (
        <Card title="How the bot got here">
          <p className="-mt-1 mb-1 text-xs text-ink-3">Open any step to see what the bot read and where it found it.</p>
          <StepList steps={a.steps} />
        </Card>
      )}

      <details className="rounded-lg border border-line p-5 text-sm">
        <summary className="cursor-pointer font-semibold text-ink">Technical details</summary>
        <div className="mt-3 space-y-3 text-ink-2">
          {p.pop && (
            <p>
              Pop model: {p.pop.model.replace("_", " ")} ({p.pop.feature_set === "text" ? "with" : "without"} text features),
              test ROC-AUC {p.pop.test_scores.auc?.toFixed(3)} on 2021–2025 IPOs it never saw in training. Probability{" "}
              {pct(p.pop.probability, 1)}.
            </p>
          )}
          {p.pop && (
            <table className="w-full text-left">
              <thead className="text-xs text-ink-3">
                <tr><th className="py-1 font-normal">Input</th><th className="py-1 text-right font-normal">SHAP contribution</th></tr>
              </thead>
              <tbody>
                {p.pop.reasons.map((r) => (
                  <tr key={r.feature} className="border-t border-line">
                    <td className="py-1.5">{r.label}</td>
                    <td className="py-1.5 text-right tabular-nums">{r.contribution >= 0 ? "+" : ""}{r.contribution.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {a.deal.underwriters && <p><span className="text-ink-3">Underwriters read: </span>{a.deal.underwriters}</p>}
          {a.source.startsWith("http") && (
            <a href={a.source} target="_blank" rel="noreferrer" className="text-accent-ink underline-offset-2 hover:underline">
              Source document on SEC EDGAR
            </a>
          )}
        </div>
      </details>

      <p className="text-xs text-ink-3">{a.disclaimer}</p>
    </div>
  );
}
