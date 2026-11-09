import { Analysis, BUCKETS, BUCKET_LABELS, BinaryPrediction, Reason, pct, signedPct } from "@/lib/api";

const BUCKET_COLORS = ["var(--b1)", "var(--b2)", "var(--b3)", "var(--b4)"];
const SECTION_LABELS: Record<string, string> = {
  summary: "Summary",
  risk_factors: "Risk Factors",
  use_of_proceeds: "Use of Proceeds",
};

function Card({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-5 ${className}`}>
      <h2 className="mb-3 text-sm font-semibold text-ink">{title}</h2>
      {children}
    </section>
  );
}

function Probability({ title, p, hint }: { title: string; p: BinaryPrediction; hint: string }) {
  const value = p.probability;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-ink-2">{title}</span>
        <span className="text-3xl font-semibold tabular-nums">{pct(value)}</span>
      </div>
      <div
        className="mt-2 h-2 w-full overflow-hidden rounded-full"
        style={{ background: "var(--neutral)" }}
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
        aria-label={title}
        title={`${title}: ${pct(value, 1)}`}
      >
        <div className="h-full rounded-full" style={{ width: `${value * 100}%`, background: "var(--accent)" }} />
      </div>
      <p className="mt-1 text-xs text-ink-3">{hint}</p>
    </div>
  );
}

function Buckets({ probs, predicted, actual }: { probs: Record<string, number>; predicted: string; actual?: string }) {
  const max = Math.max(...Object.values(probs));
  return (
    <div className="space-y-2">
      {BUCKETS.map((b, i) => (
        <div key={b} className="grid grid-cols-[minmax(0,11rem)_1fr_3rem] items-center gap-3 text-sm">
          <span className={b === predicted ? "font-medium text-ink" : "text-ink-2"}>
            {BUCKET_LABELS[b]}
            {b === actual && <span className="ml-1 text-xs text-ink-3">(actual)</span>}
          </span>
          <div className="h-3 w-full" title={`${BUCKET_LABELS[b]}: ${pct(probs[b], 1)}`}>
            <div
              className="h-full rounded-r"
              style={{ width: `${(probs[b] / max) * 100}%`, background: BUCKET_COLORS[i], minWidth: 2 }}
            />
          </div>
          <span className="text-right tabular-nums text-ink-2">{pct(probs[b])}</span>
        </div>
      ))}
    </div>
  );
}

function formatValue(r: Reason): string {
  if (r.value == null) return "missing";
  const v = r.value;
  if (r.feature === "price_revision" || r.feature.endsWith("_ret_30d") || r.feature === "prior_ipo_return_30d")
    return signedPct(v);
  if (r.feature.startsWith("lm_") && !r.feature.endsWith("_words") && !r.feature.endsWith("_fog"))
    return pct(v, 2);
  if (r.feature.startsWith("fb_")) return v.toFixed(2);
  if (r.feature.startsWith("log_")) return Math.exp(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (["above_range", "below_range", "is_nasdaq", "vc", "dual_class", "internet", "rollup"].includes(r.feature))
    return v ? "yes" : "no";
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function Reasons({ reasons, toward }: { reasons: Reason[]; toward: string }) {
  const max = Math.max(...reasons.map((r) => Math.abs(r.contribution)), 1e-9);
  return (
    <div>
      <div className="mb-2 flex justify-between text-xs text-ink-3">
        <span>
          <span className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ background: "var(--against)" }} />
          against {toward}
        </span>
        <span>
          toward {toward}
          <span className="ml-1 inline-block h-2 w-2 rounded-sm" style={{ background: "var(--toward)" }} />
        </span>
      </div>
      <ul className="space-y-2">
        {reasons.map((r) => {
          const w = (Math.abs(r.contribution) / max) * 50;
          const pos = r.contribution >= 0;
          return (
            <li key={r.feature} className="text-sm" title={`SHAP ${r.contribution.toFixed(3)}`}>
              <div className="flex justify-between gap-2">
                <span className="text-ink">{r.label}</span>
                <span className="shrink-0 tabular-nums text-ink-3">{formatValue(r)}</span>
              </div>
              <div className="relative mt-1 h-2 w-full">
                <div className="absolute left-1/2 top-0 h-full w-px" style={{ background: "var(--border)" }} />
                <div
                  className="absolute top-0 h-full"
                  style={{
                    background: pos ? "var(--toward)" : "var(--against)",
                    width: `${w}%`,
                    left: pos ? "50%" : `${50 - w}%`,
                    borderRadius: pos ? "0 3px 3px 0" : "3px 0 0 3px",
                  }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Fact({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 border-b border-line py-1.5 text-sm last:border-0">
      <dt className="text-ink-2">{k}</dt>
      <dd className="text-right tabular-nums text-ink">{v ?? "–"}</dd>
    </div>
  );
}

export default function ResultView({ a }: { a: Analysis }) {
  const { deal, predictions: p } = a;
  const money = (v: number | null) => (v == null ? "–" : `${a.market === "NG" ? "₦" : "$"}${v.toFixed(2)}`);
  return (
    <div className="space-y-5">
      {a.company && (
        <div>
          <h1 className="text-2xl font-semibold">{a.company}</h1>
          <p className="text-sm text-ink-2">
            IPO on {a.ipo_date}
            {a.actual && (
              <>
                {" · "}actual first-day return{" "}
                <span className="font-medium text-ink">{signedPct(a.actual.first_day_return)}</span> (
                {BUCKET_LABELS[a.actual.bucket]})
              </>
            )}
          </p>
        </div>
      )}

      {a.notes.length > 0 && (
        <div className="space-y-2">
          {a.notes.map((n) => (
            <p
              key={n}
              className="rounded-md px-4 py-3 text-sm"
              style={{ background: "var(--warning-bg)", color: "var(--warning-ink)" }}
            >
              <span aria-hidden>⚠ </span>
              {n}
            </p>
          ))}
        </div>
      )}

      <div className="grid gap-5 md:grid-cols-2">
        <Card title="Prediction">
          <div className="space-y-5">
            {p.pop && (
              <Probability title="Chance it closes above the offer price" p={p.pop} hint="Pop: first-day return > 0%" />
            )}
            {p.pop10 && (
              <Probability title="Chance of a pop above 10%" p={p.pop10} hint="First-day return > 10%" />
            )}
          </div>
        </Card>
        <Card title="Expected return bucket">
          {p.bucket && <Buckets probs={p.bucket.probabilities} predicted={p.bucket.predicted} actual={a.actual?.bucket} />}
        </Card>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <Card title="Why: what moved the pop prediction">
          {p.pop && <Reasons reasons={p.pop.reasons} toward="a pop" />}
          <p className="mt-3 text-xs text-ink-3">
            SHAP contributions for this prospectus, largest first. Missing values are shown as such; the model has
            learned what a missing value usually means.
          </p>
        </Card>
        <Card title="What the bot read">
          <dl>
            <Fact k="Offer price" v={money(deal.offer_price)} />
            <Fact
              k="Initial filing range"
              v={deal.price_range ? `${money(deal.price_range[0])} – ${money(deal.price_range[1])}` : "not found"}
            />
            <Fact k="Shares offered" v={deal.shares_offered?.toLocaleString()} />
            <Fact k="Exchange" v={deal.exchange} />
            <Fact k="Lead underwriter rank (0–9)" v={deal.lead_underwriter_rank?.toFixed(1)} />
            <Fact k="Syndicate size" v={deal.syndicate_size || "–"} />
            <Fact k="Date used for market conditions" v={deal.ipo_date_used} />
            <Fact
              k="Sections found"
              v={
                Object.entries(deal.sections_found)
                  .filter(([, ok]) => ok)
                  .map(([s]) => SECTION_LABELS[s] ?? s)
                  .join(", ") || "none"
              }
            />
          </dl>
          {deal.underwriters && (
            <p className="mt-3 text-xs text-ink-3">
              <span className="text-ink-2">Underwriters: </span>
              {deal.underwriters}
            </p>
          )}
        </Card>
      </div>

      {deal.risk_factors_excerpt && (
        <Card title="Risk Factors (opening)">
          <p className="whitespace-pre-line text-sm leading-relaxed text-ink-2">{deal.risk_factors_excerpt}…</p>
        </Card>
      )}

      <p className="text-xs text-ink-3">{a.disclaimer}</p>
    </div>
  );
}
