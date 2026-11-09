import { ModelCard, getModelCard } from "@/lib/api";

export const dynamic = "force-dynamic";

const TARGET_LABELS = { pop: "Pop (> 0%)", pop10: "Pop above 10%", bucket: "Return bucket (4 classes)" } as const;
const MODEL_LABELS: Record<string, string> = {
  lightgbm: "LightGBM",
  random_forest: "Random forest",
  logistic_l1: "L1 logistic regression",
};

const fmt = (v: number | undefined, d = 3) => (v == null ? "–" : v.toFixed(d));

export default async function About() {
  let card: ModelCard | null = null;
  try {
    card = await getModelCard();
  } catch {
    /* API asleep or not deployed: show the method only */
  }
  return (
    <article className="max-w-3xl space-y-8 text-sm leading-relaxed text-ink-2">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold text-ink">About the model</h1>
        <p>
          The bot predicts an IPO&apos;s first-day return (first close vs. offer price) from information public before
          trading starts. It was built for a short-course data science project:{" "}
          <em>Reading the Prospectus: an NLP-driven system for predicting first-day IPO returns in the US and Nigeria.</em>
        </p>
      </section>

      <section className="space-y-2">
        <h2 className="text-base font-semibold text-ink">Data</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>3,642 US operating-company IPOs, 2000–2025 (IPOScoop, Nasdaq, Yahoo Finance; SPACs, funds and penny stocks excluded).</li>
          <li>3,437 final prospectuses from SEC EDGAR, plus each company&apos;s earlier S-1 filings for the initial price range.</li>
          <li>Underwriter prestige ranks and firm data from Jay Ritter; VIX and index returns before each IPO.</li>
          <li>13 Nigerian (NGX) listings with returns, from the exchange&apos;s daily official lists, as a case study.</li>
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-base font-semibold text-ink">Reading the prospectus</h2>
        <p>
          The bot finds the Summary, Risk Factors and Use of Proceeds sections, the offer terms and the underwriting
          syndicate. It scores tone two ways: the Loughran-McDonald financial dictionary (share of negative, uncertain,
          litigious and modal words) and FinBERT, a finance-tuned language model applied to sentences sampled through
          each section.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-base font-semibold text-ink">How well it works</h2>
        <p>
          Models were trained on 2000–2017, tuned on 2018–2020 and tested once on 2021–2025. The served models are then
          refit on all years.
        </p>
        {card ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left tabular-nums">
              <thead className="text-xs text-ink-3">
                <tr>
                  <th className="py-1 pr-4 font-normal">Target</th>
                  <th className="py-1 pr-4 font-normal">Model</th>
                  <th className="py-1 pr-4 font-normal">Features</th>
                  <th className="py-1 pr-4 font-normal">Test score</th>
                </tr>
              </thead>
              <tbody className="text-ink">
                {(Object.keys(TARGET_LABELS) as (keyof typeof TARGET_LABELS)[]).map((t) =>
                  card?.[t] ? (
                    <tr key={t} className="border-t border-line">
                      <td className="py-1.5 pr-4">{TARGET_LABELS[t]}</td>
                      <td className="py-1.5 pr-4">{MODEL_LABELS[card[t].model_name] ?? card[t].model_name}</td>
                      <td className="py-1.5 pr-4">{card[t].feature_set === "text" ? "with text" : "without text"}</td>
                      <td className="py-1.5 pr-4">
                        {t === "bucket"
                          ? `macro-F1 ${fmt(card[t].test_scores.macro_f1)}`
                          : `ROC-AUC ${fmt(card[t].test_scores.auc)}`}
                      </td>
                    </tr>
                  ) : null,
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-ink-3">Scores load from the API, which is not reachable right now.</p>
        )}
        {card?.text_effect && (
          <>
            <h3 className="pt-2 font-semibold text-ink">Does reading the prospectus help?</h3>
            <p>Test-set gain from adding the text features, with a 95% bootstrap interval:</p>
            <ul className="space-y-1 tabular-nums">
              {card.text_effect.map((r) => (
                <li key={`${r.target}-${r.model}`}>
                  {TARGET_LABELS[r.target as keyof typeof TARGET_LABELS]}, {MODEL_LABELS[r.model] ?? r.model}:{" "}
                  <span className="text-ink">
                    {r.gain >= 0 ? "+" : ""}
                    {r.gain.toFixed(3)}
                  </span>{" "}
                  {r.metric} [{r.ci_low.toFixed(3)}, {r.ci_high.toFixed(3)}]
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-base font-semibold text-ink">Limitations</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>2021–2025 data only covers firms still listed today (delisted tickers have no price history), so test returns are biased.</li>
          <li>First-day returns are hard to predict; an ROC-AUC well below 1 is normal in this literature.</li>
          <li>
            Nigerian listings are too few to train on. NGX caps daily moves at 10%, so first-day returns there are
            censored; the project measures them up to the first session inside the limit.
          </li>
          <li>Facts the bot cannot read (for example founding year or VC backing) are left missing unless you enter them.</li>
          <li>This is a course project, not investment advice.</li>
        </ul>
      </section>
    </article>
  );
}
