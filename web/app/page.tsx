import Link from "next/link";
import Analyzer from "@/components/Analyzer";
import { BUCKET_LABELS, ExampleSummary, getExamples, signedPct } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  let examples: ExampleSummary[] = [];
  try {
    examples = await getExamples();
  } catch {
    // the API may be waking up (free hosting sleeps); the form still works once it is up
  }
  return (
    <div className="space-y-10">
      <section className="max-w-2xl space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">Will this IPO pop on day one?</h1>
        <p className="text-ink-2">
          Find a company and the bot reads its IPO prospectus: the price, the banks, the risks and the market at the
          time. It then estimates how the shares will do on their first day, and shows you why, step by step.
        </p>
        <p className="text-sm text-ink-3">Trained on 3,600 US IPOs from 2000 to 2025. A research project, not investment advice.</p>
      </section>

      <Analyzer />

      {examples.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm font-semibold">Or open a past IPO with a known outcome, already analyzed</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {examples.map((e) => (
              <Link
                key={e.id}
                href={`/examples/${e.id}`}
                className="rounded-lg border border-line p-4 transition-colors hover:bg-surface-2"
              >
                <div className="font-medium">{e.company}</div>
                <div className="mt-1 text-sm text-ink-2">
                  {e.ipo_date} · {signedPct(e.actual.first_day_return)} on day one
                </div>
                <div className="text-xs text-ink-3">{BUCKET_LABELS[e.actual.bucket]}</div>
              </Link>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
