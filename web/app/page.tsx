import Link from "next/link";
import AnalyzeForm from "@/components/AnalyzeForm";
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
          Give the bot a prospectus. It reads the deal terms, the underwriters and the tone of the Risk Factors, checks
          market conditions, and estimates the first-day return, with the reasons behind it. Trained on 3,600 US IPOs
          from 2000 to 2025, with a Nigerian (NGX) case study.
        </p>
      </section>

      <AnalyzeForm />

      {examples.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm font-semibold">Or open a past IPO with a known outcome</h2>
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
