import Link from "next/link";
import { notFound } from "next/navigation";
import ResultView from "@/components/ResultView";
import { Analysis, getExample } from "@/lib/api";

export default async function ExamplePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let a: Analysis;
  try {
    a = await getExample(id);
  } catch {
    notFound();
  }
  return (
    <div className="space-y-4">
      <Link href="/" className="text-sm text-ink-2 hover:text-ink">
        ← Analyze another prospectus
      </Link>
      <p className="text-xs text-ink-3">
        Scored from the saved EDGAR prospectus. The served model was refit on every year including this one, so this is
        an illustration; honest out-of-sample scores are on the About page.
      </p>
      <ResultView a={a} />
    </div>
  );
}
