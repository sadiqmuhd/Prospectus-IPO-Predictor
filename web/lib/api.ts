const RAW_API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").trim().replace(/\/$/, "");
// without a scheme the browser would treat the host as a path on this site
export const API_URL = /^https?:\/\//.test(RAW_API_URL) ? RAW_API_URL : `https://${RAW_API_URL}`;

export const BUCKETS = ["overpriced", "fair", "underpriced", "heavily_underpriced"] as const;
export type Bucket = (typeof BUCKETS)[number];

export const BUCKET_LABELS: Record<Bucket, string> = {
  overpriced: "Overpriced (< 0%)",
  fair: "Fair (0–10%)",
  underpriced: "Underpriced (10–30%)",
  heavily_underpriced: "Heavily underpriced (> 30%)",
};

export type Reason = {
  feature: string;
  label: string;
  group: string;
  value: number | null;
  contribution: number;
  text?: string; // plain sentence, e.g. "Priced 54% above the middle of its first price range"
};

export type StepState = "pending" | "running" | "done" | "warning" | "error";
export type Step = {
  id: string;
  state: StepState;
  title: string;
  summary?: string | null;
  details?: { label: string; value: string }[];
  quote?: string | null;
  link?: string | null;
  rows?: (Record<string, string> & { link?: string })[] | null;
};

export type Verdict = { label: string; tone: "positive" | "neutral" | "negative"; in_ten: number; sentence: string };

export type IpoStatus = "priced" | "registration" | "withdrawn" | "public" | "none";
export type Company = {
  cik: number;
  name: string;
  tickers: string[];
  status: IpoStatus;
  status_label: string;
  description: string;
  analyzable: boolean;
  date?: string;
  document?: string;
};

type Base = {
  model: string;
  feature_set: string;
  test_scores: Record<string, number>;
  reasons: Reason[];
};
export type BinaryPrediction = Base & { probability: number };
export type BucketPrediction = Base & { probabilities: Record<Bucket, number>; predicted: Bucket };

export type Analysis = {
  source: string;
  market: "US" | "NG";
  deal: {
    offer_price: number | null;
    shares_offered: number | null;
    price_range: [number, number] | null;
    price_range_source: string | null;
    ipo_date_used: string;
    underwriters: string | null;
    lead_underwriter_rank: number | null;
    syndicate_size: number | null;
    exchange: string | null;
    sections_found: Record<string, boolean>;
    risk_factors_excerpt: string | null;
    document_chars: number;
    expected_price?: boolean;
    preliminary?: boolean;
  };
  features: Record<string, number | null>;
  predictions: { pop?: BinaryPrediction; pop10?: BinaryPrediction; bucket?: BucketPrediction };
  notes: string[];
  disclaimer: string;
  verdict?: Verdict;
  steps?: Step[];
  // examples only
  company?: string;
  ticker?: string;
  ipo_date?: string;
  actual?: { first_day_return: number; pop: number; bucket: Bucket };
};

export type ExampleSummary = {
  id: string;
  company: string;
  ipo_date: string;
  actual: { first_day_return: number; pop: number; bucket: Bucket };
};

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep the status line */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function searchCompanies(q: string, signal?: AbortSignal): Promise<Company[]> {
  return asJson(await fetch(`${API_URL}/search?q=${encodeURIComponent(q)}`, { signal }));
}

export type StreamEvent =
  | ({ event: "step" } & Step)
  | { event: "result"; result: Analysis }
  | { event: "error"; message: string };

/**
 * Runs an analysis, reporting upload progress (0–1) and each step as the API finishes it.
 * XMLHttpRequest rather than fetch: only XHR reports upload progress, and it also exposes the
 * response as it streams in (one JSON event per line).
 */
export function analyzeStream(
  form: FormData,
  on: { upload?: (fraction: number) => void; event: (e: StreamEvent) => void },
): { done: Promise<void>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  let seen = 0;
  let buffer = "";
  const drain = () => {
    buffer += xhr.responseText.slice(seen);
    seen = xhr.responseText.length;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) on.event(JSON.parse(line) as StreamEvent);
  };
  const done = new Promise<void>((resolve, reject) => {
    xhr.upload.onprogress = (e) => e.lengthComputable && on.upload?.(e.loaded / e.total);
    xhr.upload.onload = () => on.upload?.(1);
    xhr.onprogress = drain;
    xhr.onload = () => {
      if (xhr.status >= 400) {
        let detail = `The server answered ${xhr.status}.`;
        try {
          const body = JSON.parse(xhr.responseText);
          if (typeof body.detail === "string") detail = body.detail;
          else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
        } catch {
          /* keep the status line */
        }
        reject(new Error(detail));
        return;
      }
      drain();
      if (buffer.trim()) on.event(JSON.parse(buffer) as StreamEvent);
      resolve();
    };
    xhr.onerror = () => reject(new Error("Couldn't reach the analysis server. Check your connection and try again."));
    xhr.onabort = () => reject(new DOMException("Cancelled", "AbortError"));
  });
  xhr.open("POST", `${API_URL}/analyze/stream`);
  xhr.send(form);
  return { done, abort: () => xhr.abort() };
}

export async function getExamples(): Promise<ExampleSummary[]> {
  return asJson(await fetch(`${API_URL}/examples`, { next: { revalidate: 3600 } }));
}

export async function getExample(id: string): Promise<Analysis> {
  return asJson(await fetch(`${API_URL}/examples/${encodeURIComponent(id)}`, { next: { revalidate: 3600 } }));
}

export type ModelCard = Record<
  "pop" | "pop10" | "bucket",
  { model_name: string; feature_set: string; trained_years: [number, number]; test_scores: Record<string, number> }
> & {
  text_effect?: { target: string; model: string; metric: string; gain: number; ci_low: number; ci_high: number }[];
};

export async function getModelCard(): Promise<ModelCard> {
  return asJson(await fetch(`${API_URL}/model`, { next: { revalidate: 3600 } }));
}

export const pct = (v: number | null | undefined, digits = 0) =>
  v == null ? "–" : `${(v * 100).toFixed(digits)}%`;

export const signedPct = (v: number, digits = 1) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;
