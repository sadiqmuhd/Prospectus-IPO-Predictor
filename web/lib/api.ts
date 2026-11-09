export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

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
  };
  features: Record<string, number | null>;
  predictions: { pop?: BinaryPrediction; pop10?: BinaryPrediction; bucket?: BucketPrediction };
  notes: string[];
  disclaimer: string;
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

export async function analyze(form: FormData): Promise<Analysis> {
  return asJson(await fetch(`${API_URL}/analyze`, { method: "POST", body: form }));
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
