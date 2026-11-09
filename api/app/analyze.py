"""Analyse one prospectus: text -> deal facts, sections, tone, market conditions -> predictions with reasons.

Everything reuses the research code in src/, so the web app scores a prospectus exactly as the models were
trained. The app runs from the project root (relative data paths in src/data assume that).
"""
from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("data", "features", "models"):
    sys.path.insert(0, str(ROOT / "src" / sub))

from build_features import FEATURE_GROUPS, RELATIVE_BASE, add_relative, derive, label  # noqa: E402
from build_us_dataset import RANK_PERIODS, load_underwriter_ranks, uw_rank_from_text  # noqa: E402
from edgar import get, get_main_document, to_text  # noqa: E402
from explain import FEATURE_GROUP, GROUP_LABELS, contributions  # noqa: E402
from fetch_price_ranges import find_range  # noqa: E402
from market import load_prices, market_features  # noqa: E402
from parse_prospectuses import parse  # noqa: E402
from text_lm import lm_features  # noqa: E402

MODELS = ROOT / "models"
TARGETS = ("pop", "pop10", "bucket")
SEC_URL = re.compile(r"^https://(www\.)?sec\.gov/Archives/edgar/data/(\d+)/[\w\-./]+$")
REG_FORMS = ("S-1", "S-1/A", "F-1", "F-1/A", "SB-2", "SB-2/A")
MIN_TEXT_CHARS = 5000
NIGERIA = re.compile(r"Nigerian Exchange|Nigerian Stock Exchange|\bNGX\b|Securities and Exchange Commission,? Nigeria|"
                     r"\bNaira\b|₦", re.I)



class AnalysisError(ValueError):
    """A problem with the input that the user can fix (shown as a 4xx)."""


@dataclass
class Document:
    text: str
    source: str
    cik: int | None = None
    filing_date: date | None = None


# ---------- loading ------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def models() -> dict:
    found = {t: joblib.load(MODELS / f"{t}.joblib") for t in TARGETS if (MODELS / f"{t}.joblib").exists()}
    if not found:
        raise RuntimeError("no trained models in models/; run src/pipeline.py first")
    return found


@lru_cache(maxsize=1)
def ranks() -> pd.DataFrame:
    return load_underwriter_ranks()


@lru_cache(maxsize=1)
def prices() -> pd.DataFrame:
    return load_prices()


@lru_cache(maxsize=1)
def past_ipos() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/processed/us_ipos.csv", parse_dates=["ipo_date"],
                    usecols=["ipo_date", "first_day_return", "px_unreliable"])
    d["first_day_return"] = d["first_day_return"].mask(d["px_unreliable"].astype(bool))
    return d


def market_row(day: pd.Timestamp) -> dict:
    ipos = past_ipos()
    # IPO activity is only known up to the end of our dataset; after that leave it missing rather than zero
    covered = day <= ipos["ipo_date"].max() + pd.Timedelta(days=1)
    return market_features(day, prices(), ipos if covered else None)


@lru_cache(maxsize=1)
def tone_history() -> pd.DataFrame:
    """Past IPOs' text features, the baseline for relative tone."""
    return pd.read_csv(ROOT / "data/processed/model_table.csv", parse_dates=["ipo_date"],
                       usecols=["ipo_date", *RELATIVE_BASE])


@lru_cache(maxsize=1)
def finbert():
    from text_finbert import FinBert
    return FinBert()


def document_from_url(url: str) -> Document:
    m = SEC_URL.match(url.strip())
    if not m:
        raise AnalysisError("Only SEC EDGAR document links are supported "
                            "(https://www.sec.gov/Archives/edgar/data/...). For other prospectuses, upload the file.")
    url = url.strip()
    if url.endswith(".txt"):
        # a full submission file carries every exhibit too; stream only the prospectus, its first document
        doc = get_main_document(url.split("/Archives/", 1)[1])
    else:
        doc = get(url).content.decode("utf-8", errors="replace")
    filed = re.search(r"FILED AS OF DATE:\s*(\d{8})", doc[:5000])
    return Document(text=to_text(doc), source=url, cik=int(m.group(2)),
                    filing_date=pd.to_datetime(filed.group(1)).date() if filed else None)


def document_from_upload(name: str, data: bytes) -> Document:
    lower = name.lower()
    if lower.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if len(text) < MIN_TEXT_CHARS:
            raise AnalysisError("This PDF has almost no text layer; it looks scanned. OCR is not enabled yet.")
    elif lower.endswith((".htm", ".html", ".txt")):
        text = to_text(data.decode("utf-8", errors="replace"))
    else:
        raise AnalysisError("Upload a PDF, HTML or TXT prospectus.")
    return Document(text=text, source=name)


# ---------- features -----------------------------------------------------------------------------------------

def initial_range(cik: int) -> tuple[float, float, str] | None:
    """The first price range the company filed, from its S-1/F-1 history on EDGAR."""
    import json
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").content)
    recent = pd.DataFrame(sub["filings"]["recent"])
    reg = recent[recent["form"].isin(REG_FORMS)].sort_values("filingDate").head(6)
    for f in reg.itertuples():
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{f.accessionNumber.replace('-', '')}/{f.primaryDocument}"
        try:
            rng = find_range(to_text(get(url).content.decode("utf-8", errors="replace")))
        except Exception:
            continue
        if rng:
            return rng[0], rng[1], f"{f.form} filed {f.filingDate}"
    return None


def build_row(doc: Document, overrides: dict) -> tuple[pd.DataFrame, dict]:
    facts, sections = parse("upload", doc.text)
    ipo_day = pd.Timestamp(overrides.get("ipo_date") or doc.filing_date or date.today())
    offer = overrides.get("offer_price") or facts["doc_offer_price"]
    shares = overrides.get("shares_offered") or facts["doc_shares"]
    if not offer:
        raise AnalysisError("Could not find the offer price in this document; enter it in the form.")

    rng_note = None
    lo, hi = overrides.get("range_low"), overrides.get("range_high")
    if not (lo and hi) and doc.cik:
        found = initial_range(doc.cik)
        if found:
            lo, hi, rng_note = found
    mid = (lo + hi) / 2 if lo and hi else None

    year = min(ipo_day.year, RANK_PERIODS[-1][2])  # newest Ritter ranks for IPOs after the last update
    row = {
        "offer_price": offer, "proceeds": offer * shares if shares else np.nan,
        "price_revision": offer / mid - 1 if mid else np.nan,
        "above_range": float(offer > hi) if mid else np.nan, "below_range": float(offer < lo) if mid else np.nan,
        "range_width": (hi - lo) / mid if mid else np.nan,
        "exchange": overrides.get("exchange") or exchange_from_text(doc.text),
        "firm_age": (ipo_day.year - overrides["founding_year"]) if overrides.get("founding_year") else np.nan,
        "vc": overrides.get("vc", np.nan), "dual_class": np.nan, "internet": np.nan, "rollup": np.nan,
        "lead_uw_rank": uw_rank_from_text(facts["doc_underwriters"], year, ranks()),
        "n_syndicate": facts["n_syndicate"] or np.nan,
        "log_doc_chars": np.log(facts["doc_chars"]) if facts["doc_chars"] else np.nan,
        "log_risk_chars": np.log(facts["risk_factors_chars"]) if facts["risk_factors_chars"] else np.nan,
        **market_row(ipo_day),
        **lm_features(doc.text, sections),
        **finbert().features(sections),
    }
    df = derive(pd.DataFrame([row]))
    df["log_proceeds"] = np.log(df["proceeds"].where(df["proceeds"] > 0))
    # relative tone needs a year of earlier IPOs; after the data ends, compare with its last year instead
    hist = tone_history()
    ref_day = min(ipo_day, hist["ipo_date"].max() + pd.Timedelta(days=1))
    df = add_relative(df.assign(ipo_date=ref_day), hist)
    info = {
        "offer_price": offer, "shares_offered": shares, "price_range": [lo, hi] if mid else None,
        "price_range_source": rng_note or ("entered" if mid else None), "ipo_date_used": str(ipo_day.date()),
        "underwriters": facts["doc_underwriters"], "lead_underwriter_rank": row["lead_uw_rank"],
        "syndicate_size": facts["n_syndicate"], "exchange": row["exchange"],
        "sections_found": {k: bool(v) for k, v in sections.items() if k != "ipo_id"},
        "risk_factors_excerpt": (sections.get("risk_factors") or "")[:1500] or None,
        "document_chars": facts["doc_chars"],
    }
    return df, info


def exchange_from_text(text: str) -> str | None:
    head = text[:20000]
    for name in ("Nasdaq", "New York Stock Exchange", "NYSE American", "NYSE"):
        if re.search(re.escape(name), head, re.I):
            return name
    return None


# ---------- prediction ---------------------------------------------------------------------------------------

def reasons(bundle, X: pd.DataFrame, k: int = 6) -> list[dict]:
    c = contributions(bundle, X, background=bundle["background"]).iloc[0]
    top = c.reindex(c.abs().sort_values(ascending=False).index).head(k)
    return [{"feature": f, "label": label(f), "group": GROUP_LABELS.get(FEATURE_GROUP.get(f), ""),
             "value": None if pd.isna(X[f].iloc[0]) else float(X[f].iloc[0]), "contribution": float(v)}
            for f, v in top.items()]


def predict(X: pd.DataFrame) -> dict:
    out = {}
    for target, bundle in models().items():
        p = bundle["model"].predict_proba(X[bundle["features"]])[0]
        res = {"model": bundle["model_name"], "feature_set": bundle["feature_set"],
               "test_scores": bundle["test_scores"], "reasons": reasons(bundle, X)}
        if bundle["kind"] == "binary":
            res["probability"] = float(p[1])
        else:
            res["probabilities"] = dict(zip(bundle["classes"], map(float, p)))
            res["predicted"] = bundle["classes"][int(np.argmax(p))]
        out[target] = res
    return out


def analyze(doc: Document, overrides: dict | None = None) -> dict:
    overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
    if len(doc.text) < MIN_TEXT_CHARS:
        raise AnalysisError("The document is too short to be a prospectus.")
    X, info = build_row(doc, overrides)
    nigerian = bool(NIGERIA.search(doc.text[:50000]))
    notes = []
    if nigerian:
        notes.append("This looks like a Nigerian (NGX) prospectus. The models were trained on US IPOs, so treat "
                     "the prediction as indicative only. NGX caps daily moves at 10%, so a large first-day pop "
                     "shows up over several sessions, not on day one. Prices are read in the document's currency.")
    if info["price_range"] is None:
        notes.append("No filing price range was found, so the strongest single predictor (price revision) is "
                     "missing. Enter the initial range for a better estimate.")
    features = {f: (None if pd.isna(X[f].iloc[0]) else float(X[f].iloc[0]))
                for g in FEATURE_GROUPS.values() for f in g if f in X}
    return {"source": doc.source, "market": "NG" if nigerian else "US", "deal": info, "features": features,
            "predictions": predict(X), "notes": notes,
            "disclaimer": "Research demo from a data science course project. Not investment advice."}
