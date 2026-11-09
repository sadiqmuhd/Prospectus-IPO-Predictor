"""Analyse one prospectus: text -> deal facts, sections, tone, market conditions -> predictions with reasons.

Everything reuses the research code in src/, so the web app scores a prospectus exactly as the models were
trained. The app runs from the project root (relative data paths in src/data assume that).
"""
from __future__ import annotations

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
from text_lm import lexicon, lm_features  # noqa: E402

from .search import REG_FORMS, doc_url, filings  # noqa: E402

MODELS = ROOT / "models"
TARGETS = ("pop", "pop10", "bucket")
SEC_URL = re.compile(r"^https://(www\.)?sec\.gov/Archives/edgar/data/(\d+)/[\w\-./]+$")
MIN_TEXT_CHARS = 5000
# "Prospectus dated May 13, 2026" / "The date of this prospectus is July 29, 2026": the pricing date
PROSPECTUS_DATE = re.compile(r"(?i)(preliminary\s+)?(?:date\s+of\s+this\s+prospectus\s+is|prospectus\s+dated)\s+"
                             r"([A-Z][a-z]+\.?\s+\d{1,2},\s*\d{4})")
# a PDF ligature glyph (ff, fi, fl, ffi, ffl) whose text mapping is lost reads "of ering" for "offering"
LIGATURE_GAP = re.compile(r"\b([A-Za-z]*f) ([a-z]+)\b")
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
    pages: int | None = None


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


def prices() -> pd.DataFrame:
    return _prices(date.today())


@lru_cache(maxsize=1)
def _prices(today: date) -> pd.DataFrame:
    """Index prices, refetched at most once a day when the cache is behind (a long-running server stays current)."""
    p = load_prices()
    if p.index.max() < pd.Timestamp(today) - pd.Timedelta(days=4):
        try:
            p = load_prices(refresh=True)
        except Exception:  # Yahoo unreachable: the cached prices still give a usable, older market row
            pass
    return p


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


def needs_finbert() -> bool:
    """FinBERT (and PyTorch) only load when a saved model uses its scores, directly or as relative tone."""
    return any(f.removeprefix("rel_").startswith("fb_") for b in models().values() for f in b["features"])


def first_trade_from_text(text: str) -> date | None:
    """The day after the final prospectus's pricing date (trading starts the next business day)."""
    for m in PROSPECTUS_DATE.finditer(text):
        if m.group(1):  # a preliminary prospectus mentioned in passing
            continue
        priced = pd.to_datetime(m.group(2).replace(".", ""), errors="coerce")
        if pd.notna(priced):
            return (priced + pd.offsets.BDay(1)).date()
    return None


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


def repair_ligatures(text: str) -> str:
    """Rejoin words split by a lost ligature, only where the fragment is not a word and the repair is one."""
    vocab = lexicon()[1]

    def fix(m: re.Match) -> str:
        left, right = m.groups()
        if right.upper() in vocab:  # "of each", "if any": two real words
            return m.group(0)
        for missing in ("f", "i", "l", "fi", "fl"):
            if (left + missing + right).upper() in vocab:
                return left + missing + right
        return m.group(0)

    return LIGATURE_GAP.sub(fix, text)


def document_from_upload(name: str, data: bytes) -> Document:
    lower = name.lower()
    if lower.endswith(".pdf"):
        # pdfium, not pdfplumber: a 200-page prospectus takes 9 s and 35 MB instead of 4 min and 2.3 GB
        import pypdfium2 as pdfium
        try:
            pdf = pdfium.PdfDocument(data)
        except pdfium.PdfiumError:
            raise AnalysisError("This file could not be read as a PDF.")
        parts, pages = [], len(pdf)
        for i in range(pages):
            page = pdf[i]
            textpage = page.get_textpage()
            parts.append(textpage.get_text_range())
            textpage.close()
            page.close()
        pdf.close()
        text = repair_ligatures("\n".join(parts).replace("\r\n", "\n"))
        if len(text) < MIN_TEXT_CHARS:
            raise AnalysisError("This PDF has almost no text layer; it looks scanned. OCR is not enabled yet.")
        return Document(text=text, source=name, pages=pages)
    if lower.endswith((".htm", ".html", ".txt")):
        return Document(text=to_text(data.decode("utf-8", errors="replace")), source=name)
    raise AnalysisError("Upload a PDF, HTML or TXT prospectus.")



# ---------- evidence helpers ---------------------------------------------------------------------------------

PRELIMINARY = re.compile(r"subject to completion|preliminary prospectus", re.I)
REG_NUMBER = re.compile(r"Registration\s+(?:Statement\s+)?No\.?\s*(333-\d{5,7})", re.I)
RANGE_HEAD_BYTES = 400_000  # the price range sits on the cover page or in the summary
MAX_RANGE_FILINGS = 10


def quote_around(text: str, start: int, end: int, width: int = 220) -> str:
    """The sentence around a match, whitespace collapsed, for showing where a fact came from."""
    lo = max(0, start - width)
    cut = max(text.rfind(".", lo, start), text.rfind("\n\n", lo, start))
    lo = cut + 1 if cut >= 0 else lo
    hi = text.find(".", end)
    hi = hi + 1 if 0 <= hi < end + width else min(len(text), end + width)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def find_quote(text: str, patterns, limit: int) -> str | None:
    for rx in patterns:
        m = rx.search(text[:limit])
        if m:
            return quote_around(text, m.start(), m.end())
    return None


def head_text(url: str) -> str:
    """The first few hundred KB of a filing: enough for its cover page, without downloading 10 MB."""
    if url.endswith(".txt"):
        return to_text(get_main_document(url.split("/Archives/", 1)[1], max_bytes=RANGE_HEAD_BYTES))
    r = get(url, stream=True)
    buf = bytearray()
    try:
        for chunk in r.iter_content(chunk_size=65536):
            buf.extend(chunk)
            if len(buf) >= RANGE_HEAD_BYTES:
                break
    finally:
        r.close()
    return to_text(bytes(buf).decode("utf-8", errors="replace"))


def cik_from_registration(text: str) -> tuple[int, str] | None:
    """An uploaded US prospectus names its SEC registration number on the cover; EDGAR maps that to the company."""
    m = REG_NUMBER.search(text[:20000])
    if not m:
        return None
    r = get(f'https://efts.sec.gov/LATEST/search-index?q=%22{m.group(1)}%22')
    for h in r.json().get("hits", {}).get("hits", []):
        ciks = h.get("_source", {}).get("ciks") or []
        if ciks:
            return int(ciks[0]), m.group(1)
    return None


def range_history(cik: int) -> list[dict]:
    """Every price range the company filed before pricing, oldest first, with links to the filings."""
    df, _, _ = filings(cik)
    reg = df[df["form"].isin(REG_FORMS)].head(MAX_RANGE_FILINGS)
    out = []
    for f in reg.itertuples():
        url = doc_url(cik, f.accessionNumber, f.primaryDocument)
        try:
            rng = find_range(head_text(url))
        except Exception:
            rng = None
        out.append({"form": f.form, "date": f.filingDate, "link": url,
                    "low": rng[0] if rng else None, "high": rng[1] if rng else None})
    return out


def money(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:.2f} billion"
    if v >= 1e6:
        return f"${v / 1e6:.0f} million"
    return f"${v:,.0f}"


def long_date(d: pd.Timestamp) -> str:
    return f"{d.day} {d:%B %Y}"  # "9 November 2026"; %-d is not portable to Windows


def pct_rank(value: float, history: pd.Series) -> float | None:
    h = history.dropna()
    return None if pd.isna(value) or len(h) < 20 else float((h < value).mean())


# ---------- plain-language reasons -----------------------------------------------------------------------------

MISSING_TEXT = {
    "price_revision": "No price range found to compare the offer price with",
    "above_range": "No price range found to compare the offer price with",
    "below_range": "No price range found to compare the offer price with",
    "range_width": "No price range found",
    "log_firm_age": "Company age not known (add the founding year under “Add details”)",
    "vc": "Venture-capital backing not known",
    "dual_class": "Dual-class shares not known",
    "internet": "Internet business not known",
    "rollup": "Roll-up not known",
    "lead_uw_rank": "Lead bank's prestige not known",
    "n_syndicate": "Number of banks not found",
    "ipo_count_30d": "Recent IPO activity not available (the IPO data ends in December 2025)",
    "prior_ipo_return_30d": "Recent IPO returns not available (the IPO data ends in December 2025)",
}


def reason_text(f: str, v: float | None) -> str:
    """One plain sentence for a model input, e.g. "Priced 54% above the middle of its first price range"."""
    if v is None or pd.isna(v):
        return MISSING_TEXT.get(f, f"{label(f)} not known")
    yes = v >= 0.5
    texts = {
        "price_revision": (f"Priced {abs(v):.0%} {'above' if v > 0 else 'below'} the middle of its first price range"
                           if abs(v) >= 0.005 else "Priced at the middle of its first price range"),
        "above_range": "Priced above its first price range" if yes else "Not priced above its first price range",
        "below_range": "Priced below its first price range" if yes else "Not priced below its first price range",
        "range_width": f"Its first price range was {v:.0%} wide",
        "log_proceeds": f"Raising {money(float(np.exp(v)))}",
        "log_offer_price": f"Offer price of ${float(np.exp(v)):,.2f} a share",
        "is_nasdaq": "Listing on Nasdaq" if yes else "Listing on the NYSE",
        "log_firm_age": f"The company is {float(np.expm1(v)):.0f} years old",
        "vc": "Backed by venture capital" if yes else "Not backed by venture capital",
        "dual_class": "Has dual-class shares" if yes else "Single class of shares",
        "internet": "An internet business" if yes else "Not an internet business",
        "rollup": "A roll-up of smaller firms" if yes else "Not a roll-up",
        "lead_uw_rank": f"Lead bank prestige {v:.0f} out of 9",
        "n_syndicate": f"{v:.0f} banks selling the shares",
        "vix_prev": f"Market volatility (VIX) at {v:.0f}",
        "sp500_ret_30d": f"S&P 500 {v:+.1%} over the past month",
        "nasdaq_ret_30d": f"Nasdaq {v:+.1%} over the past month",
        "ipo_count_30d": f"{v:.0f} IPOs in the past 30 days",
        "prior_ipo_return_30d": f"IPOs in the past month rose {v:+.0%} on day one on average",
    }
    return texts.get(f, f"{label(f)}: {v:.3g}")


def verdict(p: float) -> dict:
    """Headline wording for the pop probability. About two in three US IPOs close above their offer price."""
    if p >= 0.75:
        tone, label_ = "positive", "Likely to pop"
    elif p >= 0.6:
        tone, label_ = "positive", "Leaning towards a pop"
    elif p >= 0.45:
        tone, label_ = "neutral", "Could go either way"
    elif p >= 0.3:
        tone, label_ = "negative", "Leaning towards a fall"
    else:
        tone, label_ = "negative", "Likely to close below the offer price"
    in10 = round(p * 10)
    return {"label": label_, "tone": tone, "in_ten": in10,
            "sentence": f"About {in10} in 10 chance its shares close above the offer price on the first day."}


# ---------- prediction ---------------------------------------------------------------------------------------

def reasons(bundle, X: pd.DataFrame, k: int = 6) -> list[dict]:
    c = contributions(bundle, X, background=bundle["background"]).iloc[0]
    top = c.reindex(c.abs().sort_values(ascending=False).index).head(k)
    out = []
    for f, v in top.items():
        val = None if pd.isna(X[f].iloc[0]) else float(X[f].iloc[0])
        out.append({"feature": f, "label": label(f), "group": GROUP_LABELS.get(FEATURE_GROUP.get(f), ""),
                    "value": val, "contribution": float(v), "text": reason_text(f, val)})
    return out


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


def exchange_from_text(text: str) -> str | None:
    head = text[:20000]
    for name in ("Nasdaq", "New York Stock Exchange", "NYSE American", "NYSE"):
        if re.search(re.escape(name), head, re.I):
            return name
    return None


# ---------- the analysis, step by step ---------------------------------------------------------------------------

def step(id_: str, state: str, title: str, summary: str | None = None, details: list | None = None,
         quote: str | None = None, link: str | None = None, rows: list | None = None) -> dict:
    """One progress event. state: running, done, warning. details: [{label, value}], rows: a small table."""
    return {"event": "step", "id": id_, "state": state, "title": title, "summary": summary,
            "details": details or [], "quote": quote, "link": link, "rows": rows}


def run(load, overrides: dict | None = None):
    """Analyse a prospectus, yielding a progress event per step and finally {"event": "result", ...}.

    `load` is a callable returning the Document, so downloading is itself a visible step. Errors the user can
    fix raise AnalysisError; the caller turns that into an error event.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
    done_steps: list[dict] = []

    def emit(ev: dict) -> dict:
        if ev["state"] != "running":
            done_steps.append(ev)
        return ev

    # 1. the document
    yield emit(step("document", "running", "Reading the prospectus"))
    doc = load()
    if len(doc.text) < MIN_TEXT_CHARS:
        raise AnalysisError("The document is too short to be a prospectus.")
    preliminary = bool(PRELIMINARY.search(doc.text[:6000]))
    words = len(doc.text.split())
    facts, sections = parse("upload", doc.text)
    found = [n for k, n in (("summary", "Summary"), ("risk_factors", "Risk Factors"),
                            ("use_of_proceeds", "Use of Proceeds")) if sections.get(k)]
    yield emit(step(
        "document", "done", "Read the prospectus",
        f"{words:,} words{f' on {doc.pages} pages' if doc.pages else ''}. "
        + ("A preliminary prospectus: this IPO has not been priced yet." if preliminary
           else "The final prospectus, with the offer price."),
        details=[{"label": "Type", "value": "Preliminary (not priced yet)" if preliminary else "Final prospectus"},
                 {"label": "Sections found", "value": ", ".join(found) or "none"}],
        link=doc.source if doc.source.startswith("http") else None))

    # 2. the deal terms
    yield emit(step("deal", "running", "Finding the deal terms"))
    own_range = find_range(doc.text)
    # a preliminary prospectus quotes illustrative prices ("an assumed price of $5.00"), never the real one
    offer = overrides.get("offer_price") or (None if preliminary else facts["doc_offer_price"])
    expected = False
    if not offer and preliminary:
        if not own_range:
            raise AnalysisError("This filing has no price range yet (companies add one in a later amendment), so "
                                "there's no price to predict from. Try the company's latest filing, or enter an "
                                "expected offer price under “Add details”.")
        offer, expected = (own_range[0] + own_range[1]) / 2, True
    if not offer:
        raise AnalysisError("The offer price wasn't found in this document. Enter it under “Add details” and try "
                            "again.")
    shares = overrides.get("shares_offered") or facts["doc_shares"]
    pricing_day = first_trade_from_text(doc.text)
    ipo_day = pd.Timestamp(overrides.get("ipo_date") or pricing_day or doc.filing_date or date.today())
    exchange = overrides.get("exchange") or exchange_from_text(doc.text)
    from parse_prospectuses import PRICE_RES, PRICE_SEARCH_CHARS
    price_quote = None if expected or overrides.get("offer_price") else find_quote(doc.text, PRICE_RES,
                                                                                   PRICE_SEARCH_CHARS)
    # just the dated line: the text around it is usually the table of contents
    date_match = next((m for m in PROSPECTUS_DATE.finditer(doc.text) if not m.group(1)), None) if pricing_day else None
    date_quote = re.sub(r"\s+", " ", date_match.group(0)) if date_match else None
    date_note = ("provided" if overrides.get("ipo_date") else "the day after the prospectus date" if pricing_day
                 else "the filing date" if doc.filing_date else "today (not priced yet)" if preliminary
                 else "today (no date found)")
    yield emit(step(
        "deal", "done",
        f"Expected price ${offer:,.2f} (middle of the range)" if expected else f"Offer price ${offer:,.2f}",
        (f"{shares:,.0f} shares, raising about {money(offer * shares)}" if shares else "Share count not found")
        + (f", listing on {exchange}" if exchange else "") + ".",
        details=[{"label": "Offer price", "value": f"${offer:,.2f}" + (" (expected)" if expected else "")},
                 {"label": "Shares offered", "value": f"{shares:,.0f}" if shares else "not found"},
                 {"label": "Exchange", "value": exchange or "not found"},
                 {"label": "IPO date used", "value": f"{long_date(ipo_day)}, {date_note}"}],
        quote=price_quote or date_quote))

    # 3. the price range
    yield emit(step("range", "running", "Checking earlier filings for the price range"))
    if not doc.cik and not overrides.get("range_low"):
        try:  # uploads: find the company from the registration number, for its filing history
            hit = cik_from_registration(doc.text)
            if hit:
                doc.cik = hit[0]
        except Exception:
            pass
    history = []
    if doc.cik:
        try:
            history = range_history(doc.cik)
        except Exception:  # EDGAR unreachable: carry on without the history, the step says so
            history = []
    with_range = [h for h in history if h["low"]]
    lo, hi, rng_note = overrides.get("range_low"), overrides.get("range_high"), None
    if lo and hi:
        rng_note = overrides.get("range_source", "entered")
    elif with_range:
        first = with_range[0]
        lo, hi, rng_note = first["low"], first["high"], f"{first['form']} filed {first['date']}"
    elif own_range and preliminary:
        lo, hi, rng_note = own_range[0], own_range[1], "this filing"
    mid = (lo + hi) / 2 if lo and hi else None
    rows = [{"Filing": h["form"], "Date": h["date"],
             "Price range": f"${h['low']:,.2f}–{h['high']:,.2f}" if h["low"] else "no range yet",
             "link": h["link"]} for h in history]
    if mid:
        rev = offer / mid - 1
        where = "above" if offer > hi else "below" if offer < lo else "within"
        if expected:
            summary = (f"First range ${lo:,.2f}–{hi:,.2f}. Not priced yet; the prediction assumes it prices at "
                       f"${offer:,.2f}" + (f", {abs(rev):.0%} {'above' if rev > 0 else 'below'} the first range's "
                                           f"middle." if abs(rev) >= 0.005 else ", the middle of that range."))
        else:
            summary = (f"First range ${lo:,.2f}–{hi:,.2f} ({rng_note}). Priced {where} it"
                       + (f", {abs(rev):.0%} {'above' if rev > 0 else 'below'} its middle." if abs(rev) >= 0.005
                          else ", at its middle."))
        if where == "above" and not expected:
            summary += " Pricing above the range usually means strong demand."
        yield emit(step("range", "done", f"First price range ${lo:,.2f}–{hi:,.2f}", summary, rows=rows or None))
    else:
        yield emit(step("range", "warning", "No price range found",
                        "The strongest predictor is missing: how the offer price compares with the first price "
                        "range. Add the range under “Add details” for a better estimate.", rows=rows or None))

    # 4. underwriters
    year = min(ipo_day.year, RANK_PERIODS[-1][2])  # newest Ritter ranks for IPOs after the last update
    rank = uw_rank_from_text(facts["doc_underwriters"], year, ranks())
    banks = [b for b in re.split(r"/", facts["doc_underwriters"] or "") if b.strip()]
    yield emit(step(
        "underwriters", "done" if facts["doc_underwriters"] else "warning",
        f"{facts['n_syndicate']} banks selling the shares" if facts["n_syndicate"] else "Underwriters",
        (f"The most prestigious bank involved rates {rank:.0f} out of 9 on the Carter-Manaster scale."
         if pd.notna(rank) else "None of the banks could be matched to the prestige rankings."),
        details=[{"label": "Banks", "value": ", ".join(banks) if len(banks) > 1
                  else (facts["doc_underwriters"] or "not found")}]))

    # 5. tone of the text
    yield emit(step("tone", "running", "Reading the Risk Factors"))
    lm = lm_features(doc.text, sections)
    hist = tone_history()
    ref_day = min(ipo_day, hist["ipo_date"].max() + pd.Timedelta(days=1))
    recent = hist[(hist["ipo_date"] < ref_day) & (hist["ipo_date"] >= ref_day - pd.Timedelta(days=365))]
    neg, unc = lm.get("lm_risk_factors_negative", np.nan), lm.get("lm_risk_factors_uncertainty", np.nan)
    neg_rank = pct_rank(neg, recent["lm_risk_factors_negative"])
    unc_rank = pct_rank(unc, recent["lm_risk_factors_uncertainty"])
    uses_text = any(f.removeprefix("rel_").startswith(("lm_", "fb_")) for b in models().values()
                    for f in b["features"])
    if pd.notna(neg):
        summary = f"{neg:.1%} of the words are negative"
        if neg_rank is not None:
            summary += (f", more than in {neg_rank:.0%} of IPOs in the year before" if neg_rank >= 0.5
                        else f", fewer than in {1 - neg_rank:.0%} of IPOs in the year before")
        summary += "."
    else:
        summary = "The Risk Factors section was not found."
    if not uses_text:
        summary += " Shown for context: tone did not improve the models' accuracy, so it does not change the score."
    yield emit(step(
        "tone", "done" if pd.notna(neg) else "warning", "Read the Risk Factors", summary,
        details=[{"label": "Negative words", "value": f"{neg:.2%}" if pd.notna(neg) else "n/a"},
                 {"label": "Uncertain words", "value": f"{unc:.2%}" if pd.notna(unc) else "n/a"},
                 {"label": "Compared with the year's IPOs", "value": (
                     f"more negative than {neg_rank:.0%}, more uncertain than {unc_rank:.0%}"
                     if neg_rank is not None and unc_rank is not None else "not enough IPOs to compare")},
                 {"label": "Litigious words", "value": f"{lm.get('lm_risk_factors_litigious', np.nan):.2%}"
                  if pd.notna(lm.get("lm_risk_factors_litigious", np.nan)) else "n/a"}],
        quote=(sections.get("risk_factors") or "")[:700].strip() or None))

    # 6. market conditions
    yield emit(step("market", "running", "Checking market conditions"))
    mk = market_row(ipo_day)
    parts = [f"VIX {mk['vix_prev']:.0f}" if pd.notna(mk["vix_prev"]) else None,
             f"Nasdaq {mk['nasdaq_ret_30d']:+.1%} over the past month" if pd.notna(mk["nasdaq_ret_30d"]) else None]
    yield emit(step(
        "market", "done", "Checked market conditions",
        (", ".join(p for p in parts if p) or "Market data not available") + f", as of {long_date(ipo_day)}.",
        details=[{"label": "Volatility (VIX)", "value": f"{mk['vix_prev']:.1f}" if pd.notna(mk["vix_prev"]) else "n/a"},
                 {"label": "S&P 500, past month", "value": f"{mk['sp500_ret_30d']:+.1%}"
                  if pd.notna(mk["sp500_ret_30d"]) else "n/a"},
                 {"label": "Nasdaq, past month", "value": f"{mk['nasdaq_ret_30d']:+.1%}"
                  if pd.notna(mk["nasdaq_ret_30d"]) else "n/a"},
                 {"label": "IPOs in the past 30 days", "value": f"{mk['ipo_count_30d']:.0f}"
                  if pd.notna(mk["ipo_count_30d"]) else "not available after December 2025"},
                 {"label": "Their mean day-one return", "value": f"{mk['prior_ipo_return_30d']:+.1%}"
                  if pd.notna(mk["prior_ipo_return_30d"]) else "not available after December 2025"}]))

    # 7. the models
    yield emit(step("model", "running", "Running the models"))
    row = {
        "offer_price": offer, "proceeds": offer * shares if shares else np.nan,
        "price_revision": offer / mid - 1 if mid else np.nan,
        "above_range": float(offer > hi) if mid else np.nan, "below_range": float(offer < lo) if mid else np.nan,
        "range_width": (hi - lo) / mid if mid else np.nan, "exchange": exchange,
        "firm_age": (ipo_day.year - overrides["founding_year"]) if overrides.get("founding_year") else np.nan,
        "vc": overrides.get("vc", np.nan), "dual_class": np.nan, "internet": np.nan, "rollup": np.nan,
        "lead_uw_rank": rank, "n_syndicate": facts["n_syndicate"] or np.nan,
        "log_doc_chars": np.log(facts["doc_chars"]) if facts["doc_chars"] else np.nan,
        "log_risk_chars": np.log(facts["risk_factors_chars"]) if facts["risk_factors_chars"] else np.nan,
        **mk, **lm,
        **(finbert().features(sections) if needs_finbert() else {}),
    }
    X = derive(pd.DataFrame([row]))
    X["log_proceeds"] = np.log(X["proceeds"].where(X["proceeds"] > 0))
    # relative tone needs a year of earlier IPOs; after the data ends, compare with its last year instead
    X = add_relative(X.assign(ipo_date=ref_day), hist)
    # any feature that could not be computed is missing, not absent: the models impute it
    wanted = {f for b in models().values() for f in b["features"]}
    X = X.reindex(columns=[*X.columns, *sorted(wanted - set(X.columns))])
    preds = predict(X)
    pop = preds["pop"]
    v = verdict(pop["probability"])
    yield emit(step("model", "done", v["label"], v["sentence"],
                    details=[{"label": "Model", "value": f"{pop['model'].replace('_', ' ')}, test AUC "
                                                         f"{pop['test_scores']['auc']:.2f} on 2021–2025 IPOs"}]))

    nigerian = bool(NIGERIA.search(doc.text[:50000]))
    notes = []
    if nigerian:
        notes.append("This looks like a Nigerian (NGX) prospectus. The models were trained on US IPOs, so treat "
                     "the prediction as indicative only. NGX caps daily moves at 10%, so a large first-day pop "
                     "shows up over several sessions, not on day one. Prices are read in the document's currency.")
    if expected:
        notes.append(f"This IPO hasn't been priced yet. The prediction assumes it prices at ${offer:,.2f}, the middle "
                     "of its latest range. Pricing above the range usually signals strong demand, so check again "
                     "once the final prospectus is out.")
    if not mid:
        notes.append("No price range was found, so the strongest predictor (price vs. range) is missing. Add the "
                     "first price range under “Add details” for a better estimate.")
    if date_note.startswith("today (no date"):
        notes.append("No pricing date was found, so market conditions are as of today. Add the IPO date under "
                     "“Add details” for a past deal.")
    features = {f: (None if pd.isna(X[f].iloc[0]) else float(X[f].iloc[0]))
                for g in FEATURE_GROUPS.values() for f in g if f in X}
    info = {
        "offer_price": offer, "expected_price": expected, "shares_offered": shares,
        "price_range": [lo, hi] if mid else None, "price_range_source": rng_note, "range_history": rows,
        "ipo_date_used": str(ipo_day.date()), "underwriters": facts["doc_underwriters"],
        "lead_underwriter_rank": None if pd.isna(rank) else float(rank), "syndicate_size": facts["n_syndicate"],
        "exchange": exchange, "sections_found": {k: bool(v) for k, v in sections.items() if k != "ipo_id"},
        "risk_factors_excerpt": (sections.get("risk_factors") or "")[:1500] or None,
        "document_chars": facts["doc_chars"], "preliminary": preliminary,
    }
    yield {"event": "result", "result": {
        "source": doc.source, "market": "NG" if nigerian else "US", "deal": info, "features": features,
        "verdict": v, "predictions": preds, "notes": notes, "steps": done_steps,
        "disclaimer": "Research demo from a data science course project. Not investment advice."}}


def analyze(doc: Document, overrides: dict | None = None) -> dict:
    """The whole analysis at once (the steps are kept in the result)."""
    for ev in run(lambda: doc, overrides):
        if ev["event"] == "result":
            return ev["result"]
    raise RuntimeError("analysis ended without a result")
