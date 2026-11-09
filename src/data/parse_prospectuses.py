"""Pull sections and deal facts out of the saved prospectus texts.

Outputs: data/interim/prospectus_facts.csv          offer price, shares, underwriters, section sizes
         data/interim/prospectus_sections.jsonl.gz  Risk Factors, Use of Proceeds, Summary text per IPO
"""
import argparse
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

TEXTS = Path("data/raw/edgar/prospectus_text")
OUT = Path("data/interim")

HEADINGS = {
    "summary": r"prospectus summary|summary",
    "offering": r"the offering",
    "risk_factors": r"risk factors",
    "forward_looking": r"(special|cautionary) note (regarding|concerning|about) forward[- ]looking statements|forward[- ]looking statements",
    "use_of_proceeds": r"use of proceeds",
    "dividend_policy": r"dividend policy",
    "capitalization": r"capitalization",
    "dilution": r"dilution",
    "selected_financial": r"selected (consolidated )?(historical )?financial (and other )?data",
    "mdna": r"management.s discussion and analysis.*",
    "business": r"business|our business",
    "management": r"management",
    "underwriting": r"underwriting|underwriters|plan of distribution",
    "legal_matters": r"legal matters",
    "experts": r"experts",
}
# some filers wrap headings over two lines ("RISK\n FACTORS", common in 2007-2011), so words may be split by a newline
def heading_re(words: str) -> re.Pattern:
    words = words.replace("[- ]", r"[-\s]").replace(" ", r"\s+")
    return re.compile(rf"(?im)^[ \t]*(?:{words})(?:[ \t]*\(.*\))?[ \t]*$")


HEADING_RE = {k: heading_re(v) for k, v in HEADINGS.items()}
# shorter spans are table-of-contents entries or cross-references, not the section itself
MIN_SECTION_CHARS = {"use_of_proceeds": 300, "dividend_policy": 200}
DEFAULT_MIN_SECTION_CHARS = 1500

UW_SUFFIX = r"(?:LLC|L\.L\.C\.|Inc\.?|Incorporated|L\.P\.|LP|& Co\.?|Co\.|Corporation|Corp\.|Ltd\.?|Limited|plc|Securities|Capital Markets|Capital)"
REPS_RE = re.compile(
    r"([A-Z][^.;:]{2,400}?)\s+(?:are|is)\s+(?:acting\s+as\s+)?(?:the\s+)?(?:joint\s+)?(?:book[- ]running\s+managers?\s+(?:of\s+this\s+offering\s+)?and\s+)?"
    r"(?:as\s+)?representatives?\s+of\s+(?:the|each\s+of\s+the)\s+underwriters", re.I)
UW_ROW_RE = re.compile(rf"(?m)^[ \t]*([A-Z][\w.,&'()/ -]{{2,90}}?{UW_SUFFIX}[\w.,() -]{{0,20}}?)[ \t.]+([\d,]{{4,}})[ \t]*$")
# first dollar amount shortly after the phrase: "initial public offering price of the common stock is $20.00",
# "Public offering price ........ $15.00", "at the initial public offering price\nof $11.00 per share"
PRICE_RES = [
    re.compile(r"public offering price\b[^$]{0,60}?(?:US)?\$\s*([\d,]+(?:\.\d+)?)", re.I),
    re.compile(r"price to (?:the )?public\b[^$]{0,60}?(?:US)?\$\s*([\d,]+(?:\.\d+)?)", re.I),
]
PRICE_SEARCH_CHARS = 60000
SHARES_RE = re.compile(r"([\d,]{6,})\s+(?:Shares|American Depositary Shares|ADSs)", re.I)


def find_sections(text: str) -> dict[str, str]:
    hits = sorted((m.start(), m.end(), k) for k, rx in HEADING_RE.items() for m in rx.finditer(text))
    best: dict[str, tuple[int, int]] = {}
    for i, (start, end, key) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        # table-of-contents entries sit right next to the following heading; the real section is the longest span
        if nxt - end > best.get(key, (0, 0))[1] - best.get(key, (0, 0))[0]:
            best[key] = (end, nxt)
    return {k: text[s:e].strip() for k, (s, e) in best.items()
            if e - s >= MIN_SECTION_CHARS.get(k, DEFAULT_MIN_SECTION_CHARS)}


def underwriters(text: str, uw_section: str | None) -> tuple[str | None, int, str]:
    """Raw underwriter text plus syndicate size.

    Names are not split: commas sit inside names ("Goldman, Sachs & Co.", "Merrill Lynch,
    Pierce, Fenner & Smith Incorporated"). Prestige is scored later by finding ranked
    underwriter names in this text, and tables are sometimes alphabetical, so order is not used.
    """
    table = []
    if uw_section:
        rows = UW_ROW_RE.findall(uw_section[:20000])
        table = [n.strip(" .") for n, _ in rows if not re.match(r"total", n, re.I)]
    m = REPS_RE.search(text)
    if m:
        reps = re.sub(r"\s+", " ", m.group(1))[-300:]
        return reps, len(table), "representatives"
    if table:
        return "/".join(table), len(table), "table"
    return None, 0, "none"


def offer_price(head: str) -> float | None:
    for rx in PRICE_RES:
        m = rx.search(head)
        if m:
            v = float(m.group(1).replace(",", ""))
            if 0.5 <= v <= 2000:
                return v
    return None


def parse(ipo_id: str, text: str) -> tuple[dict, dict]:
    head = text[:15000]
    secs = find_sections(text)
    uw_text, n_syndicate, uw_src = underwriters(text, secs.get("underwriting"))
    sh = SHARES_RE.search(head)
    facts = {
        "ipo_id": ipo_id,
        "doc_offer_price": offer_price(text[:PRICE_SEARCH_CHARS]),
        "doc_shares": int(sh.group(1).replace(",", "")) if sh else None,
        "doc_underwriters": uw_text,
        "n_syndicate": n_syndicate,
        "uw_source": uw_src,
        "doc_chars": len(text),
        "risk_factors_chars": len(secs.get("risk_factors", "")),
        "use_of_proceeds_chars": len(secs.get("use_of_proceeds", "")),
        "summary_chars": len(secs.get("summary", "")),
    }
    sections = {"ipo_id": ipo_id, **{k: secs.get(k) for k in ("summary", "risk_factors", "use_of_proceeds")}}
    return facts, sections


def parse_file(path: Path) -> tuple[dict, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return parse(path.name.removesuffix(".txt.gz"), fh.read())


def main(full: bool, workers: int) -> None:
    files = sorted(TEXTS.glob("*.txt.gz"))
    OUT.mkdir(parents=True, exist_ok=True)
    facts_path, sec_path = OUT / "prospectus_facts.csv", OUT / "prospectus_sections.jsonl.gz"
    # keep earlier results unless asked to re-parse everything (do that after changing the parser)
    old = pd.read_csv(facts_path) if facts_path.exists() and sec_path.exists() and not full else pd.DataFrame()
    done = set(old["ipo_id"]) if len(old) else set()
    todo = [f for f in files if f.name.removesuffix(".txt.gz") not in done]
    print(f"{len(todo)} to parse ({len(done)} already parsed)", flush=True)
    facts = []
    if not todo and done:
        return
    # appending adds a gzip member; gzip readers handle multi-member files transparently
    with gzip.open(sec_path, "at" if done else "wt", encoding="utf-8") as sec_out:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(parse_file, todo, chunksize=4)
            for i, (fa, se) in enumerate(results, 1):
                facts.append(fa)
                sec_out.write(json.dumps(se) + "\n")
                if i % 500 == 0:
                    print(f"{i}/{len(todo)}", flush=True)
    df = pd.concat([old, pd.DataFrame(facts)], ignore_index=True)
    df.to_csv(facts_path, index=False)
    n = len(df)
    print(f"parsed {n}")
    for c in ["doc_offer_price", "doc_underwriters", "doc_shares"]:
        print(f"  {c}: {df[c].notna().mean():.1%}")
    for c in ["risk_factors_chars", "use_of_proceeds_chars", "summary_chars"]:
        print(f"  {c} > 0: {(df[c] > 0).mean():.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="re-parse every prospectus")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    main(a.full, a.workers)
