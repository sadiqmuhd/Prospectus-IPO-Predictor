"""Live company search on SEC EDGAR, with each company's IPO status from its filing history.

Any company that files with the SEC can be found, not only the IPOs in the training data. Status:
    priced        a final IPO prospectus (424B4/424B1) followed the registration statement
    registration  an S-1/F-1 is on file but the IPO has not priced yet (an upcoming IPO)
    withdrawn     the registration was withdrawn (form RW), or has been inactive for over a year
    public        already listed, but there is no IPO prospectus on EDGAR (IPO before EDGAR, direct listing, SPAC)
    none          no IPO filings at all
Blank-check companies (SPACs) and trusts are flagged and not analysed: the models were trained without them.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from urllib.parse import quote

import pandas as pd

from edgar import get  # noqa: E402 (src/data is on sys.path, set up by analyze.py)

REG_FORMS = ("S-1", "S-1/A", "F-1", "F-1/A", "SB-2", "SB-2/A")
FINAL_FORMS = ("424B4", "424B1")
WITHDRAWN_FORMS = ("RW",)
STALE_AFTER = timedelta(days=365)
MAX_RESULTS = 6
OLDER_PAGES = 2  # extra filing-history pages to read for companies with long histories

# SPACs and funds raise cash before choosing a business, so they price differently from operating companies
NOT_MODELLED = re.compile(r"\bAcquisition Corp|\bAcquisition Co\b|\bSPAC\b|\bBlank Check|\bTrust\b|\bETF\b|\bFund\b", re.I)
STATUS_LABELS = {"priced": "IPO priced", "registration": "Upcoming IPO", "withdrawn": "IPO withdrawn",
                 "public": "Already public", "none": "No IPO filing"}


def doc_url(cik: int, accession: str, document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{document}"


def filings(cik: int) -> tuple[pd.DataFrame, list[str], str]:
    """(filing history, tickers, company name), cached for the day: an upcoming IPO's status can change."""
    return _filings(cik, date.today())


@lru_cache(maxsize=512)
def _filings(cik: int, today: date) -> tuple[pd.DataFrame, list[str], str]:
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").content)
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    has_reg = frames[0]["form"].isin(REG_FORMS + FINAL_FORMS).any() if len(frames[0]) else False
    if not has_reg:
        for page in sub["filings"].get("files", [])[:OLDER_PAGES]:
            frames.append(pd.DataFrame(json.loads(get(f"https://data.sec.gov/submissions/{page['name']}").content)))
    df = pd.concat(frames, ignore_index=True)
    if len(df):
        df = df[["form", "filingDate", "accessionNumber", "primaryDocument"]].sort_values("filingDate")
    return df, list(sub.get("tickers") or []), sub.get("name") or ""


def ipo_status(cik: int) -> dict:
    """Where the company is in the IPO process, and which document the bot should read."""
    df, tickers, name = filings(cik)
    out = {"cik": cik, "name": name, "tickers": tickers}
    if df.empty:
        return {**out, "status": "public" if tickers else "none"}
    reg = df[df["form"].isin(REG_FORMS)]
    final = df[df["form"].isin(FINAL_FORMS)]
    if len(reg):
        first_reg = reg.iloc[0]
        # follow-on offerings also file 424B4s; the IPO's is the first one after the first registration statement
        ipo_final = final[final["filingDate"] >= first_reg["filingDate"]]
        if len(ipo_final):
            f = ipo_final.iloc[0]
            return {**out, "status": "priced", "date": f["filingDate"], "form": f["form"],
                    "filed": first_reg["filingDate"],
                    "document": doc_url(cik, f["accessionNumber"], f["primaryDocument"])}
        last = reg.iloc[-1]
        # a company can withdraw and later refile: only a withdrawal after its latest filing ends the IPO
        withdrawn = df[df["form"].isin(WITHDRAWN_FORMS) & (df["filingDate"] >= last["filingDate"])]
        stale = pd.Timestamp(last["filingDate"]) < pd.Timestamp(date.today() - STALE_AFTER)
        status = "withdrawn" if len(withdrawn) or stale else "registration"
        return {**out, "status": status, "date": last["filingDate"], "form": last["form"],
                "filed": first_reg["filingDate"], "amendments": int((reg["form"].str.endswith("/A")).sum()),
                "document": doc_url(cik, last["accessionNumber"], last["primaryDocument"])}
    if len(final):  # a final prospectus with no registration statement on EDGAR (very old filings)
        f = final.iloc[0]
        return {**out, "status": "priced", "date": f["filingDate"], "form": f["form"],
                "document": doc_url(cik, f["accessionNumber"], f["primaryDocument"])}
    return {**out, "status": "public" if tickers else "none"}


def describe(s: dict) -> str:
    """One plain line under the company name in the search results."""
    st = s["status"]
    if st == "priced":
        return f"IPO priced on {s['date']}. The bot reads the final prospectus."
    if st == "registration":
        n = s.get("amendments", 0)
        amended = f", amended {n} time{'s' if n != 1 else ''}" if n else ""
        return (f"Filed to go public on {s['filed']}{amended}; not priced yet. "
                "The bot predicts from the latest filing.")
    if st == "withdrawn":
        return f"Filed to go public on {s['filed']}, but the IPO was withdrawn or stalled (last filing {s['date']})."
    if st == "public":
        return "Already public, but its IPO prospectus is not on EDGAR (listed before EDGAR, or not via an IPO)."
    return "No IPO filings on EDGAR."


def search(query: str) -> list[dict]:
    query = query.strip()
    if len(query) < 2:
        return []
    r = get(f"https://efts.sec.gov/LATEST/search-index?keysTyped={quote(query)}")
    hits = r.json().get("hits", {}).get("hits", [])[:MAX_RESULTS]
    ciks = [int(h["_id"]) for h in hits if str(h.get("_id", "")).isdigit()]

    def one(cik: int) -> dict | None:
        try:
            s = ipo_status(cik)
        except Exception:  # one company's history failing should not sink the whole search
            return None
        name = re.sub(r"\s*\([A-Z0-9., ]+\)$", "", s["name"]) or str(cik)
        out = {**s, "name": name, "status_label": STATUS_LABELS[s["status"]], "description": describe(s),
               "analyzable": s["status"] in ("priced", "registration", "withdrawn")}
        if out["analyzable"] and NOT_MODELLED.search(name):
            out.update(analyzable=False, description=out["description"].split(".")[0] + ". This is a blank-check "
                       "company (SPAC) or fund; the models only cover operating companies.")
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [r for r in pool.map(one, ciks) if r]
    # people and investment vehicles that share a name have no filings worth showing; keep one so a private
    # company ("Stripe, Inc.: no IPO filing") still gets an answer
    shown = [r for r in results if r["status"] != "none"] or results[:1]
    # companies the bot can analyse first, in EDGAR's relevance order otherwise
    return sorted(shown, key=lambda r: not r["analyzable"])
