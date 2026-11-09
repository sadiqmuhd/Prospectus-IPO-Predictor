"""First trading sessions of NGX listings, from the exchange's Daily Official List (Equities) PDFs.

NGX keeps these PDFs at a fixed address back to late 2014. For each listing with a known date,
the lists for the listing day and the following weeks are downloaded, and the stock's row is read:
official close (the "Current Market Price" column) and whether it traded that day.

Outputs: data/raw/nigeria/ngx_daily_list/<YYYY-MM-DD>.pdf       cached PDFs
         data/raw/nigeria/ngx_first_sessions.csv               one row per listing per session
"""
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pdfplumber
import requests

LISTINGS = Path("data/raw/nigeria/ngx_listings.csv")
CACHE = Path("data/raw/nigeria/ngx_daily_list")
OUT = Path("data/raw/nigeria/ngx_first_sessions.csv")
URL = "https://doclib.ngxgroup.com/DownloadsContent/Daily%20Official%20List%20-%20Equities%20for%20{d}.pdf"
FIRST_ARCHIVED = date(2014, 11, 1)
SESSIONS = 10
MAX_CALENDAR_DAYS = 25

NUM = re.compile(r"^-?[\d,]+(\.\d+)?$")
DATE = re.compile(r"^\d{2}/\d{2}/\d{2,4}$")


def daily_list(day: date) -> Path | None:
    """Cached PDF for a trading day, or None when NGX has no list (weekend, holiday, not archived)."""
    path = CACHE / f"{day.isoformat()}.pdf"
    if path.exists():
        return path if path.stat().st_size > 0 else None
    for attempt in range(5):
        try:
            r = requests.get(URL.format(d=day.strftime("%d-%m-%Y")), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))  # NGX drops connections now and then; never cache a network error
    else:
        raise RuntimeError(f"NGX daily list unreachable for {day}")
    time.sleep(1.0)
    CACHE.mkdir(parents=True, exist_ok=True)
    ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf")
    path.write_bytes(r.content if ok else b"")  # an empty file marks "no list for this day"
    return path if ok else None


def list_date(pdf_path: Path) -> date | None:
    """Trading date printed in the list's title. NGX's archive has some files saved under the wrong day."""
    with pdfplumber.open(pdf_path) as pdf:
        m = re.search(r"For (\d{2})/(\d{2})/(\d{4})", pdf.pages[0].extract_text() or "")
    return date(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def column_anchors(words: list[dict]) -> dict[str, float] | None:
    """x positions of the value columns, read from the page's own header line."""
    sym = next((w for w in words if w["text"] == "Symbol"), None)
    if sym is None:
        return None
    hdr = sorted((w for w in words if abs(w["top"] - sym["top"]) < 3), key=lambda w: w["x0"])
    prices = [w["x1"] for w in hdr if w["text"] == "Price"]
    by = {w["text"]: w["x1"] for w in hdr}
    if len(prices) < 3:
        return None
    after_trade = [w for w in hdr if w["x0"] > prices[2]]
    # lists before ~2017 have no Official Open/Close columns, only the market price
    return {
        "quotation": prices[0], "open": by.get("Open"), "official_close": by.get("Close"),
        "close": prices[1], "trade_price": prices[2],
        "trade_date": next((w["x1"] for w in after_trade if w["text"] == "Date"), None),
        "qty": next((w["x1"] for w in after_trade if w["text"] == "Qty"), None),
    }


def read_row(pdf_path: Path, ticker: str) -> dict | None:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            hit = next((w for w in words if w["text"] == ticker and w["x0"] < 80), None)
            if hit is None:
                continue
            anchors = {k: v for k, v in (column_anchors(words) or {}).items() if v is not None}
            if not anchors:
                return None
            # the row runs from this symbol down to the next symbol in the first column
            below = [w["top"] for w in words if w["x0"] < 80 and w["top"] > hit["top"] + 2
                     and re.fullmatch(r"[A-Z0-9]{2,}", w["text"])]
            bottom = min(below) - 2 if below else hit["top"] + 40
            cells = [w for w in words if hit["top"] - 3 <= w["top"] < bottom and w["x0"] > 180]
            row = {}
            for w in cells:
                if not (NUM.match(w["text"]) or DATE.match(w["text"])):
                    continue
                col = min(anchors, key=lambda k: abs(anchors[k] - w["x1"]))
                if abs(anchors[col] - w["x1"]) < 30:
                    row.setdefault(col, w["text"])
            return row
    return None


def to_float(s: str | None) -> float | None:
    return float(s.replace(",", "")) if s else None


def main() -> None:
    listings = pd.read_csv(LISTINGS, dtype=str)
    listings = listings[listings["listing_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
                        & listings["ticker"].notna()]
    rows = []
    for l in listings.itertuples():
        start = date.fromisoformat(l.listing_date)
        if start < FIRST_ARCHIVED:
            continue
        session = 0
        for k in range(MAX_CALENDAR_DAYS):
            day = start + timedelta(days=k)
            if day.weekday() >= 5:
                continue
            pdf_path = daily_list(day)
            if pdf_path is None or list_date(pdf_path) != day:
                continue
            row = read_row(pdf_path, l.ticker)
            if row is None:
                if session == 0:
                    continue  # not on the list yet (listing date can be off by a day)
                break
            session += 1
            traded_on = row.get("trade_date")
            rows.append({
                "ticker": l.ticker, "listing_date": l.listing_date, "session": session, "date": day.isoformat(),
                "close": to_float(row.get("close")), "official_open": to_float(row.get("open")),
                "official_close": to_float(row.get("official_close")),
                "traded": traded_on is not None and traded_on[:5] == day.strftime("%d/%m"),
                "qty": to_float(row.get("qty")) if traded_on else None,
            })
            if session >= SESSIONS:
                break
        print(f"{l.ticker}: {session} sessions", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"saved {len(out)} session rows -> {OUT}")


if __name__ == "__main__":
    main()
