"""Download priced US IPOs from the Nasdaq IPO calendar API, one request per month."""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.nasdaq.com/api/ipo/calendar"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
OUT = Path("data/raw/us/nasdaq_priced.csv")


def fetch_month(month: str) -> list[dict]:
    for attempt in range(3):
        try:
            r = requests.get(URL, params={"date": month}, headers=HEADERS, timeout=30)
            r.raise_for_status()
            priced = (r.json().get("data") or {}).get("priced") or {}
            return priced.get("rows") or []
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    print(f"  failed: {month}")
    return []


def main(start: str, end: str) -> None:
    rows = []
    for m in pd.period_range(start, end, freq="M"):
        month_rows = fetch_month(str(m))
        rows.extend(month_rows)
        print(f"{m}: {len(month_rows)}")
        time.sleep(0.5)

    df = pd.DataFrame(rows).rename(columns={
        "dealID": "deal_id",
        "proposedTickerSymbol": "ticker",
        "companyName": "company",
        "proposedExchange": "exchange",
        "proposedSharePrice": "offer_price",
        "sharesOffered": "shares_offered",
        "pricedDate": "priced_date",
        "dollarValueOfSharesOffered": "proceeds",
    }).drop(columns=["dealStatus"], errors="ignore")
    df["offer_price"] = pd.to_numeric(df["offer_price"].str.replace(r"[$,]", "", regex=True), errors="coerce")
    df["shares_offered"] = pd.to_numeric(df["shares_offered"].str.replace(",", ""), errors="coerce")
    df["proceeds"] = pd.to_numeric(df["proceeds"].str.replace(r"[$,]", "", regex=True), errors="coerce")
    df["priced_date"] = pd.to_datetime(df["priced_date"], format="%m/%d/%Y", errors="coerce")
    df = df.drop_duplicates("deal_id")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} deals -> {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2000-01")
    p.add_argument("--end", default="2025-12")
    a = p.parse_args()
    main(a.start, a.end)
