"""First-day open/close for IPOs not covered by the IPOScoop file, from Yahoo Finance.

Yahoo prices are split-adjusted, so each close is multiplied back by the splits that
happened after the IPO. A ticker is accepted only if its first Yahoo bar falls within a
few days of the priced date; otherwise the symbol was probably reused or renamed.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

IN = Path("data/raw/us/nasdaq_priced.csv")
OUT = Path("data/raw/us/yahoo_first_day.csv")
MAX_LAG_DAYS = 7
COLUMNS = ["deal_id", "ticker", "status", "first_trade_date", "open_1", "close_1",
           "volume_1", "split_factor", "yahoo_first_date"]


def first_day(ticker: str, priced: pd.Timestamp) -> dict:
    hist = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=True)
    if hist.empty:
        return {"status": "no_data"}
    hist.index = hist.index.tz_localize(None).normalize()
    first_bar = hist.index[0]
    lag = (first_bar - priced).days
    if lag < 0 or lag > MAX_LAG_DAYS:
        return {"status": "date_mismatch", "yahoo_first_date": first_bar.date()}

    splits = hist["Stock Splits"]
    later = splits[(splits.index > first_bar) & (splits > 0)]
    factor = float(later.prod()) if len(later) else 1.0
    row = hist.iloc[0]
    return {
        "status": "ok",
        "first_trade_date": first_bar.date(),
        "open_1": row["Open"] * factor,
        "close_1": row["Close"] * factor,
        "volume_1": row["Volume"] / factor,
        "split_factor": factor,
    }


def is_spac_or_unit(row) -> bool:
    name = str(row["company"]).lower()
    t = str(row["ticker"])
    return ("acquisition" in name or "capital corp" in name and row["offer_price"] == 10
            or t.endswith(("U", ".U", "W")) and row["offer_price"] == 10)


def main(since: str, limit: int | None) -> None:
    deals = pd.read_csv(IN, parse_dates=["priced_date"])
    deals = deals[(deals["priced_date"] >= since) & deals["ticker"].notna()]
    deals = deals[~deals.apply(is_spac_or_unit, axis=1)]

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT)["deal_id"])
    todo = deals[~deals["deal_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(todo)} deals to fetch ({len(done)} cached)")

    out = []
    for i, d in enumerate(todo.itertuples(), 1):
        for attempt in range(3):
            try:
                res = first_day(d.ticker, d.priced_date)
                break
            except Exception as e:  # yfinance raises many types on rate limits
                res = {"status": f"error: {type(e).__name__}"}
                time.sleep(5 * (attempt + 1))
        out.append({"deal_id": d.deal_id, "ticker": d.ticker, **res})
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(out, columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            out = []
            print(f"{i}/{len(todo)}")
        time.sleep(0.3)

    res = pd.read_csv(OUT)
    print(res["status"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2020-01-01")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    main(a.since, a.limit)
