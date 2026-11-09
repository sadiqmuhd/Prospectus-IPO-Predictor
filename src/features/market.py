"""Market conditions before each IPO, using only information available the day before trading starts.

- vix_prev: VIX close on the last trading day before the IPO
- sp500_ret_30d / nasdaq_ret_30d: index return over the 21 trading days to that day
- ipo_count_30d: IPOs (all deals, not just the clean sample) in the previous 30 calendar days
- prior_ipo_return_30d: mean first-day return of those earlier IPOs, a "hot market" measure. Their returns were
  public before this IPO priced, so this does not leak.

`market_features(day, prices, ipos)` is also used by the web API.

Outputs: data/raw/market/<symbol>.csv   cached daily closes (refreshed with --refresh)
         data/interim/market.csv        one row per ipo_id
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/raw/market"
IPOS = ROOT / "data/processed/us_ipos.csv"
OUT = ROOT / "data/interim/market.csv"
SYMBOLS = {"vix": "^VIX", "sp500": "^GSPC", "nasdaq": "^IXIC"}
LOOKBACK_SESSIONS = 21
WINDOW_DAYS = 30


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Daily closes, one column per index, from 1999 so 2000 IPOs have a full lookback."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, sym in SYMBOLS.items():
        path = CACHE / f"{name}.csv"
        if refresh or not path.exists():
            h = yf.Ticker(sym).history(start="1999-01-01", auto_adjust=False)["Close"]
            h.index = h.index.tz_localize(None).normalize()
            h.rename("close").to_csv(path, index_label="date")
        s = pd.read_csv(path, parse_dates=["date"], index_col="date")["close"]
        cols[name] = s
    return pd.DataFrame(cols).sort_index()


def market_features(day: pd.Timestamp, prices: pd.DataFrame, ipos: pd.DataFrame | None = None) -> dict:
    before = prices[prices.index < day]
    out = dict.fromkeys(["vix_prev", "sp500_ret_30d", "nasdaq_ret_30d", "ipo_count_30d", "prior_ipo_return_30d"], np.nan)
    if len(before) > LOOKBACK_SESSIONS:
        last, base = before.iloc[-1], before.iloc[-1 - LOOKBACK_SESSIONS]
        out["vix_prev"] = last["vix"]
        out["sp500_ret_30d"] = last["sp500"] / base["sp500"] - 1
        out["nasdaq_ret_30d"] = last["nasdaq"] / base["nasdaq"] - 1
    if ipos is not None:
        w = ipos[(ipos["ipo_date"] < day) & (ipos["ipo_date"] >= day - pd.Timedelta(days=WINDOW_DAYS))]
        out["ipo_count_30d"] = len(w)
        out["prior_ipo_return_30d"] = w["first_day_return"].mean() if w["first_day_return"].notna().any() else np.nan
    return out


def main(refresh: bool) -> None:
    prices = load_prices(refresh)
    ipos = pd.read_csv(IPOS, parse_dates=["ipo_date"],
                       usecols=["ipo_id", "ipo_date", "first_day_return", "px_unreliable"])
    # reverse-split junk prices (e.g. +1667%) would swamp a month's average
    ipos["first_day_return"] = ipos["first_day_return"].mask(ipos["px_unreliable"].astype(bool))
    rows = [{"ipo_id": r.ipo_id, **market_features(r.ipo_date, prices, ipos)} for r in ipos.itertuples()]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows -> {OUT}; index data to {prices.index.max().date()}")
    print(df.describe().T[["count", "mean", "50%"]].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="re-download index prices")
    main(p.parse_args().refresh)
