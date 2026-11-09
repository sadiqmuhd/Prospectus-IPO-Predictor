"""Nigerian listings with limit-adjusted returns, in the same target format as the US dataset.

Day-1 returns on NGX are capped by the daily price limit, so a stock that is worth +60% shows +10% on
day 1. The adjusted return runs to the close of the first session whose move is inside the limit
(capped at 10 sessions); if every known session hit the limit, the row is flagged as censored.

Output: data/processed/ngx_listings.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_us_dataset import BUCKETS, BUCKET_LABELS

RAW = Path("data/raw/nigeria")
OUT = Path("data/processed/ngx_listings.csv")
AT_LIMIT = 0.95  # NGX tick rounding stops a +10% move at e.g. +9.93%, so >=95% of the limit counts as a hit


def limit_for(day: pd.Timestamp, regimes: pd.DataFrame) -> float:
    for r in regimes.itertuples():
        if day >= r.start_date and (pd.isna(r.end_date) or day <= r.end_date):
            return r.daily_limit_pct / 100
    return np.nan


def adjusted(offer: float, closes: list[float], limit: float) -> tuple[float, int, int, bool]:
    """(return to first non-limit session, that session, sessions at limit, censored)."""
    prev, at_limit = offer, 0
    for k, c in enumerate(closes, 1):
        if abs(c / prev - 1) >= AT_LIMIT * limit:
            at_limit += 1
            prev = c
            continue
        return c / offer - 1, k, at_limit, False
    return closes[-1] / offer - 1, len(closes), at_limit, True


def main() -> None:
    lst = pd.read_csv(RAW / "ngx_listings.csv")
    regimes = pd.read_csv(RAW / "price_limit_regimes.csv", parse_dates=["start_date", "end_date"])
    sess = pd.read_csv(RAW / "ngx_first_sessions.csv")

    lst["listing_date_parsed"] = pd.to_datetime(lst["listing_date"], format="%Y-%m-%d", errors="coerce")
    lst["daily_limit"] = [limit_for(d, regimes) if pd.notna(d) else np.nan for d in lst["listing_date_parsed"]]

    # sessions from the NGX official list take precedence; news figures fill days 1-2 otherwise
    sess = sess.sort_values("session")
    closes = sess.groupby("ticker")["close"].apply(list)
    first_date = sess.groupby("ticker")["date"].first()
    rows = []
    for l in lst.itertuples():
        official = closes.get(l.ticker) if isinstance(l.ticker, str) else None
        news = [c for c in (l.day1_close_ngn, l.day2_close_ngn) if pd.notna(c)]
        if official and first_date[l.ticker] > l.listing_date and pd.notna(l.day1_close_ngn):
            # the archive lacks the listing-day list (e.g. a mislabelled file): put the news day 1 in front
            official = [l.day1_close_ngn] + official[:9]
        series, src = (official, "ngx_official_list") if official else (news, "news")
        rec = {"company": l.company, "price_source": src if series else None, "sessions_known": len(series)}
        if series and pd.notna(l.offer_price_ngn) and pd.notna(l.daily_limit):
            ret, k, n_lim, cens = adjusted(l.offer_price_ngn, series, l.daily_limit)
            rec.update(day1_return=series[0] / l.offer_price_ngn - 1, adjusted_return=ret,
                       adjusted_session=k, sessions_at_limit=n_lim, censored=cens)
        rows.append(rec)

    df = lst.merge(pd.DataFrame(rows), on="company", how="left")
    df["pop"] = (df["adjusted_return"] > 0).astype("Int64").where(df["adjusted_return"].notna())
    df["bucket"] = pd.cut(df["adjusted_return"], BUCKETS, labels=BUCKET_LABELS)
    df.loc[df["adjusted_return"] == 0, "bucket"] = "fair"
    df = df.drop(columns=["listing_date_parsed"]).sort_values(["listing_type", "listing_date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    show = df[df["adjusted_return"].notna()]
    print(f"{len(df)} listings, {len(show)} with returns")
    print(show[["company", "listing_type", "offer_price_ngn", "day1_return", "adjusted_return",
                "adjusted_session", "sessions_at_limit", "censored", "price_source"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
