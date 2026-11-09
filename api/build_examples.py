"""Analyse a few well-known IPOs in advance so the web app has demos that need no upload.

Uses the saved prospectus texts and filing ranges, so it makes no network calls. Mostly test-year IPOs
(2021-2025), so the shown predictions were not fitted on those outcomes when the models were evaluated. The
served models are refit on every year, which the page notes.

Output: api/app/examples.json
"""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.app.analyze import Document, analyze  # noqa: E402
from api.app.main import clean  # noqa: E402

TEXTS = ROOT / "data/raw/edgar/prospectus_text"
OUT = ROOT / "api/app/examples.json"
WANTED = ["ABNB", "SNOW", "RIVN", "HOOD", "BMBL", "CAVA", "CART", "BIRK", "RDDT", "ALAB", "CRCL", "KVYO"]


def main() -> None:
    ipos = pd.read_csv(ROOT / "data/processed/us_ipos_clean.csv", parse_dates=["ipo_date"], low_memory=False)
    ranges = pd.read_csv(ROOT / "data/interim/price_ranges.csv")
    ipos = ipos.merge(ranges[["ipo_id", "range_low", "range_high"]], on="ipo_id", how="left",
                      suffixes=("_ds", ""))
    # tickers get reused (SNOW was Intrawest in 2014), so take the latest IPO per ticker
    picked = ipos[ipos["ticker"].isin(WANTED)].sort_values("ipo_date").drop_duplicates("ticker", keep="last")
    out = {}
    for r in picked.itertuples():
        path = TEXTS / f"{r.ipo_id}.txt.gz"
        if not path.exists():
            print(f"skip {r.ticker}: no prospectus text")
            continue
        doc = Document(text=gzip.open(path, "rt", encoding="utf-8").read(), source=f"EDGAR prospectus ({r.ticker})")
        res = analyze(doc, {"ipo_date": r.ipo_date.date(), "offer_price": r.offer_price,
                            "shares_offered": r.shares_offered if pd.notna(r.shares_offered) else None,
                            "range_low": r.range_low if pd.notna(r.range_low) else None,
                            "range_high": r.range_high if pd.notna(r.range_high) else None})
        res.update(company=r.company, ticker=r.ticker, ipo_date=str(r.ipo_date.date()),
                   actual={"first_day_return": r.first_day_return, "pop": int(r.pop), "bucket": r.bucket})
        out[r.ticker.lower()] = clean(res)
        p = res["predictions"].get("pop", {}).get("probability")
        print(f"{r.ticker:5s} actual {r.first_day_return:+.1%}  P(pop) {p:.2f}" if p is not None else r.ticker)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {len(out)} examples -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
