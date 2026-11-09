"""Initial filing price range for each IPO, from its registration statements (S-1/F-1/SB-2 and amendments).

The original S-1 usually leaves the range blank ("between $   and $   "), so filings are read in date
order and the first one that states a range is kept. Price revision = offer price vs. that range.

Output: data/interim/price_ranges.csv (resumable: IPOs already in the file are skipped)
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from edgar import get_main_document, to_text

REG = Path("data/interim/registration_filings.csv")
OUT = Path("data/interim/price_ranges.csv")
MAX_FILINGS = 6          # stop looking after this many registration filings per IPO
MAX_BYTES = 400_000      # the range is on the cover page or in the summary
SEARCH_CHARS = 40_000

NUM = r"(?:US)?\$\s*(\d{1,4}(?:\.\d{1,2})?)"
RANGE_RES = [
    re.compile(rf"between\s+{NUM}\s+and\s+{NUM}", re.I),
    re.compile(rf"{NUM}\s*(?:-|–|—|to)\s*{NUM}\s+per\s+(?:share|ADS)", re.I),
]


def find_range(text: str) -> tuple[float, float] | None:
    for rx in RANGE_RES:
        for m in rx.finditer(text[:SEARCH_CHARS]):
            lo, hi = float(m.group(1)), float(m.group(2))
            context = text[max(0, m.start() - 300):m.end() + 100].lower()
            if 0.5 <= lo <= hi <= 3 * lo and ("offering price" in context or "per share" in context
                                               or "per ads" in context):
                return lo, hi
    return None


def ipo_range(ipo_id: str, filings: pd.DataFrame) -> dict:
    tried = 0
    for f in filings.head(MAX_FILINGS).itertuples():
        tried += 1
        try:
            text = to_text(get_main_document(f.path, max_bytes=MAX_BYTES))
        except Exception:
            continue
        rng = find_range(text)
        if rng:
            return {"ipo_id": ipo_id, "range_low": rng[0], "range_high": rng[1],
                    "range_form": f.form, "range_date": f.date_filed, "filings_tried": tried}
    return {"ipo_id": ipo_id, "range_low": None, "range_high": None,
            "range_form": None, "range_date": None, "filings_tried": tried}


def main(limit: int | None, workers: int) -> None:
    reg = pd.read_csv(REG).sort_values(["ipo_id", "date_filed"])
    done = set(pd.read_csv(OUT)["ipo_id"]) if OUT.exists() else set()
    groups = [(k, g) for k, g in reg.groupby("ipo_id") if k not in done]
    print(f"{len(groups)} IPOs to check ({len(done)} done)", flush=True)
    if limit:
        groups = groups[:limit]

    batch = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(lambda kg: ipo_range(*kg), groups), 1):
            batch.append(row)
            if i % 100 == 0 or i == len(groups):
                pd.DataFrame(batch).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                batch = []
                print(f"{i}/{len(groups)}", flush=True)

    res = pd.read_csv(OUT)
    print(f"range found for {res['range_low'].notna().mean():.1%} of {len(res)} IPOs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    main(a.limit, a.workers)
