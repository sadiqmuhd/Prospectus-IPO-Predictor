"""Download EDGAR quarterly form indexes and keep IPO-related filings.

Output: data/raw/edgar/filings_index.csv with form, company, cik, date_filed, path.
"""
from pathlib import Path

import pandas as pd

from edgar import get

FORMS = {"424B1", "424B4", "S-1", "S-1/A", "F-1", "F-1/A", "SB-2", "SB-2/A"}
OUT = Path("data/raw/edgar/filings_index.csv")


def parse_form_idx(text: str) -> list[dict]:
    rows, started = [], False
    for line in text.splitlines():
        if line.startswith("-----"):
            started = True
            continue
        if not started or not line.strip():
            continue
        # column widths drift from the header, so split from the right: ... cik date path
        form = line.split(maxsplit=1)[0]
        if form not in FORMS:
            continue
        head, cik, date, path = line.rstrip().rsplit(maxsplit=3)
        rows.append({"form": form, "company": head[len(form):].strip(), "cik": int(cik),
                     "date_filed": date, "path": path})
    return rows


def main() -> None:
    rows = []
    for year in range(2000, 2026):
        for q in range(1, 5):
            url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx"
            text = get(url).content.decode("latin-1")
            got = parse_form_idx(text)
            rows.extend(got)
            print(f"{year} Q{q}: {len(got)}", flush=True)
    df = pd.DataFrame(rows)
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} filings -> {OUT}")
    print(df["form"].value_counts().to_string())


if __name__ == "__main__":
    main()
