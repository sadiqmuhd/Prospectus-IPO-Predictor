"""Download each matched final prospectus and save it as gzipped plain text.

Output: data/raw/edgar/prospectus_text/<ipo_id>.txt.gz (resumable: existing files are skipped)

HTML-to-text conversion is CPU-bound, so workers are processes, not threads. Each process
gets an equal share of the SEC rate limit through SEC_MIN_INTERVAL.
"""
import argparse
import gzip
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

MATCHES = Path("data/interim/prospectus_matches.csv")
OUT = Path("data/raw/edgar/prospectus_text")
BASE_INTERVAL = 0.15  # seconds between requests for the whole run (~6-7 req/s)


def fetch(ipo_id: str, path: str) -> str | None:
    from edgar import get_main_document, to_text  # imported in the worker, after SEC_MIN_INTERVAL is set
    try:
        text = to_text(get_main_document(path))
        tmp = OUT / f"{ipo_id}.txt.gz.tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(OUT / f"{ipo_id}.txt.gz")  # a killed run never leaves a half-written file
        return None
    except Exception as e:  # keep going; a rerun retries whatever is missing
        return f"{ipo_id}: {type(e).__name__}: {e}"


def main(limit: int | None, workers: int) -> None:
    m = pd.read_csv(MATCHES)
    OUT.mkdir(parents=True, exist_ok=True)
    for tmp in OUT.glob("*.tmp"):
        tmp.unlink()
    todo = [r for r in m.itertuples() if not (OUT / f"{r.ipo_id}.txt.gz").exists()]
    print(f"{len(todo)} to download ({len(m) - len(todo)} already saved)", flush=True)
    if limit:
        todo = todo[:limit]

    os.environ["SEC_MIN_INTERVAL"] = str(BASE_INTERVAL * workers)
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(fetch, [r.ipo_id for r in todo], [r.path for r in todo], chunksize=4)
        for i, err in enumerate(results, 1):
            if err:
                failed += 1
                print(f"  failed {err}", flush=True)
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} ({failed} failed)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    main(a.limit, a.workers)
