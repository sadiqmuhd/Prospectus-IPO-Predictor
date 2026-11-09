"""Loughran-McDonald tone and readability for each prospectus and its key sections.

For the whole document and for Summary, Risk Factors and Use of Proceeds: word count and the share of words in
each LM category (negative, positive, uncertainty, litigious, strong/weak modal, constraining), plus a Fog
readability index. Loughran & McDonald (2013) find that more uncertain S-1 language goes with larger price
revisions and higher first-day returns.

`lm_features(text, sections)` is also used by the web API for a single uploaded prospectus.

Output: data/interim/text_lm.csv, one row per ipo_id
"""
import argparse
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DICT = ROOT / "data/raw/lm/Loughran-McDonald_MasterDictionary_1993-2025.csv"
TEXTS = ROOT / "data/raw/edgar/prospectus_text"
SECTIONS = ROOT / "data/interim/prospectus_sections.jsonl.gz"
OUT = ROOT / "data/interim/text_lm.csv"

CATEGORIES = {"negative": "Negative", "positive": "Positive", "uncertainty": "Uncertainty",
              "litigious": "Litigious", "strong_modal": "Strong_Modal", "weak_modal": "Weak_Modal",
              "constraining": "Constraining"}
PARTS = ("doc", "summary", "risk_factors", "use_of_proceeds")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


@lru_cache(maxsize=1)
def lexicon() -> tuple[dict[str, frozenset], frozenset, frozenset]:
    """(category -> words, all dictionary words, words of 3+ syllables)."""
    d = pd.read_csv(DICT, keep_default_na=False, usecols=["Word", "Syllables", *CATEGORIES.values()])
    d["Word"] = d["Word"].str.upper()
    cats = {k: frozenset(d.loc[d[col] > 0, "Word"]) for k, col in CATEGORIES.items()}
    return cats, frozenset(d["Word"]), frozenset(d.loc[d["Syllables"] >= 3, "Word"])


def tone(text: str | None) -> dict[str, float]:
    if not text:
        return {"words": 0}
    cats, vocab, complex_words = lexicon()
    words = [w.upper() for w in WORD.findall(text)]
    # numbers, tickers and table debris are not words; LM count only tokens in their dictionary
    words = [w for w in words if w in vocab]
    n = len(words)
    if n == 0:
        return {"words": 0}
    out = {"words": n}
    for k, ws in cats.items():
        out[k] = sum(w in ws for w in words) / n
    sentences = max(len(SENTENCE_END.findall(text)), 1)
    out["fog"] = 0.4 * (n / sentences + 100 * sum(w in complex_words for w in words) / n)
    return out


def lm_features(text: str | None, sections: dict[str, str | None]) -> dict[str, float]:
    feats = {}
    for part in PARTS:
        src = text if part == "doc" else sections.get(part)
        for k, v in tone(src).items():
            feats[f"lm_{part}_{k}"] = v
    # net tone of the whole document, the usual single summary measure
    if feats.get("lm_doc_words"):
        feats["lm_doc_net_tone"] = feats["lm_doc_positive"] - feats["lm_doc_negative"]
    return feats


def score(record: dict) -> dict:
    path = TEXTS / f"{record['ipo_id']}.txt.gz"
    text = gzip.open(path, "rt", encoding="utf-8").read() if path.exists() else None
    return {"ipo_id": record["ipo_id"], **lm_features(text, record)}


def records():
    with gzip.open(SECTIONS, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main(workers: int) -> None:
    lexicon()  # fail early if the dictionary is missing
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(score, records(), chunksize=8), 1):
            rows.append(row)
            if i % 500 == 0:
                print(f"{i} scored", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("ipo_id", keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows -> {OUT}")
    cols = [c for c in df.columns if c.startswith("lm_doc_") or c.startswith("lm_risk_factors_")]
    print(df[cols].describe().T[["mean", "50%"]].to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    main(p.parse_args().workers)
