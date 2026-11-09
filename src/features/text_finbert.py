"""FinBERT sentence tone for each prospectus's Summary, Risk Factors and Use of Proceeds.

Running FinBERT over whole prospectuses on a CPU would take days, so each section is scored on up to
SENTENCES_PER_SECTION sentences spread evenly through it (the same sentences on every run). Features per section:
mean positive / negative / neutral probability and net tone (positive - negative).

`FinBert().features(sections)` is also used by the web API.

Output: data/interim/text_finbert.csv (resumable: appends, skips ipo_ids already scored)
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ROOT / "data/interim/prospectus_sections.jsonl.gz"
OUT = ROOT / "data/interim/text_finbert.csv"
MODEL = "ProsusAI/finbert"
SECTION_NAMES = ("summary", "risk_factors", "use_of_proceeds")
COLUMNS = ["ipo_id"] + [f"fb_{s}_{x}" for s in SECTION_NAMES for x in ("n", "pos", "neg", "neu", "net")]
SENTENCES_PER_SECTION = 16
MAX_TOKENS = 128
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(text: str | None, k: int = SENTENCES_PER_SECTION) -> list[str]:
    """k prose sentences spread evenly through the text; table rows and headings are skipped."""
    if not text:
        return []
    out = []
    for s in SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text)):
        letters = sum(c.isalpha() for c in s)
        if 40 <= len(s) <= 600 and letters / len(s) > 0.7:
            out.append(s)
    if len(out) <= k:
        return out
    return [out[i] for i in np.linspace(0, len(out) - 1, k).round().astype(int)]


class FinBert:
    def __init__(self, threads: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.idx = [labels["positive"], labels["negative"], labels["neutral"]]

    def probs(self, texts: list[str], batch: int = 32) -> np.ndarray:
        """(n, 3) array of positive / negative / neutral probabilities."""
        if not texts:
            return np.zeros((0, 3))
        order = np.argsort([len(t) for t in texts])  # similar lengths per batch: less padding
        out = np.zeros((len(texts), 3))
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                ids = order[i:i + batch]
                enc = self.tok([texts[j] for j in ids], padding=True, truncation=True,
                               max_length=MAX_TOKENS, return_tensors="pt")
                p = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
                out[ids] = p[:, self.idx]
        return out

    def features(self, sections: dict[str, str | None]) -> dict[str, float]:
        sents = {s: sentences(sections.get(s)) for s in SECTION_NAMES}
        flat = [t for s in SECTION_NAMES for t in sents[s]]
        p, feats, at = self.probs(flat), {}, 0
        for s in SECTION_NAMES:
            n = len(sents[s])
            feats[f"fb_{s}_n"] = n
            if n:
                m = p[at:at + n].mean(axis=0)
                feats.update({f"fb_{s}_pos": m[0], f"fb_{s}_neg": m[1], f"fb_{s}_neu": m[2],
                              f"fb_{s}_net": m[0] - m[1]})
            at += n
        return feats


def main(threads: int, limit: int | None) -> None:
    done = set(pd.read_csv(OUT, usecols=["ipo_id"])["ipo_id"]) if OUT.exists() else set()
    with gzip.open(SECTIONS, "rt", encoding="utf-8") as f:
        todo = [r for r in map(json.loads, f) if r["ipo_id"] not in done]
    print(f"{len(todo)} to score ({len(done)} done)", flush=True)
    if limit:
        todo = todo[:limit]
    fb, batch = FinBert(threads), []
    for i, rec in enumerate(todo, 1):
        batch.append({"ipo_id": rec["ipo_id"], **fb.features(rec)})
        if i % 50 == 0 or i == len(todo):
            # fixed column order: a batch's first row may lack a section's scores
            pd.DataFrame(batch).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            batch = []
            print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    main(a.threads, a.limit)
