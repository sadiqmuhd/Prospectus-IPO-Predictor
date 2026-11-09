"""Assemble the Hugging Face Space for the API: only the code, models and small data files the server needs.

    .venv/Scripts/python deploy/build_space.py      # writes deploy/space/

Then push deploy/space/ to a new Space (SDK: Docker). It is a separate git repository; see deploy/README.md.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "space"

FILES = [
    "api/__init__.py", "api/app/__init__.py", "api/app/analyze.py", "api/app/main.py", "api/app/examples.json",
    "api/requirements.txt", "api/requirements-finbert.txt", "api/Dockerfile",
    "src/data/build_us_dataset.py", "src/data/edgar.py", "src/data/fetch_price_ranges.py",
    "src/data/parse_prospectuses.py",
    "src/features/build_features.py", "src/features/market.py", "src/features/text_lm.py",
    "src/features/text_finbert.py", "src/models/explain.py", "src/reports/style.py",
    "models/pop.joblib", "models/pop10.joblib", "models/bucket.joblib",
    "results/metrics.csv", "results/text_effect.csv",
    "data/raw/us/Underwriter-Rank.xls", "data/raw/lm/Loughran-McDonald_MasterDictionary_1993-2025.csv",
    "data/raw/market/vix.csv", "data/raw/market/sp500.csv", "data/raw/market/nasdaq.csv",
    "data/processed/us_ipos.csv", "data/processed/model_table.csv",
]

SPACE_README = """---
title: Prospectus Bot
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

API for the IPO prospectus bot: predicts first-day returns from a prospectus.
Interactive docs at `/docs`. Research demo, not investment advice.
"""


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit("missing (run src/pipeline.py first):\n  " + "\n  ".join(missing))
    for f in FILES:
        dst = OUT / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)
    shutil.copy2(ROOT / "api/Dockerfile", OUT / "Dockerfile")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    # binary files over 10 MB need Git LFS on Hugging Face
    (OUT / ".gitattributes").write_text("*.joblib filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.xls filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.csv filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"space ready: {OUT.relative_to(ROOT)} ({len(FILES)} files, {size:.0f} MB)")


if __name__ == "__main__":
    main()
