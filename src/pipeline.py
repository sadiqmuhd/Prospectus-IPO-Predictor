"""Run the whole pipeline in order: data (US and Nigeria), features, models and figures.

    .venv/Scripts/python src/pipeline.py                  # bring everything up to date
    .venv/Scripts/python src/pipeline.py --list           # show the steps
    .venv/Scripts/python src/pipeline.py --only us_build  # run named steps only
    .venv/Scripts/python src/pipeline.py --from match     # start at a step
    .venv/Scripts/python src/pipeline.py --refresh        # re-download the source tables too
    .venv/Scripts/python src/pipeline.py --reparse        # re-parse all prospectuses (after a parser change)

Every step is safe to rerun. Source tables (Nasdaq calendar, EDGAR index, Ritter/IPOScoop files) are fetched
only when missing or with --refresh. The slow fetchers (Yahoo, prospectuses, price ranges, NGX lists) skip what
they already saved, so a rerun only does new work. Builds always run; they take seconds.

Each step's output goes to logs/<step>.log as well as the screen. The run stops at the first failing step.
"""
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
LOGS = ROOT / "logs"


@dataclass
class Step:
    name: str
    script: str
    what: str
    args: list[str] = field(default_factory=list)
    source: Path | None = None  # set for one-off downloads: skipped when this file exists, unless --refresh


STEPS = [
    Step("nasdaq", "data/fetch_nasdaq_calendar.py", "Nasdaq IPO calendar, 2000-2025 (~10 min)",
         source=ROOT / "data/raw/us/nasdaq_priced.csv"),
    Step("yahoo", "data/fetch_first_day_prices.py", "Yahoo first-day prices for post-2020 deals (resumable)"),
    Step("us_base", "data/build_us_dataset.py", "US dataset, so new IPOs can be matched to filings"),
    Step("edgar_index", "data/fetch_edgar_index.py", "EDGAR quarterly form indexes (~5 min)",
         source=ROOT / "data/raw/edgar/filings_index.csv"),
    Step("match", "data/match_prospectuses.py", "Match IPOs to final prospectuses and registration filings"),
    Step("prospectuses", "data/fetch_prospectuses.py", "Download final prospectuses (resumable)", ["--workers", "6"]),
    Step("parse", "data/parse_prospectuses.py", "Sections, underwriters and deal facts (new prospectuses only)"),
    Step("price_ranges", "data/fetch_price_ranges.py", "Initial filing price ranges (resumable)"),
    Step("us_build", "data/build_us_dataset.py", "Final US dataset with prospectus facts and price ranges"),
    Step("ngx_lists", "data/fetch_ngx_daily_lists.py", "NGX daily official lists for each listing (cached)"),
    Step("ngx_build", "data/build_ngx_dataset.py", "Nigerian listings with limit-adjusted returns"),
    Step("text_lm", "features/text_lm.py", "Loughran-McDonald tone and readability (~5 min)"),
    Step("text_finbert", "features/text_finbert.py", "FinBERT sentence tone (resumable; ~3 h the first time)"),
    Step("market", "features/market.py", "VIX, index returns and IPO activity before each IPO"),
    Step("features", "features/build_features.py", "Modelling table: clean sample + text + market features"),
    Step("eda", "reports/eda.py", "Exploratory figures and tables (reports/figures, results/eda_*)"),
    Step("train", "models/train.py", "Train, tune and test all models; save the chosen ones (~15 min)"),
    Step("explain", "models/explain.py", "SHAP importance and figures for the chosen models"),
    Step("model_figures", "reports/model_figures.py", "Model comparison and text-effect figures"),
    Step("examples", "../api/build_examples.py", "Demo IPOs for the web app (api/app/examples.json)"),
    Step("report", "reports/build_report.py", "Fill reports/report.md with the current results"),
]

# Files that come from a person's website rather than an API
DOWNLOADS = {
    "data/raw/us/IPO-age.xlsx": "https://site.warrington.ufl.edu/ritter/files/IPO-age.xlsx",
    "data/raw/us/Underwriter-Rank.xls": "https://site.warrington.ufl.edu/ritter/files/Underwriter-Rank.xls",
    "data/raw/us/SCOOP-Rating-Performance.xls":
        "https://www.iposcoop.com/wp-content/uploads/2013/11/SCOOP-Rating-Performance.xls",
}
# Files only a person can provide
MANUAL = {
    "data/raw/kaggle/IPODataFull.csv": "download 'Stocks IPO information & results' from Kaggle and save it here",
    "data/raw/us/ritter_table1_first_day_returns.csv": "transcribed from Ritter's IPO Statistics, Table 1",
    "data/raw/nigeria/ngx_listings.csv": "the hand-built list of NGX listings",
    "data/raw/nigeria/price_limit_regimes.csv": "NGX daily price limits by period",
    "data/raw/lm/Loughran-McDonald_MasterDictionary_1993-2025.csv":
        "Loughran-McDonald master dictionary CSV from sraf.nd.edu (Google Drive link on that page)",
}


def fetch_downloads(refresh: bool) -> None:
    for rel, url in DOWNLOADS.items():
        path = ROOT / rel
        if path.exists() and not refresh:
            continue
        print(f"downloading {rel}")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)


def check_manual() -> None:
    missing = [f"  {rel}: {how}" for rel, how in MANUAL.items() if not (ROOT / rel).exists()]
    if missing:
        sys.exit("missing input files:\n" + "\n".join(missing))


def run(step: Step, extra: list[str]) -> float:
    LOGS.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           # model-loading progress bars only clutter the logs
           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error", "TQDM_DISABLE": "1"}
    cmd = [sys.executable, "-W", "ignore", str(SRC / step.script), *step.args, *extra]
    start = time.time()
    with open(LOGS / f"{step.name}.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(f"  {line}", end="", flush=True)
            log.write(line)
        proc.wait()
    if proc.returncode:
        sys.exit(f"step '{step.name}' failed (exit {proc.returncode}); see logs/{step.name}.log")
    return time.time() - start


def main() -> None:
    # step output can hold any character; never let the console's code page (e.g. cp1256) stop a run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    names = [s.name for s in STEPS]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    p.add_argument("--only", help="comma-separated steps to run")
    p.add_argument("--from", dest="start", choices=names, help="first step to run")
    p.add_argument("--refresh", action="store_true", help="re-download source tables and Ritter/IPOScoop files")
    p.add_argument("--reparse", action="store_true", help="re-parse every prospectus")
    a = p.parse_args()

    if a.list:
        for s in STEPS:
            print(f"{s.name:<13} {s.what}")
        return
    steps = STEPS[names.index(a.start):] if a.start else STEPS
    if a.only:
        wanted = a.only.split(",")
        unknown = set(wanted) - set(names)
        if unknown:
            sys.exit(f"unknown steps: {', '.join(sorted(unknown))}; see --list")
        steps = [s for s in STEPS if s.name in wanted]

    fetch_downloads(a.refresh)
    check_manual()
    timings = []
    for s in steps:
        if s.source and s.source.exists() and not a.refresh:
            print(f"[{s.name}] skipped: {s.source.relative_to(ROOT)} exists (use --refresh to re-download)")
            continue
        print(f"[{s.name}] {s.what}", flush=True)
        extra = ["--full"] if s.name == "parse" and a.reparse else []
        if s.name == "text_finbert" and a.reparse:
            # scores are resumable per ipo_id, so stale ones from the old sections must go
            (ROOT / "data/interim/text_finbert.csv").unlink(missing_ok=True)
        timings.append((s.name, run(s, extra)))
    print("\ndone: " + ", ".join(f"{n} {t:.0f}s" for n, t in timings))


if __name__ == "__main__":
    main()
