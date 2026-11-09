"""Join the clean US sample with text and market features into one modelling table.

Only information public before the first trade is used. Left out on purpose:
- open/close prices, volume and IPOScoop's return: these are the outcome
- the IPOScoop rating: an outside analyst's forecast, not something read from the prospectus
- Kaggle fields: present only for firms still listed in 2018, so their presence would leak survival
- the IPO year: the test years lie outside the training years, so a year effect cannot carry over

`FEATURE_GROUPS` is shared with the models and the web API.

Output: data/processed/model_table.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data/processed/us_ipos_clean.csv"
LM = ROOT / "data/interim/text_lm.csv"
FINBERT = ROOT / "data/interim/text_finbert.csv"
MARKET = ROOT / "data/interim/market.csv"
OUT = ROOT / "data/processed/model_table.csv"

TARGETS = ["first_day_return", "pop", "pop10", "bucket"]
SECTIONS = ("doc", "risk_factors", "use_of_proceeds", "summary")
LM_CATS = ("negative", "positive", "uncertainty", "litigious", "strong_modal", "weak_modal", "constraining", "fog")

FEATURE_GROUPS = {
    "deal": ["log_proceeds", "log_offer_price", "price_revision", "above_range", "below_range", "range_width",
             "is_nasdaq"],
    "firm": ["log_firm_age", "vc", "dual_class", "internet", "rollup"],
    "underwriter": ["lead_uw_rank", "n_syndicate"],
    "market": ["vix_prev", "sp500_ret_30d", "nasdaq_ret_30d", "ipo_count_30d", "prior_ipo_return_30d"],
    "text_size": ["log_doc_chars", "log_risk_chars"],
    "text_lm": [f"lm_{s}_{c}" for s in SECTIONS for c in LM_CATS] + ["lm_doc_net_tone"],
    "text_finbert": [f"fb_{s}_{m}" for s in ("summary", "risk_factors", "use_of_proceeds")
                     for m in ("pos", "neg", "net")],
}
# Prospectus language drifts over the years (longer, more negative and uncertain), so tone is also measured
# relative to the median of IPOs in the 365 days before each one. Only earlier IPOs are used: no leakage.
RELATIVE_BASE = ["lm_doc_negative", "lm_doc_uncertainty", "lm_doc_weak_modal", "lm_doc_litigious",
                 "lm_doc_net_tone", "lm_doc_fog", "lm_risk_factors_negative", "lm_risk_factors_uncertainty",
                 "lm_risk_factors_weak_modal", "fb_risk_factors_net", "fb_summary_net", "fb_use_of_proceeds_net",
                 "log_doc_chars", "log_risk_chars"]
RELATIVE_WINDOW_DAYS = 365
FEATURE_GROUPS["text_relative"] = [f"rel_{c}" for c in RELATIVE_BASE]
TEXT_GROUPS = ("text_size", "text_lm", "text_finbert", "text_relative")

FEATURE_LABELS = {
    "log_proceeds": "Deal size", "log_offer_price": "Offer price", "price_revision": "Price vs. filing range",
    "above_range": "Priced above range", "below_range": "Priced below range", "range_width": "Filing range width",
    "is_nasdaq": "Listed on Nasdaq", "log_firm_age": "Firm age", "vc": "VC-backed", "dual_class": "Dual-class shares",
    "internet": "Internet firm", "rollup": "Roll-up", "lead_uw_rank": "Lead underwriter prestige",
    "n_syndicate": "Syndicate size", "vix_prev": "VIX before the IPO", "sp500_ret_30d": "S&P 500, past month",
    "nasdaq_ret_30d": "Nasdaq, past month", "ipo_count_30d": "IPOs in the past month",
    "prior_ipo_return_30d": "Recent IPOs' first-day returns", "log_doc_chars": "Prospectus length",
    "log_risk_chars": "Risk Factors length",
}


def label(feature: str) -> str:
    if feature.startswith("rel_"):
        return label(feature[4:]) + " vs. recent IPOs"
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    m = re.match(r"(lm|fb)_(doc|summary|risk_factors|use_of_proceeds)_(\w+)", feature)
    if m:
        src = "LM" if m.group(1) == "lm" else "FinBERT"
        sec = {"doc": "whole prospectus", "summary": "Summary", "risk_factors": "Risk Factors",
               "use_of_proceeds": "Use of Proceeds"}[m.group(2)]
        return f"{src} {m.group(3).replace('_', ' ')} ({sec})"
    return feature


def features(groups=None) -> list[str]:
    groups = groups or FEATURE_GROUPS
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Columns computed from raw deal fields; used for the training table and for a single API request."""
    df = df.copy()
    df["log_offer_price"] = np.log(df["offer_price"].where(df["offer_price"] > 0))
    df["log_firm_age"] = np.log1p(df["firm_age"].clip(lower=0))
    if "exchange" in df:
        df["is_nasdaq"] = df["exchange"].str.contains("nasdaq", case=False, na=False).astype(float)
        df.loc[df["exchange"].isna(), "is_nasdaq"] = np.nan
    for c in ("above_range", "below_range"):
        if c in df:
            df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return df


def add_relative(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """rel_<f> = f minus the median of f over `history` IPOs in the RELATIVE_WINDOW_DAYS before each row's date."""
    df = df.copy()
    hist = history.dropna(subset=["ipo_date"]).sort_values("ipo_date")
    dates = hist["ipo_date"].to_numpy()
    for c in RELATIVE_BASE:
        if c not in df:
            df[f"rel_{c}"] = np.nan
            continue
        vals = hist[c].to_numpy() if c in hist else np.full(len(hist), np.nan)
        out = np.full(len(df), np.nan)
        for i, (d, x) in enumerate(zip(df["ipo_date"].to_numpy(), df[c].to_numpy())):
            lo = np.searchsorted(dates, d - np.timedelta64(RELATIVE_WINDOW_DAYS, "D"), side="left")
            hi = np.searchsorted(dates, d, side="left")  # strictly before this IPO
            window = vals[lo:hi]
            window = window[~np.isnan(window)]
            if len(window) >= 20 and not np.isnan(x):
                out[i] = x - np.median(window)
        df[f"rel_{c}"] = out
    return df


def main() -> None:
    df = derive(pd.read_csv(CLEAN, parse_dates=["ipo_date"], low_memory=False))
    for path in (LM, FINBERT, MARKET):
        if path.exists():
            df = df.merge(pd.read_csv(path).drop_duplicates("ipo_id", keep="last"), on="ipo_id", how="left")
        else:
            print(f"warning: {path.relative_to(ROOT)} missing; its features will be empty")
    for c in RELATIVE_BASE:
        if c not in df:
            df[c] = np.nan
    df = add_relative(df, df)
    cols = features()
    for c in cols:
        if c not in df:
            df[c] = np.nan
    keep = ["ipo_id", "ipo_date", "ipo_year", "company", "ticker", "offer_price", *TARGETS, *cols]
    keep += [c for c in RELATIVE_BASE if c not in keep]
    out = df[keep]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"saved {len(out)} rows, {len(cols)} features -> {OUT.relative_to(ROOT)}")
    cover = {g: out[FEATURE_GROUPS[g]].notna().any(axis=1).mean() for g in FEATURE_GROUPS}
    print("rows with any feature in group: " + ", ".join(f"{g} {v:.0%}" for g, v in cover.items()))


if __name__ == "__main__":
    main()
