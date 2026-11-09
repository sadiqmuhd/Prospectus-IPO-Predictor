"""Link each IPO in the clean dataset to its final prospectus (424B4/424B1) on EDGAR.

A filing matches when it is dated within the window around the first trade date and
its company name is close to the IPO's name. The IPO's registration filings
(S-1, F-1, SB-2 and amendments) are listed too, for reading the filing price range.

Outputs: data/interim/prospectus_matches.csv   one row per matched IPO
         data/interim/registration_filings.csv  S-1/F-1 filings of matched CIKs before the IPO
"""
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from build_us_dataset import norm_name

IPOS = Path("data/processed/us_ipos_clean.csv")
INDEX = Path("data/raw/edgar/filings_index.csv")
OUT = Path("data/interim")
DAYS_BEFORE, DAYS_AFTER = 7, 10
MIN_SCORE = 0.8
FINAL_FORMS = ["424B4", "424B1"]
REG_FORMS = ["S-1", "S-1/A", "F-1", "F-1/A", "SB-2", "SB-2/A"]


def name_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # "alon usa partners" vs "alon usa partners lp" after normalization -> containment
    if a.startswith(b + " ") or b.startswith(a + " "):
        return 0.95
    score = SequenceMatcher(None, a, b).ratio()
    # a fuzzy score alone pairs e.g. "avenue therapeutics" with "aileron therapeutics";
    # below 0.9 also require the same first letters or one squashed name inside the other
    sa, sb = a.replace(" ", ""), b.replace(" ", "")
    if score < 0.9 and sa[:4] != sb[:4] and sa not in sb and sb not in sa:
        return 0.0
    return score


def main() -> None:
    ipos = pd.read_csv(IPOS, parse_dates=["ipo_date"])
    idx = pd.read_csv(INDEX, parse_dates=["date_filed"])
    idx["_n"] = idx["company"].map(norm_name)
    final = idx[idx["form"].isin(FINAL_FORMS)]

    matches = []
    for ipo in ipos.itertuples():
        lo = ipo.ipo_date - pd.Timedelta(days=DAYS_BEFORE)
        hi = ipo.ipo_date + pd.Timedelta(days=DAYS_AFTER)
        cand = final[(final["date_filed"] >= lo) & (final["date_filed"] <= hi)]
        if cand.empty:
            continue
        n = norm_name(ipo.company)
        scores = cand["_n"].map(lambda c: name_score(n, c))
        best = scores.max()
        if best < MIN_SCORE:
            continue
        top = cand[scores == best].copy()
        # prefer 424B4, then the filing closest to the trade date
        top["_gap"] = (top["date_filed"] - ipo.ipo_date).dt.days.abs()
        top["_form_rank"] = top["form"].map({"424B4": 0, "424B1": 1})
        row = top.sort_values(["_form_rank", "_gap"]).iloc[0]
        matches.append({"ipo_id": ipo.ipo_id, "cik": row["cik"], "edgar_name": row["company"],
                        "form": row["form"], "date_filed": row["date_filed"].date(),
                        "path": row["path"], "match_score": round(best, 3)})

    m = pd.DataFrame(matches)
    # a filing claimed by two IPO rows (duplicate listings) stays with the better name match
    m = m.sort_values("match_score", ascending=False).drop_duplicates("path").sort_index()
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "prospectus_matches.csv", index=False)

    # registration filings for the matched CIKs, up to 3 years before the IPO
    reg = idx[idx["form"].isin(REG_FORMS) & idx["cik"].isin(m["cik"])]
    reg = reg.merge(m[["ipo_id", "cik"]], on="cik").merge(ipos[["ipo_id", "ipo_date"]], on="ipo_id")
    reg = reg[(reg["date_filed"] <= reg["ipo_date"])
              & (reg["date_filed"] >= reg["ipo_date"] - pd.Timedelta(days=3 * 365))]
    reg = reg.drop(columns=["_n", "ipo_date"]).sort_values(["ipo_id", "date_filed"])
    reg.to_csv(OUT / "registration_filings.csv", index=False)

    rate = len(m) / len(ipos)
    print(f"matched {len(m)}/{len(ipos)} IPOs ({rate:.1%}); {reg['ipo_id'].nunique()} with registration filings")
    yr = ipos.assign(matched=ipos["ipo_id"].isin(m["ipo_id"])).groupby("ipo_year")["matched"].mean()
    print(yr.round(2).to_string())


if __name__ == "__main__":
    main()
