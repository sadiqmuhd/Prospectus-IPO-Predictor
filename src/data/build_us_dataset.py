"""Merge IPOScoop, Nasdaq, Yahoo and Ritter files into one US IPO table with targets.

Output: data/processed/us_ipos.csv (every deal, with exclusion flags)
        data/processed/us_ipos_clean.csv (operating-company IPOs used for modelling)
"""
import re
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/us")
OUT = Path("data/processed")
MATCH_WINDOW_DAYS = 7
MIN_SPLIT_FACTOR = 0.1

BUCKETS = [-np.inf, 0.0, 0.10, 0.30, np.inf]
BUCKET_LABELS = ["overpriced", "fair", "underpriced", "heavily_underpriced"]

# Ritter rank columns and the IPO years each one covers
RANK_PERIODS = [
    ("Rank9200", 1992, 2000), ("Rank0104", 2001, 2004), ("Rank0507", 2005, 2007),
    ("Rank0809", 2008, 2009), ("Rank1011", 2010, 2011), ("Rank1217", 2012, 2017),
    ("Rank1820", 2018, 2020), ("Rank2122", 2021, 2022), ("Rank23", 2023, 2023),
    ("Rank24", 2024, 2024), ("Rank25", 2025, 2025),
]


def norm_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(inc|corp|corporation|co|ltd|limited|holdings?|group|plc|llc|lp|the|company|sa|nv|ag)\b", " ", s)
    return " ".join(s.split())


def norm_ticker(t) -> str:
    return re.sub(r"[^A-Z]", "", str(t).upper())


def load_scoop() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "SCOOP-Rating-Performance.xls", header=None, skiprows=36)
    raw = raw[raw[0].map(lambda v: hasattr(v, "year"))].iloc[:, :11]
    raw.columns = ["trade_date", "company", "ticker", "lead_managers", "offer_price",
                   "open_1", "close_1", "ret_scoop", "chg_open", "chg_close", "scoop_rating"]
    df = raw.drop(columns=["chg_open", "chg_close"])
    # one row has the year typed as 0120; Alon USA Partners listed in Nov 2012
    df["trade_date"] = [d.replace(year=2012) if d.year < 1000 else d for d in df["trade_date"]]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ["offer_price", "open_1", "close_1", "ret_scoop"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["offer_price"] > 0]  # warrant-only listings carry a 0 offer price
    df["company"] = df["company"].str.strip()
    df["is_unit_scoop"] = df["company"].str.contains(r"\(u\)", case=False, regex=True)
    df["source"] = "iposcoop"
    return df


def load_nasdaq_with_prices() -> pd.DataFrame:
    nas = pd.read_csv(RAW / "nasdaq_priced.csv", parse_dates=["priced_date"])
    px = pd.read_csv(RAW / "yahoo_first_day.csv", parse_dates=["first_trade_date"])
    px = px[px["status"] == "ok"].drop(columns=["ticker", "status"])
    return nas.merge(px, on="deal_id", how="inner")


def nearest_match(left: pd.DataFrame, right: pd.DataFrame, ldate: str, rdate: str, cols: list[str]) -> pd.DataFrame:
    """Attach right[cols] to left on normalized ticker, nearest date within the window."""
    l = left.assign(_t=left["ticker"].map(norm_ticker), _i=np.arange(len(left)))
    r = right.assign(_t=right["ticker"].map(norm_ticker))[["_t", rdate] + cols].dropna(subset=["_t"])
    m = l[["_i", "_t", ldate]].merge(r, on="_t")
    m["_gap"] = (m[ldate] - m[rdate]).dt.days.abs()
    m = m[m["_gap"] <= MATCH_WINDOW_DAYS].sort_values("_gap").drop_duplicates("_i")
    out = l.merge(m[["_i"] + cols], on="_i", how="left", suffixes=("", "_y"))
    return out.drop(columns=["_t", "_i"])


def load_ritter_age() -> pd.DataFrame:
    a = pd.read_excel(RAW / "IPO-age.xlsx").iloc[:, :12]
    a.columns = ["offer_date", "ritter_name", "ticker", "cusip", "adr_code", "vc", "dual_class",
                 "post_issue_shares", "internet", "crsp_permno", "founding_year", "rollup"]
    a["offer_date"] = pd.to_datetime(a["offer_date"].astype(str), format="%Y%m%d", errors="coerce")
    for c in ["adr_code", "vc", "dual_class", "internet", "founding_year", "rollup"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a.loc[a["founding_year"] < 1600, "founding_year"] = np.nan
    return a[a["offer_date"] >= "1999-06-01"]


def attach_ritter(df: pd.DataFrame) -> pd.DataFrame:
    age = load_ritter_age()
    cols = ["ritter_name", "adr_code", "vc", "dual_class", "internet", "founding_year", "rollup"]
    df = nearest_match(df, age, "ipo_date", "offer_date", cols)

    # Fallback: same-year exact normalized-name match for deals whose ticker changed
    miss = df["ritter_name"].isna()
    age = age.assign(_n=age["ritter_name"].map(norm_name), _y=age["offer_date"].dt.year)
    age = age.drop_duplicates(["_n", "_y"])
    key = df.loc[miss, ["company", "ipo_date"]].assign(
        _n=lambda x: x["company"].map(norm_name), _y=lambda x: x["ipo_date"].dt.year)
    filled = key.merge(age[["_n", "_y"] + cols], on=["_n", "_y"], how="left")
    filled.index = key.index
    df.loc[miss, cols] = filled[cols]
    return df


KAGGLE = Path("data/raw/kaggle/IPODataFull.csv")
# Kaggle "Stocks IPO information & results" (CC0). Only pre-IPO fields are kept: its prices are
# split-adjusted and sometimes start on day 2, and employees/CEO/revenue were scraped ~2018.
# The file only holds firms still listed in 2018, so never use "has Kaggle data" as a feature.
KAGGLE_COLS = {
    "Sector": "kg_sector",
    "Industry": "kg_industry",
    "YearFounded": "kg_year_founded",
    "USACompany": "kg_usa_company",
    "RevenueYearBeforeIPO": "kg_revenue_pre",
    "Net_incomeYearBeforeIPO": "kg_net_income_pre",
    "Total_assetsYearBeforeIPO": "kg_total_assets_pre",
    "Total_liabilitiesYearBeforeIPO": "kg_total_liabilities_pre",
    "Net_cash_provided_by_operating_activitiesYearBeforeIPO": "kg_op_cash_flow_pre",
}


def attach_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    if not KAGGLE.exists():
        return df
    k = pd.read_csv(KAGGLE, low_memory=False, encoding_errors="replace",
                    usecols=["Symbol", "Year", "Month", "Day"] + list(KAGGLE_COLS))
    k["k_date"] = pd.to_datetime(dict(year=k.Year, month=k.Month, day=k.Day), errors="coerce")
    k = k.rename(columns={"Symbol": "ticker", **KAGGLE_COLS})
    df = nearest_match(df, k, "ipo_date", "k_date", list(KAGGLE_COLS.values()))
    df["founding_year"] = df["founding_year"].fillna(df["kg_year_founded"])
    return df.drop(columns=["kg_year_founded"])


def load_underwriter_ranks() -> pd.DataFrame:
    u = pd.read_excel(RAW / "Underwriter-Rank.xls", sheet_name="Sheet1")
    u = u.rename(columns={"Underwriter Name": "uw"}).dropna(subset=["uw"])
    u = u[~u["uw"].str.startswith("-9")]
    u["_n"] = u["uw"].map(norm_name)
    return u.replace(-9, np.nan)


# Scoop/Nasdaq spellings that Ritter files under a different name
UW_ALIASES = {
    "bofa": "bank of america merrill lynch",
    "banc of america": "bank of america merrill lynch",
    "jmp": "jmp sec",
    "citi": "citigroup",
    "cantor": "cantor fitzgerald",
    "cs first boston": "credit suisse first boston",
    "sg cowen": "cowen",
    "deutsche bank": "deutsche bank securities",
}


def match_underwriter(n: str, names: list[str]) -> list[str]:
    """Ritter names matching a normalized underwriter name, from strictest rule down."""
    for alias, target in UW_ALIASES.items():
        if n == alias or n.startswith(alias + " "):
            n = target
            break
    if n in names:
        return [n]
    # word-boundary prefix either way: "barclays" ~ "barclays capital", "wells fargo securities" ~ "wells fargo"
    if len(n.split()[0]) >= 4:
        hits = [m for m in names if m.startswith(n + " ") or n.startswith(m + " ")]
        if hits:
            return hits
    return get_close_matches(n, names, n=1, cutoff=0.9)


def lead_rank(managers: str, year: int, ranks: pd.DataFrame, cache: dict) -> float:
    """Highest Ritter rank among the listed lead managers, for the IPO's rank period."""
    if pd.isna(managers):
        return np.nan
    col = next((c for c, lo, hi in RANK_PERIODS if lo <= year <= hi), None)
    if col is None or col not in ranks:
        return np.nan
    names = ranks["_n"].tolist()
    best = np.nan
    for name in re.split(r"/|,| and |&", str(managers)):
        n = norm_name(name)
        if not n:
            continue
        if n not in cache:
            cache[n] = match_underwriter(n, names)
        v = ranks.loc[ranks["_n"].isin(cache[n]), col].max()
        if pd.notna(v):
            best = v if pd.isna(best) else max(best, v)
    return best


def uw_rank_from_text(text: str, year: int, ranks: pd.DataFrame) -> float:
    """Highest Ritter rank among ranked underwriters named anywhere in a prospectus passage."""
    if pd.isna(text):
        return np.nan
    col = next((c for c, lo, hi in RANK_PERIODS if lo <= year <= hi), None)
    if col is None or col not in ranks:
        return np.nan
    hay = f" {norm_name(text)} "
    named = ranks[ranks[col].notna() & (ranks["_n"].str.len() >= 5)]
    hits = named[[f" {n} " in hay for n in named["_n"]]]
    return hits[col].max() if len(hits) else np.nan


PROSPECTUS_FACTS = Path("data/interim/prospectus_facts.csv")
PRICE_RANGES = Path("data/interim/price_ranges.csv")


def attach_prospectus(df: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    """Deal facts parsed from the final prospectus and the initial filing price range."""
    if PROSPECTUS_FACTS.exists():
        f = pd.read_csv(PROSPECTUS_FACTS).drop(columns=["summary_chars"], errors="ignore")
        df = df.merge(f, on="ipo_id", how="left")
        doc_rank = [uw_rank_from_text(t, y, ranks) for t, y in zip(df["doc_underwriters"], df["ipo_year"])]
        df["uw_rank_from_doc"] = df["lead_uw_rank"].isna() & pd.notna(doc_rank)
        df["lead_uw_rank"] = df["lead_uw_rank"].fillna(pd.Series(doc_rank, index=df.index))
        df["log_doc_chars"] = np.log(df["doc_chars"].where(df["doc_chars"] > 0))
        df["log_risk_chars"] = np.log(df["risk_factors_chars"].where(df["risk_factors_chars"] > 0))
    if PRICE_RANGES.exists():
        r = pd.read_csv(PRICE_RANGES)[["ipo_id", "range_low", "range_high"]]
        df = df.merge(r, on="ipo_id", how="left")
        mid = (df["range_low"] + df["range_high"]) / 2
        df["price_revision"] = df["offer_price"] / mid - 1
        df["above_range"] = (df["offer_price"] > df["range_high"]).where(mid.notna())
        df["below_range"] = (df["offer_price"] < df["range_low"]).where(mid.notna())
        df["range_width"] = (df["range_high"] - df["range_low"]) / mid
    return df


def flag_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    name = df["company"].str.lower()
    tick = df["ticker"].astype(str)
    df["is_spac"] = (
        name.str.contains(r"acquisition|blank check|spac\b", regex=True)
        | (tick.str.match(r".*(U|\.U|\.UN)$") & (df["offer_price"] == 10))
        | (name.str.contains("capital") & (df["offer_price"] == 10) & tick.str.endswith("U"))
    )
    df["is_unit"] = df.get("is_unit_scoop", False).fillna(False).astype(bool) | tick.str.endswith(".U")
    df["is_fund"] = df["adr_code"].between(3, 7)
    df["is_penny"] = df["offer_price"] < 5
    df["is_adr"] = df["adr_code"] == 2
    # Undoing >10x of later reverse splits from Yahoo's rounded adjusted prices gives junk closes
    df["px_unreliable"] = df["split_factor"] < MIN_SPLIT_FACTOR
    df["in_sample"] = ~(df["is_spac"] | df["is_unit"] | df["is_fund"] | df["is_penny"]
                        | df["px_unreliable"] | df["first_day_return"].isna())
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["first_day_return"] = df["close_1"] / df["offer_price"] - 1
    df["open_return"] = df["open_1"] / df["offer_price"] - 1
    df["pop"] = (df["first_day_return"] > 0).astype("Int64")
    df["pop10"] = (df["first_day_return"] > 0.10).astype("Int64")
    df["bucket"] = pd.cut(df["first_day_return"], BUCKETS, labels=BUCKET_LABELS, right=True)
    # exactly 0% (priced and closed flat) belongs with "fair", not "overpriced"
    df.loc[df["first_day_return"] == 0, "bucket"] = "fair"
    df.loc[df["first_day_return"].isna(), ["pop", "pop10"]] = pd.NA
    return df


def main() -> None:
    scoop = load_scoop()
    nas = pd.read_csv(RAW / "nasdaq_priced.csv", parse_dates=["priced_date"])

    # Scoop deals: date is the first trade date; bring in Nasdaq deal facts by ticker/date
    scoop = scoop.rename(columns={"trade_date": "ipo_date"})
    scoop = nearest_match(scoop, nas, "ipo_date", "priced_date",
                          ["deal_id", "exchange", "shares_offered", "proceeds"])

    # Post-Scoop deals come from Nasdaq + Yahoo
    cutoff = scoop["ipo_date"].max()
    late = load_nasdaq_with_prices()
    late = late[late["first_trade_date"] > cutoff].rename(columns={"first_trade_date": "ipo_date"})
    late["source"] = "nasdaq_yahoo"
    late = late.drop(columns=["priced_date"])

    df = pd.concat([scoop, late], ignore_index=True, sort=False)
    df = df.drop_duplicates(subset=["ticker", "ipo_date"])
    df["ipo_year"] = df["ipo_date"].dt.year

    df = attach_ritter(df)
    df = attach_kaggle(df)
    ranks, cache = load_underwriter_ranks(), {}
    df["lead_uw_rank"] = [lead_rank(m, y, ranks, cache) for m, y in zip(df["lead_managers"], df["ipo_year"])]
    df["firm_age"] = (df["ipo_year"] - df["founding_year"]).where(lambda s: s >= 0)
    df["log_proceeds"] = np.log(df["proceeds"].where(df["proceeds"] > 0))

    df = add_targets(df)
    df = flag_exclusions(df)
    df = df.sort_values("ipo_date").reset_index(drop=True)
    df.insert(0, "ipo_id", df["ticker"].map(norm_ticker) + "_" + df["ipo_date"].dt.strftime("%Y%m%d"))
    df = attach_prospectus(df, ranks)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "us_ipos.csv", index=False)
    clean = df[df["in_sample"]].drop(columns=["is_unit_scoop"], errors="ignore")
    clean.to_csv(OUT / "us_ipos_clean.csv", index=False)
    print(f"all deals: {len(df)}  clean: {len(clean)}")


if __name__ == "__main__":
    main()
