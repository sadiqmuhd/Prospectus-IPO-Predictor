# Datasets

- [US IPO dataset](#us-ipo-dataset)
- [Nigerian (NGX) listings](#nigerian-ngx-listings)

# US IPO Dataset

Built by the scripts in `src/data/`. One command runs the whole pipeline (US and Nigeria) from the project root:

```bash
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/pipeline.py            # bring everything up to date
.venv/Scripts/python src/pipeline.py --list     # the steps, in order
```

Every step is safe to rerun. The Nasdaq calendar, the EDGAR index and the Ritter/IPOScoop files are downloaded only
when missing (`--refresh` re-downloads them). The slow fetchers (Yahoo, prospectuses, price ranges, NGX lists) skip
work already saved, and the parser only reads new prospectuses (`--reparse` re-parses all of them after a parser
change). `--only a,b` and `--from step` run part of it. Each step logs to `logs/<step>.log`. The Kaggle file must be
downloaded by hand into `data/raw/kaggle/`. A first run from scratch takes several hours, mostly the
SEC downloads, which are throttled to stay under SEC's 10 requests per second.

## Outputs

| File | Rows | What |
|---|---|---|
| `processed/us_ipos.csv` | 4,445 | Every deal, with exclusion flags |
| `processed/us_ipos_clean.csv` | 3,642 | Modelling sample: operating companies, offer price ≥ $5, reliable prices |

**Targets:** `first_day_return` (close / offer − 1), `pop` (> 0%), `pop10` (> 10%), `bucket`
(overpriced < 0% · fair 0–10% · underpriced 10–30% · heavily_underpriced > 30%).
Clean sample: 69% pop; buckets 1,154 fair / 888 underpriced / 868 overpriced / 732 heavily underpriced.

## Sources

| Raw file | Source | Used for |
|---|---|---|
| `raw/us/SCOOP-Rating-Performance.xls` | IPOScoop track record | Offer, open, first-day close, lead managers, May 2000 – Sep 2020 (includes later-delisted firms) |
| `raw/us/nasdaq_priced.csv` | Nasdaq IPO calendar API | Offer price, shares, proceeds, exchange, 2000–2025 |
| `raw/us/yahoo_first_day.csv` | Yahoo Finance via `yfinance` | First-day open/close after Sep 2020, split-unadjusted |
| `raw/us/IPO-age.xlsx` | Jay Ritter | Founding year, VC backing, dual class, ADR, internet flags |
| `raw/us/Underwriter-Rank.xls` | Jay Ritter (Carter-Manaster ranks) | Lead underwriter prestige (0–9) |
| `raw/us/ritter_table1_first_day_returns.csv` | Ritter, *IPO Statistics* Table 1 | Validation only |
| `raw/kaggle/IPODataFull.csv` | Kaggle "Stocks IPO information & results" (CC0) | Sector/industry, founding year fallback, pre-IPO financials (`kg_*`) |

## Validation

Yearly mean and median first-day returns (ADRs excluded) correlate with Ritter's Table 1 at 0.85 (means) and
0.88 (medians) over 2001–2025. Our medians run a little lower because our sample keeps some small deals that
Ritter excludes (banks, REITs, best-efforts offerings).

## Known limitations

- **Survivorship after Sep 2020.** Yahoo has no history for delisted tickers, so 2021–2025 holds only firms
  still trading (about 53% of priced deals). Their returns are not representative. Use 2021–2025 as a test
  period and report this bias.
- **2000 starts in May.** IPOScoop begins on 12 May 2000, so the Jan–Apr bubble peak is missing.
- **Reverse splits.** 196 Yahoo-priced deals needed more than a 10× reverse-split adjustment; their
  prices are unreliable (`px_unreliable`) and excluded.
- **Kaggle fields** cover only firms listed in 2018 (about 35% of the clean sample). Never use their presence or
  absence as a feature: that would leak survival. Its prices, employee and CEO fields are not used: the prices are
  split-adjusted and sometimes start on day 2, and the other fields were scraped after the IPO.
- **Underwriter rank** is matched by name from IPOScoop's lead managers, then filled from the underwriters
  named in the prospectus (`uw_rank_from_doc`).

## Prospectus features (SEC EDGAR)

| File | What |
|---|---|
| `raw/edgar/filings_index.csv` | 424B1/424B4 and S-1/F-1/SB-2 (and amendments) filings, 2000–2025 |
| `interim/prospectus_matches.csv` | Each IPO's final prospectus (matched by company name, −7 to +10 days of the IPO) |
| `raw/edgar/prospectus_text/<ipo_id>.txt.gz` | Prospectus as plain text (not in git; the pipeline rebuilds it) |
| `interim/prospectus_facts.csv` | Offer price, shares, underwriters, syndicate size, section lengths |
| `interim/prospectus_sections.jsonl.gz` | Summary, Risk Factors and Use of Proceeds text, for the NLP features |
| `interim/registration_filings.csv` | S-1/F-1 filings up to 3 years before each IPO |
| `interim/price_ranges.csv` | Initial filing price range: the first registration filing that states one |

`price_revision` = offer price / midpoint of the initial range − 1, with `above_range` / `below_range`. Only
documents filed before trading starts are used, so none of these features leak the outcome. Ranges were found
for 89.7% of matched IPOs and checked by hand against known deals (Airbnb, Rivian, CAVA, SentinelOne).

# Nigerian (NGX) Listings

There is no bulk source of Nigerian IPO data, so the sample is assembled listing by listing. Every row in
`raw/nigeria/ngx_listings.csv` carries the URLs it was taken from.

| File | What |
|---|---|
| `raw/nigeria/ngx_listings.csv` | One row per listing: type, offer/reference price, dates, day-1/day-2 close, status, sources |
| `raw/nigeria/price_limit_regimes.csv` | NGX daily price limits over time (5% to ~2011, 10% since ~2012) |
| `raw/nigeria/ngx_daily_list/` | Cached NGX Daily Official List (Equities) PDFs, late 2014 onward |
| `raw/nigeria/ngx_first_sessions.csv` | First 10 sessions of each listing since late 2014, read from those PDFs |

```bash
.venv/Scripts/python src/data/fetch_ngx_daily_lists.py   # re-reads cached PDFs; downloads any missing days
```

## Listing types

- `ipo`: shares sold to the public at an offer price. This is the true comparison with US IPOs.
- `introduction`: existing shares admitted with a reference price and no offer (MTN, Airtel, BUA Foods,
  Geregu, Transcorp Power, Aradel, Legend Internet). Their listing-day move is price discovery, not
  underpricing. Keep them as a separate group.
- `private_placement_listing`: shares placed privately, then listed at the placement price (Starcomms, Red Star,
  Fidson).

## Status values

`verified` (two or more sources or an NGX official list agree) · `day1_derived` (computed from a stated % move) ·
`single_source` · `conflicting_day1` · `needs_day1_price` · `needs_listing_day` · `needs_offer_price` ·
`pending_listing`.

## Daily price limit

Listing-day returns are capped by NGX's daily limit (5% before ~2012, 10% since). Every large introduction from
2019 to 2025 closed exactly +10% on day 1, and most again on day 2, so the day-1 return is censored. The adjusted
return runs from the offer/reference price to the close of the first session whose move is inside the limit,
capped at 10 sessions.

## Known gaps

- **Pre-2014 day-1 prices** come only from news reports. For most 2004–2009 IPOs (the bulk of genuine Nigerian
  IPOs) no day-1 price is reported online, and the NGX archive starts in late 2014.
- **Offer prices** for several 2005 bank IPOs (Fidelity, Diamond) and NAHCO are not online.
- **Dangote Petroleum Refinery** (offer ₦525, closed 27 Oct 2026) is expected to list in December 2026; record its
  first 10 sessions from the daily lists as they are published.
