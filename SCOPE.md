# Project Scope

## Title

**Reading the Prospectus: An NLP-Driven Machine Learning System for Predicting First-Day IPO Returns in the US and Nigeria**

## One-line summary

A tool that reads an IPO prospectus, pulls out the key deal facts and text signals, adds market conditions, and predicts (1) whether the stock will "pop" on day one and (2) how big the move will be (return bucket). It is trained on US IPOs and tested on Nigerian (NGX) IPOs.

## Research questions

1. Can deal, firm, market and prospectus-text features predict first-day IPO pops and return buckets better than a price-revision-only baseline?
2. Does prospectus language (tone, uncertainty, FinBERT sentiment of Risk Factors) add predictive power beyond the numbers?
3. Does a model trained on US IPOs transfer to Nigerian IPOs, given NGX's ±10% daily price limit and its very different market structure?

## Prediction targets

| Target | Definition |
|---|---|
| **Pop (binary)** | 1 if first-day return > 0%, else 0 (sensitivity check at > 10%) |
| **Bucket (multi-class)** | Overpriced (< 0%) · Fair (0–10%) · Underpriced (10–30%) · Heavily underpriced (> 30%) |

First-day return = (first close − offer price) / offer price.
**Nigeria adjustment:** because of the ±10% daily limit, the Nigerian return is measured up to the first trading day the price does not close at the limit (capped at 10 trading days). The raw day-one return is reported alongside it.

## Data

| Source | What it gives | Status |
|---|---|---|
| IPOScoop track record (2000–2020) | Offer price, first-day open/close, underwriter | Downloaded |
| Nasdaq IPO calendar API (2005–2025) | Offer price, size, exchange, lockup, SEC CIK | Tested |
| Yahoo Finance (`yfinance`) | First-day close for 2020–2025 IPOs, S&P 500/VIX market features | Tested |
| Jay Ritter files | Founding year (firm age), underwriter rank | Downloaded |
| SEC EDGAR 424B4 / S-1 | Prospectus text (Risk Factors, Use of Proceeds), filing price range | Tested |
| NGX / SEC Nigeria / news | Nigerian IPO list, offer prices, early trading prices, prospectus PDFs | Manual collection |

**Expected size:** ~3,000 US operating-company IPOs (SPACs, units, ADR-only and sub-$5 deals excluded), ~30–50 Nigerian IPOs.

## Features

- **Deal:** offer price, proceeds (log), shares offered, price revision vs. filing-range midpoint, underwriter rank, exchange, lockup days
- **Firm:** age, sector (SIC), VC-backed, profitable (if it can be parsed)
- **Market:** index return over the 30 days before the IPO, VIX (US) / NGX ASI momentum (Nigeria), number of IPOs in the past 30 days
- **Text:** Loughran-McDonald negative / uncertainty / litigious word ratios, FinBERT sentiment of Risk Factors, readability, prospectus length

Leakage rule: only information available **before the first trade** is used.

## The "prospectus bot" (final deliverable)

```
Prospectus (EDGAR link or PDF)
   → parse sections (Risk Factors, Use of Proceeds, Offering summary)
   → extract deal facts (price, range, shares, underwriters)
   → compute text features (LM dictionary + FinBERT)
   → fetch market conditions on the IPO date
   → model → P(pop) + return bucket + top SHAP reasons
```

Delivered as a web app with a separate backend and frontend:

| Part | Stack | Hosting |
|---|---|---|
| `api/` | FastAPI (Python). Reuses `src/data` parsing and the trained models | Hugging Face Spaces (Docker, free CPU, 16 GB RAM: fits FinBERT) |
| `web/` | Next.js (React, TypeScript) | Vercel |

API endpoints:

- `POST /analyze` with an EDGAR URL or an uploaded PDF/HTML prospectus → deal facts, sections found, text
  scores, P(pop), bucket probabilities, top SHAP reasons. Nigerian documents also get the price-limit note.
- `GET /examples`: a few past IPOs with known outcomes, so the demo works without an upload
- `GET /health`

Frontend pages: upload/paste screen; result page (facts table, tone scores, prediction with a SHAP bar chart, the
extracted Risk Factors excerpt); an "About the model" page with test-set performance and limitations. The app
states clearly that it is a research demo, not investment advice.

OCR (Tesseract) for scanned Nigerian PDFs is added only if a target document needs it.
Stretch goal only: LLM-based field extraction for messy prospectuses.

## Models and evaluation

- Baselines: majority class; logistic regression on price revision only
- Models: logistic regression (L1), Random Forest, XGBoost/LightGBM, each with and without text features
- **Time-based split:** train 2000–2017, validate 2018–2020, test 2021–2025; Nigeria is a separate out-of-sample test
- Metrics: ROC-AUC, F1 (binary); macro-F1, confusion matrix (buckets); simple "buy only predicted pops" return vs. buying every IPO
- Interpretability: SHAP

## Out of scope

- Long-run (6–12 month) returns, SPACs, direct listings
- Other African markets
- Trading or investment advice

## Timeline (8 weeks)

| Week | Work |
|---|---|
| 1 | Build the US dataset (Scoop + Nasdaq + yfinance), cleaning |
| 2 | Fetch EDGAR prospectuses, section parsing, deal-fact extraction |
| 3 | Text features (LM, FinBERT), market features |
| 4 | Nigerian data collection, adjusted returns, OCR |
| 5 | Modelling and tuning; text vs. no-text ablation |
| 6 | SHAP, Nigeria transfer test, error analysis |
| 7 | Web app: FastAPI backend (extraction endpoint can start in week 3, once parsing is stable) and Next.js frontend |
| 8 | Write-up and presentation |

## Key risks

- **Survivorship bias:** delisted tickers are missing from Yahoo; covered by the Scoop file up to 2020, documented after that
- **Small Nigerian sample:** treated as a case-study and transfer test, not a separate model
- **Web app scope:** two codebases in one week is tight; the API is built first so the model can be demoed
  through its docs page (`/docs`) even if the frontend runs late
- **Low predictability:** expected and normal; the economic metric and SHAP carry the story
