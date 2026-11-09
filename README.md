# Reading the Prospectus

An NLP-driven machine learning system for predicting first-day IPO returns in the US and Nigeria.

The project builds a dataset of 3,642 US IPOs (2000–2025) and 29 Nigerian (NGX) listings, reads each US prospectus
from SEC EDGAR, scores its tone with the Loughran-McDonald dictionary and FinBERT, and trains models that predict
whether an IPO pops on its first day and which return bucket it lands in. A web app (the "prospectus bot") applies
the models to any prospectus.

## Layout

| Path | What |
|---|---|
| `SCOPE.md` | Research questions, targets, method, timeline |
| `src/pipeline.py` | Runs everything below in order; safe to rerun |
| `src/data/` | Data collection: IPOScoop, Nasdaq, Yahoo, Ritter, EDGAR, NGX |
| `src/features/` | Loughran-McDonald and FinBERT tone, market conditions, the modelling table |
| `src/models/` | Training and evaluation (`train.py`), SHAP explanations (`explain.py`) |
| `src/reports/` | Figures for the report |
| `data/` | Raw, interim and processed data; see `data/README.md` |
| `results/` | Metrics, text-effect tests, test-set predictions, SHAP importance |
| `reports/` | Report draft (`report.md`) and figures |
| `models/` | Chosen models, used by the API |
| `api/` | FastAPI backend of the prospectus bot |
| `web/` | Next.js frontend of the prospectus bot |
| `deploy/` | How to put the API on Railway and the site on Vercel |
| `tests/` | API tests |

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/pipeline.py              # all steps; --list shows them
.venv/Scripts/python -m pytest tests -q
```

A first run from scratch takes several hours (SEC downloads are throttled; FinBERT on a CPU takes about 3 hours).
Later runs only do new work. One input must be downloaded by hand: the Kaggle "Stocks IPO information & results"
CSV, into `data/raw/kaggle/IPODataFull.csv`.

Then start the app locally:

```bash
.venv/Scripts/python -m uvicorn api.app.main:app --port 8000   # API, docs at /docs
cd web && npm install && npm run dev                             # site at http://localhost:3000
```

Deployment: see `deploy/README.md`.
