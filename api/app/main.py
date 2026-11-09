"""Prospectus bot API.

    uvicorn api.app.main:app --reload        # from the project root; docs at http://localhost:8000/docs

Endpoints:
    POST /analyze           EDGAR link (form field `url`) or uploaded PDF/HTML/TXT (`file`), plus optional
                            corrections: offer_price, shares_offered, range_low, range_high, ipo_date,
                            founding_year, vc, exchange
    GET  /examples          past IPOs with known outcomes, analysed in advance
    GET  /examples/{id}     one of them
    GET  /model             what the models are and how they scored on the test years
    GET  /health
"""
import json
import logging
import math
import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analyze import ROOT, AnalysisError, analyze, document_from_upload, document_from_url, models

EXAMPLES = Path(__file__).resolve().parent / "examples.json"
MAX_UPLOAD_MB = 40

app = FastAPI(title="Prospectus bot", version="1.0",
              description="Predicts first-day IPO returns from a prospectus. Research demo, not investment advice.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o],
    allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX"),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def clean(obj):
    """JSON-safe copy: numpy numbers to Python, NaN/inf to null."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _examples() -> dict:
    return json.loads(EXAMPLES.read_text(encoding="utf-8")) if EXAMPLES.exists() else {}


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(models())}


@app.get("/model")
def model_card():
    metrics = ROOT / "results" / "metrics.csv"
    text_effect = ROOT / "results" / "text_effect.csv"
    out = {t: {k: b[k] for k in ("model_name", "feature_set", "trained_years", "test_scores")}
           for t, b in models().items()}
    if text_effect.exists():
        import pandas as pd
        out["text_effect"] = pd.read_csv(text_effect).to_dict(orient="records")
        out["metrics"] = pd.read_csv(metrics).to_dict(orient="records") if metrics.exists() else []
    return clean(out)


@app.get("/examples")
def examples():
    return [{"id": k, "company": v["company"], "ipo_date": v["ipo_date"], "actual": v["actual"]}
            for k, v in _examples().items()]


@app.get("/examples/{example_id}")
def example(example_id: str):
    ex = _examples().get(example_id)
    if ex is None:
        raise HTTPException(404, "unknown example")
    return ex


@app.post("/analyze")
async def analyze_endpoint(
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
    offer_price: float | None = Form(None, gt=0),
    shares_offered: float | None = Form(None, gt=0),
    range_low: float | None = Form(None, gt=0),
    range_high: float | None = Form(None, gt=0),
    ipo_date: date | None = Form(None),
    founding_year: int | None = Form(None, ge=1800, le=2100),
    vc: bool | None = Form(None),
    exchange: str | None = Form(None, max_length=60),
):
    if bool(url) == bool(file):
        raise HTTPException(422, "Send either an EDGAR link or a file, not both.")
    if range_low and range_high and range_low > range_high:
        raise HTTPException(422, "range_low is above range_high.")
    try:
        if url:
            doc = document_from_url(url)
        else:
            data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
            doc = document_from_upload(file.filename or "upload", data)
        overrides = {"offer_price": offer_price, "shares_offered": shares_offered, "range_low": range_low,
                     "range_high": range_high, "ipo_date": ipo_date, "founding_year": founding_year,
                     "vc": None if vc is None else float(vc), "exchange": exchange}
        return clean(analyze(doc, overrides))
    except AnalysisError as e:
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        # an unhandled error would reach the browser without CORS headers, shown only as "Failed to fetch"
        logging.exception("analysis failed")
        raise HTTPException(500, f"The analysis failed ({type(e).__name__}). Please try again or report this document.")
