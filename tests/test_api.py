"""API tests. Run from the project root: .venv/Scripts/python -m pytest tests -q

The end-to-end test scores a saved EDGAR prospectus, so it needs the pipeline outputs (models, texts).
"""
import gzip
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.app.main import app, clean  # noqa: E402

client = TestClient(app)
HAVE_MODELS = (ROOT / "models" / "pop.joblib").exists()
TEXTS = ROOT / "data/raw/edgar/prospectus_text"


def test_clean_makes_json_safe():
    out = clean({"a": np.float64(1.5), "b": float("nan"), "c": [np.int64(2), float("inf")], 3: "x"})
    assert out == {"a": 1.5, "b": None, "c": [2, None], "3": "x"}


def test_needs_exactly_one_input():
    assert client.post("/analyze", data={}).status_code == 422
    files = {"file": ("p.txt", b"x" * 10, "text/plain")}
    r = client.post("/analyze", data={"url": "https://www.sec.gov/Archives/edgar/data/1/2/a.htm"}, files=files)
    assert r.status_code == 422


def test_rejects_non_sec_links():
    r = client.post("/analyze", data={"url": "https://example.com/prospectus.htm"})
    assert r.status_code == 422
    assert "EDGAR" in r.json()["detail"]


def test_rejects_unknown_file_type():
    r = client.post("/analyze", files={"file": ("p.docx", b"x" * 10000, "application/octet-stream")})
    assert r.status_code == 422


def test_rejects_inverted_range():
    files = {"file": ("p.txt", b"x" * 10000, "text/plain")}
    r = client.post("/analyze", data={"range_low": "20", "range_high": "10"}, files=files)
    assert r.status_code == 422


@pytest.mark.skipif(not HAVE_MODELS, reason="run the pipeline first")
def test_health_lists_models():
    assert client.get("/health").json()["models"] == ["bucket", "pop", "pop10"]


@pytest.mark.skipif(not HAVE_MODELS or not (TEXTS / "ABNB_20201210.txt.gz").exists(), reason="needs pipeline outputs")
def test_analyze_saved_prospectus():
    text = gzip.open(TEXTS / "ABNB_20201210.txt.gz", "rt", encoding="utf-8").read().encode()
    r = client.post("/analyze", files={"file": ("airbnb.txt", text, "text/plain")},
                    data={"ipo_date": "2020-12-10", "range_low": "44", "range_high": "50"})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["market"] == "US"
    assert a["deal"]["offer_price"] == 68.0
    assert a["deal"]["price_range"] == [44.0, 50.0]
    assert math.isclose(a["features"]["price_revision"], 68 / 47 - 1, rel_tol=1e-6)
    pop = a["predictions"]["pop"]
    assert 0 <= pop["probability"] <= 1 and len(pop["reasons"]) > 0
    probs = a["predictions"]["bucket"]["probabilities"]
    assert math.isclose(sum(probs.values()), 1, rel_tol=1e-6)


@pytest.mark.skipif(not HAVE_MODELS or not (TEXTS / "ABNB_20201210.txt.gz").exists(), reason="needs pipeline outputs")
def test_analyze_after_data_ends():
    # past the last IPO in the dataset, recent-IPO activity is unknown; the features must be missing, not absent
    text = gzip.open(TEXTS / "ABNB_20201210.txt.gz", "rt", encoding="utf-8").read().encode()
    r = client.post("/analyze", files={"file": ("airbnb.txt", text, "text/plain")},
                    data={"ipo_date": "2026-05-14", "range_low": "44", "range_high": "50"})
    assert r.status_code == 200, r.text
    assert r.json()["features"]["ipo_count_30d"] is None


def test_repairs_lost_pdf_ligatures():
    from api.app.analyze import repair_ligatures
    assert repair_ligatures("is of ering shares; suf icient cash") == "is offering shares; sufficient cash"
    assert repair_ligatures("one of each, if any") == "one of each, if any"


def _history(*rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["form", "filingDate", "accessionNumber", "primaryDocument"])


@pytest.mark.parametrize("rows,tickers,status", [
    ([("S-1", "2026-04-17", "a-1", "s1.htm"), ("424B4", "2026-05-14", "a-2", "p.htm")], ["CBRS"], "priced"),
    ([("S-1", "2026-10-26", "a-1", "s1.htm")], [], "registration"),
    ([("S-1", "2024-01-05", "a-1", "s1.htm"), ("RW", "2024-06-01", "a-2", "rw.htm")], [], "withdrawn"),
    # withdrew, then refiled: still an upcoming IPO
    ([("S-1", "2024-01-05", "a-1", "s1.htm"), ("RW", "2024-06-01", "a-2", "rw.htm"),
      ("S-1", "2026-10-01", "a-3", "s1b.htm")], [], "registration"),
    ([("10-K", "2026-02-01", "a-1", "k.htm")], ["AAPL"], "public"),
    ([("D", "2026-02-01", "a-1", "d.xml")], [], "none"),
])
def test_ipo_status(monkeypatch, rows, tickers, status):
    from api.app import search
    monkeypatch.setattr(search, "filings", lambda cik: (_history(*rows), tickers, "Example Inc."))
    assert search.ipo_status(1)["status"] == status


@pytest.mark.skipif(not HAVE_MODELS or not (TEXTS / "ABNB_20201210.txt.gz").exists(), reason="needs pipeline outputs")
def test_stream_reports_each_step_then_the_result():
    import json
    text = gzip.open(TEXTS / "ABNB_20201210.txt.gz", "rt", encoding="utf-8").read().encode()
    with client.stream("POST", "/analyze/stream", files={"file": ("airbnb.txt", text, "text/plain")},
                       data={"range_low": "44", "range_high": "50"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    done = [e["id"] for e in events if e["event"] == "step" and e["state"] != "running"]
    assert done == ["document", "deal", "range", "underwriters", "tone", "market", "model"]
    result = events[-1]
    assert result["event"] == "result" and result["result"]["verdict"]["label"]
    assert all(x["text"] for x in result["result"]["predictions"]["pop"]["reasons"])


def test_stream_reports_input_errors():
    import json
    with client.stream("POST", "/analyze/stream", files={"file": ("p.txt", b"too short " * 100, "text/plain")}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[-1] == {"event": "error", "message": "The document is too short to be a prospectus."}
