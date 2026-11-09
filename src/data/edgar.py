"""Shared SEC EDGAR client: declared User-Agent and a polite rate limit (SEC allows 10 req/s)."""
import os
import re
import threading
import time

import requests
from bs4 import BeautifulSoup

USER_AGENT = os.environ.get("SEC_USER_AGENT", "FTF-Project IPO research 2211402002@udusok.edu.ng")
# seconds between requests (~6-7 req/s); raise it when running two downloaders at once
MIN_INTERVAL = float(os.environ.get("SEC_MIN_INTERVAL", "0.15"))

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
_last = 0.0
_lock = threading.Lock()  # one rate limit shared by all download threads


def _throttle() -> None:
    global _last
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()


def get(url: str, retries: int = 4, stream: bool = False) -> requests.Response:
    for attempt in range(retries):
        _throttle()
        try:
            r = _session.get(url, timeout=60, stream=stream)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r
        r.close()
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"EDGAR request failed after {retries} tries: {url}")


def get_main_document(path: str, max_bytes: int | None = None) -> str:
    """First document of a full-submission .txt file, read until </DOCUMENT> or max_bytes.

    Submission files also carry exhibits and uuencoded graphics, often tens of MB;
    the prospectus itself is always the first document.
    """
    r = get(f"https://www.sec.gov/Archives/{path}", stream=True)
    buf = bytearray()
    try:
        for chunk in r.iter_content(chunk_size=65536):
            tail = len(buf)
            buf.extend(chunk)
            # search from just before the new chunk so a tag split across chunks is still found
            if buf.find(b"</DOCUMENT>", max(0, tail - 16)) >= 0 or (max_bytes and len(buf) >= max_bytes):
                break
    finally:
        r.close()
    end = buf.find(b"</DOCUMENT>")
    raw = bytes(buf if end < 0 else buf[:end])
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def to_text(doc: str) -> str:
    """Plain text of an EDGAR document (HTML filings from ~2001 on, plain text before)."""
    start = doc.find("<TEXT>")
    body = doc[start + 6:] if start >= 0 else doc
    if re.search(r"<(html|body|p|div|table|font)\b", body[:5000], re.I):
        soup = BeautifulSoup(body, "lxml")
        for t in soup(["script", "style"]):
            t.decompose()
        # keep table cells on one line so "Goldman Sachs & Co.   5,000,000" stays together
        for row in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            row.replace_with("\n" + "  ".join(c for c in cells if c) + "\n")
        body = soup.get_text("\n")
    else:
        body = re.sub(r"</?(PAGE|S|C|FN|CAPTION|TABLE)>", " ", body)
    body = body.replace("\xa0", " ")
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n\n", body)
    return body.strip()
