"""
Earnings releases from SEC EDGAR: 8-K filings with Item 2.02 and the text of their exhibit 99.1
(moved from experiments/H06_earnings_text/fetch.py; the choice of the exhibit is the same, and the
'refine' step of H06 is done right away).

SEC asks for a contact in the User-Agent and at most 10 requests per second (8 are used, shared by all threads).
"""
import html
import json
import re
import threading
import time
import urllib.error
import urllib.request

from events import config, texts

_lock, _last = threading.Lock(), [0.0]
EXHIBIT = re.compile(r"ex[-_]?99|exhibit[-_]?99|99[-_.]?1", re.I)


def get(url):
    """GET with the SEC rate limit and up to 4 retries; None for 404."""
    for attempt in range(5):
        with _lock:
            wait = _last[0] + 1 / config.SEC_MAX_PER_SECOND - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": config.sec_user_agent(), "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {url}")


def filings_of(cik, first, last):
    """8-K filings with Item 2.02 filed between first and last (older pages of the history are read when needed)."""
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))
    pages = [sub["filings"]["recent"]]
    for extra in sub["filings"].get("files", []):
        if extra["filingTo"] >= first and extra["filingFrom"] <= last:
            pages.append(json.loads(get(f"https://data.sec.gov/submissions/{extra['name']}")))
    rows = []
    for p in pages:
        for form, date, acc, items, accepted, primary in zip(p["form"], p["filingDate"], p["accessionNumber"], p["items"],
                                                            p["acceptanceDateTime"], p["primaryDocument"]):
            if form == "8-K" and "2.02" in (items or "") and first <= date <= last:
                rows.append({"cik": int(cik), "filing_date": date, "accession": acc, "accepted": accepted, "primary": primary})
    return rows


def candidate_documents(items, primary):
    return [i for i in items if i["name"].lower().endswith((".htm", ".html", ".txt")) and not i["name"].startswith(("0", "R"))
            and "index" not in i["name"].lower() and i["name"] != primary]


def choose_exhibit(items, primary):
    """The press release: an htm/txt file named like exhibit 99(.1); otherwise the largest other document."""
    docs = candidate_documents(items, primary)
    named = [i for i in docs if EXHIBIT.search(i["name"])]
    pool = named or docs
    if not pool:
        return None, "none"
    size = lambda i: int(i["size"]) if str(i.get("size", "")).isdigit() else 0
    # exhibit 99.1 before 99.2 (the second one is often a slide deck or a supplement)
    pool.sort(key=lambda i: (0 if re.search(r"99[-_.]?0?1(?!\d)", i["name"]) else 1, -size(i)))
    return pool[0]["name"], "named" if named else "largest"


def to_text(raw):
    """HTML -> plain text: scripts and tags removed, entities decoded, white space collapsed."""
    s = raw.decode("utf-8", errors="replace")
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def fetch_text(row):
    """Download the release of one filing to data/cache/edgar. Returns the filing row with document, how_chosen,
    text_path (None when there is no text). A text already saved is kept."""
    path = texts.path_for(row["cik"], row["accession"])
    if path.exists():
        return {**row, "how_chosen": row.get("how_chosen") or "cached", "text_path": texts.relative(path)}
    acc = row["accession"].replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(row['cik'])}/{acc}"
    index = get(f"{base}/index.json")
    if index is None:
        return {**row, "document": None, "how_chosen": "no index", "text_path": None}
    items = json.loads(index)["directory"]["item"]
    name, how = choose_exhibit(items, row["primary"])
    if name is None:
        return {**row, "document": None, "how_chosen": how, "text_path": None}
    text = to_text(get(f"{base}/{name}") or b"")
    if texts.doc_type(text) not in texts.MAIN_TYPES:
        # the chosen file is a supplement (EX-99.2) or another exhibit: look for the EX-99.1 among the others
        for other in candidate_documents(items, row["primary"]):
            if other["name"] == name:
                continue
            candidate = to_text(get(f"{base}/{other['name']}") or b"")
            if texts.doc_type(candidate) in texts.MAIN_TYPES:
                name, how, text = other["name"], "refined", candidate
                break
        else:
            how += "_not_ex991"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {**row, "document": name, "how_chosen": how, "text_path": texts.relative(path)}
