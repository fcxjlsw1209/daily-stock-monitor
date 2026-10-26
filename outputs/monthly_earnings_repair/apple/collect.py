"""Collect and verify Apple issuer-original quarterly release evidence.

Date-only Newsroom metadata is not a timestamp. Provider fiscal labels are not
used as actual fiscal period ends. Each successful record needs issuer body text.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / "raw"
VENDOR = ROOT / "outputs/monthly_event_confirmation/raw/AAPL.normalized.json"
OLD = ROOT / "outputs/monthly_cost_portfolio_research/events/raw"
QUARTERS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def parse_html(body, target, source_url, evidence_path):
    soup = BeautifulSoup(body, "html.parser")
    article = None
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(script.get_text())
        except (ValueError, TypeError):
            continue
        if isinstance(j, dict) and j.get("@type") == "NewsArticle":
            article = j
            break
    if not article:
        raise ValueError("No NewsArticle metadata")
    pub = article.get("datePublished", "")
    if pub[:10] != target["reported_date"]:
        raise ValueError("Official release date does not match provider target: " + pub)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}Z?", pub):
        # Do not silently treat unexpected datetime formats as validated time.
        raise ValueError("New publication precision needs explicit review: " + pub)
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    dateline = re.search(r"\bCupertino\s*,?\s+California\b", text, re.IGNORECASE)
    if not dateline:
        raise ValueError("Cupertino California dateline missing; timezone policy needs review")
    m = re.search(r"fiscal\s+(\d{4})\s+(first|second|third|fourth)\s+quarter\s+ended\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", text)
    if not m:
        raise ValueError("Actual fiscal period end not found in issuer body")
    actual = dt.datetime.strptime(m.group(3).replace(",", ""), "%B %d %Y").date().isoformat()
    return {
        "ticker": "AAPL", "release_date": pub[:10],
        "publication_timestamp": None, "precision": "date_only",
        "publication_timezone": "America/Los_Angeles",
        "publication_timezone_basis": "Conservative research policy: interpret the US issuer-local calendar date using the observed Cupertino, California dateline; the CMS timezone and exact publication time were not observed.",
        "publication_timezone_evidence": {
            "kind": "issuer_local_calendar_date_policy",
            "observed_dateline": dateline.group(0),
            "dateline_excerpt": text[dateline.start():dateline.end() + 100],
            "evidence_path": evidence_path,
            "geographic_mapping": "Cupertino, California -> America/Los_Angeles",
            "cms_timezone_observed": False,
            "exact_publication_time_inferred": False,
        },
        "first_public_time_verified": False,
        "official_date_published_raw": pub,
        "fiscal_period_end_actual": actual,
        "fiscal_year": int(m.group(1)), "fiscal_quarter": QUARTERS[m.group(2)],
        "source_url": source_url, "evidence_path": evidence_path,
        "evidence_sha256": sha(body), "evidence_excerpt": m.group(0),
        "headline": article.get("headline"),
        "vendor_fiscal_period_end_label": target["fiscal_period_end"],
        "vendor_reported_date": target["reported_date"],
        "vendor_report_time_raw": target.get("report_time_raw"),
        "vendor_period_label_matches_actual": target["fiscal_period_end"] == actual,
        "source_class": "issuer_original_press_release",
        "timestamp_note": "datePublished is date-only; dateModified, conference-call time, and provider post-market hint are not original publication timestamps.",
        "in_study_window": bool(target["in_study_window"]),
    }


def candidates(target):
    d = target["reported_date"]
    year, month, day = d.split("-")
    q = {"12": "first", "03": "second", "06": "third", "09": "fourth"}[target["fiscal_period_end"][5:7]]
    base = "https://www.apple.com/newsroom/%s/%s/" % (year, month)
    if int(year) <= 2016:
        title = "Apple-Reports-%s-Quarter-Results" % q.title()
        if q == "first":
            title = "Apple-Reports-Record-First-Quarter-Results"
        return [base + day + title + "/"]
    return [base + "apple-reports-%s-quarter-results/" % q]


def fetch(url):
    key = sha(url.encode())[:16]
    mp = RAW / (key + ".meta.json")
    hp = RAW / (key + ".html")
    if mp.exists() and hp.exists():
        meta = json.loads(mp.read_text()); body = hp.read_bytes()
        if sha(body) != meta["sha256"]:
            raise RuntimeError("Cached checksum changed: " + str(hp))
        return meta, body, hp
    stop = HERE / "SOURCE_STOP.json"
    if stop.exists() and json.loads(stop.read_text()).get("utc_date") == dt.datetime.now(dt.timezone.utc).date().isoformat():
        raise RuntimeError("Apple source already rate limited today")
    time.sleep(1.1)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        body = r.read(3_000_001)
        status = r.status; headers = r.headers; final_url = r.url
    except urllib.error.HTTPError as e:
        body = e.read(3_000_001)
        status = e.code; headers = e.headers; final_url = e.url
    if len(body) > 3_000_000:
        raise RuntimeError("Unexpected oversized response")
    meta = {"source_url": url, "final_url": final_url, "http_status": status,
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "content_type": headers.get("Content-Type"), "retry_after": headers.get("Retry-After"),
            "sha256": sha(body), "size": len(body), "public_unauthenticated": True}
    hp.write_bytes(body); dump(mp, meta)
    if status == 429:
        dump(stop, {"utc_date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
                    "source_url": url, "http_status": status, "retry_after": meta["retry_after"]})
        raise RuntimeError("Apple returned 429; stopping source today")
    return meta, body, hp


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--only"); args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    targets = [x for x in json.loads(VENDOR.read_text()) if x["reported_date"] >= "2015-01-01"]
    overrides_path = HERE / "URL_OVERRIDES.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    existing_path = HERE / "events.json"
    verified = {x["release_date"]: x for x in json.loads(existing_path.read_text())} if existing_path.exists() else {}
    manifest = []
    for target in sorted(targets, key=lambda t: t["reported_date"]):
        day = target["reported_date"]
        if args.only and day not in args.only.split(","):
            continue
        old_html = OLD / ("AAPL_" + day + ".html")
        if old_html.exists():
            old_meta = json.loads(old_html.with_suffix(".meta.json").read_text())
            body = old_html.read_bytes()
            if sha(body) != old_meta["sha256"]:
                raise RuntimeError("Old source checksum mismatch")
            source = old_meta["source"]
            hp = RAW / (sha(source.encode())[:16] + ".html")
            hp.write_bytes(body)
            meta = dict(old_meta, source_url=source, imported_from=str(old_html.relative_to(ROOT)))
            dump(hp.with_suffix(".meta.json"), meta)
            options = [(meta, body, hp)]
        else:
            options = []
        urls = overrides.get(day, candidates(target)) if not options else []
        if isinstance(urls, str):
            urls = [urls]
        for url in urls:
            options.append(fetch(url))
        for meta, body, hp in options:
            attempt = {"target_date": day, "source_url": meta["source_url"],
                       "http_status": meta["http_status"], "evidence_path": str(hp.relative_to(ROOT)),
                       "sha256": sha(body), "retrieved_at": meta["retrieved_at"]}
            if meta["http_status"] == 200:
                try:
                    event = parse_html(body, target, meta["final_url"], str(hp.relative_to(ROOT)))
                    verified[day] = event; attempt["verified"] = True
                except ValueError as e:
                    attempt["verified"] = False; attempt["error"] = str(e)
            else:
                attempt["verified"] = False; attempt["error"] = "HTTP status is not 200"
            manifest.append(attempt)
        dump(existing_path, [verified[k] for k in sorted(verified)])
        print(day, "verified" if day in verified else manifest[-1].get("error"), flush=True)
    # Include all cached attempts; preserve prior failed-URL provenance separately.
    history_path = HERE / "manifest.json"
    old = json.loads(history_path.read_text()) if history_path.exists() else []
    all_attempts = {(x["target_date"], x["source_url"]): x for x in old + manifest}
    dump(history_path, list(all_attempts.values()))
    pending = [x for x in targets if x["reported_date"] not in verified]
    dump(HERE / "PENDING.json", pending)
    print("VERIFIED", len(verified), "PENDING", len(pending), flush=True)


if __name__ == "__main__":
    main()
