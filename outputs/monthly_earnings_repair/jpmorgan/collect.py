"""JPMorgan issuer-original quarterly release evidence; no vendor API calls.

Calendar-day evidence is deliberately not promoted into a publication instant.
The official events archive is only a cross-check when the same original PDF
says its conference call is today. Scheduled release times and CMS dates are
never actual publication evidence. Cache bytes and source hashes are retained.
"""
import argparse
import calendar
import datetime as dt
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / "raw"
ORIGIN = "https://www.jpmorganchase.com"
FEED = ORIGIN + "/services/json/v1/investor-relations/quarterly-earnings.json"
EVENT_FEED = ORIGIN + "/services/json/v1/investor-relations/ir-events.json"
PDF_PYTHON = "/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
CUTOFF = "2026-09-24"
QWORDS = {1: "first", 2: "second", 3: "third", 4: "fourth"}
MONTH_NAMES = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = r"(?:" + MONTH_NAMES + r")\s+\d{1,2}\s*,\s*\d{4}"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(body):
    return hashlib.sha256(body).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def fetch(url, stem, suffix):
    path, meta_path = RAW / (stem + suffix), RAW / (stem + ".meta.json")
    if path.exists() and meta_path.exists():
        body = path.read_bytes(); meta = json.loads(meta_path.read_text())
        if meta.get("url") != url or meta.get("sha256") != sha(body):
            raise ValueError("Cache source/checksum mismatch: " + str(path))
        return body, path, meta
    stop = HERE / "RATE_LIMIT.json"
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    if stop.exists() and json.loads(stop.read_text())["utc_date"] == today:
        raise RuntimeError("Source rate-limited today; no further requests")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (public earnings research)"})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            body = response.read()
            meta = dict(url=url, final_url=response.geturl(), status=response.status,
                sha256=sha(body), retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                public_unauthenticated=True, bytes=len(body))
    except urllib.error.HTTPError as error:
        if error.code == 429:
            dump(stop, dict(utc_date=today, status=429, url=url,
                retry_after=error.headers.get("Retry-After"), policy="Stop this source for the rest of this UTC day"))
        raise
    if suffix == ".pdf" and not body.startswith(b"%PDF"):
        raise ValueError("Expected PDF, received other content")
    path.write_bytes(body); dump(meta_path, meta)
    time.sleep(.15)
    return body, path, meta


def pdf_text(path):
    target = path.with_suffix(".txt")
    sidecar = path.with_suffix(".text.meta.json")
    pdf_hash = sha(path.read_bytes())
    if target.exists() and sidecar.exists():
        info = json.loads(sidecar.read_text())
        if info["source_sha256"] == pdf_hash and info["text_sha256"] == sha(target.read_bytes()):
            return target.read_text()
    code = 'import sys; from pypdf import PdfReader; print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
    text = subprocess.check_output([PDF_PYTHON, "-c", code, str(path)], text=True)
    target.write_text(text)
    dump(sidecar, dict(source_sha256=pdf_hash, text_sha256=sha(target.read_bytes()), parser="bundled pypdf text extraction"))
    return text


def clean(text):
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ").replace("–", "-").replace("—", "-"))


def iso_date(value):
    return dt.datetime.strptime(re.sub(r"\s+", " ", value).replace(" ,", ","), "%B %d, %Y").date().isoformat()


def evidence(path, url, excerpt=None):
    result = dict(evidence_path=rel(path), sha256=sha(path.read_bytes()), source_url=url)
    if excerpt is not None:
        result["excerpt"] = excerpt
    return result


def parse_release(text, year, quarter, matching_events, event_evidence):
    text = clean(text)
    identity = re.search(r"\b" + QWORDS[quarter] + r"\s*-?\s*quarter\s+(?:and\s+full.year\s+)?" + str(year) + r"\b", text, re.I)
    if not identity:
        raise ValueError("Original release does not confirm expected fiscal year and quarter")
    location = re.search(r"New\s+York\s*,?\s*(?:NY\s+\d[\d-]*|(?:" + MONTH_NAMES + r"))", text, re.I)
    if not location:
        raise ValueError("Original New York address/dateline absent; timezone mapping needs review")
    direct = re.search(r"New\s+York\s*,?\s+(" + DATE + r")", text, re.I)
    call_today = re.search(r"(?:host|hold)\s+(?:a\s+)?conference\s+call\s+today\s*,?\s*(?:on\s+)?(" + DATE + r")", text, re.I)
    today = re.search(r".{0,70}conference\s+call\s+today.{0,220}", text, re.I)
    date_evidence = []
    if direct or call_today:
        match = direct or call_today
        release_date = iso_date(match.group(1))
        date_evidence.append(dict(kind="original_release_date_in_body", excerpt=match.group(0)))
        date_basis = "Original PDF New York dateline" if direct else "Original PDF states conference call today followed by explicit calendar date; not conference-call clock time"
    else:
        if not today:
            raise ValueError("No actual date or original-PDF conference-call-today context")
        dates = set()
        for item in matching_events:
            match = re.fullmatch(r"\w{3} (\w{3}) (\d{2}) \d{2}:\d{2}:\d{2} (?:EST|EDT) (\d{4})", item.get("date", ""))
            if match:
                dates.add(dt.datetime.strptime(" ".join(match.groups()), "%b %d %Y").date().isoformat())
        if len(dates) != 1:
            raise ValueError("Need one unambiguous official same-PDF quarterly-event date: " + repr(sorted(dates)))
        release_date = next(iter(dates))
        date_evidence.append(dict(kind="original_release_today_context", excerpt=today.group(0)))
        date_evidence.append(dict(event_evidence, kind="official_event_day_same_original_pdf", events=matching_events))
        date_basis = "Original PDF says conference call is today; official same-PDF quarterly event supplies that day, corroborating actual release date. No release instant inferred."
    if release_date <= "%04d-%02d-%02d" % (year, quarter * 3, calendar.monthrange(year, quarter * 3)[1]):
        raise ValueError("Release date not after actual quarter end")
    return dict(release_date=release_date, identity_excerpt=identity.group(0),
        location_excerpt=location.group(0), release_date_basis=date_basis,
        release_date_evidence=date_evidence)


def find_period(text, year, quarter):
    month = quarter * 3
    day = calendar.monthrange(year, month)[1]
    name = calendar.month_name[month]
    t = clean(text)
    match = re.search(r"\b" + name + r"\s+" + str(day) + r"\s*,?\s*" + str(year) + r"\b", t, re.I)
    if not match:
        return None
    return dict(date="%04d-%02d-%02d" % (year, month, day),
                excerpt=t[max(0, match.start() - 95):match.end() + 100],
                basis="Explicit current-quarter calendar end in original quarter-identified issuer financial document, not provider fiscal labels")


def main(start_year=2011):
    RAW.mkdir(parents=True, exist_ok=True)
    fb, fp, fm = fetch(FEED, "quarterly_feed", ".json")
    eb, epath, em = fetch(EVENT_FEED, "events_feed", ".json")
    feed = json.loads(fb); calendar_events = json.loads(eb)["items"]
    events, pending = [], []
    for item in sorted(feed["items"], key=lambda z:(int(z["year"]), int(z["quarter"][0]))):
        year, quarter = int(item["year"]), int(item["quarter"][0])
        if year < start_year - 1 or (year == start_year - 1 and quarter != 4) or year > 2026:
            continue
        label = "%dQ%d" % (year, quarter)
        info = dict(fiscal_year=year, fiscal_quarter=quarter, label=label)
        try:
            doc = item["docs"]["pressRelease"]
            url = urljoin(ORIGIN, doc["link"])
            body, path, meta = fetch(url, "%d_%s" % (year, item["quarter"]), ".pdf")
            text = pdf_text(path)
            matched = [z for z in calendar_events
                if urljoin(ORIGIN, z.get("quarterlyEarningsDocuments", {}).get("pressRelease", {}).get("link", "")) == url]
            parsed = parse_release(text, year, quarter, matched, evidence(epath, EVENT_FEED))
            if not "%d-01-01" % start_year <= parsed["release_date"] <= CUTOFF:
                continue
            period = find_period(text, year, quarter)
            period_path, period_url = path, url
            if not period:
                supplement = item["docs"]["supplement"]
                period_url = urljoin(ORIGIN, supplement["link"])
                _, period_path, _ = fetch(period_url, "%d_%s_supplement" % (year, item["quarter"]), ".pdf")
                period = find_period(pdf_text(period_path), year, quarter)
            if not period:
                raise ValueError("Actual fiscal end date missing in original and financial supplement")
            extra = parsed.pop("release_date_evidence")
            for proof in extra:
                if "evidence_path" not in proof:
                    proof.update(evidence(path, url))
            row = dict(ticker="JPM", release_date=parsed["release_date"], publication_timestamp=None,
                precision="date_only", publication_timezone="America/New_York",
                publication_timezone_basis="Conservative issuer-local calendar-date research mapping: original release observes New York, NY and Eastern conference-call context; no exact publication instant or CMS timezone is assumed.",
                publication_timezone_evidence=dict(kind="issuer_local_calendar_date_policy", observed_dateline=parsed["location_excerpt"],
                    geographic_mapping="New York, NY -> America/New_York", evidence_path=rel(path), sha256=sha(body),
                    cms_timezone_observed=False, exact_publication_time_inferred=False),
                first_public_time_verified=False, official_date_published_raw=parsed["release_date"],
                fiscal_period_end_actual=period["date"], fiscal_year=year, fiscal_quarter=quarter,
                source_url=url, source_class="issuer_original_press_release", evidence_path=rel(path), evidence_sha256=sha(body),
                evidence_excerpt=parsed["identity_excerpt"], headline=doc["title"],
                release_date_basis=parsed["release_date_basis"], release_date_evidence=extra,
                fiscal_period_evidence=dict(evidence(period_path, period_url, period["excerpt"]), basis=period["basis"]),
                official_archive_evidence=dict(evidence(fp, FEED), quarter_label=label, press_release_link=doc["link"]),
                timestamp_note="No PDF CreationDate/ModDate, website migration date, pre-announced time, call time, or replay time is treated as actual publication timestamp.",
                historical_limitation="Current issuer archive; not a contemporaneously captured immutable historical news feed.",
                in_study_window=parsed["release_date"] >= "2016-01-01")
            secondary = [(row["fiscal_period_evidence"], "actual_fiscal_period_end"),
                         (row["official_archive_evidence"], "official_quarter_identity_and_original_pdf_link")]
            secondary.extend((proof, "release_date_official_event_cross_confirmation") for proof in extra)
            row["additional_evidence"] = []
            for proof, role in secondary:
                if proof["evidence_path"] != row["evidence_path"]:
                    entry = dict(evidence_path=proof["evidence_path"], sha256=proof["sha256"], role=role)
                    if entry not in row["additional_evidence"]:
                        row["additional_evidence"].append(entry)
            events.append(row)
            print(label, parsed["release_date"], "ready", flush=True)
        except Exception as error:
            info.update(status="pending_review", reason=str(error))
            pending.append(info)
            print(label, "pending", str(error), flush=True)
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                break
            if isinstance(error, RuntimeError) and "rate-limited" in str(error):
                break
    events.sort(key=lambda z:z["release_date"])
    dump(HERE / "events.json", events); dump(HERE / "pending.json", pending)
    ids = {x["fiscal_year"] * 4 + x["fiscal_quarter"] - 1 for x in events}
    all_ids = range((start_year - 1) * 4 + 3, 2026 * 4 + 2)
    gaps = ["%dQ%d" % (i // 4, i % 4 + 1) for i in all_ids if i not in ids]
    dump(HERE / "STATUS.json", dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), ticker="JPM",
        requested_release_window=["%d-01-01" % start_year,CUTOFF], verified_events=len(events), pending_count=len(pending),
        missing_quarters=gaps, earliest_release=min((z["release_date"] for z in events),default=None),
        latest_release=max((z["release_date"] for z in events),default=None),
        all_events_date_only=True, approximate_times_inferred=False,
        actual_date_not_scheduled_date=True, source="Official JPMorgan original release PDFs and linked issuer archive",
        unrelated_accounts_or_prices_modified=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--start-year", type=int, default=2011)
    main(parser.parse_args().start_year)
