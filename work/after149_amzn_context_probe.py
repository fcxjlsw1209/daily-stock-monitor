"""Read-only Amazon statement-context probe; never emits financial values."""
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "outputs/monthly_financial_coverage_census_v1/CENSUS.json"
TARGET = re.compile(
    r"^(?:net income(?:\s*\(loss\))?|net loss|stock[ -]based compensation|"
    r"share[ -]based compensation|total assets|"
    r"net cash (?:provided by|used in)(?:\s*\(used in\))? operating activities(?:\s*\(\d+\))?|"
    r"adjustments to reconcile net income.*|operating activities)$", re.I)
TITLE = re.compile(r"consolidated.*(?:cash flows|balance sheets|operations)", re.I)
UNIT = re.compile(r"^\(?in (?:millions|thousands)(?:,? except .*?)?\)?$", re.I)
HEADER = re.compile(
    r"^(?:(?:three|twelve|six|nine|3|12) months ended|year(?:s)? ended|"
    r"december\s*31,?|unaudited|\(?unaudited\)?|20[0-9]{2}|19[0-9]{2}|"
    r"(?:december\s*31,?\s*)(?:20[0-9]{2}|19[0-9]{2})|"
    r"(?:three|twelve) months ended december\s*31,?|"
    r"year(?:s)? ended december\s*31,?)$", re.I)


def norm(text):
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().rstrip(":")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inspect(event):
    path = ROOT / event["evidence_path"]
    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
    raw = path.read_bytes()
    assert digest(raw) == event["evidence_sha256"]
    sidecar = path.with_suffix(".meta.json")
    if sidecar.is_file():
        receipt_raw = sidecar.read_bytes()
        receipt = json.loads(receipt_raw)
        receipt_path = sidecar.relative_to(ROOT).as_posix()
    else:
        manifest = ROOT / "outputs/monthly_earnings_repair/amazon/manifest.json"
        receipt_raw = manifest.read_bytes()
        receipt = json.loads(receipt_raw)[event["source_url"]]
        receipt_path = manifest.relative_to(ROOT).as_posix()
    assert receipt["sha256"] == event["evidence_sha256"]
    assert receipt["http_status"] == 200
    assert receipt["url"] == event["source_url"]
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    tables, rejects = [], []
    for ti, table in enumerate(soup.find_all("table")):
        try:
            rows, grid, physical = table_grid(table)
        except ValueError as exc:
            rejects.append({"table": ti, "reason": type(exc).__name__})
            continue
        targets = [(r, p, c) for (r, p), c in physical.items()
                   if TARGET.fullmatch(norm(c["text"]))]
        if not targets:
            continue
        # Header metadata is emitted only before the first financial row.
        first_target = min(r for r, _, _ in targets)
        headers = []
        for (r, p), cell in physical.items():
            text = norm(cell["text"])
            if r < first_target and (
                    HEADER.fullmatch(text) or UNIT.fullmatch(text) or
                    (len(text) < 150 and TITLE.search(text))):
                headers.append({"row": r, "physical_cell": p,
                                "columns": cell["columns"], "text": text})
        target_rows = []
        for r, p, cell in targets:
            coords = [{"physical_cell": q, "columns": c["columns"],
                       "rowspan": c["rowspan"], "colspan": len(c["columns"]),
                       "nonmonetary_role": (
                           {"": "empty", "$": "currency", "(": "open_parenthesis",
                            ")": "close_parenthesis"}.get(norm(c["text"]), "opaque"))}
                      for (rr, q), c in physical.items() if rr == r]
            target_rows.append({"row": r, "label_cell": p,
                                "label": norm(cell["text"]), "cells": coords})
        tables.append({"table": ti, "rows": len(rows),
                       "currency_dollar_symbol_present": "$" in table.get_text(),
                       "other_currency_tokens": sorted(set(re.findall(
                           r"€|£|¥|\b(?:EUR|GBP|JPY|CAD|HKD|AUD)\b",
                           table.get_text(" ", strip=True)))),
                       "headers": headers, "target_rows": target_rows})
    text = norm(soup.get_text(" ", strip=True))
    # Extract only short explicit issuer/security anchors, never financial prose.
    identity = sorted(set(m.group(0) for m in re.finditer(
        r"Amazon\.com,?\s+Inc\.?|NASDAQ\s*:\s*AMZN|Nasdaq\s+Global\s+Select\s+Market",
        text, re.I)))
    flags = sorted(set(m.group(0).lower() for m in re.finditer(
        r"\b(?:reclassif\w*|restat\w*|recast\w*|retrospect\w*|adopt\w*)\b", text, re.I)))
    contexts = []
    for match in re.finditer(
            r"\b(?:reclassif\w*|restat\w*|recast\w*|retrospect\w*|adopt\w*)\b", text, re.I):
        passage = text[max(0, match.start()-100):match.end()+240]
        # Monetary and other numeric values never leave the raw document reader.
        contexts.append(re.sub(r"\d[\d,.\s]*", "[number]", passage))
    return {k: event[k] for k in (
        "event_id", "fiscal_year", "publication_date", "available_session",
        "official_actual_period_end", "evidence_path", "evidence_sha256")} | {
        "tables": tables, "table_structure_rejections": rejects,
        "source_url": event["source_url"],
        "receipt": {"path": receipt_path, "sha256": digest(receipt_raw),
                    "http_status": receipt["http_status"],
                    "retrieved_at": receipt["retrieved_at"],
                    "source_url_and_raw_hash_match": True},
        "identity_anchors": identity, "accounting_review_terms": flags,
        "accounting_contexts_numbers_redacted": contexts,
        "financial_values_extracted": 0}


def main():
    census = json.loads(CENSUS.read_bytes())
    events = [e for e in census["annual_documents"] if
              e["symbol"] == "AMZN" and e["complete_text_label_lead"]]
    print(json.dumps({"census_sha256": digest(CENSUS.read_bytes()),
                      "events": [inspect(e) for e in events]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
