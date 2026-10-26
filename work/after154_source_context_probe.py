"""Offline statement-label probe for next source batch; no monetary parsing."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
FEASIBILITY="work/after154_remaining_source_feasibility.json"
FEASIBILITY_SHA="e3b25e067a4157ded4aba9bb980fd0c1056c6eb51e343f50c2a1151818f06a4c"
OUT="work/after154_source_context_probe.json"
PDF_PLAN="outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json"
MONTH=r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
NI=re.compile(r"\bnet\s+(?:income|earnings|loss)\b",re.I)
ASSETS=re.compile(r"\btotal\s+assets\b",re.I)
TITLE=re.compile(r"(?:consolidated|condensed).{0,70}(?:statements?.{0,15}(?:income|operations|earnings)|balance\s+sheets?)",re.I)
TITLE2=re.compile(r"(?:consolidated|condensed)|(?:statements?.{0,15}(?:income|operations|earnings)|balance\s+sheets?)",re.I)
METADATA=re.compile(r"^(?:(?:for|the|and|as|of|at|unaudited|audited|united|states|u\.s\.|gaap|years?|months?|three|six|nine|twelve|ended|end|fiscal|quarter|q4|q3|december|august|january|february|march|april|may|june|july|september|october|november|thousands|millions|euro|euros|dollars|except|share|shares|per|data|amounts|in)|\d{1,4}|[\s(),:/.*$€-])+$",re.I)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def norm(value):return re.sub(r"\s+"," ",value).strip()
def masked(value):return re.sub(r"[-+]?\d[\d,.\s]*(?:%)?","[number]",norm(value))


def main():
    pins={}
    def read(name,expected=None):
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if path.is_symlink() or "CloudStorage" in path.parts:raise ValueError("Nonlocal path")
        raw=path.read_bytes();actual=sha(raw)
        if expected and actual!=expected:raise ValueError("Changed source "+name)
        pins[name]=actual
        return raw
    feas=json.loads(read(FEASIBILITY,FEASIBILITY_SHA))
    runtime=json.loads(read(PDF_PLAN,"cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50"))
    for name,digest in runtime["pdf_runtime_hashes"].items():read(name,digest)
    sys.path.append(runtime["pdf_runtime"]["bundle_site_packages"])
    from pypdf import PdfReader
    documents=[]
    for spec in feas["verified_unscanned_pdf_sources"]:
        source=spec["source"];raw=read(source["path"],source["sha256"])
        if not raw.startswith(b"%PDF-"):raise ValueError("Expected original PDF bytes")
        receipts=[]
        for ref in source["receipt_candidates"]:
            receipt=json.loads(read(ref["path"],ref["sha256"]))
            if (receipt.get("url")==source["url"] and receipt.get("sha256")==source["sha256"]
                    and receipt.get("status",receipt.get("http_status"))==200):
                receipts.append(dict(ref,url_verified=True,hash_verified=True,http_200_verified=True))
        reader=PdfReader(io.BytesIO(raw));pages=[]
        date=datetime.strptime(spec["publication_date"],"%Y-%m-%d")
        release_pattern=re.compile(re.escape(date.strftime("%B"))+r"\s+"+str(date.day)+r",?\s+"+str(date.year),re.I)
        release_lines=[]
        for index,page in enumerate(reader.pages):
            text=page.extract_text(extraction_mode="layout") or ""
            lines=text.splitlines()
            labels=[];metadata=[]
            for line_no,line in enumerate(lines,1):
                line=norm(line)
                if not line:continue
                if TITLE2.search(line):metadata.append(dict(line=line_no,text=masked(line)))
                elif METADATA.fullmatch(line) and (re.search(r"(?:19|20)\d{2}",line) or re.search(MONTH,line,re.I)):
                    metadata.append(dict(line=line_no,text=line))
                elif "in millions" in line.lower() or "in thousands" in line.lower():
                    metadata.append(dict(line=line_no,text=masked(line)))
                if NI.search(line) or ASSETS.search(line):
                    labels.append(dict(line=line_no,text=masked(line),net_income_label=bool(NI.search(line)),total_assets_label=bool(ASSETS.search(line))))
                if index==0 and release_pattern.search(line):
                    release_lines.append(dict(page=1,line=line_no,text=masked(line),
                                              date=spec["publication_date"],matched_catalog_date=True))
            if labels or metadata:
                pages.append(dict(page=index+1,labels=labels,metadata=metadata,
                    text_sha256=sha(text.encode()),income_title_detected=bool(re.search(r"(?:consolidated|condensed).{0,70}statements?.{0,15}(?:income|operations|earnings)",norm(text),re.I)),
                    balance_title_detected=bool(re.search(r"(?:consolidated|condensed).{0,70}balance\s+sheets?",norm(text),re.I))))
        documents.append(dict(symbol=spec["symbol"],fiscal_year=spec["fiscal_year"],
            source=source,publication_date=spec["publication_date"],release_date_lines=release_lines,
            receipts=receipts,page_count=len(reader.pages),pages=pages,financial_values_parsed=0,
            official_actual_period_end=spec["official_actual_period_end"],visual_review_pending=True))
    ge=next(e for e in feas["full_body_label_leads"] if e["symbol"]=="GE")
    data=json.loads(read(ge["source"]["path"],ge["source"]["sha256"]))
    ge_review=dict(symbol="GE",fiscal_year=2012,source=ge["source"],
        original_bytes_available=data.get("original_bytes_available"),
        source_representation=data.get("source_representation"),
        inherited_catalog_full_body_claim=ge["catalog_claims_issuer_full_body"],
        disposition="hold_index_excerpt_not_original_report",
        correction="The raw JSON explicitly says original_bytes_available=false; census full-body wording must not be promoted into source admission.")
    read("work/after154_source_context_probe.py")
    result=dict(stage="source_statement_context_probe_no_amounts",created_utc=datetime.now(timezone.utc).isoformat(),
        documents=documents,ge_review=ge_review,input_hashes=pins,
        financial_values_parsed=0,ratios_computed=0,new_factors_computed=0,returns_computed=0,new_strategy_trials=0,
        limitations=["Label and header scan only; cross-period column geometry, accounting comparability and visual confirmation still required.",
                     "No first-public-version or currency ISO certification. Existing holdings and returns were not read."])
    for name,digest in list(pins.items()):read(name,digest)
    with (ROOT/OUT).open("x") as stream:json.dump(result,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n")
    print(json.dumps(dict(path=OUT,sha256=sha((ROOT/OUT).read_bytes()),
        docs=[dict(symbol=d["symbol"],year=d["fiscal_year"],pages=d["page_count"],
                   financial_pages=[p["page"] for p in d["pages"] if p["labels"]],
                   release_date_matches=len(d["release_date_lines"]),verified_receipts=len(d["receipts"])) for d in documents],
        ge_disposition=ge_review["disposition"])))

if __name__=="__main__":main()
