"""ACN semantic source context and bounded PDF visual artifacts, no values."""
from bisect import bisect_right
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT="work/after154_source_context_details.json"
ART=ROOT/"outputs/after154_source_context_pdf"
PARENT="work/after154_source_context_probe.json"
PARENT_SHA="f4e1fe0e63441eb70f71fbd7dd9840f4a17e6ddb6b4a6fa4d971e2a2ac8aaa2b"
CAL="outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json"
PDF_PLAN="outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json"


def sha(raw):return hashlib.sha256(raw).hexdigest()
def norm(text):return re.sub(r"\s+"," ",text).strip()
def redact(text):return re.sub(r"\d[\d,.]*","[number]",norm(text))
def keep_header(line):
    t=norm(line)
    return bool(re.fullmatch(r"(?:(?:Three|Months|Year|Years|Ended|August|Unaudited|unaudited)|31|(?:19|20)\d{2}|[\s,();*])+",t))


def main():
    pins={}
    def read(name,expected=None):
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if p.is_symlink() or "CloudStorage" in p.parts:raise ValueError("Nonlocal source")
        raw=p.read_bytes()
        if expected and sha(raw)!=expected:raise ValueError("Changed source")
        pins[name]=sha(raw);return raw
    parent=json.loads(read(PARENT,PARENT_SHA))
    runtime=json.loads(read(PDF_PLAN,"cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50"))
    for name,digest in runtime["pdf_runtime_hashes"].items():read(name,digest)
    sys.path.append(runtime["pdf_runtime"]["bundle_site_packages"])
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    sessions=json.loads(read(CAL))["sessions"]
    if sessions!=sorted(set(sessions)):raise ValueError("Calendar order")
    census=json.loads(read("outputs/monthly_financial_coverage_census_v1/CENSUS.json",
        "de95906ef3d752890d28027a9b52cfd7decc7aeb1bc0f09f2785f1826a3500fc"))
    rows=[]
    for doc in parent["documents"]:
        source=doc["source"];raw=read(source["path"],source["sha256"])
        reader=PdfReader(io.BytesIO(raw))
        texts=[page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
        kind=doc["symbol"];fy=doc["fiscal_year"]
        if kind=="ACN":
            inc=[i for i,t in enumerate(texts) if "consolidated income statements" in norm(t).lower()]
            bs=[i for i,t in enumerate(texts) if "consolidated balance sheets" in norm(t).lower()]
            if len(inc)!=1 or len(bs)!=1:raise ValueError("Unique ACN formal statements required")
            income,bal=texts[inc[0]].splitlines(),texts[bs[0]].splitlines()
            income_label=[dict(line=i+1,label="NET INCOME") for i,l in enumerate(income)
                          if re.match(r"^\s*NET INCOME\s+[-$()\d]",l)]
            asset_label=[dict(line=i+1,label="TOTAL ASSETS") for i,l in enumerate(bal)
                         if re.match(r"^\s*TOTAL ASSETS\s+[-$()\d]",l)]
            if len(income_label)!=1 or len(asset_label)!=1:raise ValueError("Exact consolidated total rows required")
            income_header=[dict(line=i+1,text=norm(l)) for i,l in enumerate(income[:12]) if keep_header(l)]
            asset_header=[dict(line=i+1,text=norm(l)) for i,l in enumerate(bal[:12]) if keep_header(l)]
            monetary_units=[dict(page=p+1,line=i+1,text=norm(l)) for p in (inc[0],bs[0])
                            for i,l in enumerate(texts[p].splitlines()[:12])
                            if re.fullmatch(r"\(In thousands of U\.S\. dollars(?:, except share and per share amounts)?\)",norm(l),re.I)]
            if len(monetary_units)!=2:raise ValueError("Common disclosed ACN units")
            if not all(str(y) in norm(texts[inc[0]])[:1800] and str(y) in norm(texts[bs[0]])[:1800] for y in (fy,fy-1)):
                raise ValueError("Current/prior year header not found")
            annual_word=bool(re.search(r"years?\s+ended|year\s+ended",norm(texts[inc[0]])[:1800],re.I))
            if not annual_word:raise ValueError("Annual income domain required")
            release=date.fromisoformat(doc["publication_date"])
            date_expression=re.compile(r"\b(?:Sept(?:ember)?\.?)\s+"+str(release.day)+r",?\s+"+str(release.year)+r"\b",re.I)
            matches=[dict(page=1,line=i+1,date=doc["publication_date"],surrounding_text_numbers_redacted=redact(l))
                     for i,l in enumerate(texts[0].splitlines()) if date_expression.search(norm(l))]
            # A line break may split the day/year; evidence remains first-page metadata.
            if not matches and date_expression.search(norm(texts[0])):
                matches=[dict(page=1,date=doc["publication_date"],location="normalized first page",line=None)]
            next_index=bisect_right(sessions,doc["publication_date"])
            known=sessions[next_index] if sessions[0]<=doc["publication_date"]<sessions[-1] else None
            row=dict(symbol=kind,fiscal_year=fy,source=source,
                income_page=inc[0]+1,balance_page=bs[0]+1,
                income_label=income_label[0],asset_label=asset_label[0],
                income_header=income_header,asset_header=asset_header,units=monetary_units,
                annual_current_period_end=str(fy)+"-08-31",prior_assets_end=str(fy-1)+"-08-31",
                income_start_not_quoted=True,proposed_start_assumption=str(fy-1)+"-09-01",
                net_income_definition="Total consolidated NET INCOME before attributable-to-NCI and Accenture-plc lines; not diluted-EPS numerator or adjusted reconciliation.",
                publication_date=doc["publication_date"],publication_date_matches=matches,
                first_session_after_publication=known,exact_publication_clock_unknown=True,
                receipts_verified=doc["receipts"],semantic_context_located=True,
                values_or_numeric_column_geometry_verified=False,
                unresolved=["Annual domain versus quarterly columns must be independently mapped before extraction.",
                            "Individual financial-cell amounts, signs, column ownership and units must be independently verified.",
                            "Period start proposed from consecutive annual endpoints is an assumption, not a quoted start date.",
                            "First-public version and complete restatement/correction history are not certified."])
            render_pages=(inc[0],bs[0]) if fy in (2015,2019,2025) else ()
        else:
            links=[]
            for page in reader.pages:
                for annotation in page.get("/Annots",[]):
                    value=annotation.get_object();action=value.get("/A")
                    if action and action.get("/URI"):
                        url=str(action["/URI"])
                        if "financial" in url.lower() or "quarter" in url.lower() or "asml.com" in url.lower():
                            links.append(url)
            row=dict(symbol=kind,fiscal_year=fy,source=source,
                publication_date=doc["publication_date"],publication_date_matches=doc["release_date_lines"],
                receipts_verified=doc["receipts"],
                all_pages_scanned=len(texts),total_assets_label_found=any(re.search(r"\btotal\s+assets\b",t,re.I) for t in texts),
                statement_attachment_links=sorted(set(links)),
                semantic_context_located=False,
                disposition="Hold: four-page press release has rounded headline NI; no total-assets table. It explicitly refers to separate quarterly US GAAP statements.",
                values_or_numeric_column_geometry_verified=False)
            render_pages=(0,2)
        images=[]
        if render_pages:
            ART.mkdir(parents=True,exist_ok=True)
            pdf=pdfium.PdfDocument(raw)
            for p in render_pages:
                path=ART/(kind+"_"+str(fy)+"_p"+str(p+1)+".png")
                if path.exists():raise ValueError("Preserve existing PDF render")
                page=pdf[p];image=page.render(scale=1.35).to_pil()
                image.save(path);images.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes())))
                page.close()
            pdf.close()
        row["visual_artifacts"]=images
        rows.append(row)
    catalogs={}
    for symbol in ("ACN","ASML","GE"):
        items=[x for x in census["annual_documents"] if x["symbol"]==symbol]
        projection=[]
        for item in items:
            raw=read(item["evidence_path"],item["evidence_sha256"])
            is_index=False
            if raw.lstrip().startswith(b"{"):
                metadata=json.loads(raw)
                is_index=metadata.get("original_bytes_available") is False
            projection.append({k:item.get(k) for k in ("event_id","symbol","fiscal_year","publication_date",
                    "available_session","official_actual_period_end","evidence_path","evidence_sha256","source_url","source_kind")}|
                    dict(raw_explicitly_index_not_original=is_index))
        catalogs[symbol]=projection
    read("work/after154_source_context_details.py")
    output=dict(stage="bounded_source_semantic_context_no_financial_admission",created_utc=datetime.now(timezone.utc).isoformat(),
        documents=rows,full_catalogs=catalogs,ge_review=parent["ge_review"],
        next_candidate_batch=dict(symbols=["ACN"],events=[dict(fiscal_year=r["fiscal_year"],publication_date=r["publication_date"])
            for r in rows if r["symbol"]=="ACN"],reason="First alphabetical remaining issuer with all nine original reports and matching formal income/balance statement contexts; chosen without inspecting prices, holdings or returns."),
        later_queue=["MCD","NKE","PG","XOM"],later_queue_not_semantically_reviewed_here=True,
        input_hashes=pins,financial_values_extracted=0,ratios_computed=0,new_factors_computed=0,
        orders_or_returns_computed=0,new_admissions=0,new_strategy_trials=0)
    for name,digest in list(pins.items()):read(name,digest)
    with (ROOT/OUT).open("x") as stream:json.dump(output,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n")
    print(json.dumps(dict(path=OUT,sha256=sha((ROOT/OUT).read_bytes()),
        contexts=[dict(symbol=r["symbol"],year=r["fiscal_year"],located=r["semantic_context_located"],
                       release_matches=len(r["publication_date_matches"])) for r in rows],
        images=[img["path"] for r in rows for img in r["visual_artifacts"]])))

if __name__=="__main__":main()
