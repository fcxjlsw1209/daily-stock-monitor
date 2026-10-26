"""Frozen-source MSFT annual accounting-note extraction and boundary review.

No financial amounts are parsed or recalculated: annual-cell witnesses are
referenced from the previously verified extraction. Final dispositions require
explicit source-note review; absence of a keyword never produces a pass.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_msft_revenue_basis156"
FINAL = ROOT / "work/after155_msft_revenue_basis_review.json"
CODE = "work/after155_msft_revenue_basis_review.py"
INPUTS = {
    "policy": ("work/after155_revenue_chain_policy_review.json", "8564d0d052e4a7cfb89bd73c553b46c6ea831afacd384a5d69fe63c52732ce36"),
    "proposal": ("outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json", "0f5aa563ac7eac950b9594740d14111dfa8aac1608f5e73558d39baee6066aef"),
    "review": ("work/after155_revenue_extended_review.json", "f18d530b9cea67d0fce5d82f8910b8a7a20264813e058e296058c8e783a687bd"),
    "event_review": ("work/after155_event_source_independent_review.json", "4cf546abd7e9be42c2bd7e54114c6140c7c422dc8b9d388f09ae8bcf4e177d00"),
    "event_evidence": ("outputs/monthly_msft_event_evidence156/EVENT_EVIDENCE.json", None),
}
PATTERNS = {
    "basis_change": r"\b(?:adopt\w*|recast|restate\w*|reclassif\w*|accounting standards?|revenue recognition|accounting polic\w*)\b",
    "windows_deferral": r"\b(?:Windows 10|defer\w* revenue|revenue defer\w*)\b",
    "scope_change": r"\b(?:acquisit\w*|acquir\w*|dispos\w*|discontinu\w*|divest\w*|LinkedIn|Nuance|Activision)\b",
    "measurement": r"\b(?:GAAP|constant currency|foreign currenc\w*|useful lives|useful life|segment results|segment revenue)\b",
    "statement_heading": r"(?:INCOME STATEMENTS|CASH FLOWS STATEMENTS|STATEMENTS OF CASH FLOWS|SEGMENT REVENUE AND OPERATING INCOME)",
}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def local(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_relative_to(ROOT) and p.resolve() == p and "CloudStorage" not in p.parts, "Nonlocal source")
    return p


def raw(value):
    p = local(value)
    a = p.stat()
    data = p.read_bytes()
    b = p.stat()
    require((a.st_ino, a.st_mtime_ns, a.st_size) == (b.st_ino, b.st_mtime_ns, b.st_size), "Source mutated")
    return data


def sha(value):
    return hashlib.sha256(raw(value)).hexdigest()


def read(value):
    return json.loads(raw(value))


def dump_new(value, data):
    p = local(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def normalize(text):
    return " ".join(text.split())


def freeze():
    pins = {}
    for key, (name, expected) in INPUTS.items():
        observed = sha(name)
        require(expected is None or observed == expected, "Version mismatch: "+key)
        pins[key] = {"path": name, "sha256": observed}
    proposal = read(INPUTS["proposal"][0])
    sources = []
    for record in proposal["records"]:
        year = record["fiscal_year"]
        source = record["source"]
        require(source["symbol"] == "MSFT", "Wrong issuer")
        for part in ("raw", "receipt"):
            require(sha(source[part]["path"]) == source[part]["sha256"], "Raw source mismatch")
        sources.append({"fiscal_year": year, "path": source["raw"]["path"],
                        "sha256": source["raw"]["sha256"],
                        "receipt": source["receipt"],
                        "release_date": record["availability"]["release_date"],
                        "conservative_session": record["availability"]["first_conservative_session_after_release"]})
    require([r["fiscal_year"] for r in sources] == list(range(2015, 2027)), "Exact twelve reports")
    spec = {"stage":"FROZEN_MSFT_ANNUAL_NOTE_SOURCE_REVIEW", "created_utc":datetime.now(timezone.utc).isoformat(),
        "sources":sources, "inputs":pins, "code":{"path":CODE,"sha256":sha(CODE)},
        "scope":{"issuer":"MSFT","fiscal_years":[2015,2026],"adjacent_boundaries":11},
        "extraction_patterns":PATTERNS,
        "judgment_rule":"Manually review source paragraphs and preverified annual witnesses; no pass from keyword absence.",
        "time_rule":"Each boundary uses only its two original annual reports and their available notes; disposition cannot be known before later report's conservative close. Later discoveries never overwrite earlier source levels.",
        "excluded":["new financial extraction","growth","four-level chain construction","factor","training","returns","observer modifications"]}
    dump_new(OUT/"SPEC.json",spec)
    print(json.dumps({"spec_sha256":sha(OUT/"SPEC.json")}))


def validate(expected):
    require(sha(OUT/"SPEC.json") == expected, "Exact frozen spec required")
    spec = read(OUT/"SPEC.json")
    require(sha(CODE) == spec["code"]["sha256"], "Code changed after freeze")
    for item in list(spec["inputs"].values()) + spec["sources"]:
        require(sha(item["path"]) == item["sha256"], "Frozen source changed")
    for item in spec["sources"]:
        require(sha(item["receipt"]["path"]) == item["receipt"]["sha256"], "Receipt changed")
    return spec


def extract(expected):
    spec = validate(expected)
    events = {r["fiscal_year"]:r for r in read(INPUTS["event_evidence"][0]) if r["fiscal_quarter"] == 4}
    reports = []
    for source in spec["sources"]:
        year = source["fiscal_year"]
        text = raw(source["path"]).decode("utf-8-sig")
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        headings = []
        for heading in soup.find_all(re.compile("^h[1-6]$")):
            label = normalize(heading.get_text(" ",strip=True))
            if label:
                headings.append({"html_line_1based":heading.sourceline,"text":label})
        rows = []
        seen = set()
        for element in soup.find_all(["p","li","h1","h2","h3","h4","h5","h6"]):
            paragraph = normalize(element.get_text(" ",strip=True))
            if not paragraph or paragraph in seen:
                continue
            categories = [key for key,pattern in PATTERNS.items() if re.search(pattern,paragraph,re.I)]
            if not categories:
                continue
            seen.add(paragraph)
            line = element.sourceline
            prior_headings = [h for h in headings if h["html_line_1based"] and h["html_line_1based"] <= line]
            table = element.find_parent("table")
            rows.append({"evidence_id":f"FY{year}:N{len(rows)+1:03d}",
                "html_line_1based":line,"tag":element.name,
                "heading_context":prior_headings[-1]["text"] if prior_headings else None,
                "inside_table":table is not None,"table_index_1based":soup.find_all("table").index(table)+1 if table else None,
                "categories":categories,"text":paragraph,"text_sha256":hashlib.sha256(paragraph.encode()).hexdigest()})
        event = events[year]
        require(event["source"]["sha256"] == source["sha256"], "Timing/source version mismatch")
        reports.append({"fiscal_year":year,"source":source,"known_by_at":event["clock"]["known_by_at"],
            "available_session":event["clock"]["available_session"],"notes":rows,
            "extraction_is_not_a_boundary_pass":True,"new_financial_values_extracted":0})
    dump_new(OUT/"NOTE_EVIDENCE.json",{"spec_sha256":expected,"reports":reports})
    md = ["# Source accounting-note context", "", "Source-only paragraphs; no automatic comparability decisions.", ""]
    for report in reports:
        md.extend([f"## FY{report['fiscal_year']}",f"Available: {report['known_by_at']}", ""])
        for note in report["notes"]:
            md.extend([f"{note['evidence_id']} · HTML line {note['html_line_1based']} · {note['heading_context']}",
                       note["text"],""])
    with (OUT/"NOTE_EVIDENCE.md").open("x") as f:
        f.write("\n".join(md)+"\n")
    validate(expected)
    print(json.dumps({"reports":len(reports),"notes":sum(len(r["notes"]) for r in reports),
                      "notes_sha256":sha(OUT/"NOTE_EVIDENCE.json")}))


def finalize(expected):
    spec = validate(expected)
    notes = read(OUT/"NOTE_EVIDENCE.json")
    require(notes["spec_sha256"] == expected, "Evidence spec")
    byyear = {r["fiscal_year"]:r for r in notes["reports"]}
    evid = {n["evidence_id"]:(r,n) for r in notes["reports"] for n in r["notes"]}
    decisions = read(OUT/"BOUNDARY_DECISIONS.json")
    require(decisions["note_evidence_sha256"] == sha(OUT/"NOTE_EVIDENCE.json"), "Decisions/evidence binding")
    require(decisions["policy_sha256"] == spec["inputs"]["policy"]["sha256"], "Decision policy")
    edges = decisions["boundaries"]
    require([e["later_fiscal_year"] for e in edges] == list(range(2016,2027)), "Exactly eleven boundaries")
    witnesses = {r["current_release_fy"]:r for r in read(INPUTS["proposal"][0])["cross_release_comparative_checks"]}
    output=[]
    for e in edges:
        y=e["later_fiscal_year"]
        require(e["earlier_fiscal_year"] == y-1, "Adjacent boundary")
        require(e["status"] in ("pass_with_declared_scope","blocked_known_basis_break","pending_source_review"),"Disposition")
        require(e["evidence_ids"] and e["reason"] and e["flags_resolved"], "Explicit reviewed evidence required")
        linked=[]
        for identifier in e["evidence_ids"]:
            report,note=evid[identifier]
            require(report["fiscal_year"] in (y-1,y), "Future or unrelated report note forbidden")
            linked.append(dict(note,source_path=report["source"]["path"],source_sha256=report["source"]["sha256"],
                               source_available_at=report["known_by_at"]))
        witness=witnesses[y]
        require(witness["same_period"],"Witness period")
        if y == 2018:
            require(e["status"] == "blocked_known_basis_break" and not witness["exact_scaled_amount_equal"],"Mandatory FY2018 block")
        if e["status"] == "pass_with_declared_scope":
            require(witness["exact_scaled_amount_equal"],"Changed original amount may not pass")
        output.append(dict(e,known_no_earlier_than=byyear[y]["known_by_at"],
                          earlier_report_available_at=byyear[y-1]["known_by_at"],
                          later_report_available_at=byyear[y]["known_by_at"],
                          preverified_comparative_witness=witness,evidence=linked,
                          future_cutoff_rule=f"Usable only when actual decision cutoff C >= {byyear[y]['known_by_at']}; preserve both original source vintages.",
                          four_year_chain_computed=False))
    result={"status":"MSFT_ADJACENT_REPORTED_REVENUE_BASIS_REVIEW_FOR_INDEPENDENT_CHECK",
        "spec_sha256":expected,"code_sha256":sha(CODE),"policy":spec["inputs"]["policy"],
        "note_evidence_sha256":sha(OUT/"NOTE_EVIDENCE.json"),"decisions_sha256":sha(OUT/"BOUNDARY_DECISIONS.json"),
        "source_scope":spec["scope"],"sources":spec["sources"],"boundaries":output,
        "counts":{key:sum(e["status"]==key for e in output) for key in
                  ("pass_with_declared_scope","blocked_known_basis_break","pending_source_review")},
        "interpretation":"Bounded reported consolidated GAAP total-revenue comparability under the declared original-release archival assumption; not organic growth, constant perimeter, universal accounting homogeneity or full historical amendment certification.",
        "later_discoveries":"A later note does not retroactively alter an earlier disposition. Record new evidence at its own known-by time; keep original annual levels immutable.",
        "first_public_version_verified":False,"independent_final_review_required":True,
        "four_year_chain_rows":0,"growth_ratios":0,"new_financial_values_extracted":0,
        "factor_admitted":False,"model_fits":0,"returns_computed":0,"observer_modified":False}
    dump_new(FINAL,result)
    dump_new(OUT/"MANIFEST.json",{"spec_sha256":expected,"outputs":{
        "NOTE_EVIDENCE.json":sha(OUT/"NOTE_EVIDENCE.json"),"NOTE_EVIDENCE.md":sha(OUT/"NOTE_EVIDENCE.md"),
        "BOUNDARY_DECISIONS.json":sha(OUT/"BOUNDARY_DECISIONS.json"),
        str(FINAL.relative_to(ROOT)):sha(FINAL)}})
    print(json.dumps({"review_sha256":sha(FINAL),"counts":result["counts"]}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["freeze","extract","finalize"])
    parser.add_argument("--spec-sha")
    args=parser.parse_args()
    if args.phase=="freeze":
        freeze()
    else:
        require(args.spec_sha,"--spec-sha required")
        if args.phase=="extract":extract(args.spec_sha)
        else:finalize(args.spec_sha)


if __name__=="__main__":
    main()
