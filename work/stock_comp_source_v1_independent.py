"""Independent, source-only HTML cell audit; no producer parser or strategy import.

The ten PDF reports require a separate visual receipt. This audit proves each
present HTML cell against its original bytes and previously reviewed NI/CFO
anchors. A recorded missing cell is not a proof that no alternate source exists.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = "outputs/monthly_stock_comp_source_v1"
SELF = "work/stock_comp_source_v1_independent.py"
TEST = "work/test_stock_comp_source_v1_independent.py"
OLD = "outputs/monthly_cashflow_source_union_v1/RAW_FACTS.json"
OLD_SHA = "fb26792299d6e6b75ab8ae321e1bcca65fc978876ddf4f1e0a0e983b25c41c06"
METADATA = ("event_id", "symbol", "fiscal_year", "release_date", "available_from",
            "identity", "periods", "source", "inherited_rejection_codes",
            "cash_quality_admission")
SBC = re.compile(r"^(?:stock|share)[\s\-\u2010-\u2015]+based\s+compensation(?:\s+expense)?$", re.I)
NI = re.compile(r"^net\s+(?:income|loss)(?:\s*\((?:loss|income)\))?$", re.I)
CFO = re.compile(r"^(?:net\s+)?cash\s+(?:provided\s+by(?:\s*\(used\s+(?:in|for)\))?|used\s+(?:in|for)(?:\s*\(provided\s+by\))?|generated\s+by|generated\s+from|from)\s+(?:operating\s+activities|operations)$", re.I)
ANNUAL = re.compile(r"\b(?:twelve|12)\s+months?\s+ended\b|\byears?\s+ended\b", re.I)
INTERIM = re.compile(r"\b(?:three|six|nine|3|6|9)\s+months?\s+ended\b", re.I)
MONTHS = {name: i + 1 for i, name in enumerate(
    "january february march april may june july august september october november december".split())}
DATE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(value):
    return re.sub(r"\s+", " ", value).strip()


def money(display):
    """Exact representation only: no scale conversion or financial arithmetic."""
    value = norm(display).replace("\u2212", "-")
    value = re.sub(r"\s*,\s*", ",", value)
    match = re.fullmatch(r"\$?\s*(\()?\s*(-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(\))?", value)
    require(match is not None and bool(match[1]) == bool(match[3]), "Ambiguous monetary literal")
    number = match[2].replace(" ", "").replace(",", "")
    require(not (match[1] and number.startswith("-")), "Conflicting monetary signs")
    result = Decimal(number)
    return format(-result if match[1] else result, "f")


def table_cells(table):
    """Expand spans independently; preserve physical cell identity and markup."""
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    grid, physical = {}, {}
    for row_number, row in enumerate(rows):
        column = 0
        for physical_number, cell in enumerate(row.find_all(["th", "td"], recursive=False)):
            while (row_number, column) in grid:
                column += 1
            require(cell.find("table") is None, "Nested financial cell")
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            require(1 <= rs <= 100 and 1 <= cs <= 100, "Invalid cell span")
            item = dict(row=row_number, physical=physical_number, columns=list(range(column, column + cs)),
                        rowspan=rs, text=norm(cell.get_text(" ", strip=True)),
                        markup_sha256=digest(str(cell).encode()))
            physical[(row_number, physical_number)] = item
            for rr in range(row_number, row_number + rs):
                for cc in item["columns"]:
                    require((rr, cc) not in grid, "Overlapping financial cells")
                    grid[(rr, cc)] = item
            column += cs
    return rows, grid, physical


def header_dates(text):
    return {date(int(year), MONTHS[month.lower()], int(day)).isoformat()
            for month, day, year in DATE.findall(text)}


def audit_html(raw, value, old_event):
    """Verify only a published candidate, using independent physical extraction."""
    require(value["source_sha256"] == digest(raw), "Candidate source digest")
    loc = value["locator"]
    require(loc["format"] == "html", "HTML locator required")
    current = old_event["fields"]["current"]
    old_ni, old_cfo = current["net_income"], current["operating_cash_flow"]
    require(old_ni is not None and old_cfo is not None, "Previously reviewed NI/CFO anchors required")
    ni_loc, cfo_loc = old_ni["locator"], old_cfo["locator"]
    for key in ("table_index_zero_based", "statement_segment", "logical_columns",
                "scale", "currency_literal", "period_end", "header_literals"):
        require(loc.get(key) == cfo_loc.get(key) == ni_loc.get(key), "Changed shared anchor: " + key)
    require(ni_loc.get("ni_scope") == "indirect_method_consolidated_start", "NI scope changed")
    require(value["period"] == old_event["periods"]["current"], "Current annual period changed")
    require(value["scale"] == loc["scale"] and value["currency_literal"] == loc["currency_literal"] == "$",
            "Scale or literal currency mismatch")
    require(value["currency_iso"] is None, "New ISO certification is not permitted")
    require(value["fact_admitted"] is False, "New field was prematurely admitted")

    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    table = soup.find_all("table")[loc["table_index_zero_based"]]
    _, grid, physical = table_cells(table)
    low, high, row = ni_loc["row_index_zero_based"], cfo_loc["row_index_zero_based"], loc["row_index_zero_based"]
    require(loc["indirect_start_row_zero_based"] == low and loc["operating_total_row_zero_based"] == high,
            "NI/CFO row anchor mismatch")
    require(low < row < high, "Component outside NI-to-CFO reconciliation")
    labels = lambda rr: [cell["text"].rstrip(":").strip()
                         for (r, _), cell in physical.items() if r == rr]
    require(any(NI.fullmatch(label) for label in labels(low)), "Missing physical consolidated NI label")
    require(any(CFO.fullmatch(label) for label in labels(high)), "Missing physical CFO total label")
    matches = [(rr, pi, cell) for (rr, pi), cell in physical.items()
               if low < rr < high and SBC.fullmatch(cell["text"].rstrip(":").strip())]
    require(len(matches) == 1 and matches[0][0] == row, "Duplicate or changed physical SBC label")
    label_cell = matches[0][2]
    require(label_cell["text"].rstrip(":").strip() == loc["row_label"], "Exact SBC label changed")
    require(label_cell["markup_sha256"] == loc["label_cell_sha256"], "SBC label markup changed")
    require(any(re.search(r"adjustments?\s+to\s+reconcile\s+(?:net\s+)?(?:income|loss)", text, re.I)
                for rr in range(low + 1, row) for text in labels(rr)), "Missing prior indirect adjustment heading")

    columns = loc["logical_columns"]
    require(columns and len(set(columns)) == len(columns), "Unique current columns required")
    start = (loc.get("statement_segment") or {}).get("start_row", 0)
    for col in columns:
        distinct = {}
        for rr in range(start, low):
            cell = grid.get((rr, col))
            if cell is not None:
                distinct[(cell["row"], cell["physical"])] = cell["text"]
        text = " ".join(distinct.values())
        require(header_dates(text) == {loc["period_end"]}, "Wrong or ambiguous physical date column")
        require(ANNUAL.search(text) and not INTERIM.search(text), "Current column is not solely annual")
    selected = sorted({grid[(row, col)]["physical"] for col in columns})
    require(selected == loc["physical_cell_indices_zero_based"], "Wrong physical amount cell")
    cells = [physical[(row, pi)] for pi in selected]
    require(all(cell["rowspan"] == 1 and set(cell["columns"]).issubset(columns) for cell in cells),
            "Selected amount crosses period or row")
    display = " ".join(cell["text"] for cell in cells if cell["text"]).strip()
    require(display == value["displayed_amount"], "Displayed literal mismatch")
    require(money(display) == value["value_decimal"], "Independent signed decimal mismatch")
    require([cell["markup_sha256"] for cell in cells] == value["parsed_cell_sha256"],
            "Physical amount markup mismatch")
    return dict(verified=True, table_index_zero_based=loc["table_index_zero_based"],
                row_index_zero_based=row, physical_cells=selected, period_end=loc["period_end"],
                signed_decimal_verified=True, annual_header_verified=True,
                original_reviewed_consolidated_NI_CFO_anchors_verified=True,
                source_sha256=digest(raw), accounting_admission=False)


def audit(root, plan_sha):
    root = Path(root).resolve()
    seen = {}

    def read_bytes(path, expected=None):
        resolved = Path(path) if Path(path).is_absolute() else root / path
        resolved = resolved.resolve()
        require(root in resolved.parents, "Input outside local research root")
        raw = resolved.read_bytes()
        actual = digest(raw)
        require(expected is None or expected == actual, "Hash mismatch: " + str(path))
        seen[str(resolved.relative_to(root))] = actual
        return raw

    def read(path, expected=None):
        return json.loads(read_bytes(path, expected))

    require(type(plan_sha) is str and re.fullmatch("[0-9a-f]{64}", plan_sha), "Explicit frozen PLAN SHA required")
    plan = read(OUT + "/PLAN.json", plan_sha)
    require(plan["stage"] == "frozen_stock_compensation_source_census", "Wrong source stage")
    for group in ("source_hashes", "source_code_hashes"):
        for path, sha in plan[group].items():
            read_bytes(path, sha)
    manifest = read(OUT + "/MANIFEST.json")
    require(manifest["plan_sha256"] == plan_sha, "Manifest PLAN mismatch")
    require(set(manifest["output_hashes"]) == {"RAW_FACTS.json", "COVERAGE.json", "AUDIT.json"}, "Manifest output set")
    outputs = {name: read(OUT + "/" + name, sha) for name, sha in manifest["output_hashes"].items()}
    data, coverage, process = (outputs[name] for name in ("RAW_FACTS.json", "COVERAGE.json", "AUDIT.json"))
    require(all(obj["plan_sha256"] == plan_sha for obj in outputs.values()), "Output PLAN mismatch")
    require(process["passed"] is True and process["independent_field_and_visual_review_pending"] is True,
            "Source process status changed")
    require(process["data_sha256"] == manifest["output_hashes"]["RAW_FACTS.json"] and
            process["coverage_sha256"] == manifest["output_hashes"]["COVERAGE.json"], "Process output link mismatch")
    old = {e["event_id"]: e for e in read(OLD, OLD_SHA)["events"]}
    require(len(plan["events"]) == len(data["events"]) == len(coverage["events"]) == len(old) == 44, "44-event census")
    require(len({e["event_id"] for e in data["events"]}) == 44, "Duplicate event")
    results = []
    for spec, event, cov in zip(plan["events"], data["events"], coverage["events"]):
        require({k: event[k] for k in METADATA} == {k: spec[k] for k in METADATA}, "Original event metadata changed")
        eid = event["event_id"]
        require(cov["event_id"] == eid and eid in old, "Event order changed")
        require(all(event[k] is False for k in ("fact_admitted", "model_admitted", "strategy_admitted")),
                "Event prematurely admitted")
        cell = event["stock_compensation"]
        require(cell["candidate_only"] is True and all(cell[k] is False
                for k in ("fact_admitted", "model_admitted", "strategy_admitted")), "Cell prematurely admitted")
        require(all(cell[k] is False for k in ("first_public_version_verified", "complete_correction_history_verified",
                                             "certified_daily_security_master")), "Unsupported source certification")
        value = cell["value"]
        require(cov["present"] == (value is not None) and cov["new_field_admitted"] is False and
                cov["inherited_hold"] == (not event["cash_quality_admission"]["admitted"]) and
                cov["gaps"] == cell["gaps"], "Coverage disagreement")
        detail = dict(event_id=eid, symbol=event["symbol"], inherited_hold=cov["inherited_hold"],
                      accounting_review_flags=deepcopy(cell["accounting_review_flags"]))
        if value is None:
            require(bool(cell["gaps"]), "Missing value without explicit gap")
            detail.update(status="explicit_gap_preserved", gaps=deepcopy(cell["gaps"]),
                          gap_is_not_proof_of_global_source_absence=True)
        else:
            require(not cell["gaps"], "Published value coexists with ambiguity")
            require(value["available_from"] == event["available_from"] and value["source"] == event["source"],
                    "Current-report availability/source changed")
            if value["locator"]["format"] == "html":
                raw = read_bytes(event["source"]["path"], event["source"]["sha256"])
                detail.update(status="HTML_physical_cell_verified",
                              cell=audit_html(raw, value, old[eid]))
            else:
                require(value["locator"]["format"] == "pdf", "Unsupported source format")
                detail.update(status="PDF_visual_review_required")
        results.append(detail)
    counts = dict(Counter(e["status"] for e in results))
    require(sum(e["inherited_hold"] for e in results) == 13, "Inherited hold count changed")
    require(sum(e["symbol"] != "AAPL" for e in results) == 34, "HTML census changed")
    for path in (SELF, TEST):
        read_bytes(path)
    return dict(passed=True, stage="independent_HTML_physical_cell_review_not_admission",
                source_only=True, plan_sha256=plan_sha, events=44, HTML_events=34, inherited_holds=13,
                counts=counts, event_reviews=results, input_hashes=seen,
                financial_ratios_computed=0, factors_computed=0, ranks_computed=0, returns_computed=0,
                new_field_admissions=0, producer_parser_imported=False, PDF_visual_review_pending=True,
                limitations=["Only current annual reported reconciliation components are checked.",
                             "Existing holds and source-version/accounting limitations are unchanged.",
                             "Explicit gaps are retained; no missing cell is inferred as zero.",
                             "SBC addback is not a measured cash payment or share dilution."])


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--plan-sha", required=True)
    args = cli.parse_args()
    result = audit(ROOT, args.plan_sha)
    result["reviewed_utc"] = datetime.now(timezone.utc).isoformat()
    target = ROOT / OUT / "HTML_INDEPENDENT_REVIEW.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(passed=True, counts=result["counts"], receipt_sha256=digest(target.read_bytes()))))


if __name__ == "__main__":
    main()
