"""Read-only census/context comparison for batch4 admission diagnostics."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
context = json.loads((ROOT / "work/after152_next_source_context.json").read_text())
census = json.loads((ROOT / "outputs/monthly_financial_coverage_census_v1/CENSUS.json").read_text())
rows = [r for r in census["annual_documents"] if r["symbol"] in ("MU", "PEP")]
for raw, ctx in zip(rows, context["events"]):
    diffs = {k: {"census": v, "context": ctx.get(k)}
             for k, v in raw.items() if ctx.get(k) != v}
    if diffs:
        print(json.dumps({"event_id": raw["event_id"], "differences": diffs},
                         ensure_ascii=False, sort_keys=True))
