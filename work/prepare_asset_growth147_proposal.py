"""Write the fixed asset-expansion specification before any actual new scores."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import monthly_asset_growth147_runner as r

ROOT = Path(__file__).resolve().parents[1]


def ref(name):
    return dict(path=name, sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest())


def main():
    p = json.loads((ROOT / "work/monthly_cash_quality146_proposal.json").read_text())
    p.update(round=147, created_utc=datetime.now(timezone.utc).isoformat(),
             arms=list(r.NAMES), primary_challenger=r.NAMES[3], dose=r.DOSE,
             concept=ref("work/monthly_asset_growth147_concept.json"),
             selection_contract=r.selection_contract(),
             hypothesis="A small preference for lower annual reported asset expansion may improve unchanged long-momentum selection beyond fixed profitability and cash-quality adjustments.")
    p["measurement"] = dict(
        average_assets="(A_current+A_comparison)/2",
        earnings="NI_current/average_assets",
        low_accrual="(CFO_current-NI_current)/average_assets",
        asset_growth="(A_current-A_comparison)/A_comparison",
        low_asset_growth="-h(asset_growth)",
        bounded="h(x)=x/(1+abs(x))",
        placement="Add0.01*bounded feature to original local67 pre-score only for originally eligible candidates; exactly one original episode/boundary selector. No extra GP overlay.",
        arithmetic="Exact rational scaled amounts; convert once after bounded mapping. No annualization or rescaling.",
        asset_growth_range="For positive endpoints g>-1; -h(g) in(-1,0.5). Float endpoints from rounding are permitted. The score correction has an asymmetric theoretical range(-0.01,0.005).",
        common_mask="Exactly146 four-field source validity shared by all three corrections; no newly released source holds.",
        time=p["measurement"]["time"],
        missing=p["measurement"]["missing"])
    p["sources"].update(
        asset_use_policy=ref("work/monthly_asset_growth147_source_use.json"),
        asset_use_review=ref("work/monthly_asset_growth147_source_review.json"))
    p["novelty"] = ref("work/monthly_asset_growth147_novelty.json")
    p["interpretation"] = [
        "Reported asset expansion only; acquisitions, disposals, FX and accounting changes are not removed. Not organic investment or capital expenditure.",
        "Sparse four-technology-issuer evidence; unchanged coverage31 admitted and13 held events. Noncovered candidates receive zero correction, potentially introducing covered-issuer exposure effects.",
        "Earnings and low-accrual controls test fixed alternatives, not statistical isolation of a unique investment mechanism.",
        "First-public versions, complete correction histories and daily security master remain unverified.",
        "Later34 and all history have been repeatedly explored; no pristine out-of-sample claim.",
        "Same fixed dose does not equalize empirical variance or turnover; no rescaling after seeing outcomes.",
        "Prior-code search incomplete:49 of146 prior result directories present, no standalone asset-growth signal found in inspected sources. Global novelty not certified.",
        "Failure or unchanged portfolios provides no incremental efficacy evidence. Keep failed results; no dose tuning to rescue them."
    ]
    p["literature"] = json.loads((ROOT / "work/monthly_asset_growth147_concept.json").read_text())["literature"]
    r.validate_proposal(p)
    target = ROOT / "work/monthly_asset_growth147_proposal.json"
    with target.open("x") as f:
        json.dump(p, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps(ref(str(target.relative_to(ROOT)))))


if __name__ == "__main__":
    main()
