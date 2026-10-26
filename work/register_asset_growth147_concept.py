"""Record the new hypothesis before any asset-growth score or returns are computed."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, obj):
    p = ROOT / name
    with p.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return {"path": name, "sha256": sha(p)}


def main():
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_text())
    assert len(state["completed"]) == 146 and sum(x["trials"] for x in state["completed"]) == 939
    assert not state["registered_queue"]
    refs = ["work/monthly_profitability141_factor.py",
            "work/monthly_same_report144_factor.py",
            "work/monthly_cash_quality146_factor.py",
            "work/monthly_cash_quality146_proposal.json",
            "outputs/monthly_cash_quality146/COMPLETION.json"]
    novelty = {
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "not_found_in_available_sources_incomplete_global_coverage",
        "completed_rounds": 146, "completed_directories_present": 49,
        "completed_directories_missing": 97, "directory_files_searched": 106,
        "work_files_searched": 308,
        "reviewer": "index_union_gap_review",
        "finding": "No explicit standalone annual total-asset growth signal found in searched sources. Earlier tests use assets as profitability or accrual denominators.",
        "not_a_claim": "Not a complete 146-round global nonduplication certification. Missing archive code was not inferred or reconstructed.",
        "nearby_verified_sources": {p: sha(ROOT / p) for p in refs},
        "actual_new_factor_calculations": 0, "actual_new_orders": 0,
        "actual_new_return_calculations": 0,
    }
    novelty_ref = write("work/monthly_asset_growth147_novelty.json", novelty)
    concept = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prospective_concept_not_registration",
        "round": 147,
        "hypothesis": "A small preference for lower reported annual asset expansion may improve the unchanged long-momentum candidate ranking beyond fixed earnings and cash-quality corrections.",
        "arms": ["incumbent_local67", "earnings_control", "low_accrual_control", "low_asset_growth"],
        "primary_challenger": "low_asset_growth", "dose": .01,
        "formula": {
            "average_assets": "(Ac+Ap)/2",
            "earnings_control": "h(NI_current/average_assets)",
            "low_accrual_control": "h((CFO_current-NI_current)/average_assets)",
            "low_asset_growth": "-h((Ac-Ap)/Ap)",
            "h": "x/(1+abs(x))",
            "arithmetic": "Exact scaled decimal Fractions; convert to float64 after bounded mapping.",
        },
        "asymmetry": "Positive assets imply g>-1 and low_asset_growth in (-1,0.5). Correction is mathematically (-0.01,0.005); rounded float endpoints are permitted. No empirical rescaling or centering.",
        "unchanged": {
            "core": "Original dominant 50%/52% long-term momentum, skipping last42 exchange sessions.",
            "universe": "Original historical monthly32 stocks plus17 ETFs.",
            "holdings": "Original EW3, existing two/cash fallback; fixed adjusted units intramonth.",
            "score_placement": "Add dose*feature to original local67 pre-score for originally eligible cells, once before unchanged episode/boundary selector.",
            "measurement_mask": "Exactly146 admitted44-event dataset and common four-field validity; retain31 admissions and13 holds. Missing/ETF/noncovered slots use exact incumbent score.",
            "timing": "Latest known current report chosen before validity; invalidlatest blocks fallback. First pinned session after release, contiguous330..400-day years,550-day expiry. No backdating comparative values.",
            "selection": "Same146 first88/early53/recent35 and36-60/60-76/76-88 folds. Sameclose strictly beats all3controls; nextopen5bp noninferior incumbent and strictly beats2controls. Same risk/segment tolerances.",
            "diagnostic": "Later34 reused, never pristineOOS or used to rescue failed primary.",
        },
        "limits": [
            "Sparse four technology-heavy issuers; negative expansion adjustment may change covered-issuer exposure.",
            "Reported assets include acquisitions, disposals, FX and accounting presentation; not organic growth or capital expenditure.",
            "Controls do not statistically isolate a unique investment mechanism.",
            "No new information source, issuer expansion, parameter search or new models.",
            "Incomplete original public-version and correction-history certification retained.",
            "Global prior-code coverage incomplete; novel only relative to inspected code.",
        ],
        "literature": {
            "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.French/Data_Library/six_portfolios_me_inv.html",
            "consulted_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Supports percentage change in annual total assets as an investment measure. This monthly four-issuer overlay is not a replication of June-sorted CMA portfolios."
        },
        "novelty": novelty_ref,
        "new_source_use_review_required": True,
        "actual_new_factors": 0, "actual_new_orders": 0, "actual_new_returns": 0,
        "observer_changes": 0,
        "writer": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(Path(__file__))},
    }
    print(json.dumps(write("work/monthly_asset_growth147_concept.json", concept)))


if __name__ == "__main__":
    main()
