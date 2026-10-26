"""Write the prospective four-arm specification before real factor computation."""
from pathlib import Path
from datetime import datetime, timezone
import json
import monthly_cash_quality146_runner as r

ROOT = Path(__file__).resolve().parents[1]


def ref(path):
    return dict(path=path, sha256=r.sha(ROOT/path))


def main():
    p = dict(round=146, status="FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
        created_utc=datetime.now(timezone.utc).isoformat(), arms=list(r.NAMES),
        primary_challenger=r.NAMES[3], dose=r.DOSE,
        concept=ref("work/after145_cashflow_experiment_concept.json"),
        selection_contract=r.selection_contract(),
        hypothesis="Cash flow minus consolidated earnings, scaled by average annual assets, "
            "may distinguish durable long-momentum candidates beyond either earnings or cash flow alone.",
        fixed=dict(period=["2016-08-17", "2026-09-24"], original_arrays=27,
            signal_months=122, width=49, training_months=88,
            core="Original dominant50%/52% long-term momentum; latest42 sessions skipped.",
            pool="Original historical monthly32 stocks plus17 ETFs unchanged.",
            holdings="Original EW3 with two/cash fallback; fixed adjusted units intramonth.",
            execution="Original month-end same-close zero-cost comparison; next-open zero and5bp stress.",
            new_model_fits=0, new_downloads=0, observer_changes=0),
        measurement=dict(average_assets="(A_current+A_comparison)/2",
            earnings="NI_current/average_assets", cash_flow="CFO_current/average_assets",
            low_accrual="(CFO_current-NI_current)/average_assets",
            bounded="h(x)=x/(1+abs(x))",
            placement="Add0.01*h to original local67 pre-score only for originally eligible "
                "candidates; exactly one original episode/boundary selector. No old GP overlay.",
            arithmetic="Exact rational decimal-scaled amounts; convert bounded ratio to float once.",
            common_mask="Identical four-field validity for all three experimental arms.",
            time="Latest report known by signal selected before validity; invalid latest blocks older. "
                "First pinned exchange session strictly after release; fiscal-end age<=550days; "
                "contiguous330..400-day annual periods. No comparison-flow-derived growth.",
            missing="Unknown features leave exact original score/choice; ETFs have no financial metric."),
        sources=dict(policy=ref("outputs/monthly_cash_quality_source_admission_v1/POLICY.json"),
            dataset=ref("outputs/monthly_cash_quality_source_admission_v1/DATASET.json"),
            admission=ref("outputs/monthly_cash_quality_source_admission_v1/ADMISSION.json"),
            full_inventory=44, admitted=31, held=13),
        target="Training-selected model full-period CAGR>=70%, independently verified. "
            "Rejected or diagnostic-only controls do not satisfy the goal.",
        interpretation=[
            "Sparse four-technology-issuer annual reports, not a full-universe cash-quality sample.",
            "First-public versions, complete correction histories and daily security master remain unverified.",
            "Later34 and all history have been repeatedly explored; no pristine out-of-sample claim.",
            "Equal score cap does not equalize variance or turnover. No post-result rescaling.",
            "Working capital, noncash charges, acquisitions, taxes and unequal fiscal years can affect ratios.",
            "Any unchanged portfolios or failure against matched controls is no evidence of incremental mechanism."],
        actual_new_factors=0, actual_new_orders=0, actual_new_returns=0)
    r.validate_proposal(p)
    target=ROOT/"work/monthly_cash_quality146_proposal.json"
    with target.open("x") as f:
        json.dump(p,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(path=str(target),sha256=r.sha(target))))


if __name__ == "__main__":
    main()
