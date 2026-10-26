"""Write the prospective coverage-only experiment; never compute a factor."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import monthly_financial_coverage150_runner as experiment

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "work/monthly_financial_coverage150_proposal.json"


def ref(name):
    path = ROOT / name
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build():
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    assert len(state["completed"]) == 149 and not state["registered_queue"]
    assert sum(x["trials"] for x in state["completed"]) == 950
    source = {k: ref(v) for k, v in experiment.COVERAGE_REFS.items()}
    admission = json.loads((ROOT / source["admission"]["path"]).read_bytes())
    assert admission["passed"] is True
    assert admission["new_strategy_trials"] == admission["actual_factors_calculated"] == 0
    legacy = {k: ref(experiment.LEGACY_OUT + "/" + v)
              for k, v in experiment.LEGACY_FILES.items()}
    concept = ref("work/after149_coverage_extension_concept.json")
    return dict(
        round=150, status="FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
        created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="Extending the same small reported-profitability correction to independently reviewed AMZN and CSCO reports may improve monthly long-momentum selection beyond the unchanged four-issuer earnings control.",
        arms=list(experiment.NAMES), primary_challenger=experiment.NAMES[2],
        dose=experiment.DOSE,
        selection_contract=experiment.selection_contract(),
        measurement=experiment.measurement_contract(),
        sources=dict(legacy149=legacy, coverage_source=source),
        concept=concept,
        fixed=dict(
            core="Original dominant 50%/52% long-term momentum; latest 42 sessions excluded.",
            holdings="Original EW3 with two/cash fallback; fixed units within each month.",
            period=["2016-08-17", "2026-09-24"], signal_months=122,
            training_months=88, width=49, original_arrays=27,
            pool="Original monthly 32 stocks and 17 ETFs unchanged; new sources never exclude a stock.",
            execution="Same-close zero-cost, next-open zero-cost and next-open 5bp each side.",
            new_model_fits=0, new_downloads=0, observer_changes=0),
        interpretation=[
            "Only source coverage changes: economic definition, 1% correction dose, sign, selector and execution stay fixed.",
            "Original earnings uses the finalized round149 cache and its SBC common mask exactly; this is not a freshly recomputed or broadened original control.",
            "New reports were chosen by original-source availability and alphabetic issuer order, not historical holdings, realized returns or later34 performance.",
            "The full 73-stock census, six unprocessed AMZN catalog reports, all old44 records and13 old holds remain; latest invalid or unprocessed known reports block fallback.",
            "The CSCO2011 two missing extracted asset fields remain null even though independent review found the original values. No frozen extraction is edited.",
            "New source admission retains early-calendar and FY2019 accounting-transition holds, 550-day expiry and source-local dollar-unit assumptions.",
            "Coverage is conditional historical research, not certification of first-public versions, complete correction histories or continuous security identities.",
            "The small earnings modifier is tested as auxiliary company quality. It is not claimed to be the author's undisclosed rule.",
            "Selection uses only first88 summaries, segmented Sharpe/drawdown guards and next-open cost checks. Later34 diagnoses cannot rescue a rejected candidate.",
            "Repeatedly inspected development history is not pristine out-of-sample evidence.",
            "No threshold, dose, sign, source-year or source-subset search is authorized. Failed candidates and all three paths remain reported.",
        ],
        actual_new_factors=0, actual_new_orders=0, actual_new_returns=0,
        target="A training-selected, independently verified full-period same-close CAGR at least70%; diagnostics alone do not satisfy the goal.")


if __name__ == "__main__":
    proposal = build()
    experiment.validate_proposal(proposal)
    with DEST.open("x") as stream:
        json.dump(proposal, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(ref(str(DEST.relative_to(ROOT)))))
