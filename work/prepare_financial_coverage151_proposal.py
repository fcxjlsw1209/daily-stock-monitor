"""Prepare the fixed cumulative-coverage proposal; never calculate a factor."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import monthly_financial_coverage151_runner as experiment

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "work/monthly_financial_coverage151_proposal.json"
PREVIOUS_DONE = "28546cec6459a8ca73df5b434d06878ca7bb16008fc0e13b1a6b870cee8ffb92"
CONCEPT_SHA = "60c3afd837cd32b75a70733b70c777084a9c124cf917e8052628947f1cb7d6ee"


def ref(name):
    path = ROOT / name
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build():
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    experiment.require(len(state["completed"]) == 150 and not state["registered_queue"] and
                       sum(x["trials"] for x in state["completed"]) == 953, "Final150 /953 with empty queue required")
    experiment.require(state["completed"][-1]["directory"] == "monthly_financial_coverage150" and
                       state["completed"][-1]["completion"]["sha256"] == PREVIOUS_DONE,
                       "Exact finalized150 predecessor required")
    sources = {batch:{k:ref(v) for k,v in entries.items()}
               for batch,entries in experiment.COVERAGE_REFS.items()}
    for batch,source in sources.items():
        admission = json.loads((ROOT / source["admission"]["path"]).read_bytes())
        experiment.require(admission["passed"] is True and
                           all(type(admission.get(k)) is int and admission[k] == 0
                               for k in ("new_strategy_trials", "actual_factors_calculated", "actual_returns_calculated")),
                           "Accepted source-only admission required: "+batch)
    legacy = {k:ref(experiment.LEGACY_OUT+"/"+v) for k,v in experiment.LEGACY_FILES.items()}
    prior = {k:ref(experiment.PRIOR_OUT+"/"+v) for k,v in experiment.LEGACY_FILES.items()}
    experiment.require(prior["completion"]["sha256"] == PREVIOUS_DONE, "Prior150 completion changed")
    concept = ref("work/after149_coverage_extension_concept.json")
    experiment.require(concept["sha256"] == CONCEPT_SHA, "Coverage-only concept changed")
    return dict(
        round=151, status="FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
        created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="Adding independently reviewed INTC and KLAC annual NI/assets coverage to the unchanged AMZN/CSCO extension may improve the same small reported-profitability correction beyond all fixed controls.",
        arms=list(experiment.NAMES), primary_challenger=experiment.NAMES[3], dose=experiment.DOSE,
        selection_contract=experiment.selection_contract(), measurement=experiment.measurement_contract(),
        sources=dict(legacy149=legacy, prior150=prior, coverage_sources=sources), concept=concept,
        fixed=dict(core="Original dominant 50%/52% long-term momentum excluding latest42 sessions.",
            holdings="Original EW3/two/cash fallback, fixed units within month.",
            period=["2016-08-17","2026-09-24"], signal_months=122, training_months=88,
            width=49, original_arrays=27,
            pool="Original monthly32 stocks plus17 ETFs; source absence never excludes a stock.",
            execution="Same-close zero-cost, next-open zero-cost, next-open5bp each side.",
            new_model_fits=0, new_downloads=0, observer_changes=0),
        interpretation=[
            "Only cumulative source coverage changes; earnings definition,1% dose, sign, selector and execution remain fixed.",
            "Original earnings reproduces finalized149 earnings and its shared SBC mask; batch1 earnings reproduces finalized150 expanded earnings and mask. No old factor is rebuilt.",
            "New issuers were chosen by original-source availability and alphabetic census order, not realized returns or later34 outcomes.",
            "All44 old records,26 batch1 records and19 batch2 records retain their source decisions; the two complete32-event catalogs are retained. Latest known held/unprocessed events block fallback.",
            "All3904 stock-month identities and17 ETFs remain. Added financial availability is not stock-pool eligibility.",
            "KLAC2011 three PRE nulls,KLAC2019 ASC606 hold and early-calendar holds remain; no source gap is filled by inference.",
            "INTC annual starts use the disclosed consecutive prior-year asset endpoint plus one day assumption, corroborated by independent boundary evidence; they are not quoted dates.",
            "Common-dollar source units,550-day expiry and report-local identity windows do not certify first-public versions, complete correction histories or a continuous security master.",
            "The first88 training gate compares expanded earnings to the incumbent, original earnings and frozen batch1 earnings under both execution conventions. Later34 is diagnostic only.",
            "Repeated development-history inspection is not pristine out-of-sample evidence. No parameter, dose, sign, year or source-subset search is authorized.",
            "All four arms and twelve execution paths remain reported; no observer promotion is authorized."],
        actual_new_factors=0, actual_new_orders=0, actual_new_returns=0,
        target="A training-selected independently verified full-period same-close CAGR at least70%; diagnostics alone do not satisfy the goal.")


if __name__ == "__main__":
    proposal = build()
    experiment.validate_proposal(proposal)
    with DEST.open("x") as stream:
        json.dump(proposal, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(ref(str(DEST.relative_to(ROOT)))))
