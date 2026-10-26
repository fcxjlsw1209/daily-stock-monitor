"""Prepare a fixed-dose coverage proposal; no factors, orders or returns."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import monthly_financial_coverage152_runner as experiment

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "work/monthly_financial_coverage152_proposal.json"
PREVIOUS_DONE = "7dd764a81ff112d2706e4f4216685832830e899ca19af65b77e5dbc5a6648cf5"
CONCEPT_SHA = "60c3afd837cd32b75a70733b70c777084a9c124cf917e8052628947f1cb7d6ee"


def ref(name):
    path = ROOT / name
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build():
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    experiment.require(len(state["completed"]) == 151 and not state["registered_queue"]
                       and sum(x["trials"] for x in state["completed"]) == 957,
                       "Final151 /957 with empty queue required")
    experiment.require(state["completed"][-1]["directory"] == "monthly_financial_coverage151"
                       and state["completed"][-1]["completion"]["sha256"] == PREVIOUS_DONE,
                       "Exact finalized151 predecessor required")
    sources = {batch: {key: ref(path) for key, path in entries.items()}
               for batch, entries in experiment.COVERAGE_REFS.items()}
    for batch, source in sources.items():
        admission = json.loads((ROOT / source["admission"]["path"]).read_bytes())
        experiment.require(admission["passed"] is True
                           and all(type(admission.get(key)) is int and admission[key] == 0
                                   for key in ("new_strategy_trials", "actual_factors_calculated",
                                               "actual_returns_calculated")),
                           "Accepted source-only admission required: " + batch)
    legacy = {k: ref(experiment.LEGACY_OUT + "/" + v) for k, v in experiment.LEGACY_FILES.items()}
    prior = {k: ref(experiment.PRIOR_OUT + "/" + v) for k, v in experiment.LEGACY_FILES.items()}
    experiment.require(prior["completion"]["sha256"] == PREVIOUS_DONE, "Prior151 completion changed")
    concept = ref("work/after149_coverage_extension_concept.json")
    experiment.require(concept["sha256"] == CONCEPT_SHA, "Coverage-only concept changed")
    return dict(
        round=152, status="FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
        created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="Separately reviewed KO/LRCX annual earnings/assets coverage may improve the unchanged small profitability correction beyond all frozen controls.",
        arms=list(experiment.NAMES), primary_challenger=experiment.NAMES[4], dose=experiment.DOSE,
        selection_contract=experiment.selection_contract(), measurement=experiment.measurement_contract(),
        sources=dict(legacy149=legacy, prior151=prior, coverage_sources=sources), concept=concept,
        fixed=dict(
            core="Original dominant50%/52% long-term momentum excluding latest42 sessions.",
            holdings="Original EW3/two/cash fallback; fixed units within month.",
            period=["2016-08-17", "2026-09-24"], signal_months=122, training_months=88,
            width=49, original_arrays=27,
            pool="Original monthly32 stocks plus17 ETFs; financial source absence never excludes a stock.",
            execution="Same-close zero-cost; next-open zero-cost; next-open5bp each side.",
            new_model_fits=0, new_downloads=0, observer_changes=0),
        interpretation=[
            "Only cumulative source coverage changes; formula,1% dose, sign, score placement, selector and execution remain fixed.",
            "Original149 earnings, batch1 final150 and batch2 final151 earnings/masks/scores/ordered choices are preserved exactly.",
            "KO/LRCX were chosen by source-availability and alphabetical census order before outcomes, not by holdings, ranks or realized returns.",
            "All44 original records and26+19+10 batch records retain their source decisions. All three complete32-event catalogs remain; latest held/unprocessed releases block fallback.",
            "All3904 stock-month identities and17 ETFs remain. Financial availability is not stock eligibility.",
            "KO FY2012 three nulls remain producer gaps despite readable adjusted-GAAP source. Early calendar holds remain; no values or dates are silently filled.",
            "KO calendar-year and LRCX52/53-week start dates are explicit same-report boundary assumptions, not quoted source facts.",
            "Common-dollar units,550-day expiry and report-local identity windows do not certify original-publication vintage, complete correction history or a continuous security master.",
            "First88 training selection requires the expanded arm to clear the incumbent and all three earnings controls; later34 is diagnostic only.",
            "Previously explored historical data is not pristine out-of-sample evidence. No sign, dose, year, issuer-subset or parameter search.",
            "All five arms and15 execution paths must be reported. No frozen observer changes or automatic promotion.",
        ],
        actual_new_factors=0, actual_new_orders=0, actual_new_returns=0,
        target="A training-selected independently verified full-period same-close CAGR at least70%; a diagnostic-only result is insufficient.")


if __name__ == "__main__":
    proposal = build()
    experiment.validate_proposal(proposal)
    with DEST.open("x") as stream:
        json.dump(proposal, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(ref(str(DEST.relative_to(ROOT)))))
