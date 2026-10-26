"""Save the prospective round148 contract without computing market features."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import monthly_session_cluster148_runner as runner

ROOT = Path(__file__).resolve().parents[1]


def reference(path, expected=None):
    digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, path
    return dict(path=path, sha256=digest)


def main():
    assert not (ROOT / runner.OUT / "PLAN.json").exists()
    concept_ref = reference(runner.CONCEPT, runner.CONCEPT_SHA)
    concept = json.loads((ROOT / runner.CONCEPT).read_text())
    proposal = dict(
        round=148,
        created_utc=datetime.now(timezone.utc).isoformat(),
        arms=list(runner.NAMES),
        primary_challenger=runner.NAMES[2],
        dose=runner.DOSE,
        concept=concept_ref,
        selection_contract=runner.selection_contract(),
        feature_contract=runner.feature_contract(),
        hypothesis=concept["hypothesis"],
        fixed=dict(
            period=["2016-08-17", "2026-09-24"],
            original_arrays=27,
            signal_months=122,
            width=49,
            training_months=88,
            core="Unchanged local67 dominant long momentum excluding latest42 sessions.",
            pool="Original historical estimated issuer Top32 plus theme17.",
            holdings="Original EW3 and two/cash fallback; fixed intramonth adjusted units.",
            new_model_fits=0, new_downloads=0, observer_changes=0,
        ),
        measurement=dict(
            overnight="log(adjusted_open[s]/adjusted_close[s-1])",
            intraday="log(adjusted_close[s]/adjusted_open[s])",
            event="overnight>1e-12 and intraday< -1e-12",
            frequency="sum(event)/210",
            covariance="(N11*209-Nleft*Nright)/(209*209)",
            integer_counts=True,
            placement="Add0.01*centered_low_rank to unchanged pre-score before exactly one original episode/boundary selector.",
            rank_population="All common-valid candidates in the original49 pool before original trading eligibility.",
            time="210events t-251 through t-42;211closes t-252 through t-42.",
            missing="Invalidate both raw features; neutral zero correction, no dropped pairs, extended window, fill or fallback.",
            adjustment="Preserved total-return-adjusted price convention; adjusted overnight includes dividend effects, not pure quotation gaps.",
        ),
        sources=dict(
            prices=reference(runner.PRICE, runner.PRICE_SHA),
            sessions=reference(runner.SESSIONS, runner.SESSIONS_SHA),
            calendar=reference(runner.CALENDAR, runner.CALENDAR_SHA),
            predecessor=reference(
                "outputs/monthly_asset_growth147/COMPLETION.json",
                "ac0bc2e3b0aa5860c1878b3f1f0e746837cc591ea170682ec059c53fa7b2accb",
            ),
        ),
        novelty=concept["prior_novelty_record"],
        bounded_archive_novelty=concept["bounded_archive_novelty"],
        target="First88-selected challenger full-period CAGR>=70%, independently verified. Rejected or diagnostic-only arms do not meet the target.",
        interpretation=concept["limitations"] + [
            "Event frequency control does not equalize conditional variance or isolate covariance causally.",
            "No tuning of fixed210window,42skip,1e-12sign threshold,direction,dose or rank mapping after results.",
            "An empty, singleton or all-tied rank population gives zero adjustment.",
            "The first signal is the inherited2016-08-17 partial-month start; later signals retain original month-end dates.",
            "The existing universe and historical-price-vintage qualifications are preserved.",
            "Same-close zero-cost is a research comparison; next-open zero and5bp per-side paths test execution sensitivity.",
        ],
        actual_new_factors=0, actual_new_orders=0, actual_new_returns=0,
    )
    runner.validate_proposal(proposal)
    for key in ("novelty", "bounded_archive_novelty"):
        reference(**dict(path=proposal[key]["path"], expected=proposal[key]["sha256"]))
    destination = ROOT / "work/monthly_session_cluster148_proposal.json"
    with destination.open("x") as stream:
        json.dump(proposal, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(reference(str(destination.relative_to(ROOT)))))


if __name__ == "__main__":
    main()
