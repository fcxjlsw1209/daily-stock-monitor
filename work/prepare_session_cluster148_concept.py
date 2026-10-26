"""Record fixed next-hypothesis choices, without loading market arrays."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def ref(relative):
    return dict(path=relative, sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest())


def main():
    concept = dict(
        round=148, stage="prospective_concept_not_registered",
        created_utc=datetime.now(timezone.utc).isoformat(),
        name="adjusted_overnight_positive_intraday_negative_event_clustering",
        hypothesis="Among long-momentum candidates, persistent clustering of adjusted overnight gains followed by intraday losses may identify weaker trend continuation.",
        unchanged=["incumbent_local67 primary score and original selection machinery",
                   "original historical estimated issuer Top32 plus theme17",
                   "dominant long momentum, excluding most recent42 sessions",
                   "month-end EW3, original two-security/cash fallback, fixed intramonth units"],
        arms=["incumbent_local67", "event_frequency_control", "low_event_clustering"],
        direction="Prefer low frequency in the control and low adjacent covariance in the challenger.",
        input_contract=dict(
            preferred="Round147 registered adjusted close/open cache, subject to complete211-close and210-open warmup.",
            fallback="No automatic fallback; inventory missing warmup explicitly before registration.",
            observations="Only signals, pool identities and already verified local prices. No new financial data or online requests."),
        window=dict(
            event_sessions="t-251 through t-42 inclusive,210 consecutive exchange sessions",
            required_prices="close t-252 through t-42 inclusive,211 closes; opening t-251 through t-42",
            overnight="log(adjusted_open[s]/adjusted_close[s-1])",
            intraday="log(adjusted_close[s]/adjusted_open[s])",
            event="overnight>1e-12 and intraday< -1e-12",
            frequency="number of events/210",
            covariance="(N11*209-Nleft*Nright)/(209*209), using209 adjacent pairs",
            missing="Any nonfinite/nonpositive required price invalidates both features; no pair dropping, zero filling or extended window."),
        ranking=dict(
            population="Common fully observed candidates in that signal's existing49-security pool; ETF included.",
            ascending=True,
            formula="0.5-average_zero_based_rank/(n_valid-1)",
            singleton_or_empty="0",
            all_equal="0",
            unavailable="0",
            scope="This normalization only applies to the new corrections. All incumbent ranks unchanged."),
        score=dict(dose=0.01, max_abs_correction=0.005,
                   control="incumbent_pre_score +0.01*frequency_rank",
                   challenger="incumbent_pre_score +0.01*cluster_rank",
                   no_tuning=["window", "lag", "direction", "dose", "threshold", "rank mapping"]),
        validation=dict(
            first88="Reuse round147 selection risk and segment tolerances, replacing its financial controls with event_frequency_control.",
            later34="Diagnostic only; cannot rescue a failed first88 choice.",
            executions=["same_close_zero_cost", "next_open_zero_cost", "next_open_5bp_per_side"],
            independent_review_required=True,
            actual_calculation_requires_registered_plan=True),
        limitations=[
            "This is a predictive association hypothesis, not evidence of identifiable investor selling.",
            "Adjusted overnight return includes dividend adjustments; it is not a pure quoted-price gap.",
            "Low covariance may prefer alternation; a frequency control is not full conditional residualization.",
            "Historical prices use their preserved retrieved vintage, not certified first-public prices.",
            "This direction was proposed after140; missing older experiment files prevent exhaustive novelty certification.",
            "Repeatedly reused historical periods are not pristine out of sample."],
        prior_novelty_record=ref("work/after140_path_novelty.json"),
        bounded_archive_novelty=ref("work/monthly_asset_growth147_novelty.json"),
        actual_features_calculated=0, actual_orders_calculated=0, actual_returns_calculated=0,
        observer_writes=False)
    path = ROOT/"work/monthly_session_cluster148_concept.json"
    with path.open("x") as stream:
        json.dump(concept, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(ref(str(path.relative_to(ROOT)))))


if __name__ == "__main__":
    main()
