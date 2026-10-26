"""Hash/contract review and synthetic date arithmetic; no actual event anchors.

Only --write creates the new validation receipt. No strategy producer, source
collector, actual calendar, price array, ordering or return engine is imported.
"""
import argparse
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REVIEW = "work/after156_delayed_clock_review.json"
REVIEW_SHA = "dadf7abdc8ceb2f5ed811be313cd62b76da2e3f5687970007bd6513fe226fee4"
OUT = "work/after156_delayed_clock_validation.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def strict_date(value):
    require(isinstance(value, str), "Date must be an ISO string")
    result = date.fromisoformat(value)
    require(result.isoformat() == value, "Canonical YYYY-MM-DD required")
    return result


def calendar(values):
    require(isinstance(values, (list, tuple)) and len(values) >= 3, "Calendar too short")
    parsed = tuple(strict_date(value) for value in values)
    require(tuple(sorted(set(parsed))) == parsed, "Calendar must be strictly increasing")
    return parsed


def delayed_anchor(sessions, filing_date):
    """Independent candidate implementation on a supplied complete date axis."""
    filed = strict_date(filing_date)
    require(sessions[0] <= filed, "Left calendar boundary unknown")
    index = bisect_right(sessions, filed) + 1
    require(index < len(sessions), "Fewer than two later sessions")
    return index


def enumerate_anchor(sessions, filed):
    """Slow oracle: walk forward until two strictly later sessions are counted."""
    seen = 0
    for index, session in enumerate(sessions):
        if session > filed:
            seen += 1
            if seen == 2:
                return index
    raise ValueError("Oracle calendar exhausted")


def synthetic_checks():
    names = []

    def check(name, ok):
        require(ok, name)
        names.append(name)

    def rejects(name, call):
        try:
            call()
        except (TypeError, ValueError):
            names.append(name)
        else:
            raise ValueError("Expected rejection: " + name)

    # Fabricated calendar: a Friday holiday and an exceptional closure on Monday.
    small = calendar(["2020-07-01", "2020-07-02", "2020-07-07", "2020-07-08",
                      "2020-07-09", "2020-07-10", "2020-07-13", "2020-07-14"])
    check("filing day excluded even if an exchange session", delayed_anchor(small, "2020-07-01") == 2)
    check("holiday and exceptional closure do not count", small[delayed_anchor(small, "2020-07-02")] == date(2020, 7, 8))
    check("Saturday waits for two actual sessions", small[delayed_anchor(small, "2020-07-04")] == date(2020, 7, 8))
    check("Sunday waits for two actual sessions", delayed_anchor(small, "2020-07-05") == delayed_anchor(small, "2020-07-04"))
    check("nontrading dates may share one anchor", len({delayed_anchor(small, value) for value in ("2020-07-02", "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06")}) == 1)
    check("next filing session advances anchor", delayed_anchor(small, "2020-07-07") == delayed_anchor(small, "2020-07-06") + 1)
    check("shortened session still counts once", delayed_anchor(small, "2020-07-09") == 6)
    rejects("left calendar truncation rejected", lambda: delayed_anchor(small, "2020-06-30"))
    rejects("one remaining session is insufficient", lambda: delayed_anchor(small, "2020-07-13"))
    rejects("end of calendar rejected", lambda: delayed_anchor(small, "2020-07-14"))
    rejects("timestamp not accepted as date", lambda: delayed_anchor(small, "2020-07-01T00:00:00Z"))
    rejects("invalid leap date rejected", lambda: strict_date("2021-02-29"))
    rejects("unsorted calendar rejected", lambda: calendar(["2020-07-02", "2020-07-01", "2020-07-03"]))
    rejects("duplicate calendar rejected", lambda: calendar(["2020-07-01", "2020-07-01", "2020-07-02"]))
    check("valid leap date preserved", strict_date("2020-02-29") == date(2020, 2, 29))
    check("anchor plus two means four sessions after session-valued F", delayed_anchor(small, "2020-07-07") + 2 == 2 + 4)

    # Every day (including excluded days), several long axes, many decision t's.
    # One axis has weekends; others add regular fake holidays and a long closure.
    start = date(2016, 1, 1)
    days = [start + timedelta(days=i) for i in range(1900)]
    axes = [tuple(d for i, d in enumerate(days) if d.weekday() < 5
                  and (kind == 0 or i % 47 != 0)
                  and (kind < 2 or not 600 <= i < 610)) for kind in range(3)]
    cases, boundary_cases = 0, 0
    for sessions in axes:
        for t in (300, 401, 555, 701, 888, 1111):
            low, high = sessions[t - 273], sessions[t - 43]
            filed = low - timedelta(days=7)
            while filed <= high + timedelta(days=7):
                direct = enumerate_anchor(sessions, filed)
                via_search = delayed_anchor(sessions, filed.isoformat())
                require(direct == via_search, "Binary map differs from two-session counting oracle")
                in_anchor_range = t - 271 <= direct <= t - 42
                in_inverse_range = low <= filed < high
                require(in_anchor_range == in_inverse_range, "Inverse interval is not exact")
                cases += 1
                filed += timedelta(days=1)
            require(enumerate_anchor(sessions, low) == t - 271, "Inclusive lower bound")
            require(enumerate_anchor(sessions, low - timedelta(days=1)) < t - 271, "Before lower bound")
            require(enumerate_anchor(sessions, high) == t - 41, "Exclusive upper bound")
            require(enumerate_anchor(sessions, high - timedelta(days=1)) == t - 42, "Before upper bound")
            boundary_cases += 4
    check("exhaustive inverse range matches enumerated second session", cases > 5000)
    check("inclusive lower and exclusive upper boundaries across all axes", boundary_cases == 72)
    return {"named_checks_passed": len(names), "names": names,
            "synthetic_calendar_axes": len(axes), "calendar_day_cases": cases,
            "explicit_inverse_boundary_cases": boundary_cases,
            "actual_calendar_rows_used": 0, "actual_filing_rows_anchored": 0}


def recommendation():
    return {
        "answer": "Authentic public SEC 10-Q/10-K filing-date fields plus prospectively fixed checks for known anomalies can support an explicitly disclosed assumption-based historical study. A proof that all historical data is error-free is not a reasonable research prerequisite. This does not itself satisfy the frozen exact-clock design or certify point-in-time source data.",
        "two_different_time_claims": {
            "decision_time_no_lookahead": "The metadata and version actually used to form a historical portfolio at signal t must be available by that decision cutoff, or this availability must be an explicitly declared reconstruction assumption. Known later disclosures/corrections must not be backfilled into earlier decisions.",
            "post_publication_interpretation": "Availability before anchor open is a stronger condition needed to call every measured window post-publication. A filing-date-label study may instead disclose that this is assumed and may include pre-publication price movement. It must not call the assumption a verified causal clock.",
            "implication_for_saved_review": "The review's before-anchor requirement is sufficient for both meanings and appropriate to its stronger post-publication interpretation; it is stronger than the minimum decision-time cutoff of a clearly labeled reconstructed filing-date predictor. This clarification is advice for a successor, not an edit or waiver of that saved review."
        },
        "finite_evidence_standard": [
            "Use the real accession-bound public original/form/filingDate in authentic saved SEC sources and retain their provenance; record whether dates are reconstructed from a later snapshot.",
            "Predeclare exceptions from source evidence: invalid dates, conflicting original keys, disclosed date adjustments, known delayed/nonpublic dissemination, contradictory accession versions and known post-signal metadata. An unexplained Z/date mismatch is a review flag, not proof of invalidity or permission to guess a timezone.",
            "Review the source field's domain and known exception mechanisms proportionately. A documented representative original-source check can support the stated research assumption; it is not a logical proof about every row. Do not demand a new exact-time header for each event once exact-time precision is no longer the estimand.",
            "Keep unresolved known contradictions neutral under the common mask and count them in the unchanged denominator. Freeze exception rules before actual feature/return examination; do not reject or keep records based on their effect on coverage thresholds or performance.",
            "Stop escalating the burden after the stated source policy, exception checks, identity, complete chronology and fixed breadth requirements have been met. Residual unknown source error remains a disclosed limitation, not a demand to prove a universal negative."
        ],
        "still_required": [
            "Positive review of the explicit research assumption and evidence scope in a separately named prospective protocol; never label it a passed exact-clock certification.",
            "Unchanged historical security/class mapping, canonical original-plus-amendment enumeration, authentic source and receipt bindings, duplicate/conflict handling and systematic original-universe coverage gates.",
            "One fixed date-lag and partition chosen without new price/return comparisons; actual common-mask metadata coverage must be recomputed for the new inverse interval.",
            "Unchanged beta timing, score dose, fixed intramonth holdings, first88 selection and later34 diagnostic discipline, independent execution review and protected files."
        ],
        "not_justified": "No automatic promotion of current ticker links to historical identities, no treating missing chronology as non-event days, no assumption overriding a known late disclosure, no shrinking the denominator, and no claim that two sessions mathematically bound all SEC dissemination delays.",
        "recommendation": "Proceed with a clearly labeled reconstructed SEC-filing-date study if the coordinator adopts that evidence standard prospectively. State the two-session availability assumption, known-exception policy and decision-time reconstruction limits in the successor. A certified historical dissemination-time product would require stronger source/version evidence; do not impose that product's universal coverage claim on an exploratory study.",
        "status": "RECOMMENDATION_ONLY_NO_DESIGN_RELAXED_OR_ASSUMPTION_ADOPTED"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    inputs = {}

    def read(relative, expected=None, parse=False):
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts, "Relative local input required")
        path = ROOT / rel
        require(not path.is_symlink() and path.resolve().is_relative_to(ROOT), "Nonlocal input")
        raw = path.read_bytes()
        observed = digest(raw)
        require(expected is None or observed == expected, "Changed input: " + relative)
        require(relative not in inputs or inputs[relative] == observed, "Concurrent input change")
        inputs[relative] = observed
        return json.JSONDecoder().decode(raw.decode()) if parse else raw

    review = read(REVIEW, REVIEW_SHA, True)
    for relative, expected in review["input_hashes"].items():
        read(relative, expected)
    for relative in ("work/after156_delayed_clock_validation.py", "run_local.sh", "local_run.py", "local.sandbox.sb"):
        read(relative)
    design = read("work/after156_filing_drift_design.json", parse=True)
    require(design["prospective_round"] is None, "Original design unexpectedly registered")
    require(review["decision"]["can_replace_exact_acceptance_instant_requirement_in_a_new_design"] is True
            and review["decision"]["can_replace_historical_public_availability_evidence"] is False
            and not any(review["admission"].values()), "Recommendation incorrectly claims admission")
    require(all(review["activity_counts"][k] == 0 for k in review["activity_counts"] if k != "new_files"),
            "Review reports real activity")
    require(review["exact_proposed_clock"]["clock_id"] == "sec_reported_filing_date_plus_two_xnys_sessions_v1",
            "Unexpected clock rule")
    require("[date(S[t-273]),date(S[t-43]))" in review["exact_proposed_clock"]["boundary_enumeration"]
            and "j+4 through j+22" in review["exact_proposed_clock"]["additional_delay_disclosure"],
            "Saved inverse or double-delay disclosure changed")
    floor = design["gates_before_any_real_factor"]["broad_coverage_floor"]
    require([floor[k] for k in ("minimum_distinct_issuers_in_first88",
            "minimum_distinct_issuers_in_each_early53_and_recent35",
            "minimum_usable_slot_fraction_in_first88",
            "minimum_usable_slot_fraction_in_each_early53_and_recent35",
            "minimum_usable_issuers_per_qualifying_month",
            "minimum_qualifying_months_in_early53", "minimum_qualifying_months_in_recent35")]
            == [12, 8, .5, .4, 8, 40, 26], "Original coverage floor changed")
    tests = synthetic_checks()
    for relative, expected in inputs.items():
        require(digest((ROOT / relative).read_bytes()) == expected, "Input changed during validation")
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "REFERENCE_HASH_AND_SYNTHETIC_RULE_VALIDATION_PASSED", "passed": True,
        "scope": "Saved review references, frozen floor consistency and synthetic two-session/inverse arithmetic only; no real source timing, identity or chronology admission.",
        "review": {"path": REVIEW, "sha256": REVIEW_SHA}, "input_hashes": inputs,
        "input_files_verified": len(inputs), "runtime_python": sys.version,
        "synthetic_validation": tests, "research_evidence_standard_review": recommendation(),
        "retained_design": {"path": "work/after156_filing_drift_design.json",
                            "sha256": inputs["work/after156_filing_drift_design.json"], "changed": False},
        "admission": {"exact_clock": False, "two_session_clock": False,
                      "availability_assumption_accepted": False, "strategy_registered": False},
        "activity_counts": {"actual_event_anchors": 0, "price_values_read": 0,
                            "factor_calculations": 0, "returns": 0, "registrations": 0,
                            "new_trials": 0, "network_requests": 0, "existing_files_modified": 0,
                            "observer_actions": 0}}
    if args.write:
        with (ROOT / OUT).open("xb") as stream:
            stream.write(encode(result))
            stream.flush()
            os.fsync(stream.fileno())
    print(encode({"output": OUT if args.write else None, "passed": True,
                  "receipt_sha256": digest(encode(result)), "input_files_verified": len(inputs),
                  "synthetic_validation": tests, "admission": result["admission"]}).decode(), end="")


if __name__ == "__main__":
    main()
