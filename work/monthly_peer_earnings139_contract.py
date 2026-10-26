"""Pure prospective round139 arm/selection contract; no data or market imports."""
from collections.abc import Mapping
import math
from numbers import Real

NAMES = ("incumbent_local67", "peer_recent10", "negative_own_event10", "peer_event10")
PRIMARY_CHALLENGER = NAMES[3]
REFERENCE_NAMES = NAMES[:3]
ROLES = {
    NAMES[0]: "incumbent_control",
    NAMES[1]: "matched_recent_control",
    NAMES[2]: "negative_own_event_control",
    NAMES[3]: "primary_challenger",
}
METRIC_FIELDS = (("training", "cagr"), ("training", "sharpe"),
                 ("training", "max_drawdown"), ("early", "cagr"), ("recent", "cagr"))
CAGR_EPSILON = 1e-12
GUARD_TOLERANCES = (.05, .03, .025, .025)
TRAIN_MONTHS, LATER_MONTHS = 88, 34
AUXILIARY_BUDGET = .025


def selection_schema():
    """Return fresh registration metadata; this is not a registered PLAN."""
    return {
        "round": 139, "arms": list(NAMES), "roles": dict(ROLES),
        "primary_challenger": PRIMARY_CHALLENGER,
        "reference_arms": list(REFERENCE_NAMES), "candidate_index": 3,
        "reference_indices": [0, 1, 2], "metric_fields": [list(pair) for pair in METRIC_FIELDS],
        "strict_cagr_epsilon": CAGR_EPSILON,
        "maximum_shortfalls": {"training.sharpe": .05, "training.max_drawdown": .03,
                               "early.cagr": .025, "recent.cagr": .025},
        "training_months": TRAIN_MONTHS, "later_diagnostic_months": LATER_MONTHS,
        "fallback": NAMES[0], "controls_can_be_promoted": False,
    }


def _metrics(row):
    return tuple(row[group].get(field) if isinstance(row.get(group), Mapping) else None
                 for group, field in METRIC_FIELDS)


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def select(rows):
    """Only peer_event10 may replace incumbent, after beating all references.

    Arm count/order is a schema error. Missing, nonnumeric or nonfinite required
    metrics fail the affected comparison. Extra later-period metrics are ignored.
    Inputs are neither modified nor used to fit or compute any return series.
    """
    if not isinstance(rows, (list, tuple)) or len(rows) != len(NAMES) or any(
            not isinstance(row, Mapping) or row.get("name") != name for row, name in zip(rows, NAMES)):
        raise ValueError("Four fixed arms in order required")
    candidate = _metrics(rows[3])
    guards = {}
    for index, name in enumerate(REFERENCE_NAMES):
        reference = _metrics(rows[index])
        good = False
        if all(_finite_number(value) for value in candidate + reference):
            good = bool(candidate[0] > reference[0] + CAGR_EPSILON and all(
                value >= baseline - tolerance
                for value, baseline, tolerance in zip(candidate[1:], reference[1:], GUARD_TOLERANCES)))
        guards[name] = good
    admitted = all(guards.values())
    return {"primary": PRIMARY_CHALLENGER if admitted else NAMES[0],
            "admitted": admitted, "guards": admitted, "guards_against": guards}
