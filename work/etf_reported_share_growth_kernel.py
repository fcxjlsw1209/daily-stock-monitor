"""Synthetic-ready reported-share growth, with no data access or score weights.

Fixed 63-session window ending five supplied exchange sessions before query.
Source units and reporting clocks are assertions supplied with evidence IDs;
this kernel checks those assertions, never infers them from a price/share jump.
Passing this function does not independently certify source provenance or PIT.
No assumed-publication mode is implemented. Missing clocks or evidence are neutral.

Skip42 concerns the existing price-momentum core. This non-price auxiliary does
not itself violate skip42; its own availability and unit clocks remain required.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

WINDOW = 63
LAG = 5
MARKET_ZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ShareObservation:
    session: str
    shares: object
    known_at: Optional[str]
    version_id: str
    identity: str


@dataclass(frozen=True)
class UnitAdmission:
    identity: str
    unit_basis: str
    start_session: str
    end_session: str
    known_at: Optional[str]
    evidence_id: str
    units_verified: bool = False
    action_ledger_verified: bool = False
    reporting_clock_verified: bool = False


@dataclass(frozen=True)
class SplitAction:
    event_id: str
    legal_effective_session: str
    source_unit_session: Optional[str]
    new_per_old: object
    known_at: Optional[str]
    evidence_id: str
    verified: bool = False


def _day(value):
    if not isinstance(value, str):
        raise ValueError("Session must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("Invalid ISO session") from None
    if parsed.isoformat() != value:
        raise ValueError("Noncanonical ISO session")
    return value


def _stamp(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit timezone-aware timestamp required")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise ValueError("Invalid timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Naive timestamp is not an availability clock")
    return parsed.astimezone(timezone.utc)


def _clock_reason(value, cutoff, kind):
    if value is None:
        return kind + "_clock_missing"
    try:
        instant = _stamp(value)
    except ValueError:
        return kind + "_clock_invalid"
    return kind + "_not_known" if instant > cutoff else None


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _positive(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0. else None


def reported_share_growth63(sessions: Sequence[str],
                            observations: Sequence[ShareObservation],
                            query_session: str, decision_at: str, *,
                            admission: Optional[UnitAdmission],
                            actions: Sequence[SplitAction] = ()):
    """Return paired raw features or one shared unavailable/neutral result.

    G = ending / initial common-unit shares - 1.
    A = sum(abs(daily NET common-unit share differences)) / initial shares.
    A is not gross creations plus redemptions. Neither feature is cash flow.

    For native series, each pre-action balance is multiplied by the verified
    new-per-old factor to express it in the window-ending share unit. The
    source_unit_session is the independently proved row date on which the
    underlying native share balance reflects the legal action. It can be later
    than legal_effective_session; the kernel does not guess that mapping.
    For an already common-unit series, no action multiplier is applied again.

    Only relevant window observations are read. Future/nonwindow values and
    actions cannot change the result. An unresolved in-window action invalidates
    BOTH arms. Neutral numbers require the caller to bypass any ranking/scoring
    transformation when available is false; missing is never an observed zero.
    """
    axis = tuple(_day(s) for s in sessions)
    if not axis or axis != tuple(sorted(set(axis))):
        raise ValueError("Calendar must be nonempty, unique and increasing")
    query_session = _day(query_session)
    if query_session not in axis:
        raise ValueError("Query must be a supplied exchange session")
    cutoff = _stamp(decision_at)
    if cutoff.astimezone(MARKET_ZONE).date().isoformat() != query_session:
        raise ValueError("Decision timestamp must belong to query's New York date")
    result = dict(
        available=False, reason="insufficient_history",
        candidate=0., control=0., raw_candidate=None, raw_control=None,
        query_session=query_session, decision_at=cutoff.isoformat(),
        lag_sessions=LAG, change_sessions=WINDOW,
        source_start=None, source_end=None, observations_used=0,
        applied_split_ids=[], supplied_clock_checks_passed=False,
        strict_PIT_certified=False,
        provenance_note="Supplied evidence assertions checked; no independent source certification",
    )

    def neutral(reason):
        result["reason"] = reason
        return result

    query_index = axis.index(query_session)
    if query_index < WINDOW + LAG:
        return result
    window = axis[query_index - LAG - WINDOW:query_index - LAG + 1]
    first, last = window[0], window[-1]
    result.update(source_start=first, source_end=last)
    if admission is None:
        return neutral("unit_admission_missing")
    if not isinstance(admission, UnitAdmission):
        raise TypeError("UnitAdmission required")
    if admission.unit_basis not in ("native", "common"):
        return neutral("unit_basis_unknown")
    if not all(v is True for v in (admission.units_verified,
                                  admission.action_ledger_verified,
                                  admission.reporting_clock_verified)):
        return neutral("unit_action_or_reporting_evidence_unverified")
    if not _text(admission.evidence_id) or not _text(admission.identity):
        return neutral("unit_evidence_or_identity_missing")
    if not (_day(admission.start_session) <= first and _day(admission.end_session) >= last):
        return neutral("unit_evidence_outside_window")
    reason = _clock_reason(admission.known_at, cutoff, "unit_evidence")
    if reason:
        return neutral(reason)

    by_day = {}
    needed = set(window)
    for obs in observations:
        if not isinstance(obs, ShareObservation):
            raise TypeError("ShareObservation required")
        day = _day(obs.session)
        if day not in needed:
            continue
        if day in by_day:
            return neutral("duplicate_observation")
        by_day[day] = obs
    if set(by_day) != needed:
        return neutral("missing_observation")
    balances = []
    for day in window:
        obs = by_day[day]
        if obs.identity != admission.identity:
            return neutral("observation_identity_mismatch")
        if not _text(obs.version_id):
            return neutral("observation_version_missing")
        reason = _clock_reason(obs.known_at, cutoff, "observation")
        if reason:
            return neutral(reason)
        value = _positive(obs.shares)
        if value is None:
            return neutral("invalid_share_balance")
        balances.append(value)

    relevant = []
    ids, transition_days = set(), set()
    for action in actions:
        if not isinstance(action, SplitAction):
            raise TypeError("SplitAction required")
        legal = _day(action.legal_effective_session)
        # An announced action outside this source window must not require a
        # future reporting-unit mapping or change any earlier prefix result.
        if legal > last:
            continue
        try:
            transition = _day(action.source_unit_session)
        except ValueError:
            return neutral("action_source_unit_session_unknown")
        if not (first < transition <= last):
            continue
        if transition < legal:
            return neutral("source_unit_precedes_legal_effective")
        if transition not in needed or legal not in axis:
            return neutral("action_not_on_known_session_axis")
        if not _text(action.event_id) or not _text(action.evidence_id) or action.verified is not True:
            return neutral("action_evidence_unverified")
        if action.event_id in ids or transition in transition_days:
            return neutral("duplicate_or_ambiguous_action")
        ids.add(action.event_id)
        transition_days.add(transition)
        reason = _clock_reason(action.known_at, cutoff, "action")
        if reason:
            return neutral(reason)
        factor = _positive(action.new_per_old)
        if factor is None:
            return neutral("invalid_split_factor")
        relevant.append((transition, action.event_id, factor))
    relevant.sort()

    try:
        common = []
        for day, balance in zip(window, balances):
            multiplier = math.prod(f for transition, _, f in relevant if day < transition) if admission.unit_basis == "native" else 1.
            value = balance * multiplier
            if not math.isfinite(multiplier) or multiplier <= 0. or not math.isfinite(value) or value <= 0.:
                return neutral("nonfinite_or_underflow_common_unit")
            common.append(value)
        candidate = common[-1] / common[0] - 1.
        variation = math.fsum(abs(b - a) for a, b in zip(common, common[1:]))
        control = variation / common[0]
        if not math.isfinite(candidate) or not math.isfinite(control):
            return neutral("nonfinite_feature_arithmetic")
    except (OverflowError, ValueError, ZeroDivisionError):
        return neutral("nonfinite_feature_arithmetic")
    result.update(
        available=True, reason="available",
        candidate=candidate, control=control,
        raw_candidate=candidate, raw_control=control,
        observations_used=len(window),
        applied_split_ids=[name for _, name, _ in relevant] if admission.unit_basis == "native" else [],
        supplied_clock_checks_passed=True,
    )
    return result
