"""Auditable earnings publication availability and conservative bar reactions.

Vendor fiscal labels are not issuer fiscal-period ends. Date-only publication
evidence is not midnight publication, and an availability bound is not an
observed first-public timestamp. This module neither downloads nor trades.
"""
from functools import lru_cache
import re

from research_calendar import market_calendar
import numpy as np
import pandas as pd

POLICY_VERSION = "earnings_publication_availability_v1"
EXCHANGE = "XNYS"
RELEASE_KINDS = {"release", "earnings_release", "issuer_release", "newswire_release"}


@lru_cache(maxsize=1)
def _default_calendar():
    return market_calendar()


def _date(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must have YYYY-MM-DD precision")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("invalid date")
    return result


def _aware(value, name):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is None:
        raise ValueError(name + " requires an explicit timezone or UTC offset")
    return result.tz_convert("UTC")


def _publication(value, timezone):
    if not isinstance(value, str) or not re.search(r"[T ]\d{2}:\d{2}", value):
        raise ValueError("publication_at is not a precise timestamp")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("invalid publication timestamp")
    if result.tz is None:
        if not timezone:
            raise ValueError("naive publication timestamp needs publication_timezone")
        # Ambiguous autumn or nonexistent spring wall times require source
        # clarification; never guess DST or choose one fold automatically.
        result = result.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    elif timezone:
        # Validate an explicitly supplied zone even if timestamp has an offset.
        pd.Timestamp("2024-01-01").tz_localize(timezone)
    return result.tz_convert("UTC")


def _first_close(calendar, timestamp, strictly_after_date=None):
    closes = calendar.schedule["close"]
    good = closes >= timestamp
    if strictly_after_date is not None:
        good &= (calendar.schedule.index > strictly_after_date) & (closes > timestamp)
    matches = np.flatnonzero(good.to_numpy())
    if not len(matches):
        raise ValueError("availability falls outside supplied XNYS calendar")
    i = int(matches[0])
    return calendar.schedule.index[i], closes.iloc[i]


def normalize_event(record, calendar=None):
    """Normalize one explicit event-evidence record to a JSON-safe dictionary.

    record: symbol, asset_type ('equity' or 'etf'), vendor{fiscalDateEnding,
    reportedDate,reportTime}, official{actual_period_end, publication_at OR
    publication_date, publication_timezone, source_url, verified:True,
    publication_kind:'release'}. A timezone-aware publication_at supplies its
    own offset. Date-only evidence always requires publication_timezone.

    official.verified asserts publication evidence was actually checked by the
    caller; this function cannot verify a website. dateModified, acceptance and
    conference-call fields are never used as publication substitutes. Without
    verified release evidence or actual fiscal-period end: pending_review.
    Vendor-vs-official fiscal-label differences are retained, never excluded.
    """
    if not isinstance(record, dict):
        raise ValueError("event record must be a dictionary")
    vendor = record.get("vendor") or {}
    official = record.get("official") or {}
    if not isinstance(vendor, dict) or not isinstance(official, dict):
        raise ValueError("vendor and official metadata must be dictionaries")
    asset_type = str(record.get("asset_type", "equity")).lower()
    result = dict(symbol=record.get("symbol"), asset_type=asset_type, status="pending_review",
        vendor_fiscal_date_ending=vendor.get("fiscalDateEnding"),
        vendor_reported_date=vendor.get("reportedDate"), vendor_report_time=vendor.get("reportTime"),
        official_actual_period_end=official.get("actual_period_end"), fiscal_labels_differ=None,
        publication_at=None, publication_date=None, publication_timezone=official.get("publication_timezone"),
        precision="unknown", known_by_at=None, available_session=None,
        known_by_is_actual_publication=False, date_only_upper_bound_at=None,
        source_url=official.get("source_url"), policy_version=POLICY_VERSION,
        exchange=EXCHANGE, issues=[])
    if asset_type == "etf":
        result.update(status="not_applicable", precision="not_applicable")
        return result
    if asset_type != "equity":
        result["issues"].append("unsupported_asset_type")
    if not result["symbol"]:
        result["issues"].append("missing_symbol")
    if official.get("verified") is not True or not official.get("source_url"):
        result["issues"].append("release_publication_evidence_not_verified")
    if official.get("publication_kind", "release") not in RELEASE_KINDS:
        result["issues"].append("timestamp_is_not_release_publication")
    try:
        actual_period = _date(official.get("actual_period_end"))
    except (ValueError, TypeError):
        actual_period = None
        result["issues"].append("invalid_official_actual_period_end")
    if actual_period is None:
        result["issues"].append("official_actual_period_end_pending_review")
    if actual_period is not None and vendor.get("fiscalDateEnding") is not None:
        result["fiscal_labels_differ"] = str(actual_period.date()) != str(vendor["fiscalDateEnding"])
    if result["issues"]:
        return result

    calendar = calendar or _default_calendar()
    raw_at = official.get("publication_at")
    raw_date = official.get("publication_date")
    zone = official.get("publication_timezone")
    # A date string put in a timestamp field stays date-only, not UTC midnight.
    if isinstance(raw_at, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_at):
        if raw_date and raw_date != raw_at:
            result["issues"].append("conflicting_publication_dates")
            return result
        raw_date, raw_at = raw_at, None
    try:
        if raw_at:
            published = _publication(raw_at, zone)
            source_zone = zone or pd.Timestamp(raw_at).tz
            local = published.tz_convert(source_zone)
            pub_date = local.tz_localize(None).normalize()
            if raw_date and _date(raw_date) != pub_date:
                raise ValueError("publication_date conflicts with exact timestamp in its source timezone")
            session, _ = _first_close(calendar, published)
            result.update(publication_at=published.isoformat(), publication_date=str(pub_date.date()),
                publication_timezone=str(source_zone),
                precision="exact", known_by_at=published.isoformat(), available_session=str(session.date()),
                known_by_is_actual_publication=True)
        else:
            pub_date = _date(raw_date)
            if pub_date is None:
                raise ValueError("release publication date/timestamp is missing; vendor hints are insufficient")
            if not zone:
                raise ValueError("date-only publication evidence requires a source timezone")
            # First instant of following LOCAL calendar day, minus1ns. Do not
            # add24hours to a localized midnight across a DST transition.
            next_local_midnight = (pub_date + pd.Timedelta(days=1)).tz_localize(
                zone, ambiguous="raise", nonexistent="raise")
            upper = (next_local_midnight - pd.Timedelta(nanoseconds=1)).tz_convert("UTC")
            session, conservative = _first_close(calendar, upper, strictly_after_date=pub_date)
            result.update(publication_date=str(pub_date.date()), precision="date_only",
                known_by_at=conservative.isoformat(), available_session=str(session.date()),
                date_only_upper_bound_at=upper.isoformat(), known_by_is_actual_publication=False)
        if actual_period > pub_date:
            raise ValueError("official fiscal period ends after publication; may be a calendar forecast")
    except Exception as exc:
        # Calendar bounds, invalid zones and DST ambiguities are unresolved
        # evidence, not a reason to fabricate an instant or availability.
        result.update(status="pending_review", known_by_at=None, available_session=None)
        result["issues"].append(str(exc))
        return result
    result["status"] = "ready"
    return result


def event_available(event, signal_asof):
    """Knowledge availability; not a claim of fillability or a reaction bar."""
    if event.get("status") != "ready" or not event.get("known_by_at"):
        return False
    if event.get("policy_version") != POLICY_VERSION:
        return False
    return bool(_aware(signal_asof, "signal_asof") >= _aware(event["known_by_at"], "known_by_at"))


def _bar_values(bars, sessions, column):
    if bars is None or not isinstance(bars, pd.DataFrame) or column not in bars:
        return None
    if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is not None
            or not bars.index.is_unique or not bars.index.equals(bars.index.normalize())):
        raise ValueError("bars need unique timezone-naive midnight session labels")
    try:
        values = bars.reindex(sessions)[column].to_numpy(dtype=float)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(values).all():
        return None
    return values


def reaction_features(event, bars, signal_asof, horizon=1, calendar=None, benchmark_bars=None):
    """Post-known close-to-close reaction after horizon complete XNYS sessions.

    Anchor is first session close at/after known_by, then endpoint is horizon
    sessions later. Thus even premarket exact releases use release-day close as
    anchor and do not claim to measure the announcement jump. A date-only event
    known conservatively at tomorrow's close needs one more session for horizon1.
    All observed prices/volumes used are at/after known_by and closed by asof.
    bars/benchmark_bars use session-indexed 'close', optional 'volume'. Missing
    or immature observations remain None; no filling and no zero substitute.
    """
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer session count")
    asof = _aware(signal_asof, "signal_asof")
    result = dict(status="pending_review", policy_version=POLICY_VERSION, horizon=int(horizon),
        signal_asof=asof.isoformat(), known_by_at=event.get("known_by_at"),
        anchor_session=None, end_session=None, anchor_close_at=None, end_close_at=None,
        post_known_close_to_close_return=None, benchmark_return=None, market_adjusted_return=None,
        reaction_volume_sum=None, reaction_volume_mean=None, volume_status="unavailable",
        benchmark_status="not_supplied" if benchmark_bars is None else "unavailable", issues=[])
    if event.get("status") == "not_applicable":
        result["status"] = "not_applicable"
        return result
    if event.get("status") != "ready" or event.get("policy_version") != POLICY_VERSION:
        result["issues"].append("event_not_verified_under_current_policy")
        return result
    if not event_available(event, asof):
        result["status"] = "not_yet_available"
        return result
    calendar = calendar or _default_calendar()
    known = _aware(event["known_by_at"], "known_by_at")
    try:
        anchor, anchor_close = _first_close(calendar, known)
        anchor_i = int(calendar.sessions.get_loc(anchor))
        end_i = anchor_i + int(horizon)
        if end_i >= len(calendar.sessions):
            raise ValueError("reaction endpoint outside supplied XNYS calendar")
        sessions = calendar.sessions[anchor_i:end_i + 1]
        end = sessions[-1]
        end_close = calendar.session_close(end)
    except Exception as exc:
        result["issues"].append(str(exc))
        return result
    result.update(anchor_session=str(anchor.date()), end_session=str(end.date()),
                  anchor_close_at=anchor_close.isoformat(), end_close_at=end_close.isoformat())
    if end_close > asof:
        result["status"] = "waiting_for_closed_window"
        return result
    assert known <= anchor_close <= end_close <= asof
    prices = _bar_values(bars, sessions, "close")
    if prices is None or (prices <= 0).any():
        result.update(status="missing_price_data")
        result["issues"].append("complete_positive_reaction_price_path_required")
        return result
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reaction = prices[-1] / prices[0] - 1.0
    if not np.isfinite(reaction):
        result["status"] = "missing_price_data"
        result["issues"].append("nonfinite_reaction_return")
        return result
    result.update(status="ready", post_known_close_to_close_return=float(reaction))
    volume = _bar_values(bars, sessions[1:], "volume")
    if volume is not None and (volume >= 0).all() and np.isfinite(volume.sum()):
        result.update(reaction_volume_sum=float(volume.sum()), reaction_volume_mean=float(volume.mean()),
                      volume_status="ready")
    benchmark = _bar_values(benchmark_bars, sessions, "close")
    if benchmark is not None and (benchmark > 0).all():
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            bench = benchmark[-1] / benchmark[0] - 1.0
        if np.isfinite(bench):
            result.update(benchmark_return=float(bench), market_adjusted_return=float(reaction - bench),
                          benchmark_status="ready")
    return result
