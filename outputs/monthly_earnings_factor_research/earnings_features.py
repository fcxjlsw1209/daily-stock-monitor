"""Strict, point-in-time extraction of the frozen earnings reaction panel.

No downloads, filling, candidate selection, fitting, or source-file writes.
Prices are adjusted closes. Event values retain their repaired-source values;
the old backtest price panel is used for matched recent returns and a separate
non-mutating event-price consistency audit.
"""
from collections import Counter
from functools import lru_cache
import importlib.util
from pathlib import Path
import re

import numpy as np
import pandas as pd

POLICY_VERSION = "earnings_publication_availability_v1"
HORIZONS = (10, 21)
ML_FEATURE_NAMES = ("event10", "event21", "event21_minus_event10")
RETURN_KEYS = ("post_known_close_to_close_return", "market_adjusted_return",
               "benchmark_return", "reaction_volume_sum", "reaction_volume_mean")
CROSSCHECK_THRESHOLD = 0.0005


@lru_cache(maxsize=1)
def _calendar():
    path = Path(__file__).resolve().parent.parent / "monthly_earnings_repair/research_calendar.py"
    spec = importlib.util.spec_from_file_location("earnings_factor_research_calendar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.market_calendar()


def _session(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(label + " must be an exact YYYY-MM-DD session label")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(label + " is invalid")
    return result


def _aware(value, label):
    if value is None:
        raise ValueError(label + " is missing")
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is None:
        raise ValueError(label + " needs an explicit timezone")
    return result.tz_convert("UTC")


def _finite(value, label):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(label + " must be a finite numeric value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(label + " must be a finite numeric value")
    return result


def _mapping(feature_document, meta):
    if not isinstance(meta, dict) or not isinstance(feature_document, dict):
        raise ValueError("feature_document and meta must be dictionaries")
    dates, symbols = meta.get("dates"), meta.get("symbols")
    if not isinstance(dates, list) or not dates or not isinstance(symbols, list):
        raise ValueError("meta requires nonempty dates and per-date symbols lists")
    if len(dates) != len(symbols):
        raise ValueError("meta dates/symbols length mismatch")
    parsed = [_session(day, "meta date") for day in dates]
    if len(set(dates)) != len(dates) or parsed != sorted(parsed):
        raise ValueError("meta dates must be unique and increasing")
    expected = {}
    width = None
    for i, (day, names) in enumerate(zip(dates, symbols)):
        if not isinstance(names, list) or not names:
            raise ValueError("meta symbols must contain nonempty per-date lists")
        if width is None:
            width = len(names)
        if len(names) != width:
            raise ValueError("meta candidate counts must be rectangular")
        for j, symbol in enumerate(names):
            if not isinstance(symbol, str) or not symbol or symbol != symbol.strip():
                raise ValueError("meta symbols must be nonempty exact identifiers")
            key = (day, symbol)
            if key in expected:
                raise ValueError("duplicate meta candidate: " + repr(key))
            expected[key] = (i, j)
    rows = feature_document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("feature_document.rows must be a list")
    mapped = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every feature candidate must be a dictionary")
        day, symbol = row.get("signal_date"), row.get("symbol")
        _session(day, "feature signal_date")
        if not isinstance(symbol, str):
            raise ValueError("feature symbol must be an exact string identifier")
        key = (day, symbol)
        if key in mapped:
            raise ValueError("duplicate feature candidate: " + repr(key))
        if key not in expected:
            raise ValueError("unmatched feature candidate: " + repr(key))
        mapped[key] = row
    missing = set(expected) - set(mapped)
    if missing:
        raise ValueError("missing feature candidates: " + repr(sorted(missing)[:10]))
    return dates, expected, mapped, (len(dates), width)


def _price_path(px, sessions, symbol):
    if symbol not in px.columns or "SPY" not in px.columns:
        return None
    # Selecting SPY as both stock and benchmark is harmless, though ETFs are
    # excluded before paths are requested. Reindex detects missing whole bars.
    try:
        values = px.reindex(sessions)[[symbol, "SPY"]].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(values).all() or (values <= 0).any():
        return None
    return values


def _excess_return(values):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = (values[-1, 0] / values[0, 0] - 1.0
                 - (values[-1, 1] / values[0, 1] - 1.0))
    if not np.isfinite(value):
        raise ValueError("complete price path produced a nonfinite return")
    return float(value)


def _reaction_window(reaction, horizon, asof, calendar, context):
    if reaction.get("policy_version") != POLICY_VERSION:
        raise ValueError(context + ": unexpected reaction policy")
    if reaction.get("horizon") != horizon or isinstance(reaction.get("horizon"), bool):
        raise ValueError(context + ": reaction horizon mismatch")
    if _aware(reaction.get("signal_asof"), context + " reaction asof") != asof:
        raise ValueError(context + ": reaction signal_asof mismatch")
    known = _aware(reaction.get("known_by_at"), context + " known_by_at")
    anchor = _session(reaction.get("anchor_session"), context + " anchor_session")
    end = _session(reaction.get("end_session"), context + " end_session")
    anchor_close = _aware(reaction.get("anchor_close_at"), context + " anchor_close_at")
    end_close = _aware(reaction.get("end_close_at"), context + " end_close_at")
    if not known <= anchor_close < end_close <= asof:
        raise ValueError(context + ": future or invalid reaction window")
    if anchor not in calendar.sessions or end not in calendar.sessions:
        raise ValueError(context + ": reaction session is not XNYS")
    if calendar.session_close(anchor) != anchor_close or calendar.session_close(end) != end_close:
        raise ValueError(context + ": reaction timestamp is not the canonical session close")
    anchor_i = calendar.sessions.get_loc(anchor)
    end_i = calendar.sessions.get_loc(end)
    if end_i - anchor_i != horizon:
        raise ValueError(context + ": reaction is not the stated completed-session horizon")
    previous = calendar.sessions[anchor_i - 1] if anchor_i > 0 else None
    if previous is not None and calendar.session_close(previous) >= known:
        raise ValueError(context + ": reaction anchor is not first close at/after known_by_at")
    return calendar.sessions[anchor_i:end_i + 1], known


def build_features(feature_document, meta, px, etfs):
    """Return ``(arrays, audit)`` aligned exactly to ``meta`` candidate order.

    ``meta`` is one scope's dictionary with dates and per-date symbols (49 in
    the frozen study). ``px`` has unique, ordered, timezone-naive midnight
    session labels and adjusted-close symbol columns including SPY. ``etfs``
    is the frozen set/list of ETF identifiers, never inferred from prices.

    event10/event21 and recent10/recent21 are float arrays (months,candidates).
    matched_recent10/matched_recent21 alias recent10/recent21. valid10/valid21
    describe earnings-feature availability, not trading eligibility. Invalid
    values remain NaN. valid_ml requires both horizons; ml_features has shape
    (months,candidates,3) and all three fields are NaN outside valid_ml.
    Unknown/ETF/immature rows never receive scores. An incomplete recent path
    for a usable event is an error, so the matched control cannot change masks.
    """
    dates, positions, rows, shape = _mapping(feature_document, meta)
    if not isinstance(px, pd.DataFrame) or not isinstance(px.index, pd.DatetimeIndex):
        raise ValueError("px must be a DataFrame with a DatetimeIndex")
    if (px.index.tz is not None or not px.index.is_unique or not px.index.is_monotonic_increasing
            or not px.index.equals(px.index.normalize()) or not px.columns.is_unique):
        raise ValueError("px requires unique ordered midnight sessions and unique columns")
    if isinstance(etfs, str) or etfs is None:
        raise ValueError("etfs must explicitly list the frozen ETF symbols")
    etfs = set(etfs)
    if not all(isinstance(symbol, str) for symbol in etfs):
        raise ValueError("etfs must contain exact string identifiers")
    calendar = _calendar()
    arrays = {name: np.full(shape, np.nan, dtype=float)
              for name in ("event10", "event21", "recent10", "recent21")}
    row_counts, reaction_counts, exclusion_counts = Counter(), Counter(), Counter()
    crosscheck = dict(threshold=CROSSCHECK_THRESHOLD, checked=0, incomplete_paths=0,
                      max_abs_diff=None, differences_over_5bp=0, examples=[])
    for (day, symbol), position in positions.items():
        row = rows[day, symbol]
        context = day + " " + symbol
        session = _session(day, "signal_date")
        if session not in calendar.sessions:
            raise ValueError(context + ": signal_date is not an XNYS session")
        asof = _aware(row.get("signal_asof"), context + " signal_asof")
        if asof != calendar.session_close(session):
            raise ValueError(context + ": signal_asof is not the canonical XNYS close")
        status = row.get("status")
        if not isinstance(status, str):
            raise ValueError(context + ": row status is missing")
        row_counts[status] += 1
        reactions = row.get("reactions")
        if not isinstance(reactions, dict):
            raise ValueError(context + ": reactions must be a dictionary")
        # Audit even unused horizons, before selecting usable factor values.
        for key, reaction in reactions.items():
            if not isinstance(reaction, dict):
                raise ValueError(context + ": reaction must be a dictionary")
            if reaction.get("status") != "ready" and any(reaction.get(k) is not None for k in RETURN_KEYS):
                raise ValueError(context + ": non-ready reaction contains a numeric value (horizon " + str(key) + ")")
        if symbol in etfs or status != "verified_event_available":
            if reactions:
                raise ValueError(context + ": ETF/unverified candidate must not contain reactions")
            exclusion_counts["etf" if symbol in etfs else status] += 1
            continue
        if not row.get("event_id"):
            raise ValueError(context + ": verified candidate lacks event_id")
        if not {str(h) for h in HORIZONS}.issubset(reactions):
            raise ValueError(context + ": verified candidate lacks a required reaction horizon")
        known_times = set()
        for horizon in HORIZONS:
            reaction = reactions[str(horizon)]
            reaction_status = reaction.get("status")
            reaction_counts[str(horizon) + ":" + str(reaction_status)] += 1
            if reaction_status != "ready":
                exclusion_counts[str(horizon) + ":" + str(reaction_status)] += 1
                continue
            sessions, known = _reaction_window(reaction, horizon, asof, calendar, context)
            known_times.add(known)
            raw_value = reaction.get("market_adjusted_return")
            if raw_value is None:
                exclusion_counts[str(horizon) + ":missing_market_adjusted_return"] += 1
                continue
            event_value = _finite(raw_value, context + " market_adjusted_return")
            if reaction.get("benchmark_status") != "ready":
                raise ValueError(context + ": market-adjusted value lacks a ready benchmark")
            stock_value = _finite(reaction.get("post_known_close_to_close_return"), context + " stock return")
            benchmark_value = _finite(reaction.get("benchmark_return"), context + " benchmark return")
            if not np.isclose(event_value, stock_value - benchmark_value, atol=1e-12, rtol=1e-12):
                raise ValueError(context + ": inconsistent source market-adjusted return")
            signal_i = calendar.sessions.get_loc(session)
            if signal_i < horizon:
                raise ValueError(context + ": insufficient calendar for recent path")
            recent_sessions = calendar.sessions[signal_i - horizon:signal_i + 1]
            recent_path = _price_path(px, recent_sessions, symbol)
            if recent_path is None:
                raise ValueError(context + ": incomplete recent" + str(horizon)
                                 + " price path for a valid event; matched mask cannot change")
            arrays["event" + str(horizon)][position] = event_value
            arrays["recent" + str(horizon)][position] = _excess_return(recent_path)
            event_path = _price_path(px, sessions, symbol)
            if event_path is None:
                crosscheck["incomplete_paths"] += 1
            else:
                recomputed = _excess_return(event_path)
                difference = abs(recomputed - event_value)
                crosscheck["checked"] += 1
                crosscheck["max_abs_diff"] = max(crosscheck["max_abs_diff"] or 0.0, difference)
                if difference > CROSSCHECK_THRESHOLD:
                    crosscheck["differences_over_5bp"] += 1
                    if len(crosscheck["examples"]) < 25:
                        crosscheck["examples"].append(dict(signal_date=day, symbol=symbol, horizon=horizon,
                            event_id=row["event_id"], anchor_session=reaction["anchor_session"],
                            end_session=reaction["end_session"], source_value=event_value,
                            panel_recomputed_value=recomputed, abs_diff=difference))
        if len(known_times) > 1:
            raise ValueError(context + ": horizons use different event knowledge times")
    for horizon in HORIZONS:
        arrays["valid" + str(horizon)] = np.isfinite(arrays["event" + str(horizon)])
        if not np.array_equal(arrays["valid" + str(horizon)], np.isfinite(arrays["recent" + str(horizon)])):
            raise AssertionError("matched recent control changed the event mask")
        arrays["matched_recent" + str(horizon)] = arrays["recent" + str(horizon)]
    valid_ml = arrays["valid10"] & arrays["valid21"]
    ml = np.full(shape + (3,), np.nan, dtype=float)
    ml[valid_ml] = np.column_stack((arrays["event10"][valid_ml], arrays["event21"][valid_ml],
                                   arrays["event21"][valid_ml] - arrays["event10"][valid_ml]))
    arrays.update(valid_ml=valid_ml, ml_features=ml)
    audit = dict(candidate_rows=len(positions), signal_dates=len(dates), candidates_per_date=shape[1],
                 exact_candidate_mapping=True, duplicate_candidates=0, future_reaction_values=0,
                 zero_filled_missing_reactions=0, matched_masks_equal=True,
                 valid10=int(arrays["valid10"].sum()), valid21=int(arrays["valid21"].sum()),
                 valid_ml=int(valid_ml.sum()), row_status_counts=dict(row_counts),
                 reaction_status_counts=dict(reaction_counts), exclusion_counts=dict(exclusion_counts),
                 ml_feature_names=list(ML_FEATURE_NAMES), event_price_crosscheck=crosscheck)
    return arrays, audit
