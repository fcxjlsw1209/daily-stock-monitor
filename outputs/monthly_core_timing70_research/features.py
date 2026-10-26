"""Fixed calendar-anchor and raw-core-smoothing hypotheses; no strategy fitting.

Calendar anchors come from the verified research XNYS calendar, never from the
last available price. Trading smoothers preserve the original observation-day
shift convention. The caller keeps its original trading eligibility and core
weight and falls back to its original core wherever a returned rank is NaN.
"""
from pathlib import Path
import hashlib
import importlib
import importlib.util

import numpy as np
import pandas as pd


CALENDAR_PATH = Path(__file__).resolve().parents[1]/"monthly_earnings_repair/research_calendar.py"
_SPEC = importlib.util.spec_from_file_location("core_timing70_verified_xnys", CALENDAR_PATH)
_CAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CAL)
FEATURE_NAMES = ("calendar_mixed", "calendar12_all", "trading_smooth5", "trading_smooth21")


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tz is not None or day != day.normalize():
        raise ValueError("Signal dates must be nonmissing timezone-naive midnight dates")
    return day


def _validate(adj, meta, calm_flags):
    if not isinstance(adj, pd.DataFrame) or not isinstance(adj.index, pd.DatetimeIndex):
        raise TypeError("adj must be a DataFrame indexed by exchange-session dates")
    if (adj.index.hasnans or adj.index.tz is not None or not adj.index.equals(adj.index.normalize())
            or not adj.index.is_unique or not adj.index.is_monotonic_increasing
            or not adj.columns.is_unique):
        raise ValueError("Price axes must be unique; dates naive, midnight and increasing")
    if not isinstance(meta, dict) or "dates" not in meta or "symbols" not in meta:
        raise ValueError("meta must supply dates and same-date symbol pools")
    dates = [_day(day) for day in meta["dates"]]
    pools = [list(symbols) for symbols in meta["symbols"]]
    if len(dates) != len(pools) or dates != sorted(set(dates)):
        raise ValueError("Signal dates must be unique, increasing and aligned with pools")
    width = len(pools[0]) if pools else 0
    if any(len(pool) != width for pool in pools):
        raise ValueError("All monthly symbol pools must have the same width")
    if any(len(set(pool)) != len(pool) or any(not isinstance(s, str) or not s for s in pool) for pool in pools):
        raise ValueError("Each pool must contain unique nonempty symbol strings")
    calm = np.asarray(calm_flags)
    if calm.shape != (len(dates),) or calm.dtype.kind != "b":
        raise ValueError("calm_flags must be an exact boolean vector aligned with signal dates")
    if any(day not in adj.index for day in dates):
        raise ValueError("Every signal must be an exact supplied price-index session")
    try:
        values = adj.to_numpy(dtype=float, copy=True, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise TypeError("Prices must be numeric or missing") from exc
    values[~np.isfinite(values) | (values <= 0.)] = np.nan
    return pd.DataFrame(values, index=adj.index.copy(), columns=adj.columns.copy()), dates, pools, calm.copy()


def _calendar(dates, prices):
    candidates = dates + ([prices.index[0], prices.index[-1]] if len(prices) else [])
    left = min(candidates + [pd.Timestamp("2000-01-01")])-pd.DateOffset(months=15)
    right = max(candidates + [pd.Timestamp("2035-12-31")])
    return _CAL.market_calendar(str(left.to_period("M").start_time.date()),
                                str(right.to_period("M").end_time.normalize().date()))


def _month_end(calendar, month):
    sessions = calendar.sessions_in_range(month.start_time.normalize(), month.end_time.normalize())
    if not len(sessions):
        raise ValueError("No XNYS sessions in anchor month " + str(month))
    return pd.Timestamp(sessions[-1])


def _anchors(calendar, day):
    if not calendar.is_session(day):
        raise ValueError("Signal is not an official XNYS session: " + str(day.date()))
    month_end = _month_end(calendar, day.to_period("M"))
    is_month_end = day == month_end
    actual, requested = {}, {}
    for lag in (2, 12, 14):
        if is_month_end:
            month = day.to_period("M")-lag
            requested[lag] = month.end_time.normalize()
            actual[lag] = _month_end(calendar, month)
        else:
            requested[lag] = day-pd.DateOffset(months=lag)
            actual[lag] = pd.Timestamp(calendar.date_to_session(requested[lag], direction="previous"))
        if actual[lag] >= day:
            raise AssertionError("Calendar anchor must precede its signal")
    return actual, requested, is_month_end, month_end


def _trading_panels(prices):
    """Original raw factors first; complete trailing means precede ranking."""
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=63).std(ddof=1)
    vol = vol.where(np.isfinite(vol) & (vol > 0.))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        raw = prices.shift(42)/prices.shift(252)-1.
        risk = (prices.shift(42)/prices.shift(294)-1.)/vol
    raw = raw.where(np.isfinite(raw))
    risk = risk.where(np.isfinite(risk))
    return vol, {window: (raw.rolling(window, min_periods=window).mean(),
                          risk.rolling(window, min_periods=window).mean()) for window in (5, 21)}


def _raw_features(prices, dates, pools, calm, calendar):
    """Expose raw values internally so formula tests can precede normalization."""
    width = len(pools[0]) if pools else 0
    raw = {name: np.full((len(dates), width), np.nan) for name in FEATURE_NAMES}
    vol, smoothed = _trading_panels(prices)
    records = []
    for i, (day, symbols) in enumerate(zip(dates, pools)):
        anchors, requests, is_month_end, scheduled_end = _anchors(calendar, day)
        p = prices.reindex(index=[anchors[lag] for lag in (2, 12, 14)], columns=symbols).to_numpy()
        sigma = vol.loc[day].reindex(symbols).to_numpy()
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            twelve = p[0]/p[1]-1.
            fourteen = p[0]/p[2]-1.
            raw["calendar_mixed"][i] = twelve if calm[i] else fourteen/sigma
            raw["calendar12_all"][i] = twelve if calm[i] else twelve/sigma
        for window in (5, 21):
            # The CURRENT flag selects a raw-factor family for the entire mean.
            # Historical market flags are deliberately not involved.
            selected = smoothed[window][0 if calm[i] else 1]
            raw["trading_smooth"+str(window)][i] = selected.loc[day].reindex(symbols).to_numpy()
        t = prices.index.get_loc(day)
        records.append(dict(signal_date=str(day.date()), calm=bool(calm[i]),
            official_month_end=str(scheduled_end.date()), signal_is_official_month_end=is_month_end,
            anchor_rule="calendar_month_end" if is_month_end else "dateoffset_then_previous_XNYS_session",
            requested_anchors={str(lag): str(requests[lag].date()) for lag in (2, 12, 14)},
            actual_anchors={str(lag): str(anchors[lag].date()) for lag in (2, 12, 14)},
            anchor_sessions_present={str(lag): anchors[lag] in prices.index for lag in (2, 12, 14)},
            smoothing_observation_starts={str(w): str(prices.index[t-w+1].date()) if t >= w-1 else None for w in (5, 21)}))
    for value in raw.values():
        value[~np.isfinite(value)] = np.nan
    return raw, records


def _source_hashes(provenance):
    paths = {Path(__file__).resolve(), CALENDAR_PATH.resolve(), Path(provenance["manifest"])}
    paths.update(Path(row["source_file"]) for row in provenance["closures"])
    for name in ("exchange_calendars.exchange_calendar_xnys", "exchange_calendars.exchange_calendar",
                 "exchange_calendars.us_holidays", "exchange_calendars.calendar_helpers", "pandas.tseries.holiday"):
        paths.add(Path(importlib.import_module(name).__file__).resolve())
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def build_features(adj, meta, calm_flags):
    """Return four [signals,pool-width] arrays of finite-only percentile ranks.

    Usual monthends use official monthends 2/12/14 months earlier. An explicitly
    supplied non-monthend signal (including 2016-08-17) instead subtracts each
    calendar-month offset from that day and uses the previous official session.
    Missing anchor prices do not move the scheduled anchor or shorten a month.
    calendar_mixed uses raw R(12,2) in calm markets and R(14,2)/recent_sigma63
    otherwise. calendar12_all uses R(12,2) in both cases, dividing by sigma only
    outside calm markets. Risk sigma uses 63 complete simple returns, ddof=1.
    Smoothers average 5/21 consecutive raw-factor observation days, including
    the signal day; the current state chooses raw252 or risk294 for all days.
    Rank each resulting signal row across its entire supplied pool, ignoring
    only NaNs. Return NaN for missing factors, never an eligibility filter.
    """
    prices, dates, pools, calm = _validate(adj, meta, calm_flags)
    calendar = _calendar(dates, prices)
    raw, records = _raw_features(prices, dates, pools, calm, calendar)
    ranked = {name: pd.DataFrame(value).rank(axis=1, pct=True, method="average").sub(.5).to_numpy()
              for name, value in raw.items()}
    provenance = _CAL.provenance()
    audit = dict(feature_names=list(FEATURE_NAMES), signals=len(dates), symbols_per_signal=len(pools[0]) if pools else 0,
        calendar=provenance, anchors=records,
        calendar_formula="calm:R(12,2); otherwise calendar_mixed:R(14,2)/sigma63, calendar12_all:R(12,2)/sigma63",
        trading_formula="raw252=P.shift(42)/P.shift(252)-1; risk294=(P.shift(42)/P.shift(294)-1)/sigma63",
        smoothing="Complete trailing5/21 raw observations including signal; current calm flag chooses family before averaging; rank AFTER mean",
        trading_clock="Supplied observation-day index, preserving original pandas shift convention; no resampling or price filling",
        volatility="63 complete simple daily returns, sample std(ddof=1), ending at each raw observation date",
        missing_policy="Missing/nonpositive anchors, incomplete raw smoothing windows, or zero/nonfinite risk denominator produce NaN; caller falls back to original core",
        normalization="Average-tie percentile rank across all finite values in each supplied same-date pool, minus0.5; no eligible-only filter",
        current_market_state_smoothed=False, trading_eligibility_modified=False,
        finite_feature_cells={name: int(np.isfinite(value).sum()) for name, value in ranked.items()},
        missing_feature_cells={name: int(np.isnan(value).sum()) for name, value in ranked.items()},
        source_hashes=_source_hashes(provenance), dependency_versions={"numpy": np.__version__, "pandas": pd.__version__})
    return ranked, audit
