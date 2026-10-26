"""Fixed one/two/three-calendar-month labels; monthly trading remains external.

XNYS fixes each future month-end before examining prices. Only completed,
positive, finite paths become training labels. Labels never mutate signal-time
eligibility or shorten an unknown future horizon to the cache's final row.
"""
from pathlib import Path
import importlib.util
import hashlib

import numpy as np
import pandas as pd

CALENDAR_PATH = Path(__file__).resolve().parents[1] / "monthly_earnings_repair/research_calendar.py"
_SPEC = importlib.util.spec_from_file_location("horizon70_verified_xnys", CALENDAR_PATH)
_CAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CAL)
HORIZONS = (1, 2, 3)


def _day(value, name):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tz is not None or day != day.normalize():
        raise ValueError(name + " must be a timezone-naive midnight date")
    return day


def _validate(px, meta, train_eligible):
    if not isinstance(px, pd.DataFrame) or not isinstance(px.index, pd.DatetimeIndex):
        raise ValueError("px must be a DataFrame indexed by price-session dates")
    if (px.index.tz is not None or not px.index.equals(px.index.normalize())
            or not px.index.is_unique or not px.index.is_monotonic_increasing
            or not px.columns.is_unique):
        raise ValueError("Price axes must be unique; session dates naive, midnight and increasing")
    if not isinstance(meta, dict) or "dates" not in meta or "symbols" not in meta:
        raise ValueError("meta needs signal dates and per-signal symbol pools")
    dates = [_day(day, "signal") for day in meta["dates"]]
    pools = [list(symbols) for symbols in meta["symbols"]]
    if len(dates) != len(pools) or len(set(dates)) != len(dates):
        raise ValueError("Signal dates and symbol pools must align; dates must be unique")
    if len({day.to_period("M") for day in dates}) != len(dates):
        raise ValueError("At most one training signal per calendar month")
    for pool in pools:
        if not all(isinstance(symbol, str) and symbol for symbol in pool) or len(set(pool)) != len(pool):
            raise ValueError("Each symbol pool needs unique nonempty string identifiers")
    eligible = np.asarray(train_eligible)
    if eligible.dtype.kind != "b" or eligible.shape != (sum(map(len, pools)),):
        raise ValueError("train_eligible must be a flat boolean array matching supplied pools")
    if "ends" in meta and len(meta["ends"]) != len(dates):
        raise ValueError("Optional existing one-month ends must align with signals")
    return dates, pools, eligible


def _calendar(dates, px):
    left = min(dates + ([px.index[0]] if len(px) else []) + [pd.Timestamp("2000-01-01")])
    right = max([day.to_period("M").end_time.normalize() for day in dates]
                + ([px.index[-1]] if len(px) else []) + [pd.Timestamp("2035-12-31")])
    right = max(right, max([day.to_period("M") + 3 for day in dates], default=pd.Period("2035-12", "M")).end_time.normalize())
    return _CAL.market_calendar(start=str(left.to_period("M").start_time.date()),
                                end=str(right.to_period("M").end_time.normalize().date()))


def _month_sessions(calendar, month):
    return calendar.sessions_in_range(month.start_time.normalize(), month.end_time.normalize())


def _month_end(calendar, month):
    sessions = _month_sessions(calendar, month)
    if not len(sessions):
        raise ValueError("XNYS has no scheduled sessions in target month " + str(month))
    return pd.Timestamp(sessions[-1])


def _price_coverage(calendar, px):
    if not len(px):
        return dict(price_first_session=None, price_last_session=None,
                    final_price_month_scheduled_end=None, final_price_month_complete=False,
                    last_complete_price_calendar_month=None)
    last_month = px.index[-1].to_period("M")
    last_end = _month_end(calendar, last_month)
    complete = bool(_month_sessions(calendar, last_month).isin(px.index).all())
    month = last_month
    first_month = px.index[0].to_period("M")
    last_complete = None
    while month >= first_month:
        if _month_sessions(calendar, month).isin(px.index).all():
            last_complete = str(month)
            break
        month -= 1
    return dict(price_first_session=str(px.index[0].date()), price_last_session=str(px.index[-1].date()),
                final_price_month_scheduled_end=str(last_end.date()), final_price_month_complete=complete,
                last_complete_price_calendar_month=last_complete)


def _source_hashes(provenance):
    paths = {Path(__file__).resolve(), CALENDAR_PATH.resolve(), Path(provenance["manifest"])}
    paths.update(Path(row["source_file"]) for row in provenance["closures"])
    for name in ("exchange_calendars.exchange_calendar_xnys", "exchange_calendars.exchange_calendar",
                 "exchange_calendars.us_holidays", "exchange_calendars.calendar_helpers", "pandas.tseries.holiday"):
        module = importlib.import_module(name)
        paths.add(Path(module.__file__).resolve())
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def build_labels(px, meta, train_eligible):
    """Return ``(arrays, audit)`` with flat target/end/good for horizons1/2/3.

    Candidate order is the concatenation of meta['symbols']. Each end_hN is a
    real XNYS session date, including beyond-price-cache unknown labels. Within
    a signal/horizon, only signal-eligible names with complete finite positive
    paths enter centering, and at least two such names are required. SPY must
    also have a complete path. Optional meta['ends'] is verified against H1.
    """
    dates, pools, eligible = _validate(px, meta, train_eligible)
    calendar = _calendar(dates, px)
    n = len(eligible)
    arrays = {}
    for h in HORIZONS:
        arrays["target_h" + str(h)] = np.full(n, np.nan)
        arrays["end_h" + str(h)] = np.empty(n, dtype="datetime64[D]")
        arrays["good_h" + str(h)] = np.zeros(n, dtype=bool)
    records = []
    offset = 0
    metadata_end_checks = 0
    for i, (day, symbols) in enumerate(zip(dates, pools)):
        if not calendar.is_session(day):
            raise ValueError("Training signal is not an XNYS session: " + str(day.date()))
        sl = slice(offset, offset + len(symbols))
        local_eligible = eligible[sl]
        for h in HORIZONS:
            month = day.to_period("M") + h
            end = _month_end(calendar, month)
            arrays["end_h" + str(h)][sl] = np.datetime64(end.date(), "D")
            if h == 1 and "ends" in meta and meta["ends"][i] is not None:
                existing = _day(meta["ends"][i], "existing one-month label end")
                if existing != end:
                    raise ValueError("Existing one-month end conflicts with XNYS for " + str(day.date()))
                metadata_end_checks += 1
            path_sessions = calendar.sessions_in_range(day, end)
            target_sessions = _month_sessions(calendar, month)
            missing_sessions = path_sessions[~path_sessions.isin(px.index)]
            target_complete = bool(target_sessions.isin(px.index).all())
            row = dict(signal_date=str(day.date()), horizon=h, target_month=str(month),
                       label_end=str(end.date()), required_sessions=len(path_sessions),
                       eligible_rows=int(local_eligible.sum()), target_month_complete=target_complete,
                       missing_session_count=len(missing_sessions),
                       first_missing_session=str(missing_sessions[0].date()) if len(missing_sessions) else None,
                       status="unknown_price_path", valid_eligible_rows=0, centered_rows=0,
                       missing_symbols=[], invalid_path_symbols=[], benchmark_path_valid=False)
            # Calendar dates are fixed even if the horizon has not elapsed in
            # the input cache. Never replace end by the last observed close.
            if not len(missing_sessions) and target_complete:
                names = list(symbols) + ([] if "SPY" in symbols else ["SPY"])
                missing_symbols = [symbol for symbol in names if symbol not in px.columns]
                row["missing_symbols"] = missing_symbols
                # Reindex exposes absent prices but never fills them.
                try:
                    values = px.reindex(index=path_sessions, columns=names).to_numpy(dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Prices must be numeric") from exc
                positive = np.isfinite(values) & (values > 0)
                complete = positive.all(axis=0)
                benchmark = names.index("SPY")
                row["benchmark_path_valid"] = bool(complete[benchmark])
                row["invalid_path_symbols"] = [symbol for j, symbol in enumerate(symbols)
                                                if local_eligible[j] and not complete[j]]
                supported = local_eligible & complete[:len(symbols)] & complete[benchmark]
                row["valid_eligible_rows"] = int(supported.sum())
                if not complete[benchmark]:
                    row["status"] = "invalid_benchmark_path"
                elif supported.sum() < 2:
                    row["status"] = "fewer_than_two_complete_eligible"
                else:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        gross = values[-1, :len(symbols)][supported] / values[0, :len(symbols)][supported]
                        spy_gross = values[-1, benchmark] / values[0, benchmark]
                        # The -1 terms cancel. H1 therefore matches the older
                        # clip(simple_return-SPY_return) definition exactly.
                        excess = np.power(gross, 1. / h) - np.power(spy_gross, 1. / h)
                    finite = np.isfinite(excess)
                    if not finite.all():
                        indices = np.flatnonzero(supported)
                        supported[indices[~finite]] = False
                        excess = excess[finite]
                    row["valid_eligible_rows"] = int(supported.sum())
                    if supported.sum() >= 2:
                        clipped = np.clip(excess, -1., 1.)
                        centered = clipped - clipped.mean()
                        target = arrays["target_h" + str(h)][sl]
                        good = arrays["good_h" + str(h)][sl]
                        target[supported] = centered
                        good[supported] = True
                        row.update(status="complete", centered_rows=int(supported.sum()),
                                   clipped_rows=int((np.abs(excess) > 1.).sum()),
                                   centering_mean=float(clipped.mean()))
                    else:
                        row["status"] = "fewer_than_two_finite_payoffs"
            records.append(row)
        offset += len(symbols)
    provenance = _CAL.provenance()
    audit = dict(
        method="xnys_calendar_month_1_2_3_equivalent_monthly_excess_labels",
        horizons=list(HORIZONS), rows=n, signal_months=len(dates),
        formula="clip((P_end/P_signal)**(1/h)-(SPY_end/SPY_signal)**(1/h),-1,1), then mean-center",
        horizon_units="calendar months; target end is exact last XNYS session of that month",
        path_rule="every XNYS session from signal through exact end; finite strictly positive asset and SPY prices; no filling",
        centering="independent per horizon among signal-eligible complete assets, minimum2",
        eligibility_policy="train_eligible untouched; future completeness only controls training labels",
        missing_end_policy="preserve scheduled datetime64[D] end beyond cache; targetNaN/goodFalse; never shorten horizon",
        h1_old_target_relation="same clipping/centering formula on complete common months; old ordinal_model drops whole month if any eligible asset path is missing, while this per-asset policy retains at least2 complete eligible names",
        maturity="caller must require end_hN<=annual previousDec31; for matched H1/H2/H3 use shared good and end_h3 maturity",
        common_comparison="caller intersects good_h1/h2/h3 and re-centers each target on that common month cohort before shared H3 maturity",
        overlap={"h1": 0, "h2": 1, "h3": 2},
        overlap_units="adjacent monthly anchors share this many future return months; not independent label samples",
        minimum_centering_assets=2, benchmark="SPY", metadata_end_checks=metadata_end_checks,
        good_rows={str(h): int(arrays["good_h" + str(h)].sum()) for h in HORIZONS},
        complete_signal_months={str(h): sum(row["horizon"] == h and row["status"] == "complete" for row in records) for h in HORIZONS},
        calendar={**provenance, "source_module": str(CALENDAR_PATH)},
        source_hashes=_source_hashes(provenance),
        price_coverage=_price_coverage(calendar, px), months=records,
    )
    return arrays, audit
