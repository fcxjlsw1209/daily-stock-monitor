"""Four fixed continuous next-calendar-month targets on one cached cohort.

This module builds labels only. It does not fit models, select parameters, or
change signal-time trading eligibility. Existing ``train_good`` preserves the
upstream complete-price-path rule; new validity checks only restrict that mask.
"""
import numpy as np
import pandas as pd


ARMS = ("simple_excess", "log_excess", "vol_scaled", "loss_utility")
CLIP_LIMITS = dict(simple_excess=1.0, log_excess=1.0,
                   vol_scaled=5.0, loss_utility=1.0)
VOL_RETURNS = 63
VOL_DDOF = 1
MONTH_SCALE = np.sqrt(21.0)
LOSS_PENALTY = 0.5
MIN_MONTH_ROWS = 2


def _days(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind in "biufc":
        raise ValueError(name + " must be a one-dimensional date array")
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must contain valid dates") from exc
    if index.hasnans or index.tz is not None or not index.equals(index.normalize()):
        raise ValueError(name + " must contain timezone-naive midnight dates without NaT")
    return index.to_numpy(dtype="datetime64[D]")


def _bool(values, n, name):
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype.kind != "b":
        raise ValueError(name + " must be a length-N boolean array")
    return result


def _validate(adj, dates, ends, symbols, train_eligible, train_good, features_finite):
    dates, ends = _days(dates, "dates"), _days(ends, "ends")
    if dates.shape != ends.shape:
        raise ValueError("dates and ends shapes differ")
    n = len(dates)
    months = dates.astype("datetime64[M]")
    if np.any(ends <= dates) or np.any(ends.astype("datetime64[M]") - months != 1):
        raise ValueError("labels must end after the signal in the next calendar month")
    symbols = np.asarray(symbols)
    if symbols.shape != (n,):
        raise ValueError("symbols must be a flattened length-N array")
    if any(not isinstance(s, str) or not s or s != s.strip() for s in symbols):
        raise ValueError("symbols must contain nonempty exact string identifiers")
    masks = [_bool(value, n, name) for value, name in
             [(train_eligible, "train_eligible"), (train_good, "train_good"),
              (features_finite, "features_finite")]]
    for month in np.unique(months):
        rows = months == month
        if len(np.unique(dates[rows])) != 1 or len(np.unique(ends[rows])) != 1:
            raise ValueError("each calendar month must have one signal date and one label end")
        if len(set(symbols[rows])) != int(rows.sum()):
            raise ValueError("duplicate security ID within a signal month")
    if not isinstance(adj, pd.DataFrame) or not isinstance(adj.index, pd.DatetimeIndex):
        raise ValueError("adj must be a DataFrame with a DatetimeIndex")
    index = adj.index
    if index.hasnans or index.tz is not None or not index.equals(index.normalize()):
        raise ValueError("adj index must contain timezone-naive midnight dates without NaT")
    if not index.is_unique or not index.is_monotonic_increasing or not adj.columns.is_unique:
        raise ValueError("adj index must be unique and sorted, and columns must be unique")
    return dates, ends, symbols, months, masks


def _prices(adj, row_positions, symbol):
    """Read only requested rows; unsupported prices remain missing, never filled."""
    if symbol not in adj.columns:
        return None
    try:
        prices = adj.iloc[row_positions][symbol].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.all(np.isfinite(prices) & (prices > 0)):
        return None
    return prices


def build_targets(adj, dates, ends, symbols, train_eligible, train_good, features_finite):
    """Return ``targets``, common ``good``, ``vol63`` and a JSON-safe ``audit``.

    All row inputs are flat arrays in the existing cache order. ``features_finite``
    is a boolean row mask supplied by the caller. Four continuous raw labels use
    exact signal/end adjusted prices. They are clipped to their fixed limits,
    then mean-centered over the SAME valid cohort in each signal month; months
    with fewer than two valid rows have no labels. Centering may exceed a clip
    bound, because the bounds apply before centering.

    Volatility uses 63 simple returns from the last 64 positive finite session
    prices up to and including signal close, with sample std (ddof=1). Missing
    exact endpoints are not shortened or filled. Only these historical prices
    and the specified label end are read; cached ``train_good`` independently
    retains the upstream full future-path requirement.
    """
    dates, ends, symbols, months, masks = _validate(
        adj, dates, ends, symbols, train_eligible, train_good, features_finite)
    eligible, cached_good, finite_X = masks
    n = len(dates)
    raw = {arm: np.full(n, np.nan) for arm in ARMS}
    targets = {arm: np.full(n, np.nan) for arm in ARMS}
    vol63 = np.full(n, np.nan)
    endpoint_valid = np.zeros(n, dtype=bool)
    good = np.zeros(n, dtype=bool)
    base_cohort = eligible & cached_good & finite_X
    audit_months = []

    for month in np.unique(months):
        rows = np.flatnonzero(months == month)
        date, end = pd.Timestamp(dates[rows[0]]), pd.Timestamp(ends[rows[0]])
        signal_pos, end_pos = adj.index.get_indexer([date, end])
        if signal_pos >= 0:
            for row in rows:
                if signal_pos < VOL_RETURNS:
                    continue
                history = _prices(adj, slice(signal_pos - VOL_RETURNS, signal_pos + 1), symbols[row])
                if history is None:
                    continue
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    returns = history[1:] / history[:-1] - 1.0
                    value = np.std(returns, ddof=VOL_DDOF)
                if np.isfinite(returns).all() and np.isfinite(value):
                    vol63[row] = value

        if signal_pos >= 0 and end_pos >= 0:
            positions = [signal_pos, end_pos]
            spy_prices = _prices(adj, positions, "SPY")
            if spy_prices is not None:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    spy_gross = spy_prices[1] / spy_prices[0]
                    spy_return = spy_gross - 1.0
                    spy_log = np.log(spy_gross)
                for row in rows:
                    prices = _prices(adj, positions, symbols[row])
                    if prices is None:
                        continue
                    endpoint_valid[row] = True
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        gross = prices[1] / prices[0]
                        asset_return = gross - 1.0
                        excess = asset_return - spy_return
                        raw["simple_excess"][row] = excess
                        raw["log_excess"][row] = np.log(gross) - spy_log
                        raw["vol_scaled"][row] = excess / (vol63[row] * MONTH_SCALE)
                        raw["loss_utility"][row] = excess - LOSS_PENALTY * max(0.0, -asset_return)

        finite_raw = np.logical_and.reduce([np.isfinite(raw[arm][rows]) for arm in ARMS])
        positive_vol = np.isfinite(vol63[rows]) & (vol63[rows] > 0)
        common = base_cohort[rows] & endpoint_valid[rows] & positive_vol & finite_raw
        selected = rows[common]
        clip_counts = {arm: 0 for arm in ARMS}
        if len(selected) >= MIN_MONTH_ROWS:
            good[selected] = True
            for arm in ARMS:
                limit = CLIP_LIMITS[arm]
                values = raw[arm][selected]
                clip_counts[arm] = int(np.count_nonzero(np.abs(values) > limit))
                clipped = np.clip(values, -limit, limit)
                targets[arm][selected] = clipped - clipped.mean()
        audit_months.append(dict(
            month=str(month), signal_date=str(dates[rows[0]]), label_end=str(ends[rows[0]]),
            rows=int(len(rows)), train_eligible_rows=int(eligible[rows].sum()),
            cached_train_good_rows=int(cached_good[rows].sum()),
            features_finite_rows=int(finite_X[rows].sum()), base_cohort_rows=int(base_cohort[rows].sum()),
            positive_endpoint_rows=int(endpoint_valid[rows].sum()),
            positive_vol63_rows=int(positive_vol.sum()), finite_raw_target_rows=int(finite_raw.sum()),
            common_before_minimum_rows=int(len(selected)), common_rows=int(good[rows].sum()),
            status="complete" if len(selected) >= MIN_MONTH_ROWS else "fewer_than_two_common_rows",
            clip_counts=clip_counts))

    audit = dict(
        target_type="continuous magnitudes; no target ranks or return-based parameter selection",
        formulas=dict(simple_excess="r_asset-r_SPY", log_excess="log(asset_gross)-log(SPY_gross)",
                      vol_scaled="(r_asset-r_SPY)/(vol63*sqrt(21))",
                      loss_utility="r_asset-r_SPY-0.5*max(0,-r_asset)"),
        clipping="fixed symmetric limits before common-cohort signal-month mean centering",
        clip_limits=dict(CLIP_LIMITS), volatility="63 simple returns from 64 positive finite prices through signal close; ddof=1; no filling",
        common_cohort="cached train_good & train_eligible & features_finite & positive exact endpoints & finite four raw targets & finite positive vol63",
        existing_full_path_rule="retained through cached train_good; never relaxed or recomputed here",
        min_month_rows=MIN_MONTH_ROWS, input_rows=n, base_cohort_rows=int(base_cohort.sum()),
        common_rows=int(good.sum()), complete_months=sum(m["status"] == "complete" for m in audit_months),
        excluded_months=sum(m["status"] != "complete" for m in audit_months),
        clip_counts={arm: sum(m["clip_counts"][arm] for m in audit_months) for arm in ARMS},
        clip_count_scope="retained common cohort before centering", months=audit_months)
    return dict(targets=targets, good=good, vol63=vol63, audit=audit)
