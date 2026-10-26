"""Pure three-block trend state and routing of one existing local increment.

No rank, eligibility, model, or episode calculations are performed here. The
caller supplies the upstream score B, unchanged Local67 pre-score P, and the
separately computed original local increment D. Their original addition order
is verified exactly before any routing. Missing states retain original P.
"""
import math

import numpy as np
import pandas as pd


PRICE_COUNT = 211
SKIP = 42
BLOCK = 70
ARMS = ('trend_gated_local', 'uniform_local_shrink')
INCREMENT_LIMIT = .0125
TOLERANCE = 1e-12


def _empty(reason):
    return dict(available=False, consistent=False, reason=reason,
                block_log_returns=[], block_return_estimates=0)


def measure(adjusted):
    if np.shape(adjusted) != (PRICE_COUNT,):
        raise ValueError('Expected exactly 211 adjusted session prices')
    try:
        prices = np.asarray(adjusted, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _empty('nonnumeric_adjusted_price')
    if not (np.isfinite(prices) & (prices > 0.)).all():
        return _empty('invalid_adjusted_price')
    logs = [math.log(float(prices[j])) for j in (0, 70, 140, 210)]
    moves = [logs[j+1] - logs[j] for j in range(3)]
    if not all(math.isfinite(v) for v in logs + moves):
        raise FloatingPointError('Positive finite prices produced nonfinite block logs')
    return dict(available=True, consistent=bool(all(v > 0. for v in moves)),
                reason='available', block_log_returns=moves, block_return_estimates=3)


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ' requires a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + ' requires a naive midnight date')
    return day


def _panel(panel):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    frame = panel['adj']
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adjusted source requires DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Panel dates must be unique increasing naive midnight sessions')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                           for s in frame.columns)):
        raise ValueError('Panel symbols must be unique nonempty strings')


def _calendar(calendar):
    dates = []
    for value in calendar:
        day = _day(value, 'Calendar')
        if not isinstance(value, str) or value != str(day.date()) or day.weekday() >= 5:
            raise ValueError('Calendar requires ISO weekday sessions')
        if dates and day <= dates[-1]:
            raise ValueError('Calendar must be unique and increasing')
        dates.append(day)
    if not dates:
        raise ValueError('Calendar cannot be empty')
    return pd.DatetimeIndex(dates)


def _queries(meta, sessions):
    if not isinstance(meta, dict) or any(k not in meta for k in ('dates', 'symbols', 'etfs')):
        raise ValueError('Metadata requires dates, symbols and etfs')
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)):
        raise ValueError('ETF IDs require an explicit sequence')
    etfs = list(meta['etfs'])
    if any(not isinstance(s, str) or not s or s.strip() != s for s in etfs) or len(set(etfs)) != len(etfs):
        raise ValueError('ETF IDs must be unique nonempty strings')
    positions = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in positions or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Candidate IDs require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)
                or len(set(symbols)) != len(symbols)):
            raise ValueError('Candidate IDs must be unique nonempty strings')
        if width is not None and width != len(symbols):
            raise ValueError('Candidate width must remain constant')
        width = len(symbols)
        rows.append((day, symbols, positions[day]))
    return rows, width or 0


def build_features(panel, meta, calendar, limit=None):
    _panel(panel)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is None:
        limit = len(queries)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or not 0 <= limit <= len(queries):
        raise ValueError('Limit must be an integer event prefix')
    values = {key: np.zeros((limit, width), dtype=bool) for key in ('consistent', 'available')}
    frame, history, reasons, estimates = panel['adj'], [], {}, 0
    for row, (day, symbols, position) in enumerate(queries[:limit]):
        window = sessions[position-252:position-SKIP+1] if position >= 252 else None
        missing, windows = [], None
        if window is not None:
            missing = [str(d.date()) for d in window if d not in frame.index]
            windows = dict(
                formation=dict(price_start=str(window[0].date()), price_end=str(window[-1].date()), return_count=210),
                blocks=[dict(start_index=start, end_index=start+BLOCK,
                             start_date=str(window[start].date()), end_date=str(window[start+BLOCK].date()))
                        for start in (0, 70, 140)])
        cells = []
        for j, symbol in enumerate(symbols):
            if window is None:
                cell = _empty('insufficient_history')
            elif symbol not in frame.columns:
                cell = _empty('missing_symbol')
            elif missing:
                cell = _empty('missing_official_session')
            else:
                cell = measure(frame.loc[window, symbol].to_numpy())
            for key in values:
                values[key][row, j] = cell[key]
            estimates += cell['block_return_estimates']
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(dict(cell, symbol=symbol, date=str(day.date()), windows=windows))
        history.append(dict(date=str(day.date()), symbols=symbols, session_position=position,
                            price_start=str(window[0].date()) if window is not None else None,
                            price_end=str(window[-1].date()) if window is not None else None,
                            windows=windows, missing_official_sessions=missing, cells=cells))
    available = int(values['available'].sum())
    audit = dict(events=int(limit), symbols_per_event=width, checked_cells=int(limit)*width,
                 price_count=PRICE_COUNT, skip_sessions=SKIP, formation_return_count=210,
                 block_count=3, block_sessions=BLOCK, available_cells=available,
                 unavailable_cells=int(limit)*width-available, consistent_cells=int(values['consistent'].sum()),
                 reason_counts=reasons, block_return_estimates=estimates, predictive_model_fits=0,
                 portfolio_returns_calculated=False, ranking_performed=False,
                 trading_eligibility_modified=False, event_history=history)
    return values, audit


def _numeric_matrix(value, shape, name):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape != shape or raw.dtype.kind not in 'fiu':
        raise ValueError(name + ' requires a real numeric matrix of the common shape')
    return raw.astype(float, copy=False)


def _sum(values):
    try:
        total = math.fsum(values)
    except (ValueError, OverflowError) as exc:
        raise FloatingPointError('Invalid increment-budget sum') from exc
    if not math.isfinite(total):
        raise FloatingPointError('Nonfinite increment-budget sum')
    return total


def route_scores(pre, upstream, increment, consistent, available):
    """Route D only, retaining exact P copies on every unchanged cell.

The absolute budgets are theoretical local-increment L1 amounts, not score
differences after floating-point addition and not portfolio exposures.
"""
    shape = np.shape(pre)
    if len(shape) != 2:
        raise ValueError('Pre-score requires a two-dimensional prefix')
    p = _numeric_matrix(pre, shape, 'Pre-score')
    b = _numeric_matrix(upstream, shape, 'Upstream score')
    d = _numeric_matrix(increment, shape, 'Local increment')
    flags, valid = np.asarray(consistent), np.asarray(available)
    if flags.shape != shape or valid.shape != shape or flags.dtype.kind != 'b' or valid.dtype.kind != 'b':
        raise ValueError('State and availability require exact common-shape boolean matrices')
    if (flags & ~valid).any():
        raise ValueError('Unavailable states must not be consistent')
    if np.isinf(p).any() or np.isinf(b).any() or not np.array_equal(np.isnan(p), np.isnan(b)):
        raise ValueError('P and B require matching NaNs and no infinity')
    if not np.isfinite(d).all() or (np.abs(d) > INCREMENT_LIMIT + TOLERANCE).any():
        raise ValueError('Original local increment must be finite and within its existing slot')
    if (d[np.isnan(b)] != 0.).any():
        raise ValueError('Inaccessible scores must have zero increment')
    expected = b.copy()
    changed = d != 0.
    with np.errstate(over='raise', invalid='raise'):
        expected[changed] += d[changed]
    if not np.array_equal(p, expected, equal_nan=True):
        raise ValueError('P must exactly equal B with the original nonzero D additions')
    scores = {arm: p.copy() for arm in ARMS}
    history, fractions, candidate_budgets, control_budgets = [], [], [], []
    total_changed = {arm: 0 for arm in ARMS}
    for i in range(shape[0]):
        use = valid[i]
        num = _sum(abs(float(v)) for v in d[i, use & flags[i]])
        den = _sum(abs(float(v)) for v in d[i, use])
        fraction = num / den if i and den else 0.
        if not math.isfinite(fraction) or not 0. <= fraction <= 1.:
            raise FloatingPointError('Uniform removed fraction outside [0,1]')
        candidate_removed = num if i else 0.
        control_removed = _sum(fraction * abs(float(v)) for v in d[i, use])
        budget_error = abs(candidate_removed - control_removed)
        if budget_error > TOLERANCE:
            raise FloatingPointError('Matched local-increment budgets differ')
        if i:
            cut = use & flags[i] & changed[i]
            scores[ARMS[0]][i, cut] = b[i, cut]
            target = use & changed[i]
            if fraction == 1.:
                scores[ARMS[1]][i, target] = b[i, target]
            elif fraction:
                with np.errstate(over='raise', invalid='raise'):
                    scores[ARMS[1]][i, target] = b[i, target] + (1. - fraction)*d[i, target]
        row_changed = {}
        for arm in ARMS:
            same = (scores[arm][i] == p[i]) | (np.isnan(scores[arm][i]) & np.isnan(p[i]))
            row_changed[arm] = int((~same).sum())
            total_changed[arm] += row_changed[arm]
            must_copy = ~use | (d[i] == 0.)
            if not i or (arm == ARMS[1] and fraction == 0.):
                must_copy = np.ones(shape[1], dtype=bool)
            elif arm == ARMS[0]:
                must_copy |= ~flags[i]
            if not np.array_equal(scores[arm][i, must_copy], p[i, must_copy], equal_nan=True):
                raise AssertionError('Unchanged score cells must be copied exactly')
        fractions.append(fraction)
        candidate_budgets.append(candidate_removed)
        control_budgets.append(control_removed)
        history.append(dict(row=i, initial_neutral=(i == 0), valid_cells=int(use.sum()),
                            consistent_valid_cells=int((use & flags[i]).sum()),
                            available_abs_increment=den, consistent_abs_increment=num,
                            uniform_removed_fraction=fraction,
                            candidate_removed_abs_increment=candidate_removed,
                            control_removed_abs_increment=control_removed,
                            matched_budget_error=budget_error, changed_cells=row_changed))
    audit = dict(events=shape[0], symbols_per_event=shape[1], checked_cells=shape[0]*shape[1],
                 increment_abs_limit=INCREMENT_LIMIT, tolerance=TOLERANCE,
                 original_pre_reconstruction_exact=True, initial_scores_unchanged=True,
                 unchanged_cell_copy_exact=True, matched_removed_budgets=True,
                 no_cross_sectional_ranking=True, predictive_model_fits=0,
                 portfolio_returns_calculated=False, episode_applications=0,
                 uniform_removed_fraction=fractions, candidate_removed_abs_increment=candidate_budgets,
                 control_removed_abs_increment=control_budgets, changed_cells=total_changed,
                 event_history=history)
    return scores, audit
