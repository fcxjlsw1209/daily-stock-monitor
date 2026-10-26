"""Half replacement of the frozen core by eligible-domain or affine ranks.

Pure source reconstruction and score transforms. No model fitting, I/O,
portfolio calculations, eligibility changes, or episode applications.
"""
import numpy as np
import pandas as pd

FRACTION = .5


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adjusted prices require a DataFrame with DatetimeIndex')
    idx = frame.index
    if (idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing or idx.tz is not None
            or not idx.equals(idx.normalize())):
        raise ValueError('Price sessions must be unique increasing naive dates')
    if not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s for s in frame):
        raise ValueError('Price columns must be unique security IDs')


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Expected a naive midnight date')
    return day


def _axes(meta, calendar, limit):
    if not isinstance(meta, dict) or any(k not in meta for k in ('dates', 'symbols', 'etfs')):
        raise ValueError('Metadata requires dates, symbols and etfs')
    dates = [_day(x) for x in calendar]
    if (not dates or any(not isinstance(x, str) or x != str(d.date()) or d.weekday() >= 5
                         for x, d in zip(calendar, dates))
            or any(a >= b for a, b in zip(dates, dates[1:]))):
        raise ValueError('Calendar must contain increasing unique ISO weekday sessions')
    sessions = pd.DatetimeIndex(dates); lookup = {d: i for i, d in enumerate(dates)}
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Event dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)) or len(set(meta['etfs'])) != len(meta['etfs']):
        raise ValueError('ETF identities require a unique sequence')
    if any(not isinstance(s, str) or not s or s.strip() != s for s in meta['etfs']):
        raise ValueError('Invalid ETF identity')
    rows, width = [], None
    for date, ss in zip(meta['dates'], meta['symbols']):
        day = _day(date)
        if day not in lookup or (rows and rows[-1][0] >= day):
            raise ValueError('Events must be increasing official sessions')
        if isinstance(ss, (str, bytes)):
            raise ValueError('Symbols require a sequence')
        ss = list(ss)
        if (not ss or len(ss) != len(set(ss)) or
                any(not isinstance(s, str) or not s or s.strip() != s for s in ss)):
            raise ValueError('Event security IDs must be unique')
        if width is not None and len(ss) != width:
            raise ValueError('Event width must be constant')
        width = len(ss); rows.append((day, ss, lookup[day]))
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(rows)):
        raise ValueError('limit must be an integer event prefix')
    return sessions, rows if limit is None else rows[:limit], (len(rows), width or 0)


def _matrix(value, shape, n, name, boolean=False):
    arr = np.asarray(value)
    if arr.shape != shape or (boolean and arr.dtype.kind != 'b'):
        raise ValueError(name + ' must match full event matrix' + (' and be boolean' if boolean else ''))
    try:
        return arr[:n].astype(bool if boolean else float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be numeric') from exc


def _state(value, rows, n, name):
    arr = np.asarray(value)
    if arr.shape != (rows,) or arr.dtype.kind != 'b':
        raise ValueError(name + ' must be a full boolean event vector')
    return arr[:n]


def _weight(value, shape, n, name):
    arr = np.asarray(value)
    if arr.shape == (shape[0],):
        arr = arr[:, None]
    if arr.shape not in ((shape[0], 1), shape):
        raise ValueError(name + ' shape must be n, n-by-1 or n-by-width')
    ans = np.broadcast_to(arr[:n].astype(float, copy=False), (n, shape[1]))
    if not np.isfinite(ans).all():
        raise ValueError(name + ' must be finite')
    return ans


def _sources(original, source, shape, n):
    if not isinstance(original, dict) or not isinstance(source, dict):
        raise ValueError('Original and source must be dictionaries')
    calm = _state(original['state_calm10'], shape[0], n, 'state_calm10')
    cw = _weight(original['core_weight'], shape, n, 'core_weight')
    expected = np.broadcast_to(np.where(calm, .5, .52)[:, None], cw.shape)
    if not np.array_equal(cw, expected):
        raise ValueError('Original core weight disagrees with calm state')
    core = _matrix(original['core'], shape, n, 'core')
    eligible = _matrix(original['eligible'], shape, n, 'eligible', True)
    if not np.isfinite(core).all() or (np.abs(core/cw) > .5+1e-12).any():
        raise ValueError('Original weighted core must be finite and bounded')
    if 'core' in source and not np.array_equal(_matrix(source['core'], shape, n, 'source core'), core):
        raise ValueError('Source and original core differ')
    if 'core_weight' in source and not np.array_equal(_weight(source['core_weight'], shape, n, 'source cw'), cw):
        raise ValueError('Source and original core weight differ')
    # Other source fields are intentionally not read: this is not a new
    # volatility, model, industry, or eligibility transformation.
    return core, cw, calm, eligible


def _rank(values):
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).rank(
        method='average', pct=True).fillna(.5).to_numpy()-.5


def eligible_comparisons(core_rank, raw_valid, eligible, initial=False):
    """Return two target ranks and a shared whole-eligible-row support mask."""
    u = np.asarray(core_rank, dtype=float)
    finite, good = np.asarray(raw_valid), np.asarray(eligible)
    if (u.ndim != 1 or finite.shape != u.shape or good.shape != u.shape
            or finite.dtype.kind != 'b' or good.dtype.kind != 'b'
            or type(initial) is not bool):
        raise ValueError('Rank inputs require aligned vectors and exact booleans')
    if not np.isfinite(u).all() or (np.abs(u) > .5+1e-12).any():
        raise ValueError('Canonical centered core ranks must be finite and bounded')
    count = int(good.sum())
    levels = int(len(np.unique(u[good])))
    missing = int((good & ~finite).sum())
    candidate = np.full(u.shape, np.nan)
    control = np.full(u.shape, np.nan)
    available = np.zeros(u.shape, dtype=bool)
    audit = dict(eligible_count=count, distinct_eligible_levels=levels,
        missing_eligible_core_count=missing, row_ready=False, reason=None,
        original_min=None, original_max=None, target_min=None, target_max=None,
        affine_slope=None, control_two_level_copy=False, exact_equivalent=False,
        max_candidate_control_difference=None)
    if initial:
        audit['reason'] = 'initial_allocation'
    elif missing:
        audit['reason'] = 'missing_eligible_core'
    elif count < 2:
        audit['reason'] = 'fewer_than_two_eligible'
    elif count == len(u):
        audit['reason'] = 'all_eligible'
    elif levels == 1:
        audit['reason'] = 'equal_eligible_core'
    else:
        selected = u[good]
        e = _rank(selected)
        low, high = float(selected.min()), float(selected.max())
        emin, emax = float(e.min()), float(e.max())
        slope = (emax-emin)/(high-low)
        a = emin+(selected-low)*(emax-emin)/(high-low)
        a[selected == low] = emin
        a[selected == high] = emax
        two = levels == 2
        if two:
            a = e.copy()
        if not (np.isfinite(e).all() and np.isfinite(a).all()):
            raise FloatingPointError('Nonfinite core rank transformation')
        if (np.abs(e) > .5+1e-12).any() or (np.abs(a) > .5+1e-12).any():
            raise FloatingPointError('Core transformation exceeds bounded rank scale')
        candidate[good] = e
        control[good] = a
        available[good] = True
        diff = float(np.max(np.abs(e-a)))
        audit.update(row_ready=True, reason='available', original_min=low,
            original_max=high, target_min=emin, target_max=emax, affine_slope=slope,
            control_two_level_copy=two, exact_equivalent=bool(np.array_equal(e, a)),
            max_candidate_control_difference=diff)
    return candidate, control, available, audit


def _json_numbers(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def build_features(panel, meta, calendar, original, source, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    _frame(panel['adj'])
    sessions, queries, shape = _axes(meta, calendar, limit)
    n, width = len(queries), shape[1]
    core, cw, calm, eligible = _sources(original, source, shape, n)
    values = dict(candidate=np.full((n, width), np.nan),
        control=np.full((n, width), np.nan), available=np.zeros((n, width), bool))
    history, reasons = [], {}
    if queries:
        last = queries[-1][0]
        prefix = panel['adj'].loc[:last]
        if prefix.empty or 'SPY' not in prefix:
            raise ValueError('Original source requires SPY history')
        needed = sorted({'SPY'} | {s for _, ss, _ in queries for s in ss})
        start = prefix.index[0]
        official = sessions[(sessions >= start) & (sessions <= last)]
        overlap = prefix.index[prefix.index >= sessions[0]]
        if len(overlap.difference(sessions)):
            raise ValueError('Source includes nonofficial covered sessions')
        prefix = prefix.reindex(index=prefix.index.union(official), columns=needed)
        try:
            px = prefix.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Price prefix must be numeric') from exc
        x = px.to_numpy()
        supplied = ~np.isnan(x)
        if not (np.isfinite(x[supplied]) & (x[supplied] > 0.)).all():
            raise ValueError('Observed prices must be finite positive; NaN is missing')
        # Preserve the original raw-core expressions, including their missing
        # and zero-volatility conventions; no extra formation warmup is added.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            r = px.pct_change(fill_method=None)
            sigma63 = r.rolling(63).std(ddof=1)
            momentum = px.shift(42)/px.shift(252)-1.
            risk = (px.shift(42)/px.shift(294)-1.)/sigma63.replace(0., np.nan)
        momentum = momentum.replace([np.inf, -np.inf], np.nan)
        risk = risk.replace([np.inf, -np.inf], np.nan)
        rebuilt_calm = sigma63['SPY']*np.sqrt(252.) < .10
        # The research source audit.py::feature_panel first replaces infinite
        # raw252 by NaN, then snapshot applies its strict-positive comparison.
        rebuilt_eligible = (px.notna().rolling(295, min_periods=295).sum().eq(295)
                            & px.gt(0.) & momentum.gt(0.))
        for i, (day, symbols, position) in enumerate(queries):
            if bool(rebuilt_calm.at[day]) != bool(calm[i]):
                raise ValueError('Original calm state parity failed')
            good = rebuilt_eligible.loc[day, symbols].to_numpy(dtype=bool)
            if not np.array_equal(good, eligible[i]):
                raise ValueError('Original eligibility parity failed')
            raw = (momentum if calm[i] else risk).loc[day, symbols].to_numpy(dtype=float)
            ranked = _rank(raw)
            if not np.allclose(ranked*cw[i], core[i], rtol=0., atol=1e-12):
                raise ValueError('Original weighted core parity failed')
            valid = np.isfinite(raw)
            result = eligible_comparisons(ranked, valid, good, initial=i == 0)
            values['candidate'][i], values['control'][i], values['available'][i], detail = result
            reasons[detail['reason']] = reasons.get(detail['reason'], 0)+1
            history.append(dict(date=str(day.date()), symbols=symbols,
                session_position=position, calm=bool(calm[i]), core_horizon=252 if calm[i] else 294,
                raw_core=_json_numbers(raw), core_valid=valid.tolist(),
                core_rank=ranked.tolist(), eligible=good.tolist(),
                candidate=_json_numbers(values['candidate'][i]),
                control=_json_numbers(values['control'][i]), **detail))
    available = int(values['available'].sum())
    ready = sum(row['row_ready'] for row in history)
    equivalent = sum(row['exact_equivalent'] for row in history)
    return values, dict(events=n, symbols_per_event=width, checked_cells=n*width,
        eligible_cells=int(eligible.sum()),
        raw_core_finite_cells=sum(sum(row['core_valid']) for row in history),
        available_cells=available, unavailable_cells=n*width-available,
        ready_rows=ready, fallback_rows=n-ready, equivalent_ready_rows=equivalent,
        candidate_control_different_rows=ready-equivalent,
        max_candidate_control_difference=max(
            (row['max_candidate_control_difference'] for row in history if row['row_ready']), default=0.),
        core_rank_reconstructions=n, predictive_model_fits=0, replacement_fraction=FRACTION,
        reason_counts=reasons, event_history=history, common_availability=True,
        ranking_eligibility_filtered=True, trading_eligibility_modified=False,
        portfolio_returns_calculated=False, original_core_parity_checked=True,
        eligibility_parity_checked=True, calm_state_parity_checked=True)


def build_deltas(values, original, source, limit=None):
    if not isinstance(values, dict) or set(values) != {'candidate', 'control', 'available'}:
        raise ValueError('Values require candidate, control and available')
    shape = np.asarray(original['core']).shape
    mask = np.asarray(values['available'])
    if (len(shape) != 2 or mask.ndim != 2 or mask.shape[1] != shape[1]
            or mask.shape[0] > shape[0] or mask.dtype.kind != 'b'):
        raise ValueError('Availability must match an original event prefix')
    if limit is not None and (type(limit) is not int or not 0 <= limit <= mask.shape[0]):
        raise ValueError('limit must be an integer prefix of values')
    n = shape[0] if limit is None else limit
    if mask.shape[0] < n:
        raise ValueError('A partial feature prefix requires explicit limit')
    core, cw, _, eligible = _sources(original, source, shape, n)
    available = mask[:n]
    if n and available[0].any():
        raise ValueError('Initial partial allocation must have no new rank availability')
    if (available & ~eligible).any():
        raise ValueError('New core rank cannot include ineligible securities')
    for i in range(n):
        if available[i].any() and not np.array_equal(available[i], eligible[i]):
            raise ValueError('Core transform support must cover the entire eligible row')
    deltas = {}
    for key in ('candidate', 'control'):
        val = np.asarray(values[key])
        if val.shape != mask.shape:
            raise ValueError('Feature value shape changed')
        val = val[:n].astype(float, copy=False)
        if np.isinf(val).any() or not np.array_equal(np.isfinite(val), available):
            raise ValueError('Both arms must share exactly the finite availability mask')
        if (np.abs(val[available]) > .5+1e-12).any():
            raise ValueError('New core ranks must remain bounded')
        delta = np.zeros(available.shape)
        delta[available] = FRACTION*(cw[available]*val[available]-core[available])
        if n:
            delta[0] = 0.
        if not np.isfinite(delta).all():
            raise FloatingPointError('Nonfinite score correction')
        deltas[key] = delta
    return deltas, dict(events=n, width=shape[1], dose=FRACTION,
        available_cells=int(available.sum()), core_weights=cw.tolist(),
        changed_cells={key: int(np.count_nonzero(x)) for key, x in deltas.items()},
        max_abs_delta={key: float(np.max(np.abs(x), initial=0.)) for key, x in deltas.items()},
        initial_delta_zero=True, core_weight_preserved=True, original_support_unchanged=True,
        original_model_corrections_unchanged=True, episode_applied=False)


