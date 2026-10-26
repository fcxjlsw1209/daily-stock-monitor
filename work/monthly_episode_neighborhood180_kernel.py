"""Pure replacement of the frozen episode's hard ranking neighborhood.

No source loading, fitting, portfolio returns, or original episode invocation.
The caller supplies the unchanged Local67 pre-score and original support data.
An episode coefficient cap does not bound corrections for arbitrary unbounded
support; the host separately verifies the frozen support's [-.5, .5] contract.
"""
from datetime import date
import math

import numpy as np


ARMS = ('candidate', 'control')
GAUSSIAN_WIDTH = .02
BUDGET_CAP = .05


def _matrix(value, shape, n, name, boolean=False):
    a = np.asarray(value)
    if a.shape != shape or (boolean and a.dtype.kind != 'b'):
        raise ValueError(name + ' requires the exact monthly candidate shape'
                         + (' and boolean dtype' if boolean else ''))
    try:
        return a[:n].astype(bool if boolean else float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be numeric within the selected prefix') from exc


def _state(arrays, key, rows, n):
    value = np.asarray(arrays[key])
    if value.shape != (rows,) or value.dtype.kind != 'b':
        raise ValueError(key + ' requires an exact monthly boolean vector')
    return value[:n].copy()


def _metadata(meta, rows, width, n):
    if not isinstance(meta, dict) or not all(k in meta for k in ('dates', 'symbols')):
        raise ValueError('Metadata requires dates and symbols')
    if len(meta['dates']) != rows or len(meta['symbols']) != rows:
        raise ValueError('Metadata rows must match all source rows')
    out = []
    for i in range(n):
        value = meta['dates'][i]
        try:
            day = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError('Dates require ISO calendar strings') from exc
        if value != day.isoformat() or (out and value <= out[-1][0]):
            raise ValueError('Dates require unique increasing ISO strings')
        supplied = meta['symbols'][i]
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Candidate symbols require explicit rows')
        symbols = list(supplied)
        if (len(symbols) != width or len(set(symbols)) != width
                or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)):
            raise ValueError('Each candidate row requires unique nonempty symbols')
        out.append((value, symbols))
    return out


def _weight(score, anchor):
    # Overflow in an extreme finite score distance means the Gaussian rounds
    # to zero. It is not invalid input and never yields inf*0 downstream.
    z = (score - anchor) / GAUSSIAN_WIDTH
    if not math.isfinite(z):
        return 0.
    return math.exp(-.5 * z * z)


def _float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def apply_episode(pre, arrays, meta, limit=None):
    """Return scores/corrections/column-index choices and a JSON-safe audit.

Only the selected prefix is converted and examined. The first input row is
the preserved initial allocation; prefix callers must not rebase later rows
as a new initial row. NaN effects are missing and neutral; infinite effects
are contract errors. Nonfinite ineligible pre/support inputs stay untouched.
Core weights broadcast from [rows], [rows,1], or [rows,width].
"""
    if not isinstance(arrays, dict):
        raise TypeError('Original arrays require a dictionary')
    base_source = np.asarray(arrays['baseline'])
    if base_source.ndim != 2 or base_source.shape[1] == 0:
        raise ValueError('Baseline requires a nonempty candidate dimension')
    rows, width = shape = base_source.shape
    n = rows if limit is None else limit
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 0 <= n <= rows:
        raise ValueError('Limit requires an integer prefix from zero through all rows')
    n = int(n)
    queries = _metadata(meta, rows, width, n)
    baseline = _matrix(base_source, shape, n, 'baseline')
    core = _matrix(arrays['core'], shape, n, 'core')
    scores = _matrix(pre, shape, n, 'pre')
    eligible = _matrix(arrays['eligible'], shape, n, 'eligible', boolean=True)
    effects = _matrix(arrays['effects'], shape, n, 'effects')
    calm = _state(arrays, 'state_calm10', rows, n)
    weak = _state(arrays, 'state_weak_narrow', rows, n)
    supplied = np.asarray(arrays['core_weight'])
    if supplied.shape == (rows,):
        supplied = supplied[:, None]
    if supplied.shape not in ((rows, 1), shape):
        raise ValueError('Core weights require [rows], [rows,1], or [rows,width]')
    try:
        weights = np.broadcast_to(supplied[:n].astype(float), (n, width))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Core weights must be numeric in the selected prefix') from exc
    if not (np.isfinite(weights) & (weights > 0.) & (weights < 1.)).all():
        raise ValueError('Core weights must be finite and strictly between zero and one')
    if np.isinf(effects).any():
        raise ValueError('Infinite episode effects are invalid, not missing')
    if not all(np.isfinite(value[eligible]).all() for value in (baseline, core, scores)):
        raise ValueError('Eligible baseline, core, and pre-score must be finite')
    support = np.full((n, width), np.nan)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        support[eligible] = (baseline[eligible] - core[eligible]) / (1. - weights[eligible])
    if not np.isfinite(support[eligible]).all():
        raise FloatingPointError('Eligible original support must be finite')
    result = dict(scores={arm: scores.copy() for arm in ARMS},
                  corrections={arm: np.zeros((n, width)) for arm in ARMS},
                  choices={arm: [] for arm in ARMS}, eligible=eligible.copy())
    audit = dict(events=n, symbols_per_event=width, checked_cells=n*width,
                 eligible_cells=int(eligible.sum()),
                 finite_effect_eligible_cells=int((eligible & np.isfinite(effects)).sum()),
                 missing_effect_eligible_cells=int((eligible & np.isnan(effects)).sum()),
                 active_rows=0, initial_rows_suppressed=int(n > 0),
                 low_cardinality_rows=0, correction_cells={arm: 0 for arm in ARMS},
                 gaussian_width=GAUSSIAN_WIDTH, budget_cap=BUDGET_CAP,
                 predictive_model_fits=0, portfolio_returns_calculated=False,
                 eligibility_modified=False, episode_applications=1, event_history=[])
    for i, (day, symbols) in enumerate(queries):
        indices = [int(j) for j in np.flatnonzero(eligible[i])]
        ranked = sorted(indices, key=lambda j: (-scores[i, j], symbols[j]))
        q = float(scores[i, ranked[2]]) if len(ranked) >= 3 else None
        budget = min(BUDGET_CAP, max(0., .10 - .025*bool(calm[i]) - .05*bool(weak[i])))
        active = i != 0 and len(ranked) >= 3
        audit['active_rows'] += int(active)
        audit['low_cardinality_rows'] += int(len(ranked) < 3)
        cells = []
        for j, symbol in enumerate(symbols):
            unweighted = weight = 0.
            if active and eligible[i, j]:
                weight = _weight(float(scores[i, j]), q)
                if math.isfinite(float(effects[i, j])):
                    s, e = float(support[i, j]), float(effects[i, j])
                    shifted = s + e
                    if not math.isfinite(shifted):
                        raise FloatingPointError('Support plus effect must remain finite before clipping')
                    target = min(.5, max(-.5, shifted))
                    unweighted = budget * (target - s)
                result['corrections']['candidate'][i, j] = weight * unweighted
                result['corrections']['control'][i, j] = unweighted
                for arm in ARMS:
                    delta = float(result['corrections'][arm][i, j])
                    if delta != 0.:
                        new = float(scores[i, j]) + delta
                        if not math.isfinite(new):
                            raise FloatingPointError('Episode correction produced a nonfinite eligible score')
                        result['scores'][arm][i, j] = new
            cells.append(dict(symbol=symbol, eligible=bool(eligible[i, j]),
                pre_score=_float(scores[i, j]), support=_float(support[i, j]),
                effect=_float(effects[i, j]), unweighted_correction=unweighted,
                neighborhood_weight=weight,
                candidate_correction=float(result['corrections']['candidate'][i, j]),
                control_correction=float(result['corrections']['control'][i, j]),
                candidate_score=_float(result['scores']['candidate'][i, j]),
                control_score=_float(result['scores']['control'][i, j])))
        for arm in ARMS:
            selected = sorted(indices, key=lambda j: (-result['scores'][arm][i, j], symbols[j]))
            result['choices'][arm].append(selected[:3] if len(selected) >= 2 else [])
        audit['event_history'].append(dict(date=day, symbols=symbols,
            eligible_indices=indices, pre_order=ranked, third_score=q, budget=budget,
            active=active, suppression_reason=('initial_row' if i == 0 else
                                              'fewer_than_three_eligible' if not active else None),
            choices={arm: result['choices'][arm][-1].copy() for arm in ARMS}, cells=cells))
    audit['correction_cells'] = {arm: int(np.count_nonzero(result['corrections'][arm])) for arm in ARMS}
    return result, audit
