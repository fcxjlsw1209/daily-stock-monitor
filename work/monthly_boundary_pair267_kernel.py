"""Prospective267 caller-supplied pair Ridge and original-episode adapters.

No file access, source loading, legacy imports, registration or portfolio replay.
A caller must authenticate original Local67 P/E, identity, adjusted endpoints
and plain10 provenance separately. This module makes no actual coverage claim.
"""
from copy import deepcopy
from datetime import date
from itertools import combinations
import math
import numbers
import numpy as np

INITIAL_PARTIAL = '2016-08-17'
ARMS = ('top5_all_pairs_control', 'original_p34_pair_candidate')
FEATURES = 10
ALPHA = 10.0
MIN_MONTHS = 18
HALF_LIFE_MONTHS = 12.0
GAP = .02
RETURN_SCALE = .05
POLICY = dict(budget=.05, ceiling=5, gap=.02)
BODY_KEYS = ('symbols', 'pre_scores', 'eligible', 'features', 'signal_prices', 'end_prices')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _schema(name):
    return 'monthly_boundary_pair267_' + name + '_v1'


def _day(value, label):
    need(type(value) is str and len(value) == 10, label + ' ISO date')
    try:
        out = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(label + ' valid date') from exc
    need(out.isoformat() == value, label + ' canonical date')
    return out


def _real(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return None
    try:
        x = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _vector(values, n, label):
    need(isinstance(values, (list, tuple, np.ndarray)) and len(values) == n, label + ' width')
    if isinstance(values, np.ndarray):
        need(values.ndim == 1, label + ' one axis')
    return values


def _pack(values):
    a = np.asarray(values)
    need(a.dtype == np.dtype('float64') and np.isfinite(a).all(), 'Finite float64 audit')
    return dict(dtype=a.dtype.str, shape=list(a.shape), values=[float(x).hex() for x in a.flat])


def original_topk(pre_scores, eligible, symbols):
    """Original E and (-P, symbol) order; never filter by label/feature validity."""
    need(isinstance(symbols, (list, tuple)) and len(symbols) > 0, 'Security identity vector')
    n = len(symbols)
    need(all(type(s) is str and s and s.strip() == s for s in symbols)
         and len(set(symbols)) == n, 'Unique nonempty security identities')
    _vector(pre_scores, n, 'Original P')
    _vector(eligible, n, 'Original E')
    ids = []
    numeric = {}
    for i in range(n):
        need(isinstance(eligible[i], (bool, np.bool_)), 'Original E Boolean')
        if eligible[i]:
            value = _real(pre_scores[i])
            need(value is not None, 'Eligible original P must be finite; no new eligibility filter')
            numeric[i] = value
            ids.append(i)
    ids.sort(key=lambda i: (-numeric[i], symbols[i]))
    return ids[:min(5, len(ids))], len(ids)


def _feature(row):
    if not isinstance(row, (list, tuple, np.ndarray)):
        return None
    if isinstance(row, np.ndarray) and row.ndim != 1:
        return None
    if len(row) != FEATURES:
        return None
    out = [_real(x) for x in row]
    if any(x is None or not -.5 <= x <= .5 for x in out):
        return None
    return np.asarray(out, dtype=np.float64)


def _headers(months, year):
    need(type(year) is int and 7 <= year <= 9999, 'Query year native integer')
    need(isinstance(months, (list, tuple)), 'Chronological month records')
    cutoff, lower = date(year - 1, 12, 31), date(year - 6, 12, 31)
    headers = []
    previous = None
    seen = set()
    for i, row in enumerate(months):
        need(type(row) is dict and 'signal' in row and 'end' in row, 'Month date headers')
        signal, end = _day(row['signal'], 'Signal'), _day(row['end'], 'Label end')
        need(previous is None or signal > previous, 'Strict chronological signal order')
        previous = signal
        partial = row['signal'] == INITIAL_PARTIAL
        if partial:
            need(end > signal, 'Partial endpoint after signal')
        else:
            key = (signal.year, signal.month)
            need(key not in seen, 'One original signal per natural month')
            seen.add(key)
            next_month = (signal.year + (signal.month == 12), signal.month % 12 + 1)
            need((end.year, end.month) == next_month and end > signal, 'Next natural-month endpoint')
        status = ('initial_partial' if partial else 'outside_window' if signal <= lower
                  else 'not_mature' if end > cutoff else 'eligible_dates')
        headers.append((i, signal, end, status))
    return cutoff, lower, headers


def build_cohort(months, query_year):
    """Shared TopK-valid months, then different pair-loss rows on those months.

    Unused future/old/partial records expose only date headers. A selected name
    with invalid features/prices excludes the whole month for both arms.
    Structural P/E/identity errors fail; they never invent replacement names.
    """
    cutoff, lower, headers = _headers(months, query_year)
    audit = []
    retained = []
    for source_index, signal, end, status in headers:
        item = dict(source_index=source_index, signal=signal.isoformat(), end=end.isoformat(), status=status)
        audit.append(item)
        if status != 'eligible_dates':
            continue
        source = months[source_index]
        need(all(k in source for k in BODY_KEYS), 'Complete consumed original month body')
        names = source['symbols']
        chosen, n = original_topk(source['pre_scores'], source['eligible'], names)
        item.update(eligible_count=n, selected_indices=chosen, selected_symbols=[names[j] for j in chosen])
        if n < 4:
            item['status'] = 'fewer_than_four_eligible'
            continue
        width = len(names)
        need(isinstance(source['features'], (list, tuple, np.ndarray)) and len(source['features']) == width,
             'Feature source rows')
        _vector(source['signal_prices'], width, 'Signal prices')
        _vector(source['end_prices'], width, 'End prices')
        selected = []
        invalid = []
        for j in chosen:
            x = _feature(source['features'][j])
            start, stop = _real(source['signal_prices'][j]), _real(source['end_prices'][j])
            if x is None or start is None or stop is None or start <= 0. or stop <= 0.:
                invalid.append(j)
            else:
                selected.append((j, x, start, stop))
        if invalid:
            item.update(status='invalid_selected_name', invalid_selected_indices=invalid)
            continue
        # Every selected source passed before evaluating any endpoint return.
        checked = []
        for j, x, start, stop in selected:
            ratio = stop / start
            need(math.isfinite(ratio) and ratio > 0., 'Finite positive adjusted gross; no underflow to zero')
            r = float(ratio) - 1.0
            need(math.isfinite(r), 'Finite unmodified endpoint return')
            checked.append((j, x, r))
        item['status'] = 'common_valid_month'
        item['boundary_pair_indices'] = [chosen[2], chosen[3]]
        retained.append((source_index, signal, end, names, checked))
    M = len(retained)
    ages = [(cutoff.year - signal.year) * 12 + cutoff.month - signal.month for _, signal, _, _, _ in retained]
    need(all(type(a) is int and a >= 1 for a in ages), 'Mature monthly decay age')
    decays = [float(np.exp2(-a / HALF_LIFE_MONTHS)) for a in ages]
    need(all(math.isfinite(d) and d > 0. for d in decays), 'Finite positive month decay')
    mass = math.fsum(decays)
    wm = [(M * d) / mass for d in decays] if M else []
    need(all(math.isfinite(w) and w > 0. for w in wm), 'Finite positive common-month weights')
    buckets = {arm: dict(X=[], y=[], weights=[], pairs=[]) for arm in ARMS}
    for m, (source_index, signal, end, names, selected) in enumerate(retained):
        pair_positions = list(combinations(range(len(selected)), 2))
        for arm, pairs in ((ARMS[0], pair_positions), (ARMS[1], [(2, 3)])):
            b = buckets[arm]
            weight = wm[m] / len(pairs)
            need(math.isfinite(weight) and weight > 0., 'Positive pair weight')
            for a, c in pairs:
                i, xi, ri = selected[a]
                j, xj, rj = selected[c]
                x = xi - xj
                y = ri - rj
                need(np.isfinite(x).all() and math.isfinite(y), 'Finite unscaled pair design/label')
                b['X'].append(x)
                b['y'].append(y)
                b['weights'].append(weight)
                b['pairs'].append(dict(source_index=source_index, signal=signal.isoformat(), end=end.isoformat(),
                                       indices=[i, j], symbols=[names[i], names[j]], original_ranks=[a + 1, c + 1],
                                       common_month_index=m))
    for b in buckets.values():
        b['X'] = np.asarray(b['X'], dtype=np.float64).reshape((-1, FEATURES))
        for key in ('y', 'weights'):
            b[key] = np.asarray(b[key], dtype=np.float64)
        b['weight_mass'] = math.fsum(float(v) for v in b['weights'])
        b['weight_mass_hex'] = b['weight_mass'].hex()
        b['float_evidence'] = {k: _pack(b[k]) for k in ('X', 'y', 'weights')}
    return dict(schema=_schema('cohort'), query_year=query_year, cutoff=cutoff.isoformat(),
                signal_lower_exclusive=lower.isoformat(), minimum_months=MIN_MONTHS, common_months=M,
                status='ready' if M >= MIN_MONTHS else 'cold', months=audit,
                common_source_indices=[x[0] for x in retained], decay_ages=ages, decays=decays,
                month_weights=wm, month_weights_hex=[x.hex() for x in wm], mathematical_weight_mass=M,
                weight_rule='d=exp2(-age/12); wm=(M*d)/fsum(d); control=wm/C(K,2), candidate=wm; no residual correction',
                training_gap_filter=False, returns_rule='float(end/signal)-1.0, then ri-rj; no clipping/excess/third',
                source_authentication='Caller responsibility; official endpoints and original P/E are not authenticated here',
                actual_coverage_claim=False, arms=buckets)


def _fit(X, y, weights):
    need(X.dtype == y.dtype == weights.dtype == np.dtype('float64') and X.ndim == 2
         and X.shape[1] == FEATURES and y.shape == weights.shape == (len(X),), 'Pair Ridge shapes/dtypes')
    need(np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(weights).all()
         and np.all(weights > 0.), 'Finite weighted pair Ridge')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        gram = np.einsum('ni,n,nj->ij', X, weights, X, optimize=False)
        gram += ALPHA * np.eye(FEATURES)
        rhs = np.einsum('ni,n,n->i', X, weights, y, optimize=False)
    need(np.isfinite(gram).all() and np.isfinite(rhs).all(), 'Finite Ridge equations')
    beta = np.linalg.solve(gram, rhs)
    need(np.isfinite(beta).all(), 'Finite Ridge solution; no solver fallback')
    return beta, gram, rhs


def fit_models(months, query_year):
    """Fit exactly two models iff the common selected-name cohort has18 months."""
    cohort = build_cohort(months, query_year)
    models = {}
    for arm in ARMS:
        rows = cohort['arms'][arm]
        if cohort['status'] == 'ready':
            beta, gram, rhs = _fit(rows['X'], rows['y'], rows['weights'])
            fitted = 1
        else:
            beta, gram, rhs, fitted = np.zeros(FEATURES), None, None, 0
        evidence = dict(beta=_pack(beta), gram=None if gram is None else _pack(gram), rhs=None if rhs is None else _pack(rhs))
        models[arm] = dict(schema=_schema('model'), arm=arm, query_year=query_year, status=cohort['status'],
                           common_months=cohort['common_months'], alpha=ALPHA, intercept=0., rescaled=False,
                           reverse_augmented=False, beta=beta, gram=gram, rhs=rhs, float_evidence=evidence,
                           rows=len(rows['y']), actual_solve_calls=fitted)
    return dict(schema=_schema('models'), query_year=query_year, cohort=cohort, models=models,
                actual_solve_calls=sum(x['actual_solve_calls'] for x in models.values()),
                pair_loss_set_only_difference=True, actual_coverage_claim=False)


def adjust_effects(event, model):
    """Only original ranks3/4 change; never choose a winner or create new ranks."""
    need(type(event) is dict and type(model) is dict, 'Caller event/model')
    signal = _day(event['signal'], 'Deployment signal')
    need(model.get('schema') == _schema('model') and model.get('arm') in ARMS
         and type(model.get('query_year')) is int and model['query_year'] == signal.year
         and model.get('status') in ('ready', 'cold'), 'Same-year fixed pair model')
    months = model.get('common_months')
    need(type(months) is int and months >= 0 and (months >= MIN_MONTHS) == (model['status'] == 'ready'),
         'Model availability from common eighteen-month support')
    effects = event['effects']
    need(isinstance(effects, np.ndarray) and effects.dtype == np.dtype('float64') and effects.ndim == 1
         and np.isfinite(effects).all(), 'Original finite float64 effects')
    result = effects.copy()
    audit = dict(signal=event['signal'], arm=model['arm'], status='initial_partial' if event['signal'] == INITIAL_PARTIAL else 'cold',
                 pair_indices=[], predicted_return_gap=None, q=0., changed_indices=[], original_selector_calls=0,
                 original_pre_scores_unchanged=True, new_global_score_ranks_computed=False, winner_selected=False)
    if event['signal'] == INITIAL_PARTIAL or model['status'] == 'cold':
        return result, audit
    chosen, n = original_topk(event['pre_scores'], event['eligible'], event['symbols'])
    need(len(effects) == len(event['symbols']), 'Original effects width')
    if n < 4:
        audit['status'] = 'fewer_than_four_eligible'
        return result, audit
    i, j = chosen[2:4]
    gap = float(event['pre_scores'][i]) - float(event['pre_scores'][j])
    need(math.isfinite(gap) and gap >= 0., 'Finite nonnegative original boundary gap')
    audit.update(pair_indices=[i, j], original_gap=gap)
    if gap > GAP:
        audit['status'] = 'outside_deployment_gap'
        return result, audit
    features = event['features']
    need(isinstance(features, (list, tuple, np.ndarray)) and len(features) == len(effects), 'Deployment feature rows')
    xi, xj = _feature(features[i]), _feature(features[j])
    need(xi is not None and xj is not None, 'Invalid required boundary features')
    beta = model['beta']
    need(isinstance(beta, np.ndarray) and beta.shape == (FEATURES,) and beta.dtype == np.dtype('float64')
         and np.isfinite(beta).all() and model.get('alpha') == ALPHA and model.get('intercept') == 0.
         and model.get('rescaled') is False and model.get('reverse_augmented') is False, 'Fixed no-intercept pair Ridge')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        predicted = float(np.dot(xi - xj, beta))
    need(math.isfinite(predicted), 'Finite predicted return gap')
    scaled = predicted / RETURN_SCALE
    need(math.isfinite(scaled), 'Finite effect scale')
    q = .5 * math.tanh(scaled)
    need(math.isfinite(q) and -.5 <= q <= .5, 'Finite bounded pair effect')
    audit.update(predicted_return_gap=predicted, predicted_return_gap_hex=predicted.hex(), q=q, q_hex=q.hex())
    if q == 0.:
        audit['status'] = 'zero_prediction'
        return result, audit
    result[i] = min(.5, max(-.5, float(effects[i]) + q))
    result[j] = min(.5, max(-.5, float(effects[j]) - q))
    audit['status'] = 'adjusted'
    audit['changed_indices'] = [k for k in (i, j) if result[k].tobytes() != effects[k].tobytes()]
    return result, audit


def select_once(selector, pre_scores, eligible, symbols, effects, source):
    """One caller-provided original selector, with the unchanged episode policy.

    The callback owns Top2 preservation and rank3..5 competition. This adapter
    never implements a new ranking/choice rule or forces rank3/4 to win.
    """
    need(callable(selector), 'Original selector callable')
    need(isinstance(pre_scores, np.ndarray) and pre_scores.dtype == np.dtype('float64') and pre_scores.ndim == 2,
         'Original P matrix')
    need(isinstance(eligible, np.ndarray) and eligible.dtype == np.dtype(bool) and eligible.shape == pre_scores.shape,
         'Original E matrix')
    need(isinstance(effects, np.ndarray) and effects.dtype == np.dtype('float64') and effects.shape == pre_scores.shape
         and np.isfinite(effects).all(), 'Effects matrix')
    need(type(source) is dict and isinstance(symbols, (list, tuple)) and len(symbols) == len(pre_scores), 'Caller source/monthly identities')
    for p, e, ss in zip(pre_scores, eligible, symbols):
        original_topk(p, e, ss)
    choices, final = selector(pre_scores.copy(), eligible.copy(), deepcopy(symbols), effects.copy(), deepcopy(source), dict(POLICY))
    need(isinstance(final, np.ndarray) and final.dtype == np.dtype('float64') and final.shape == pre_scores.shape,
         'Original selector score shape/dtype')
    need(isinstance(choices, (list, tuple)) and len(choices) == len(pre_scores), 'Original selector choices axis')
    for row, use in zip(choices, eligible):
        need(isinstance(row, (list, tuple, np.ndarray)) and len(row) in (0, 2, 3)
             and all(isinstance(j, (int, np.integer)) and not isinstance(j, (bool, np.bool_)) and 0 <= j < len(use) and use[j] for j in row)
             and len(set(row)) == len(row), 'Original selector valid eligible choices')
    return choices, final, dict(original_selector_calls=1, policy=dict(POLICY), new_global_score_ranks_computed=False,
                                forced_pair_winner=False, original_pre_scores_preserved=True,
                                source_authentication='Caller must supply and bind the original selector and source')
