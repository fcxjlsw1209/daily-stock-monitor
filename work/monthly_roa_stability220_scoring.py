"""Pure .01 ROA correction to original pre-score, with one frozen episode pass.

No sources, fits, I/O or outcomes. The host authenticates original inputs and
injects the frozen study, policy, and adapter. Core and support are unchanged.
"""
import numpy as np

NAMES = ('incumbent_local67', 'roa_stability_penalty', 'three_year_roa_mean_control')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
DOSE = .01


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scale_and_delta(values, arrays, n):
    require(set(values) == {'available', 'candidate', 'control'}, 'Exact feature schema required')
    shape = np.asarray(arrays['baseline']).shape
    require(len(shape) == 2 and shape[1] > 0 and type(n) is int and 0 <= n <= shape[0],
            'Invalid score prefix')
    available = np.asarray(values['available'])
    eligible = np.asarray(arrays['eligible'])
    require(available.shape == eligible.shape == shape and
            available.dtype == eligible.dtype == np.dtype(bool), 'Boolean support axes required')
    targets, deltas = {}, {}
    for name, key in RAW.items():
        field = np.asarray(values[key])
        require(field.shape == shape and field.dtype == np.dtype('float64'), 'Float64 target axes required')
        target = field[:n].copy()
        require(not np.isinf(target).any() and
                np.array_equal(np.isfinite(target), available[:n]), 'Common exact source mask required')
        require((np.abs(target[available[:n]]) <= 1.).all(), 'h must stay within binary64 bounded range')
        active = eligible[:n] & available[:n] & (target != 0.)
        delta = np.zeros((n, shape[1]), dtype=float)
        delta[active] = DOSE * target[active]
        require(np.isfinite(delta).all() and np.max(np.abs(delta), initial=0.) <= DOSE,
                'Correction exceeded fixed budget')
        targets[name], deltas[name] = target, delta
    require((targets[NAMES[1]][available[:n]] <= targets[NAMES[2]][available[:n]]).all(),
            'Dispersion penalty increased source score')
    return targets, deltas


def decisions(study, arrays, meta, values, n, *, policy, adapter):
    targets, deltas = scale_and_delta(values, arrays, n)
    shape = np.asarray(arrays['baseline']).shape
    require(len(meta['symbols']) == shape[0], 'Monthly candidate axes required')
    etfs = set(meta['etfs'])
    for i, symbols in enumerate(meta['symbols'][:n]):
        require(len(symbols) == shape[1], 'Candidate width differs')
        require(all(not values['available'][i, j] for j, symbol in enumerate(symbols) if symbol in etfs),
                'ETF financial correction forbidden')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    # h is bounded by1, not a centered rank. The zero-blend adapter receives
    # missing ranks; the separately bounded h enters only original pre-score.
    absent_rank = np.full(shape, np.nan)
    for arm in NAMES[1:]:
        pre = np.asarray(study.modified_pre_score(study.configs()[4], arrays, n))
        require(pre.shape in (shape, (n, shape[1])), 'Original pre-score axes differ')
        prefix = np.array(pre[:n], dtype=float, copy=True)
        require(np.isfinite(prefix).all(), 'Finite original pre-score required')
        changed = deltas[arm] != 0.
        prefix[changed] += deltas[arm][changed]
        supplied = np.zeros(shape)
        supplied[:n] = prefix
        chosen, score = adapter.score_and_choose('incumbent66', 0., absent_rank,
            arrays, meta, n, pre_score=supplied, effects=arrays['effects'],
            episode_policy=policy, boundary_selector=study.ep.select_boundary)
        orders[arm] = study.rp.make_orders(chosen, score, meta)
    return orders, targets, deltas
