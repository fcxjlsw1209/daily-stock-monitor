"""Frozen251 leader-domain bridge into one original Local67 selector.

No IO, source loading, fitting or returns. Caller-supplied frozen251 features
are verified against the original source before the old endpoint gates. Only
the eligible argument of the bridge episode changes; its pre-score and all
support/effect/model inputs retain the original all49 computation.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = ROOT / 'work/monthly_basket_core251_kernel.py'
OLD_SHA = 'dc70755222267891da5efe8acbbe8cf388ff4d481e92dbb5789e57ec21ea6ccd'
MATH_PATH = ROOT / 'work/monthly_basket_core251_math.py'
MATH_SHA = '217f5f9eebd68e40cca94057ed2b550e6751aee056a83ecdab41faa541e7f6e4'
for _path, _sha in ((OLD_PATH, OLD_SHA), (MATH_PATH, MATH_SHA)):
    if hashlib.sha256(_path.read_bytes()).hexdigest() != _sha:
        raise ValueError('Changed frozen251 dependency: ' + _path.name)
_spec = importlib.util.spec_from_file_location('leader253_frozen251', OLD_PATH)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
require = old.require
INITIAL_PARTIAL = old.INITIAL_PARTIAL
POLICY = dict(old.POLICY)
ARMS = ('incumbent_local67', 'long_leader_local67', 'basket_core251_control')
ENDPOINT_ARMS = ('incumbent_local67', 'long_momentum_basket_core')
SCHEMA = 'monthly_leader_bridge253_decisions_v1'
AUDIT_EVENT_KEYS = ('query_index', 'date', 'branch', 'original_eligible_indices',
    'new_eligible_indices', 'shortlist_indices', 'selector_eligible_indices',
    'pre_order', 'boundary_candidates', 'episode_budget', 'original_remaining_support',
    'incumbent_choices', 'candidate_choices', 'basket_choices')
COUNT_FIELDS = ('events', 'symbols_per_event', 'initial_partial_events', 'formal_events',
    'original_eligible_cells', 'positive_momentum_cells', 'shortlist_cells',
    'restricted_eligible_cells', 'removed_eligible_cells', 'cash_events', 'two_name_events',
    'threeplus_name_events', 'candidate_changed_holdings_events', 'basket_changed_holdings_events',
    'candidate_vs_basket_changed_holdings_events', 'original_episode_calls',
    'candidate_episode_calls', 'diagnostic_episode_calls', 'reused_feature_events',
    'new_factor_definitions', 'new_model_fits', 'predictive_model_fits')


def dependency_files():
    return tuple(dict.fromkeys((OLD_PATH, MATH_PATH,
        ROOT / 'work/test_monthly_basket_core251_kernel.py') + old.dependency_files()))


def source_view(features, n):
    """Only consumed source evidence; do not inspect future feature bodies."""
    require(type(n) is int and n >= 0 and type(features) is list and len(features) >= n,
            'Frozen251 feature prefix')
    out = deepcopy(features[:n])
    for e in out:
        require(type(e) is dict and tuple(e) == old.EVENT_KEYS, 'Frozen251 event schema')
        if not e['initial_partial']:
            e['branch'] = 'source_verified'
            for key in old.CANDIDATE_KEYS:
                e[key] = None
    return out


def _bytes_equal(actual, expected, n, label):
    require(isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray), label + ' arrays')
    require(actual.shape == expected.shape == (n, 49), label + ' shape')
    require(actual.dtype == expected.dtype == np.dtype('float64'), label + ' dtype')
    require(actual.tobytes() == expected.tobytes(), label + ' exact binary64')


def _expected(expected, n):
    if expected is None:
        return
    require(type(expected) is dict and set(expected) == {'orders', 'scores', 'pre_scores'},
            'Expected endpoint envelope')
    for key in ('orders', 'scores', 'pre_scores'):
        require(type(expected[key]) is dict and set(expected[key]) == set(ENDPOINT_ARMS),
                'Expected endpoint arms')
    for arm in ENDPOINT_ARMS:
        require(type(expected['orders'][arm]) is list and len(expected['orders'][arm]) == n,
                'Expected endpoint order axis')


def _decision(study, arrays, meta, features, n, b, *, incumbent_validator=None,
              endpoint_validator=None, expected_endpoints=None):
    _expected(expected_endpoints, n)
    # The frozen251 incumbent gate precedes its exact cached shortlist/trio check.
    ff = deepcopy(features[:n])
    oo, ss, pp, old_audit = old._select(study, meta, b, ff, n, False,
        incumbent_validator=incumbent_validator,
        expected_incumbent_orders=None if expected_endpoints is None else
            expected_endpoints['orders'][ENDPOINT_ARMS[0]])
    endpoint_orders = {arm: deepcopy(oo[arm]) for arm in ENDPOINT_ARMS}
    endpoint_scores = {arm: ss[arm].copy() for arm in ENDPOINT_ARMS}
    endpoint_pres = {arm: pp[arm].copy() for arm in ENDPOINT_ARMS}
    if expected_endpoints is not None:
        for arm in ENDPOINT_ARMS:
            old._strict_equal(endpoint_orders[arm], expected_endpoints['orders'][arm],
                              'Frozen251 endpoint orders ' + arm)
            _bytes_equal(endpoint_scores[arm], expected_endpoints['scores'][arm], n,
                         'Frozen251 endpoint scores ' + arm)
            _bytes_equal(endpoint_pres[arm], expected_endpoints['pre_scores'][arm], n,
                         'Frozen251 endpoint pre ' + arm)
    if endpoint_validator is not None:
        require(callable(endpoint_validator), 'Endpoint validator callable')
        receipt = endpoint_validator(deepcopy(endpoint_orders),
            {k: v.copy() for k, v in endpoint_scores.items()},
            {k: v.copy() for k, v in endpoint_pres.items()})
        require(type(receipt) is dict and receipt.get('passed') is True,
                'Independent frozen251 endpoint gate failed')

    # Exact251 finalized evidence has been checked, and BOTH old endpoints are
    # gated. Only now construct the bridge's restricted selector domain.
    eligible = b['good'].copy()
    for i, e in enumerate(ff):
        if e['initial_partial']:
            continue
        eligible[i] = False
        eligible[i, e['shortlist_indices']] = b['good'][i, e['shortlist_indices']]
    pre = b['pre'].copy()
    source = dict(baseline=b['base'].copy(), core=b['core'].copy(),
        core_weight=b['cw'].copy(), eligible=b['good'].copy(),
        state_calm10=b['calm'].copy(), state_weak_narrow=b['weak'].copy())
    selected, post = study.ep.select_boundary(pre.copy(), eligible.copy(),
        deepcopy(b['meta']['symbols']), b['effects'].copy(), source, POLICY.copy())
    post = old.old._matrix(post, (n, 49), n, 'Bridge final score')
    chosen = [[int(j) for j in row] for row in selected]
    require(len(chosen) == n, 'Bridge choices axis')
    orders = {ARMS[0]: endpoint_orders[ENDPOINT_ARMS[0]],
              ARMS[1]: study.rp.make_orders(chosen, post, b['meta']),
              ARMS[2]: endpoint_orders[ENDPOINT_ARMS[1]]}
    scores = {ARMS[0]: endpoint_scores[ENDPOINT_ARMS[0]], ARMS[1]: post,
              ARMS[2]: endpoint_scores[ENDPOINT_ARMS[1]]}
    pres = {ARMS[0]: endpoint_pres[ENDPOINT_ARMS[0]], ARMS[1]: pre,
            ARMS[2]: endpoint_pres[ENDPOINT_ARMS[1]]}
    events = []
    for i, e in enumerate(ff):
        partial = e['initial_partial']
        inc = old_audit['events'][i]['arms'][ENDPOINT_ARMS[0]]
        basket = old_audit['events'][i]['arms'][ENDPOINT_ARMS[1]]
        ids = np.flatnonzero(eligible[i]).tolist()
        ranked = sorted(ids, key=lambda j: (-pre[i, j], e['symbols'][j]))
        boundary = [] if len(ranked) < 3 else [j for j in ranked[2:POLICY['ceiling']]
            if pre[i, ranked[2]] - pre[i, j] <= POLICY['gap']]
        require(len(chosen[i]) == (3 if len(ids) >= 3 else (2 if len(ids) == 2 else 0)),
                'Original three/two/cash rule')
        require(len(set(chosen[i])) == len(chosen[i]) and set(chosen[i]) <= set(ids),
                'Bridge choices in exact restricted domain')
        if partial:
            old._strict_equal(orders[ARMS[1]][i], orders[ARMS[0]][i], 'Partial orders exact')
            require(post[i].tobytes() == scores[ARMS[0]][i].tobytes() and
                    pre[i].tobytes() == pres[ARMS[0]][i].tobytes(), 'Partial scores/pre exact')
        branch = 'initial_partial' if partial else ('cash' if len(ids) < 2 else
            ('two_names' if len(ids) == 2 else 'restricted_episode'))
        events.append(dict(query_index=i, date=e['date'], branch=branch,
            original_eligible_indices=np.flatnonzero(b['good'][i]).tolist(),
            new_eligible_indices=None if partial else e['new_eligible_indices'].copy(),
            shortlist_indices=None if partial else e['shortlist_indices'].copy(),
            selector_eligible_indices=ids, pre_order=ranked, boundary_candidates=boundary,
            episode_budget=float(b['episode'][i]), original_remaining_support=b['remaining'][i].tolist(),
            incumbent_choices=inc.copy(), candidate_choices=chosen[i], basket_choices=basket.copy()))
    formal = [e for e in events if e['branch'] != 'initial_partial']
    counts = dict(events=n, symbols_per_event=49, initial_partial_events=n-len(formal),
        formal_events=len(formal), original_eligible_cells=int(b['good'].sum()),
        positive_momentum_cells=sum(len(e['new_eligible_indices']) for e in formal),
        shortlist_cells=sum(len(e['shortlist_indices']) for e in formal),
        restricted_eligible_cells=int(eligible.sum()),
        removed_eligible_cells=int((b['good'] & ~eligible).sum()),
        cash_events=sum(e['branch'] == 'cash' for e in formal),
        two_name_events=sum(e['branch'] == 'two_names' for e in formal),
        threeplus_name_events=sum(e['branch'] == 'restricted_episode' for e in formal),
        candidate_changed_holdings_events=sum(set(e['candidate_choices']) != set(e['incumbent_choices']) for e in events),
        basket_changed_holdings_events=sum(set(e['basket_choices']) != set(e['incumbent_choices']) for e in events),
        candidate_vs_basket_changed_holdings_events=sum(set(e['candidate_choices']) != set(e['basket_choices']) for e in events),
        original_episode_calls=1, candidate_episode_calls=1, diagnostic_episode_calls=0,
        reused_feature_events=n, new_factor_definitions=0, new_model_fits=0, predictive_model_fits=0)
    require(tuple(counts) == COUNT_FIELDS and all(type(v) is int for v in counts.values()),
            'Bridge count schema')
    require(all(tuple(e) == AUDIT_EVENT_KEYS for e in events), 'Bridge event schema')
    return orders, scores, pres, dict(schema=SCHEMA, counts=counts, events=events)



def _consumed_snapshot(meta, features, n):
    """Protect consumed metadata without copying or comparing future bodies.

    Array byte fingerprints intentionally cover the complete caller cache as
    mutation evidence. They do not interpret future array values or feed them
    into a prefix score; object-bearing future metadata is never traversed.
    """
    require(type(n) is int and n >= 0 and type(meta) is dict, 'Prefix snapshot')
    require(type(features) is list and len(features) >= n, 'Feature prefix axis')
    for key in ('dates', 'ends', 'symbols'):
        require(type(meta.get(key)) is list and len(meta[key]) >= n, 'Metadata prefix axis')
    mm = {key: deepcopy(meta[key][:n]) for key in ('dates', 'ends', 'symbols')}
    mm['etfs'] = deepcopy(meta.get('etfs'))
    return mm, deepcopy(features[:n])


def decisions(study, arrays, meta, features, n, *, incumbent_validator=None,
              endpoint_validator=None, expected_endpoints=None):
    before = old.old._snapshot(arrays)
    mm, frozen = _consumed_snapshot(meta, features, n)
    b = old.old._cache(study, arrays, meta, n)
    require(b['width'] == 49, 'Original49 symbol axis')
    out = _decision(study, arrays, meta, features, n, b,
        incumbent_validator=incumbent_validator, endpoint_validator=endpoint_validator,
        expected_endpoints=expected_endpoints)
    require(before == old.old._snapshot(arrays) and _consumed_snapshot(meta, features, n) == (mm, frozen),
            'Original cache/metadata/frozen251 features mutated')
    return out


def prepare(study, arrays, meta, panel, calendar, source_review, features, n, *,
            source_validator=None, incumbent_validator=None, endpoint_validator=None,
            expected_endpoints=None):
    before = old.old._snapshot(arrays)
    mm, frozen = _consumed_snapshot(meta, features, n)
    b = old.old._cache(study, arrays, meta, n)
    require(b['width'] == 49, 'Original49 symbol axis')
    rebuilt = old._features(panel, meta, calendar, source_review, b, n)
    cached_source = source_view(features, n)
    old._cached(cached_source, meta, b, n, True)
    old._strict_equal(rebuilt, cached_source, 'Rebuilt frozen251 source evidence')
    if source_validator is not None:
        require(callable(source_validator), 'Source validator callable')
        receipt = source_validator(deepcopy(rebuilt))
        require(type(receipt) is dict and receipt.get('passed') is True,
                'Independent frozen251 source gate failed')
    out = _decision(study, arrays, meta, features, n, b,
        incumbent_validator=incumbent_validator, endpoint_validator=endpoint_validator,
        expected_endpoints=expected_endpoints)
    require(before == old.old._snapshot(arrays) and _consumed_snapshot(meta, features, n) == (mm, frozen),
            'Source preparation input mutation')
    return out
