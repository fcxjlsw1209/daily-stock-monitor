"""Caller-supplied 271 source/control adapter; no data loader or outcome replay.

Reuses only the frozen 251 raw/source layer and its independent source/control
checks. It never invokes 251 candidate finalization or a variance objective.
The 251 layer does compute its old per-security momentum/block statistics;
these are disclosed separately from the new 271 selection work.

File authentication and full-file decode inventories belong to the host. This
module copies only consumed monthly arrays and consumed original price history.
Calendar and listing records are metadata, not predictive/source numeric reads.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import monthly_joint_growth271_kernel as kernel
import monthly_joint_growth271_independent as independent

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / 'work/monthly_basket_core251_kernel.py'
SOURCE_SHA = 'dc70755222267891da5efe8acbbe8cf388ff4d481e92dbb5789e57ec21ea6ccd'
ORACLE_PATH = ROOT / 'work/monthly_basket_core251_independent.py'
ORACLE_SHA = '76bdbe2cc662bf4cabe2e8ac9a090d32e0e007b1b413f34cdf55780f747b67cb'
SCHEMA = 'monthly_joint_growth271_adapter_v1'
ARMS = kernel.ARMS
INITIAL_PARTIAL = '2016-08-17'
RESULT_KEYS = ('schema', 'n', 'source_features', 'events', 'choices', 'orders',
               'scores', 'pre_scores', 'audits', 'source_proof', 'incumbent_proof',
               'selection_proofs', 'work', 'scope')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _load(path, expected, name):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Frozen adapter dependency changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = _load(SOURCE_PATH, SOURCE_SHA, 'growth271_source251')
oracle = _load(ORACLE_PATH, ORACLE_SHA, 'growth271_independent251')


def dependency_files():
    return tuple(sorted(set(source.dependency_files()) | set(oracle.dependency_files()) |
                        {SOURCE_PATH, ORACLE_PATH, Path(kernel.__file__).resolve(),
                         Path(independent.__file__).resolve(), Path(__file__).resolve()}))


def _same_array(a, b, label):
    require(type(a) is np.ndarray and type(b) is np.ndarray and a.shape == b.shape
            and a.dtype == b.dtype and a.tobytes() == b.tobytes(), label + ' exact array')


def source_gate_check(check, n):
    """Enforce the frozen 251 host contract before any original episode."""
    require(type(check) is dict and all(check.get(k) is True for k in oracle.SOURCE_GATE_FLAGS),
            'Independent source gate flags')
    require(type(check.get('checked_events')) is int and check['checked_events'] == n
            and type(check.get('episode_calls')) is int and check['episode_calls'] == 0,
            'Independent source gate complete event/zero episode counts')
    for key, expected in (('C_source_atol', 1e-12), ('C_source_rtol', 0.),
                          ('continuous_atol', 2e-10), ('continuous_rtol', 2e-10),
                          ('rank_tolerance', 0.)):
        require(type(check.get(key)) is float and check[key] == expected,
                'Independent source gate numerical contract: ' + key)
    require(all(check.get(k) is False for k in
                ('new_target_ranks_computed', 'new_basket_selection_computed',
                 'producer_modules_imported', 'computes_returns', 'candidate_selection_computed')),
            'Independent source gate source-only scope')


def incumbent_gate_check(check, n):
    """Require complete exact original-only evidence before new 271 work."""
    require(type(check) is dict and all(check.get(k) is True for k in oracle.INCUMBENT_GATE_FLAGS),
            'Independent incumbent gate flags')
    require(all(type(check.get(k)) is int and check[k] == n
                for k in ('checked_events', 'ordered_choices_checked')),
            'Independent incumbent gate complete counts')
    require(type(check.get('numerical_tolerance')) is float and check['numerical_tolerance'] == 0.,
            'Independent incumbent gate exact numerical contract')
    require(all(check.get(k) is False for k in
                ('producer_modules_imported', 'computes_returns', 'candidate_selection_computed')),
            'Independent incumbent gate original-only scope')


def _prefix(arrays, meta, n):
    require(type(arrays) is dict and len(arrays) == 27 and type(meta) is dict,
            'Exactly 27 caller cache arrays and metadata')
    base = arrays.get('baseline')
    require(type(base) is np.ndarray and base.ndim == 2 and base.shape[1] == 49,
            'Original 49-column matrix')
    total = base.shape[0]
    if n is None:
        n = total
    require(type(n) is int and 0 <= n <= total, 'Native prefix length')
    aa = {}
    for k, value in arrays.items():
        require(type(k) is str and type(value) is np.ndarray and value.ndim >= 1
                and value.shape[0] == total, 'Monthly array axis ' + str(k))
        aa[k] = value[:n].copy()
    mm = {}
    for k in ('dates', 'ends', 'symbols'):
        require(type(meta.get(k)) is list and len(meta[k]) == total, 'Metadata axis ' + k)
        mm[k] = deepcopy(meta[k][:n])
    require(type(meta.get('etfs')) in (list, tuple), 'Static ETF metadata')
    mm['etfs'] = deepcopy(meta['etfs'])
    return aa, mm, n


def _review_prefix(review, n):
    require(type(review) is dict and review.get('schema') == source.SOURCE_SCHEMA,
            'Frozen 251 source review schema')
    require(type(review.get('clock_events')) is list and len(review['clock_events']) >= n
            and type(review.get('identity_cells')) is list, 'Source metadata inventory')
    clocks = deepcopy(review['clock_events'][:n])
    require(all(type(row) is dict and type(row.get('query_index')) is int
                and row['query_index'] == i for i, row in enumerate(clocks)), 'Ordered consumed clocks')
    # Only query-index metadata is inspected for unconsumed identity records.
    # Their body, and every unconsumed clock body, are never copied or hashed.
    ids = []
    if n:
        for row in review['identity_cells']:
            require(type(row) is dict and type(row.get('query_index')) is int
                    and row['query_index'] >= 0, 'Identity query metadata')
            if row['query_index'] < n:
                ids.append(deepcopy(row))
    return dict(schema=source.SOURCE_SCHEMA, panel_first_session=review.get('panel_first_session'),
                clock_events=clocks, listing_bounds=deepcopy(review.get('listing_bounds')),
                identity_cells=ids)


def physical_inputs(arrays, meta, panel, calendar, source_review, n=None):
    """Physically short numeric inputs; no full-file I/O is performed here."""
    aa, mm, n = _prefix(arrays, meta, n)
    require(type(panel) is dict and set(panel) == {'adj'}, 'Only adjusted-price panel')
    frame = panel['adj']
    source._frame(frame)
    sessions = source.old._calendar(calendar)
    formal = [i for i, d in enumerate(mm['dates']) if d != INITIAL_PARTIAL]
    for d in mm['dates']:
        source.old._iso(d, 'Consumed query')
    if formal:
        last = mm['dates'][formal[-1]]
        names = sorted({s for i in formal for s in mm['symbols'][i]})
        require(set(names) <= set(frame.columns), 'Original source symbol coverage')
        pp = {'adj': frame.loc[:last, names].copy(deep=True)}
    else:
        pp = {'adj': frame.iloc[:0].copy(deep=True)}
    if n:
        require(mm['dates'][-1] in sessions, 'Last query official session')
        end = sessions.index(mm['dates'][-1])
        require(end + 1 < len(sessions), 'Next official witness')
        cc = sessions[:end + 2]
    else:
        cc = []
    return aa, mm, pp, cc, _review_prefix(source_review, n)


def _incumbent(study, b, n):
    """The original-episode fragment of 251._select, without its finalizer."""
    pre = b['pre'].copy()
    original = dict(baseline=b['base'], core=b['core'], core_weight=b['cw'],
                    eligible=b['good'], state_calm10=b['calm'], state_weak_narrow=b['weak'])
    calls = []
    def boundary(score, eligible, names, effects, supplied, policy):
        require(not calls and policy == source.POLICY, 'One original episode and policy')
        _same_array(score, pre, 'Original pre-score')
        _same_array(eligible, b['good'], 'Original E')
        _same_array(effects, b['effects'], 'Original effects')
        require(names == b['meta']['symbols'], 'Original symbols')
        for k, v in original.items():
            if k != 'eligible':
                _same_array(supplied[k], v, 'Original support ' + k)
        calls.append(True)
        return study.ep.select_boundary(score, eligible, names, effects, supplied, policy)
    choices, scores = source.old.core_adapter.score_and_choose(
        'incumbent66', 0., np.full((n, b['width']), np.nan), original, b['meta'], n,
        pre_score=pre, effects=b['effects'], episode_policy=source.POLICY.copy(), boundary_selector=boundary)
    require(len(calls) == 1, 'Exactly one producer original episode')
    scores = source.old._matrix(scores, (n, b['width']), n, 'Original final scores')
    choices = [[int(j) for j in row] for row in choices]
    return choices, scores, pre, study.rp.make_orders(choices, scores, b['meta'])


def _windows(features, panel, calendar, review, incumbent):
    frame = panel['adj']
    bounds = {x['symbol']: x for x in review['listing_bounds']}
    events, observed = [], 0
    for i, feature in enumerate(features):
        paths = [None] * len(feature['symbols'])
        if not feature['initial_partial']:
            t = feature['query_position']
            days = [pd.Timestamp(d) for d in calendar[t - 294:t - 41]]
            require(len(days) == 253, 'Complete formation price window')
            for j in feature['required_indices']:
                name = feature['symbols'][j]
                producer = [source.old._number(frame.at[d, name], 'Adapter formation price') for d in days]
                require(all(v > 0 for v in producer), 'Positive adapter formation prices')
                other = oracle.required_prices(frame, name, days, bounds.get(name), feature['date'])
                _same_array(np.asarray(producer, dtype=np.float64), np.asarray(other, dtype=np.float64),
                            'Independent required-price window')
                observed += len(producer)
                paths[j] = producer
        events.append(dict(date=feature['date'], symbols=feature['symbols'].copy(),
                           eligible=[j in feature['original_eligible_indices'] for j in range(len(paths))],
                           prices253=paths, incumbent_choices=incumbent[i].copy()))
    return events, observed


def prepare(study, arrays, meta, panel, calendar, source_review, n=None, *, expected_incumbent_orders=None):
    aa, mm, pp, cc, rr = physical_inputs(arrays, meta, panel, calendar, source_review, n)
    n = len(mm['dates'])
    before = source.old._snapshot(aa)
    meta_before = deepcopy(mm)
    b = source.old._cache(study, aa, mm, n)
    features = source._features(pp, mm, cc, rr, b, n)
    source_proof = oracle.verify_source_features(pp, mm, cc, aa, rr, features, n)
    source_gate_check(source_proof, n)
    incumbent, post, pre, inc_orders = _incumbent(study, b, n)
    if expected_incumbent_orders is not None:
        require(type(expected_incumbent_orders) is list and len(expected_incumbent_orders) >= n,
                'Frozen incumbent prefix')
        source._strict_equal(inc_orders, expected_incumbent_orders[:n], 'Frozen incumbent orders')
    inc_proof = oracle.verify_incumbent(aa, mm, inc_orders, post, pre, n)
    incumbent_gate_check(inc_proof, n)
    events, window_observations = _windows(features, pp, cc, rr, incumbent)
    choices, orders, scores, pres, audits, proofs = {}, {}, {}, {}, {}, {}
    for arm in ARMS:
        chosen, audit = kernel.select(events, arm)
        proof = independent.verify_selection(events, arm, chosen, audit)
        require(type(proof) is dict and proof.get('passed') is True, 'Independent 271 selection gate failed')
        if arm == ARMS[0]:
            require(chosen == incumbent, 'Pure incumbent ordered copy')
            display, pscore, out_orders = post.copy(), pre.copy(), deepcopy(inc_orders)
        else:
            display = np.zeros((n, b['width']), dtype=np.float64)
            pscore = display.copy()
            for i, row in enumerate(chosen):
                if events[i]['date'] == INITIAL_PARTIAL:
                    require(row == incumbent[i], 'Partial choices exact')
                    display[i], pscore[i] = post[i], pre[i]
                else:
                    for k, j in enumerate(row):
                        display[i, j] = pscore[i, j] = float(len(row) - k)
            out_orders = study.rp.make_orders(chosen, display, mm)
            for i, e in enumerate(events):
                if e['date'] == INITIAL_PARTIAL:
                    source._strict_equal(out_orders[i], inc_orders[i], 'Partial exact order')
        choices[arm], orders[arm], scores[arm], pres[arm] = chosen, out_orders, display, pscore
        audits[arm], proofs[arm] = audit, proof
    require(before == source.old._snapshot(aa) and mm == meta_before, 'Consumed source arrays/metadata mutation')
    formal = [f for f in features if not f['initial_partial']]
    required = sum(len(f['required_indices']) for f in formal)
    producer_past = int(pp['adj'].size) if formal else 0
    independent_past = sum(int(pp['adj'].loc[:f['date'], f['symbols']].size) for f in formal)
    work = dict(events=n, symbols_per_event=b['width'], source_feature_calls=1,
                independent_source_feature_calls=1, producer_original_episode_calls=1,
                independent_original_episode_calls=1, candidate_episode_calls=0,
                consumed_panel_copy_cells=producer_past,
                producer_original_history_price_observations=producer_past,
                independent_original_history_price_observations=independent_past,
                producer251_required_price_observations=253 * required,
                independent251_required_price_observations=253 * required,
                producer251_momentum_evaluations=required, independent251_momentum_evaluations=required,
                producer251_block_return_evaluations=12 * required, independent251_block_return_evaluations=12 * required,
                adapter_required_price_observations=window_observations,
                independent_window_price_observations=window_observations,
                exact_window_value_comparisons=window_observations,
                producer_selection_counts={a: deepcopy(audits[a]['counts']) for a in ARMS},
                independent_selection_counts={a: {k: proofs[a][k] for k in
                                                  ('checked_events', 'checked_required_cells', 'checked_trios',
                                                   'exact_scalar_checks', 'log_display_checks')} for a in ARMS},
                new_model_fits=0, predictive_model_fits=0)
    scope = dict(source_files_authenticated=False, full_files_decoded=False, original_arrays_mutated=False,
                 source_and_core_numerically_checked=True, original_incumbent_numerically_checked=True,
                 required_windows_independently_checked=True, computes_strategy_returns=False,
                 computes_predictive_labels=False, candidate_scores_are_order_displays_only=True,
                 source251_candidate_finalizer_called=False, new_objective_evaluated_only_after_incumbent_gate=True,
                 full_history_work='Producer one union matrix; independent separately per formal query; both 251 per-security measures counted.',
                 metadata_scope='Full calendar/listing metadata and identity query indices; only consumed clock/identity bodies.',
                 numeric_scope='Only consumed 27-array rows and union-symbol original price history through last formal query.')
    return dict(schema=SCHEMA, n=n, source_features=features, events=events, choices=choices,
                orders=orders, scores=scores, pre_scores=pres, audits=audits,
                source_proof=source_proof, incumbent_proof=inc_proof, selection_proofs=proofs, work=work, scope=scope)
