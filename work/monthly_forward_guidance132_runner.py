"""Prospective round132 orchestration; no registration and no work on import.

Only ``prepare``, ``select_first88`` and ``diagnose_later34`` are executable.
Root must first create a reviewed, frozen PLAN and source-admission manifest.
The runner never fits a model or constructs a successful-issuer-only universe.

PLAN schema (all referenced bytes must also occur in ``hashes``):
  round=132, status=REGISTERED, arms=NAMES, primary_challenger=NAMES[-1],
  output_directory, runtime={python,numpy,pandas},
  inputs={proposal,admission,events,availability,meta,sessions,policy,
          original_orders,prices,protected_manifest}, each {path,sha256},
  hashes, raw_source_hashes, independent_prepare_source_hashes.
Admission schema: status=PASS, semantic_review_passed=True,
  availability_review_passed=True, calendar_review_passed=True,
  events_sha256, availability_sha256, reviewed_event_ids, source_hashes.
This explicit protocol is provisional until root freezes its independent review.

CLI example AFTER registration: python -B work/monthly_forward_guidance132_runner.py
  prepare --plan /absolute/new/PLAN.json --plan-sha <sha256> --root <workspace>
Then run the independent prepare reviewer, then select_first88, then
diagnose_later34 using the same arguments. The latter replays all 122 frozen
orders for full-period/cost accounting, but cannot change the first88 selection.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import monthly_forward_guidance_factor132 as factor

NAMES = ('incumbent_local67', 'zero_forward_control',
         'matched_realized_growth_control', 'next_quarter_forward_growth')
TRAIN, MONTHS, WIDTH = 88, 122, 49
BUDGET = .025
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'FEATURE_AUDIT.json',
               'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json')
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json',
                'TRAIN_TRIALS.json', 'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
CHECKS = ('original27_arrays_byte_identical', 'all_features_independently_reconstructed',
          'all_four_score_matrices_and_ordered_choices_independently_verified',
          'one_episode_and_prefix_isolation_verified', 'zero_and_missing_exact_fallback')
INPUTS = ('proposal', 'admission', 'events', 'availability', 'meta', 'sessions',
          'policy', 'original_orders', 'prices', 'protected_manifest')
PROTECTED_SHA = '89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb'
LOCAL_SHA = 'e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb'
TOLERANCES = dict(pre_delta=1e-12, scores_atol=1e-14, scores_rtol=0.,
    price_atol=1e-10, price_rtol=1e-10, ledger_atol=2e-9, ledger_rtol=2e-10,
    dollar_atol=1e-5, dollar_rtol=2e-11, dimensionless=1e-10, training_cagr=1e-10)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def source_events(value):
    events = value.get('events') if isinstance(value, dict) else value
    require(isinstance(events, list) and all(isinstance(e, dict) for e in events),
            'Source event list required')
    return events


def normalized_hashes(mapping, root):
    result = {}
    for path, digest in mapping.items():
        p = Path(path)
        key = str((p if p.is_absolute() else Path(root)/p).resolve())
        require(key not in result or result[key] == digest, 'Conflicting normalized source hashes')
        result[key] = digest
    return result


def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def source_files(root):
    """Minimum code/runtime pins; Local67Backend also requires its old closure."""
    root = Path(root)
    paths = [Path(__file__), Path(__file__).with_name('test_monthly_forward_guidance132_runner.py'),
        root/'work/monthly_forward_guidance_factor132.py',
        root/'work/monthly_forward_guidance_contract_v1.py',
        root/'outputs/monthly_path70_research/incumbent_adapter.py',
        root/'outputs/monthly_guidance130/study.py',
        root/'outputs/monthly_local_linear70/study.py',
        Path(sys.executable), Path(np.__file__), Path(pd.__file__)]
    return {str(p.resolve()): sha(p) for p in paths}


def validate_meta(meta):
    require(len(meta['dates']) == MONTHS and meta.get('train_months') == TRAIN,
            'Original 122 months / first88 split required')
    require(len(meta['ends']) == MONTHS and len(meta['symbols']) == MONTHS, 'Original axes')
    require(meta['dates'] == sorted(set(meta['dates'])), 'Unique ordered monthly dates')
    etfs = set(meta['etfs'])
    require(len(etfs) == 17, 'Original 17 ETFs remain in the investment pool')
    for names in meta['symbols']:
        require(len(names) == WIDTH and len(set(names)) == WIDTH and etfs <= set(names),
                'Each original pool must retain 32 equities plus 17 ETFs')


def build_features(events, rows, reviewed, meta):
    """Convert an already independently reviewed source-month grid, never rejoin it."""
    validate_meta(meta)
    expected = [(i, j, day, symbol) for i, (day, names) in enumerate(
        zip(meta['dates'], meta['symbols'])) for j, symbol in enumerate(names)]
    require(len(rows) == len(expected), 'All 5978 original source-month rows required')
    for row, key in zip(rows, expected):
        require(tuple(row[k] for k in ('signal_index', 'pool_index', 'signal_date', 'symbol')) == key,
                'Source-month axes/order changed')
        require(type(row.get('is_etf')) is bool and row['is_etf'] == (row['symbol'] in meta['etfs']),
                'Explicit original ETF identity required')
    cells = factor.features_for_rows(events, rows, set(reviewed))
    forward = np.full((MONTHS, WIDTH), np.nan)
    realized = forward.copy()
    for cell in cells:
        i, j = cell['signal_index'], cell['pool_index']
        require((cell['forward'] is None) == (cell['realized'] is None), 'Matched source mask')
        if cell['forward'] is not None:
            forward[i, j], realized[i, j] = cell['forward'], cell['realized']
    require(np.array_equal(np.isfinite(forward), np.isfinite(realized)), 'Matched source mask')
    return forward, realized, cells


def corrected_choices(pre, hand, arrays, meta, n, adapter, selector, policy):
    """130 formula with continuous bounded H; original episode is called once."""
    raw = np.asarray(hand)
    require(raw.shape == np.asarray(arrays['baseline']).shape and all(
            isinstance(v, (int, float, np.integer, np.floating)) and
            not isinstance(v, (bool, np.bool_)) for v in raw[:n].flat),
            'Numeric hand matrix with exact original axes required')
    _, shape, _, _, cw, h, finite, support = adapter._inputs('incumbent66', BUDGET, raw, arrays, n)
    require(not np.isinf(h).any(), 'Infinity is not missing')
    good = adapter._matrix(arrays['eligible'], shape, n, 'eligible', boolean=True)
    for i, names in enumerate(meta['symbols'][:n]):
        require(all(not finite[i, j] for j, name in enumerate(names) if name in meta['etfs']),
                'ETFs must have missing guidance')
    changed = finite & (h != 0.) & good
    require(not np.any(1. - cw - adapter.OLD_LIMIT - BUDGET * changed < .355 - 1e-12),
            'Original support reserve')
    view = adapter._matrix(pre, shape, n, 'pre_score').copy()
    delta = np.zeros(view.shape)
    delta[changed] = BUDGET * (np.clip(support[changed] + h[changed], -.5, .5) - support[changed])
    require(np.isfinite(delta).all() and not np.any(np.abs(delta) > .0125 + 1e-12), 'Pre-episode bound')
    nz = delta != 0.
    view[nz] += delta[nz]  # no writes/additions for measured zero or missing
    full = np.zeros(shape, dtype=np.float64)
    full[:n] = view
    return adapter.score_and_choose('incumbent66', 0., hand, arrays, meta, n,
        pre_score=full, effects=arrays['effects'], episode_policy=policy, boundary_selector=selector)


def choose(name, arrays, meta, n, forward, realized, backend, policy):
    require(name in NAMES, 'Unknown arm')
    if name == NAMES[0]:
        return backend.baseline(arrays, meta, n, policy)
    hand = forward if name in (NAMES[1], NAMES[3]) else realized
    if name == NAMES[1]:
        hand = hand.copy()
        hand[:n] = np.where(np.isfinite(hand[:n]), 0., np.nan)
    return corrected_choices(backend.pre(arrays, n), hand, arrays, meta, n,
        backend.adapter, backend.selector, policy)


def order_panels(arrays, meta, forward, realized, backend, policy):
    orders, scores = {}, {}
    original = {k: (np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
                for k, v in arrays.items()}
    for name in NAMES:
        choices, score = choose(name, arrays, meta, MONTHS, forward, realized, backend, policy)
        orders[name] = backend.make_orders(choices, score, meta)
        scores[name] = score
        for n in (0, 1, 36, 60, 76, 88):
            pc, ps = choose(name, arrays, meta, n, forward, realized, backend, policy)
            require(pc == choices[:n] and ps.tobytes() == score[:n].tobytes(), 'Prefix isolation failed')
    require(orders[NAMES[0]] == orders[NAMES[1]] and
            scores[NAMES[0]].tobytes() == scores[NAMES[1]].tobytes(), 'Exact zero control failed')
    for hand in (np.full(forward.shape, np.nan),
                 np.where(np.isfinite(forward), 0., np.nan)):
        for name in NAMES[2:]:
            cc, ss = choose(name, arrays, meta, MONTHS, hand, hand, backend, policy)
            require(backend.make_orders(cc, ss, meta) == orders[NAMES[0]] and
                    ss.tobytes() == scores[NAMES[0]].tobytes(), 'Missing/zero fallback failed')
    require(original == {k: (np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
                         for k, v in arrays.items()}, 'Original arrays mutated')
    return orders, scores


def select(rows):
    require([r['name'] for r in rows] == list(NAMES), 'Four fixed arms in order required')
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [r.get(k, {}).get(s) for r in (rows[3], rows[0]) for k, s in fields]
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and
                  not isinstance(v, (bool, np.bool_)) for v in values)
    good = False
    if numeric and np.isfinite(values).all():
        v, b = values[:5], values[5:]
        good = bool(v[0] > b[0] + 1e-12 and all(x >= y - t for x, y, t in
                    zip(v, b, (1e-12, .05, .03, .025, .025))))
    return dict(primary=NAMES[3] if good else NAMES[0], admitted=good, guards=good,
                guards_against={NAMES[0]: good})


class Local67Backend:
    """Exact existing calculation hooks, loaded only after PLAN/admission checks."""
    def __init__(self, root, runner):
        root = Path(root)
        self.root = root
        path = root/'outputs/monthly_local_linear70/study.py'
        require(sha(path) == LOCAL_SHA, 'Frozen local67 implementation changed')
        self.local = module('forward132_local67', path)
        self.adapter = module('forward132_adapter', root/'outputs/monthly_path70_research/incumbent_adapter.py')
        self.old130 = module('forward132_old130', root/'outputs/monthly_guidance130/study.py')
        local = self.local
        # Metadata-only closure check before local.load reads any market arrays.
        dependencies = list(local.implementations()) + list(local.source_files())
        dependencies += [local.P/x for x in ('PLAN.json', 'CACHE_HASHES.json', 'INPUT_HASHES.json',
                                             'INCUMBENT_POLICY.json', 'local_linear_ORDERS.json')]
        dependencies += [local.P/x for x in read(local.P/'CACHE_HASHES.json')]
        wiring = dict(meta=local.P/'META.json', policy=local.P/'INCUMBENT_POLICY.json',
                      original_orders=local.P/'local_linear_ORDERS.json',
                      prices=root/'outputs/monthly_2016_extension/PRICES.npz')
        dependencies += list(wiring.values())
        for dep in dependencies:
            runner.bound_path(dep)
        for key, path in wiring.items():
            require(runner.inputs[key] == Path(path).resolve(), 'Incorrect original input wiring: '+key)
        local.registered()
        local.verify_source()
        self.selector = local.ep.select_boundary
        self.make_orders = local.rp.make_orders

    def load(self):
        return self.local.load()  # (meta, original arrays, closes, opens)

    def baseline(self, a, m, n, policy):
        return self.local.choices(self.local.configs()[4], a, m, n, policy=policy)

    def pre(self, a, n):
        return self.local.modified_pre_score(self.local.configs()[4], a, n)

    def price_parity(self, m, px, op, expected_raw):
        panel, info = self.local.arch.old.panel_load(m['etfs'])
        require(normalized_hashes(self.local.raw_source_hashes(info), self.root) ==
                normalized_hashes(expected_raw, self.root), 'Original raw provenance changed')
        return self.local.verify_panel_parity(panel, px, op)

    def training(self, orders, m, px):
        local, rp = self.local, self.local.rp
        blocks, _, bounds, valid = rp.paths(px, m, TRAIN)
        returns, rows = [], []
        for name in NAMES:
            choices = [[m['symbols'][i].index(s) for s in row['symbols']]
                       for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(choices, blocks, valid)
            returns.append(rr)
            rows.append(dict(name=name, **local.old.session.measure(rr, m, bounds)))
        require(np.array_equal(returns[0], returns[1]) and rows[0]['training'] == rows[1]['training'],
                'Zero control training differs')
        folds, stitched, control = [], [], []
        for n, end in ((36, 60), (60, 76), (76, 88)):
            prefix = [dict(name=name, **local.old.session.measure(rr, m, bounds, n))
                      for name, rr in zip(NAMES, returns)]
            pick = select(prefix)
            stitched.extend(returns[NAMES.index(pick['primary'])][bounds[n]:bounds[end]])
            control.extend(returns[0][bounds[n]:bounds[end]])
            folds.append(dict(n=n, end=end, selection=pick))
        return rows, dict(folds=folds, pristine_oos=False,
            metrics=rp.stats_rr(np.asarray(stitched), m['dates'][36], m['ends'][87]),
            incumbent=rp.stats_rr(np.asarray(control), m['dates'][36], m['ends'][87]))

    def diagnostics(self, orders, m, px, op, writer):
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'),
                                    ('open', .0005, 'next_open_cost')):
                met, eq, tr, hold = self.local.ex.e.engine.replay(px, op, orders[name], cost=cost,
                    execution=mode, start=pd.Timestamp(m['dates'][0]), end=pd.Timestamp(m['ends'][-1]))
                for year in met['annual_returns']:
                    if year['year'] in (2016, 2026):
                        year['partial'] = True
                results[name][key] = dict(metrics=met,
                    slices=self.local.arch.old.prior.old.previous.cuts(eq, m))
                writer(name+'_'+key+'_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
                writer(name+'_'+key+'_TRADES.json', tr)
                writer(name+'_'+key+'_HOLDINGS.json', hold)
        periods = []
        for i, day in enumerate(m['dates']):
            held = {name: orders[name][i]['symbols'] for name in NAMES}
            returns = {name: float((px.loc[m['ends'][i], ss]/px.loc[day, ss]-1).mean()) if ss else 0.
                       for name, ss in held.items()}
            periods.append(dict(date=day, end=m['ends'][i], split='early53' if i < 53 else
                'recent35' if i < TRAIN else 'later34', holdings=held, returns=returns))
        return results, periods


class Runner:
    def __init__(self, root, plan_path, plan_sha, backend_factory=Local67Backend):
        self.root = Path(root).resolve()
        self.plan_path = Path(plan_path).resolve()
        require(sha(self.plan_path) == plan_sha, 'PLAN hash mismatch')
        self.plan_sha = plan_sha
        self.plan = read(self.plan_path)
        p = self.plan
        require(p.get('round') == 132 and p.get('status') == 'REGISTERED' and
                p.get('arms') == list(NAMES) and p.get('primary_challenger') == NAMES[3], 'Not registered round132')
        require(p.get('tolerances') == TOLERANCES, 'Frozen tolerances required')
        require(p.get('runtime') == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                'Pinned runtime mismatch')
        self.pins = self.verify(p.get('hashes'))
        require(set(p.get('inputs', {})) == set(INPUTS), 'Exact named input contract required')
        self.inputs = {}
        for name, ref in p['inputs'].items():
            path = self.bound_path(ref['path'])
            require(self.pins[str(path)] == ref['sha256'], 'Input pin mismatch: '+name)
            self.inputs[name] = path
        require(p['inputs']['protected_manifest']['sha256'] == PROTECTED_SHA, 'Observer protection changed')
        for path, digest in source_files(self.root).items():
            require(self.pins.get(path) == digest, 'Implementation/runtime not PLAN-bound: '+path)
        for path, digest in self.verify(p.get('raw_source_hashes')).items():
            require(self.pins.get(path) == digest, 'Raw source not PLAN-bound')
        for path, digest in self.verify(p.get('independent_prepare_source_hashes')).items():
            require(self.pins.get(path) == digest, 'Independent prepare source not PLAN-bound')
        admission = read(self.inputs['admission'])
        require(admission.get('status') == 'PASS' and all(admission.get(k) is True for k in
                ('semantic_review_passed', 'availability_review_passed', 'calendar_review_passed')),
                'Independent source admission incomplete')
        for key in ('events', 'availability'):
            require(admission.get(key+'_sha256') == p['inputs'][key]['sha256'], 'Admission input mismatch')
        for path, digest in self.verify(admission.get('source_hashes')).items():
            require(self.pins.get(path) == digest, 'Admission evidence not PLAN-bound')
        ids = admission.get('reviewed_event_ids')
        require(isinstance(ids, list) and all(isinstance(x, str) and x for x in ids) and
                len(ids) == len(set(ids)), 'Explicit unique reviewed IDs required')
        self.reviewed = ids
        proposal = read(self.inputs['proposal'])
        require(proposal.get('prospective_round') == 132 and proposal.get('arms') == list(NAMES) and
                proposal.get('primary_challenger') == NAMES[3], 'Proposal mismatch')
        self.out = self.path(p['output_directory'])
        require(self.out == self.root/'outputs/monthly_forward_guidance132' and
                self.plan_path == self.out/'PLAN.json', 'Dedicated new output directory with PLAN required')
        self.backend_factory = backend_factory

    def path(self, path):
        path = Path(path)
        return (path if path.is_absolute() else self.root/path).resolve()

    def verify(self, mapping):
        require(isinstance(mapping, dict) and bool(mapping), 'Nonempty hash closure required')
        result = {}
        for path, digest in mapping.items():
            path = str(self.path(path))
            require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid digest')
            require(path not in result or result[path] == digest, 'Conflicting normalized pins')
            require(sha(path) == digest, 'Changed dependency: '+path)
            result[path] = digest
        return result

    def bound_path(self, path):
        path = self.path(path)
        require(self.pins.get(str(path)) == sha(path), 'Unbound dependency: '+str(path))
        return path

    def write(self, name, value):
        dump(self.out/name, value)

    def refresh(self):
        require(sha(self.plan_path) == self.plan_sha, 'PLAN changed after loading')
        self.verify(self.pins)

    def prepare(self):
        self.refresh()
        require(not any((self.out/x).exists() for x in (*CACHE_FILES, 'CACHE.json')), 'Prepare already started')
        meta = read(self.inputs['meta'])
        events = source_events(read(self.inputs['events']))
        available = read(self.inputs['availability'])
        forward, realized, cells = build_features(events, available['rows'], self.reviewed, meta)
        backend = self.backend_factory(self.root, self)
        m, arrays, px, op = backend.load()
        require(m == meta and len(arrays) == 27, 'Original META / 27 cached arrays required')
        parity = backend.price_parity(meta, px, op, self.plan['raw_source_hashes'])
        orders, scores = order_panels(arrays, meta, forward, realized, backend, read(self.inputs['policy']))
        require(orders[NAMES[0]] == read(self.inputs['original_orders']), 'Original local67 orders changed')
        with (self.out/'DATA.npz').open('xb') as f:
            np.savez_compressed(f, **arrays, forward_hand=forward, realized_hand=realized)
        with (self.out/'SCORES.npz').open('xb') as f:
            np.savez_compressed(f, **scores)
        self.write('META.json', meta)
        self.write('FEATURE_AUDIT.json', dict(hand_cells=cells, source_mask_matched=True,
            original_arrays_unchanged=True, original_months=MONTHS, original_width=WIDTH,
            new_model_fits=0, returns_or_nav_computed=False, availability_summary=available.get('summary'),
            source_limitations=read(self.inputs['proposal']).get('known_source_limitations', [])))
        self.write('PRETRAIN_ORDERS.json', orders)
        self.write('PRICE_PARITY.json', parity)
        self.write('CACHE.json', {name: sha(self.out/name) for name in CACHE_FILES})

    def cache(self):
        cache = read(self.out/'CACHE.json')
        require(set(cache) == set(CACHE_FILES), 'Exact six prepared artifacts required')
        return self.verify({str(self.out/name): digest for name, digest in cache.items()})

    def require_prepare_review(self):
        cache = self.cache()
        review = read(self.out/'INDEPENDENT_PREPARE.json')
        require(review.get('passed') is True and review.get('stage') == 'prepare' and
                all(review.get(k) is True for k in CHECKS), 'Independent prepare guarantees incomplete')
        require(all(review.get(k) is False for k in ('producer_modules_imported', 'returns_or_nav_computed')),
                'Independent prepare isolation')
        require(type(review.get('monthly_label_models_fit')) is int and
                review['monthly_label_models_fit'] == 0, 'No new fits')
        require(self.verify(review.get('source_hashes')) ==
                self.verify(self.plan['independent_prepare_source_hashes']), 'Reviewer closure differs')
        required = dict(cache)
        required.update({str(self.out/x): sha(self.out/x) for x in ('PLAN.json', 'CACHE.json')})
        artifacts = self.verify(review.get('artifact_hashes'))
        require(all(artifacts.get(k) == v for k, v in required.items()), 'Incomplete reviewed artifact closure')
        allowed = dict(self.pins, **required)
        require(all(allowed.get(k) == v for k, v in artifacts.items()), 'Unbound reviewer artifact')
        for key, file in (('plan_sha256', 'PLAN.json'), ('cache_sha256', 'CACHE.json'),
                          ('orders_sha256', 'PRETRAIN_ORDERS.json')):
            require(review.get(key) == sha(self.out/file), 'Review receipt mismatch')

    def loaded(self):
        self.cache()
        backend = self.backend_factory(self.root, self)
        m, old, px, op = backend.load()
        require(m == read(self.out/'META.json'), 'Original metadata changed')
        with np.load(self.out/'DATA.npz', allow_pickle=False) as z:
            require(set(z.files) == set(old) | {'forward_hand', 'realized_hand'}, 'Unexpected cached arrays')
            require(all(z[k].dtype == v.dtype and z[k].shape == v.shape and z[k].tobytes() == v.tobytes()
                        for k, v in old.items()), 'Original arrays changed')
        return backend, m, px, op

    def select_first88(self):
        self.refresh()
        require(not any((self.out/x).exists() for x in ('TRAIN_TRIALS.json', 'WALKFORWARD.json', 'FROZEN.json')),
                'Selection already started')
        self.require_prepare_review()  # before backend load, training paths or any returns
        backend, meta, px, _ = self.loaded()
        rows, folds = backend.training(read(self.out/'PRETRAIN_ORDERS.json'), meta, px)
        self.write('TRAIN_TRIALS.json', rows)
        self.write('WALKFORWARD.json', folds)
        self.write('FROZEN.json', dict(selection=select(rows), pristine_oos=False,
            all_diagnostic_orders_frozen=True, hashes={x: sha(self.out/x) for x in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh()
        require(not (self.out/'RESULTS.json').exists(), 'Diagnostic already completed')
        frozen = read(self.out/'FROZEN.json')
        require(set(frozen.get('hashes', {})) == set(FROZEN_FILES), 'Exact six frozen inputs required')
        self.verify({str(self.out/x): h for x, h in frozen['hashes'].items()})
        require(frozen.get('selection') == select(read(self.out/'TRAIN_TRIALS.json')) and
                frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,
                'Frozen first88 selection invalid')
        self.require_prepare_review()
        backend, meta, px, op = self.loaded()
        results, periods = backend.diagnostics(read(self.out/'PRETRAIN_ORDERS.json'), meta, px, op, self.write)
        train = read(self.out/'TRAIN_TRIALS.json')
        for i, name in enumerate(NAMES):
            require(abs(results[name]['same_close']['slices']['training2016_2023']['cagr'] -
                        train[i]['training']['cagr']) < TOLERANCES['training_cagr'], 'Training replay differs')
        cells = read(self.out/'FEATURE_AUDIT.json')['hand_cells']
        for i, row in enumerate(periods):
            row['guidance_events'] = [x for x in cells if x['signal_index'] == i]
        comparisons = {label: backend.old130.attribution(periods, results, new=new, old=old)
            for label, new, old in (('forward_vs_incumbent', NAMES[3], NAMES[0]),
                                   ('realized_vs_incumbent', NAMES[2], NAMES[0]),
                                   ('forward_vs_realized', NAMES[3], NAMES[2]))}
        self.write('RESULTS.json', results)
        self.write('MONTH_COMPARISON.json', periods)
        self.write('ATTRIBUTION.json', dict(comparisons=comparisons))
        self.write('VALIDATION.json', dict(primary=frozen['selection']['primary'], new_model_fits=0,
            original_arrays_unchanged=True, independent_outcome_review_pending=True,
            no_observer_writes=True, no_intramonth_trades=True, pristine_oos=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'select_first88', 'diagnose_later34'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    getattr(Runner(args.root, args.plan, args.plan_sha), args.phase)()


if __name__ == '__main__':
    main()
