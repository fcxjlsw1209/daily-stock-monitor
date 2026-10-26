"""Three prospectively fixed target arms; no market work on import.

Real labels, forecasts, orders and returns require an independently reviewed
registration and the explicit ordered prepare/select/diagnose phases.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_formation142_runner as prior
import after142_paired_target_labels as labels
import monthly_paired_target143_model as model

require, sha, read, dump, module = base.require, base.sha, base.read, base.dump, base.module
TRAIN, MONTHS, WIDTH = 88, 122, 49
PROTECTED_SHA, LOCAL_SHA = base.PROTECTED_SHA, base.LOCAL_SHA
TOLERANCES = dict(prior.TOLERANCES)
NAMES = ('incumbent_local67', 'paired_close_target', 'paired_open_target')
ARMS = ('close', 'open')
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'protected_manifest', 'train_inputs', 'training_meta')
EXTRA = ('paired_raw_close', 'paired_raw_open', 'paired_pooled_close', 'paired_pooled_open',
         'paired_available', 'paired_local_available')
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json',
               'LABELS.npz', 'LABEL_AUDIT.json', 'MODEL_AUDIT.json', 'IDENTITY_AUDIT.json',
               'CALENDAR.json', 'RANK_AUDIT.json')
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
                'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
CHECKS = ('original27_arrays_byte_identical', 'training_identity_and_plain10_verified',
          'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
          'paired_labels_independently_reconstructed', 'shared_maturity_and_cohort_verified',
          'annual_forecasts_independently_reconstructed', 'all_three_score_matrices_and_ordered_choices_independently_verified',
          'all_signal_prefixes_verified', 'whole_month_incumbent_fallback_verified',
          'one_episode_and_top2_lock_verified')
PROPOSAL_SHA = 'd1b93c8ae18e11998e011a3b91a76af761d79be0ada8f3435258b3365cfe5256'


def guard(candidate, control, strict=True):
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [row.get(k, {}).get(s) for row in (candidate, control) for k, s in fields]
    if not all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in values):
        return False
    if not np.isfinite(values).all():
        return False
    v, b = values[:5], values[5:]
    first = v[0] > b[0]+1e-12 if strict else v[0] >= b[0]-1e-12
    return bool(first and all(x >= y-t for x, y, t in zip(v[1:], b[1:], (.05, .03, .025, .025))))


def select(rows, open_rows):
    for group in (rows, open_rows):
        require([r['name'] for r in group] == list(NAMES), 'Three fixed arms in order required')
    close_guards = {NAMES[i]: guard(rows[2], rows[i]) for i in (0, 1)}
    open_guards = {NAMES[i]: guard(open_rows[2], open_rows[i], strict=(i == 1)) for i in (0, 1)}
    passed = all(close_guards.values()) and all(open_guards.values())
    return dict(primary=NAMES[2] if passed else NAMES[0], admitted=passed, guards=passed,
                guards_against=close_guards, next_open_guards_against=open_guards,
                mechanism_confirmed=all(open_guards.values()))


def ranked_pair(arrays, n):
    shape = np.asarray(arrays['baseline']).shape
    require(len(shape) == 2 and type(n) is int and 0 <= n <= shape[0], 'Valid monthly prefix')
    def matrix(key, boolean=False):
        raw = np.asarray(arrays[key])
        require(raw.shape == shape, 'Invalid shape '+key)
        if boolean:
            require(raw.dtype.kind == 'b', 'Invalid boolean '+key)
            return raw[:n].copy()
        prefix = raw[:n]
        require(all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in prefix.flat), 'Invalid numeric '+key)
        return prefix.astype(float, copy=True)
    old = matrix('rank_log_excess')
    use = matrix('forecast_available', True) & matrix('eligible', True)
    available = matrix('paired_available', True)
    values = {key: matrix('paired_raw_'+key) for key in ARMS}
    require(np.isfinite(old).all() and (np.abs(old) <= .5+1e-12).all(), 'Invalid original log ranks')
    require(not any(np.isinf(v).any() for v in values.values()), 'Infinite paired prediction')
    ready = np.zeros(n, bool)
    result = {key: old.copy() for key in ARMS}
    for i in range(n):
        ready[i] = bool(use[i].any() and available[i, use[i]].all() and
                        all(np.isfinite(v[i, use[i]]).all() for v in values.values()))
        if ready[i]:
            for key in ARMS:
                rank = pd.Series(values[key][i]).where(use[i]).rank(method='average', pct=True).fillna(.5).to_numpy()-.5
                result[key][i, use[i]] = rank[use[i]]
    return result, ready, use


def choices_all(arrays, meta, n, backend, policy):
    original, original_scores = backend.baseline(arrays, meta, n, policy)
    ranks, ready, use = ranked_pair(arrays, n)
    shape = np.asarray(arrays['baseline']).shape
    scope = backend.scope_pre(arrays, n)
    require(scope.shape == shape, 'Scope pre-score full shape required')
    original_rank = np.asarray(arrays['rank_log_excess'])[:n]
    weak = backend.adapter._state(arrays, 'state_weak_narrow', shape[0], n)
    calm = backend.adapter._state(arrays, 'state_calm10', shape[0], n)
    ext = backend.adapter._matrix(arrays['extension_available'], shape, n, 'extension_available', boolean=True)
    g = .25*(.05*weak[:, None]+.01*(calm & ~weak)[:, None]*ext)
    choices, scores = {NAMES[0]: original}, {NAMES[0]: original_scores}
    for name, key in zip(NAMES[1:], ARMS):
        pre = scope.copy()
        delta = np.zeros((n, shape[1]))
        active = use & ready[:, None]
        delta[active] = (g*(ranks[key]-original_rank))[active]
        require(np.isfinite(delta).all() and (np.abs(delta) <= .0125+1e-12).all(), 'Existing slot bound')
        view = pre[:n].copy(); changed = delta != 0.
        view[changed] += delta[changed]; pre[:n] = view
        new, score = backend.adapter.score_and_choose('incumbent66', 0., np.full(shape, np.nan), arrays, meta, n,
            pre_score=pre, effects=arrays['effects'], episode_policy=policy, boundary_selector=backend.selector)
        # Incomplete paired rows must preserve the LOCAL incumbent, not scope67.
        for i in np.flatnonzero(~ready):
            new[i] = list(original[i]); score[i] = original_scores[i]
        choices[name], scores[name] = new, score
    audit = [dict(signal_index=i, date=meta['dates'][i], original_population=int(use[i].sum()),
                  paired_ready=bool(ready[i]), fallback_to_local_incumbent=not bool(ready[i])) for i in range(n)]
    return choices, scores, audit


def panels(arrays, meta, backend, policy):
    snapshot = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    choices, scores, audit = choices_all(arrays, meta, MONTHS, backend, policy)
    for n in range(MONTHS+1):
        short, ss, _ = choices_all(arrays, meta, n, backend, policy)
        for name in NAMES:
            require(short[name] == choices[name][:n] and ss[name].tobytes() == scores[name][:n].tobytes(), 'All-prefix parity')
    unavailable = dict(arrays, paired_available=np.zeros_like(arrays['paired_available']))
    cc, sc, _ = choices_all(unavailable, meta, MONTHS, backend, policy)
    for name in NAMES[1:]:
        require(cc[name] == choices[NAMES[0]] and sc[name].tobytes() == scores[NAMES[0]].tobytes(), 'Missing whole-month fallback')
    require(snapshot == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Input mutation')
    return {name: backend.make_orders(choices[name], scores[name], meta) for name in NAMES}, scores, audit


class Backend(base.Local67Backend):
    price_parity = prior.Backend.price_parity

    def scope_pre(self, arrays, n):
        return self.local.scope.modified_pre_score(self.local.SOURCE_SCOPE_SPEC, arrays, n)

    def build(self, meta, arrays, px, op):
        local, target = self.local, self.local.target
        require(self.runner.inputs['train_inputs'] == target.P/'TRAIN_INPUTS.npz', 'Training input wiring')
        require(self.runner.inputs['training_meta'] == target.old.P/'META.json', 'Training metadata wiring')
        with np.load(self.runner.inputs['train_inputs'], allow_pickle=False) as store:
            cached = {key: store[key].copy() for key in store.files}
        with np.load(self.runner.bound_path(target.old.P/'DATA.npz'), allow_pickle=False) as store:
            original = {key: store[key].copy() for key in target.CACHED_KEYS}
        training_meta = read(self.runner.inputs['training_meta'])
        identity = local.validate_identity(training_meta, cached, original, read(self.runner.bound_path(local.arch.P/'META.json')))
        target.validate_axes(meta, training_meta, cached, arrays['eligible'])
        require(cached['train_plain'].shape == (8820, 10) and cached['event_plain'].shape == (5978, 10), 'Original plain10 axes')
        provenance = local.calendar_provenance()
        require(provenance == self.runner.plan['calendar'], 'Original official calendar provenance')
        calendar = local.calendar_labels._CAL.market_calendar(start='2008-01-01', end='2035-12-31')
        sessions = calendar.sessions
        require(sessions.tz is None and sessions.is_unique and sessions.is_monotonic_increasing, 'Official naive complete calendar')
        calendar_record = dict(start='2008-01-01', end='2035-12-31', provenance=provenance,
                               sessions=[str(d.date()) for d in sessions])
        symbols = np.array([s for row in training_meta['training']['symbols'] for s in row])
        paired = labels.build_paired_targets(px, op, calendar, cached['dates'], symbols,
            cached['common_good'], cached['train_eligible'], np.isfinite(cached['train_plain']).all(axis=1))
        require(np.array_equal(paired['calendar_ends'], np.asarray(cached['ends'], dtype='datetime64[D]')),
                'Original training calendar ends changed')
        require(not (paired['good'] & ~cached['common_good']).any(), 'Paired cohort expands original common good')
        # Same unmodified 10 features and source query population in both arms.
        event_symbols = np.array([s for row in meta['symbols'] for s in row])
        event_good = arrays['eligible'].reshape(-1) & np.isfinite(cached['event_plain']).all(axis=1)
        forecasts = model.annual_forecasts(paired['dates'], paired['calendar_ends'], paired['available_sessions'], symbols,
            cached['train_plain'], {key: paired['targets']['paired_'+key] for key in ARMS}, paired['good'],
            cached['event_dates'], event_symbols, cached['event_plain'], event_good)
        validate_forecasts(forecasts, len(event_symbols))
        shape = arrays['baseline'].shape
        extra = {'paired_raw_'+key: forecasts['raw'][key].reshape(shape) for key in ARMS}
        extra.update({'paired_pooled_'+key: forecasts['pooled'][key].reshape(shape) for key in ARMS})
        extra.update(paired_available=forecasts['available'].reshape(shape), paired_local_available=forecasts['local_available'].reshape(shape))
        label_arrays = {key: paired[key] for key in ('dates', 'calendar_ends', 'entry_sessions', 'exit_sessions', 'available_sessions', 'symbols', 'good', 'base_good', 'endpoint_good')}
        label_arrays.update({'target_'+key: paired['targets']['paired_'+key] for key in ARMS})
        return extra, label_arrays, paired['audit'], forecasts['audit'], identity, calendar_record

    def __init__(self, root, runner):
        super().__init__(root, runner)
        self.runner = runner

    def open_training(self, orders, meta, px, op, n):
        split = 53 if n >= 88 else n//2
        rows = []
        for name in NAMES:
            _, eq, _, _ = self.local.ex.e.engine.replay(px, op, orders[name][:n], cost=.0005,
                execution='open', start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][n-1]))
            intervals = (('training', meta['dates'][0], meta['ends'][n-1]),
                         ('early', meta['dates'][0], meta['ends'][split-1]),
                         ('recent', meta['dates'][split], meta['ends'][n-1]))
            row = dict(name=name)
            for key, lo, hi in intervals:
                part = eq.loc[lo:hi]
                require(len(part) >= 2, 'Training open interval too short')
                row[key] = self.local.rp.stats_rr(part.to_numpy()[1:]/part.to_numpy()[:-1]-1, lo, hi)
            rows.append(row)
        return rows

    def training(self, orders, meta, px, op):
        rp = self.local.rp
        blocks, _, bounds, valid = rp.paths(px, meta, TRAIN)
        returns, rows = [], []
        for name in NAMES:
            indices = [[meta['symbols'][i].index(s) for s in row['symbols']] for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(indices, blocks, valid); returns.append(rr)
            rows.append(dict(name=name, **self.local.old.session.measure(rr, meta, bounds)))
        open_rows = self.open_training(orders, meta, px, op, TRAIN)
        folds, stitched, control = [], [], []
        for n, end in ((36, 60), (60, 76), (76, 88)):
            prefix = [dict(name=name, **self.local.old.session.measure(rr, meta, bounds, n)) for name, rr in zip(NAMES, returns)]
            oprefix = self.open_training(orders, meta, px, op, n)
            selection = select(prefix, oprefix)
            stitched.extend(returns[NAMES.index(selection['primary'])][bounds[n]:bounds[end]])
            control.extend(returns[0][bounds[n]:bounds[end]])
            folds.append(dict(n=n, end=end, selection=selection, open_training=oprefix))
        return rows, open_rows, dict(folds=folds, pristine_oos=False,
            metrics=rp.stats_rr(np.asarray(stitched), meta['dates'][36], meta['ends'][87]),
            incumbent=rp.stats_rr(np.asarray(control), meta['dates'][36], meta['ends'][87]))

    def diagnostics(self, orders, meta, px, op, writer):
        # Reuse the established identical accounting implementation without
        # changing its module-level names or frozen bytes.
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'), ('open', .0005, 'next_open_cost')):
                met, eq, trades, holds = self.local.ex.e.engine.replay(px, op, orders[name], cost=cost,
                    execution=mode, start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][-1]))
                for year in met['annual_returns']:
                    if year['year'] in (2016, 2026): year['partial'] = True
                results[name][key] = dict(metrics=met, slices=self.local.arch.old.prior.old.previous.cuts(eq, meta))
                writer(name+'_'+key+'_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
                writer(name+'_'+key+'_TRADES.json', trades); writer(name+'_'+key+'_HOLDINGS.json', holds)
        periods = []
        for i, day in enumerate(meta['dates']):
            held = {name: orders[name][i]['symbols'] for name in NAMES}
            rr = {name: float((px.loc[meta['ends'][i], ss]/px.loc[day, ss]-1).mean()) if ss else 0. for name, ss in held.items()}
            periods.append(dict(date=day, end=meta['ends'][i], split='early53' if i < 53 else 'recent35' if i < TRAIN else 'later34', holdings=held, returns=rr))
        return results, periods


def validate_forecasts(forecast, size):
    require(set(forecast['raw']) == set(ARMS) and set(forecast['pooled']) == set(ARMS), 'Exactly paired model outputs')
    for key in ('available', 'local_available'):
        a = np.asarray(forecast[key]); require(a.shape == (size,) and a.dtype.kind == 'b', 'Invalid '+key)
    require(not (forecast['local_available'] & ~forecast['available']).any(), 'Local availability expansion')
    for group in ('raw', 'pooled'):
        for key in ARMS:
            a = np.asarray(forecast[group][key]); require(a.shape == (size,) and a.dtype.kind in 'fiu', 'Invalid forecast shape')
            require(not np.isinf(a).any() and np.array_equal(np.isfinite(a), forecast['available']), 'Finite exactly where model available')
    cold = forecast['available'] & ~forecast['local_available']
    for key in ARMS:
        require(np.array_equal(forecast['raw'][key][cold], forecast['pooled'][key][cold]), 'Cold own history must use pooled forecast')


class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha, backend_factory=Backend):
        self.root = Path(root).resolve(); self.plan_path = Path(plan_path).resolve(); self.plan_sha = plan_sha
        require(sha(self.plan_path) == plan_sha, 'PLAN hash mismatch')
        self.plan = read(self.plan_path); p = self.plan
        require(p.get('round') == 143 and p.get('status') == 'REGISTERED' and p.get('arms') == list(NAMES), 'Wrong round')
        require(p.get('primary_challenger') == NAMES[2] and p.get('tolerances') == TOLERANCES, 'Selection contract mismatch')
        require(p['runtime'] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Runtime mismatch')
        self.pins = self.verify(p['hashes'])
        require(set(p['inputs']) == set(INPUTS), 'Input schema')
        self.inputs = {k: self.bound_path(v['path']) for k, v in p['inputs'].items()}
        require(all(sha(self.inputs[k]) == v['sha256'] for k, v in p['inputs'].items()), 'Input digest')
        require(sha(self.inputs['proposal']) == PROPOSAL_SHA and sha(self.inputs['protected_manifest']) == PROTECTED_SHA, 'Immutable policy mismatch')
        require(all(self.pins.get(k) == v for k, v in source_files(self.root).items()), 'Implementation not bound')
        for section in ('raw_source_hashes', 'independent_prepare_source_hashes', 'independent_outcome_source_hashes'):
            require(all(self.pins.get(k) == v for k, v in self.verify(p[section]).items()), 'Unbound '+section)
        self.backend_factory = backend_factory
        self.out = self.path(p['output_directory'])
        require(self.out == self.root/'outputs/monthly_paired_target143' and self.plan_path == self.out/'PLAN.json', 'Output path')

    def prepare(self):
        self.refresh(); require(not any((self.out/x).exists() for x in (*CACHE_FILES, 'CACHE.json')), 'Prepare already started')
        backend = self.backend_factory(self.root, self); meta, arrays, px, op = backend.load()
        require(len(arrays) == 27 and meta == read(self.inputs['meta']), 'Original metadata/27 arrays')
        base.validate_meta(meta)
        parity = backend.price_parity(meta, px, op, self.plan['raw_source_hashes'])
        extra, target_arrays, label_audit, model_audit, identity, calendar = backend.build(meta, arrays, px, op)
        require(set(extra) == set(EXTRA), 'New array schema')
        arrays = dict(arrays, **extra)
        orders, scores, audit = panels(arrays, meta, backend, read(self.inputs['policy']))
        require(orders[NAMES[0]] == read(self.inputs['original_orders']), 'Baseline changed')
        for name, values in (('DATA.npz', arrays), ('LABELS.npz', target_arrays), ('SCORES.npz', scores)):
            with (self.out/name).open('xb') as stream: np.savez_compressed(stream, **values)
        for name, value in (('META.json', meta), ('PRETRAIN_ORDERS.json', orders), ('PRICE_PARITY.json', parity),
            ('LABEL_AUDIT.json', label_audit), ('MODEL_AUDIT.json', model_audit), ('IDENTITY_AUDIT.json', identity),
            ('CALENDAR.json', calendar), ('RANK_AUDIT.json', dict(months=audit))): self.write(name, value)
        self.write('CACHE.json', {name: sha(self.out/name) for name in CACHE_FILES})

    def cache(self):
        data = read(self.out/'CACHE.json'); require(set(data) == set(CACHE_FILES), 'Cache schema')
        return self.verify({str(self.out/key): value for key, value in data.items()})

    def require_prepare_review(self):
        cache = self.cache(); review = read(self.out/'INDEPENDENT_PREPARE.json')
        require(review.get('passed') is True and review.get('stage') == 'prepare' and all(review.get(k) is True for k in CHECKS), 'Prepare review incomplete')
        require(review.get('producer_modules_imported') is False and review.get('returns_or_nav_computed') is False, 'Independent isolation')
        require(self.verify(review['source_hashes']) == self.verify(self.plan['independent_prepare_source_hashes']), 'Review code mismatch')
        required = dict(cache, **{str(self.out/x): sha(self.out/x) for x in ('PLAN.json', 'CACHE.json')})
        got = self.verify(review['artifact_hashes']); allowed = dict(self.pins, **required)
        require(all(got.get(k) == v for k, v in required.items()) and all(allowed.get(k) == v for k, v in got.items()), 'Review artifact closure')
        for key, file in (('plan_sha256', 'PLAN.json'), ('cache_sha256', 'CACHE.json'), ('orders_sha256', 'PRETRAIN_ORDERS.json')):
            require(review.get(key) == sha(self.out/file), 'Review receipt binding')

    def loaded(self):
        self.cache(); backend = self.backend_factory(self.root, self); meta, old, px, op = backend.load()
        require(meta == read(self.out/'META.json'), 'Metadata changed')
        with np.load(self.out/'DATA.npz', allow_pickle=False) as store:
            require(set(store.files) == set(old) | set(EXTRA), 'Array schema')
            require(all(store[k].dtype == v.dtype and store[k].shape == v.shape and store[k].tobytes() == v.tobytes() for k, v in old.items()), 'Original arrays changed')
        return backend, meta, px, op

    def select_first88(self):
        self.refresh(); require(not any((self.out/x).exists() for x in ('TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'FROZEN.json')), 'Already selected')
        self.require_prepare_review(); backend, meta, px, op = self.loaded()
        rows, open_rows, folds = backend.training(read(self.out/'PRETRAIN_ORDERS.json'), meta, px, op)
        self.write('TRAIN_TRIALS.json', rows); self.write('TRAIN_OPEN_TRIALS.json', open_rows); self.write('WALKFORWARD.json', folds)
        self.write('FROZEN.json', dict(selection=select(rows, open_rows), pristine_oos=False, all_diagnostic_orders_frozen=True,
            hashes={x: sha(self.out/x) for x in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh()
        outputs = ['DIAGNOSE_STARTED.json', 'RESULTS.json', 'MONTH_COMPARISON.json', 'ATTRIBUTION.json', 'VALIDATION.json']
        outputs.extend(name+'_'+mode+'_'+kind+'.json' for name in NAMES for mode in ('same_close', 'next_open_zero', 'next_open_cost') for kind in ('DAILY', 'TRADES', 'HOLDINGS'))
        require(not any((self.out/x).exists() for x in outputs), 'Diagnostic stage already started; preserve partial outputs')
        frozen = read(self.out/'FROZEN.json'); require(set(frozen['hashes']) == set(FROZEN_FILES), 'Frozen schema')
        self.verify({str(self.out/k): v for k, v in frozen['hashes'].items()})
        require(frozen['selection'] == select(read(self.out/'TRAIN_TRIALS.json'), read(self.out/'TRAIN_OPEN_TRIALS.json')) and frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True, 'Selection changed')
        self.require_prepare_review(); backend, meta, px, op = self.loaded()
        self.write('DIAGNOSE_STARTED.json', dict(plan_sha256=self.plan_sha, frozen_sha256=sha(self.out/'FROZEN.json'), started_utc=datetime.now(timezone.utc).isoformat()))
        results, periods = backend.diagnostics(read(self.out/'PRETRAIN_ORDERS.json'), meta, px, op, self.write)
        for filename, mode in (('TRAIN_TRIALS.json', 'same_close'), ('TRAIN_OPEN_TRIALS.json', 'next_open_cost')):
            for row in read(self.out/filename):
                require(abs(results[row['name']][mode]['slices']['training2016_2023']['cagr']-row['training']['cagr']) < TOLERANCES['training_cagr'], 'Replay training parity')
        months = read(self.out/'RANK_AUDIT.json')['months']; require(len(months) == MONTHS and len(periods) == MONTHS, 'All122 rows')
        for i, row in enumerate(periods):
            require(months[i]['signal_index'] == i and months[i]['date'] == row['date'], 'Rank/month axes')
            row['paired_target'] = months[i]
        comparisons = {label: backend.old130.attribution(periods, results, new=new, old=old) for label, new, old in
            (('open_vs_incumbent', NAMES[2], NAMES[0]), ('close_vs_incumbent', NAMES[1], NAMES[0]), ('open_vs_close', NAMES[2], NAMES[1]))}
        by_date = {row['date']: row['paired_target'] for row in periods}
        for comparison in comparisons.values():
            for change in comparison['changes']: change['paired_target'] = by_date[change['date']]
        self.write('RESULTS.json', results); self.write('MONTH_COMPARISON.json', periods); self.write('ATTRIBUTION.json', dict(comparisons=comparisons))
        self.write('VALIDATION.json', dict(primary=frozen['selection']['primary'], selection=frozen['selection'],
            original_arrays_unchanged=True, independent_outcome_review_pending=True, no_observer_writes=True,
            no_intramonth_trades=True, pristine_oos=False, model_audit_sha256=sha(self.out/'MODEL_AUDIT.json')))


def source_files(root):
    root = Path(root)
    paths = [Path(__file__), Path(prior.__file__), root/'work/test_monthly_paired_target143_runner.py',
             root/'work/test_monthly_paired_target143_model.py', root/'work/after142_paired_target_labels_tests.py',
             *model.dependency_files()]
    return dict(base.source_files(root), **{str(p.resolve()): sha(p) for p in paths})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'select_first88', 'diagnose_later34'))
    parser.add_argument('--plan', required=True); parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    getattr(Runner(args.root, args.plan, args.plan_sha), args.phase)()

if __name__ == '__main__': main()
