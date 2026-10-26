"""Matched absolute-return/loss Ridge target retest; no market work on import.

Real labels, forecasts, orders and returns require an independently reviewed
registration and the explicit ordered prepare/select/diagnose phases.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import hashlib
import json
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_formation142_runner as prior
import monthly_endpoint155_model as model

require, sha, read, dump, module = base.require, base.sha, base.read, base.dump, base.module
TRAIN, MONTHS, WIDTH = 88, 122, 49
PROTECTED_SHA, LOCAL_SHA = base.PROTECTED_SHA, base.LOCAL_SHA
TOLERANCES = dict(prior.TOLERANCES)
NAMES = ('incumbent_local67', 'matched_mean_return_ridge', 'matched_loss_indicator_ridge')
ARMS = ('return', 'loss')
DOSE = .01
OUT = 'outputs/monthly_endpoint155'
DESIGN_PATH = 'work/after154_endpoint_retest_design_v2.json'
DESIGN_SHA = '91d8ed23f6aa53cde1ac62728bae2b08376f2f7ecc36076d6893d1bf37f2ba63'
PREVIOUS_COMPLETION = 'outputs/monthly_financial_coverage154/COMPLETION.json'
PREVIOUS_COMPLETION_SHA = 'cbc61cf6bd4501dedb3142be5c672a1b2f196ade55f42f9a3b14df66190fb749'
SOURCE_PLAN = 'outputs/monthly_paired_target143/PLAN.json'
SOURCE_PLAN_SHA = '2d73a3afb4745f5106f6cc870d0c25a775721181cd3c99ad78a7e1f56b58500b'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'protected_manifest',
          'train_inputs', 'training_meta', 'design', 'calendar')
EXTRA = ('mean_return_prediction', 'loss_prediction', 'mean_return_rank',
         'negative_loss_rank', 'common_available')
LABELS = ('dates', 'ends', 'symbols', 'return_target', 'loss_target', 'base_good', 'good')
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json',
               'LABELS.npz', 'LABEL_AUDIT.json', 'MODEL_AUDIT.json', 'IDENTITY_AUDIT.json',
               'CALENDAR.json', 'RANK_AUDIT.json')
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
                'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
CHECKS = ('original27_arrays_byte_identical', 'training_identity_and_plain10_verified',
          'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
          'endpoint_labels_independently_reconstructed', 'shared_maturity_rows_weights_scalers_verified',
          'annual_forecasts_independently_reconstructed', 'all_three_score_matrices_and_ordered_choices_independently_verified',
          'shared_query_mask_and_centered_ranks_verified',
          'all_signal_prefixes_verified', 'initial_partial_and_missing_exact_fallback_verified',
          'one_episode_and_top2_lock_verified')


def selection_contract():
    return dict(primary=NAMES[2], controls=list(NAMES[:2]), first_months=88,
        same_close='strict_cagr_over_both_controls',
        next_open_5bp='noninferior_incumbent_strict_over_mean_return',
        cagr_epsilon=1e-12, sharpe_tolerance=.05, max_drawdown_tolerance=.03,
        early_cagr_tolerance=.025, recent_cagr_tolerance=.025,
        walk_forward=[[36, 60], [60, 76], [76, 88]], later34_diagnostic_only=True)


def measurement_contract():
    return dict(design_sha256=DESIGN_SHA, predictor='unchanged_plain10',
        label='adjusted_close_next_calendar_month_end_over_signal_close_minus_one',
        loss='endpoint_close_strictly_below_signal_close', zero_is_loss=False,
        raw_simple_return=True, clip=False, center=False, spy_relative=False,
        shared_inherited_common_good=True, min_names_per_month=2,
        annual_prior_december31=True, exclusive_five_year_signal_bound=True,
        maturity='natural_month_endpoint_close_not_next_open', min_months=18,
        alpha=10., half_life_months=12., month_balanced=True, weighted_scaler=True,
        local_residual=False, class_balancing=False, probability_calibration=False,
        rank='average_rank0_over_n_minus1_minus_half', dose=DOSE,
        max_absolute_correction=.005, query_future_outcomes_used=False,
        first_partial='2016-08-17_exact_incumbent_fallback',
        final_partial='2026-09-24_replay_only_never_training_endpoint',
        original_arrays=27, extra_arrays=list(EXTRA), label_arrays=list(LABELS),
        score_placement='unchanged_local67_pre_plus_dose_rank_then_one_episode')


def validate_proposal(proposal):
    require(proposal.get('round') == 155 and proposal.get('arms') == list(NAMES)
            and proposal.get('primary_challenger') == NAMES[2], 'Wrong endpoint proposal')
    require(proposal.get('status') == 'FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
            'Final prospective proposal required')
    require(proposal.get('measurement') == measurement_contract()
            and proposal.get('selection_contract') == selection_contract(), 'Changed endpoint contracts')
    require(proposal.get('design') == dict(path=DESIGN_PATH, sha256=DESIGN_SHA)
            and proposal.get('previous_completion') == dict(path=PREVIOUS_COMPLETION,
                    sha256=PREVIOUS_COMPLETION_SHA), 'Changed design/predecessor')
    for key in ('actual_new_labels', 'actual_new_model_fits', 'actual_new_orders', 'actual_new_returns'):
        require(type(proposal.get(key)) is int and proposal[key] == 0, 'Premature actual work')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


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
    eligible = matrix('eligible', True)
    available = matrix('common_available', True)
    values = {'return': matrix('mean_return_prediction'), 'loss': matrix('loss_prediction')}
    require(not any(np.isinf(v).any() for v in values.values()), 'Infinite endpoint prediction')
    require(not (available & ~eligible).any(), 'Forecast outside eligible query population')
    require(all(np.array_equal(np.isfinite(v), available) for v in values.values()),
            'Both forecasts finite exactly on shared available query rows')
    use = available & eligible
    ready = use.any(axis=1)
    result = {key: np.zeros((n, shape[1])) for key in ARMS}
    for i in range(n):
        count = int(use[i].sum())
        if count > 1:
            for key in ARMS:
                signed = values[key][i, use[i]] * (-1. if key == 'loss' else 1.)
                result[key][i, use[i]] = (pd.Series(signed).rank(method='average').to_numpy()-1.)/(count-1.)-.5
    for key, saved_key in (('return', 'mean_return_rank'), ('loss', 'negative_loss_rank')):
        cached = matrix(saved_key)
        require(np.isfinite(cached).all() and (np.abs(cached) <= .5).all(), 'Invalid centered rank')
        require(np.array_equal(cached, result[key]), 'Cached endpoint rank differs from shared-population reconstruction')
    return result, ready, use


def choices_all(arrays, meta, n, backend, policy):
    original, original_scores = backend.baseline(arrays, meta, n, policy)
    ranks, ready, use = ranked_pair(arrays, n)
    shape = np.asarray(arrays['baseline']).shape
    pre_original = backend.pre(arrays, n)
    require(pre_original.shape == shape, 'Local67 pre-score full shape required')
    initial = np.array([d == '2016-08-17' for d in meta['dates'][:n]], dtype=bool)
    require(not use[initial].any(), 'Initial partial query must have unavailable corrections')
    choices, scores = {NAMES[0]: original}, {NAMES[0]: original_scores}
    for name, key in zip(NAMES[1:], ARMS):
        pre = pre_original.copy()
        delta = DOSE*ranks[key]
        require(np.isfinite(delta).all() and (np.abs(delta) <= .005+1e-12).all(), 'Fixed endpoint correction bound')
        view = pre[:n].copy(); changed = delta != 0.
        view[changed] += delta[changed]; pre[:n] = view
        new, score = backend.adapter.score_and_choose('incumbent66', 0., np.full(shape, np.nan), arrays, meta, n,
            pre_score=pre, effects=arrays['effects'], episode_policy=policy, boundary_selector=backend.selector)
        for i in np.flatnonzero(~changed.any(axis=1)):
            require(new[i] == original[i] and score[i].tobytes() == original_scores[i].tobytes(),
                    'Unchanged month must preserve exact local incumbent')
        choices[name], scores[name] = new, score
    audit = [dict(signal_index=i, date=meta['dates'][i], shared_query_population=int(use[i].sum()),
                  available=bool(ready[i]), initial_partial_fallback=bool(initial[i]),
                  unchanged_return_correction=not bool(np.any(ranks['return'][i])),
                  unchanged_loss_correction=not bool(np.any(ranks['loss'][i]))) for i in range(n)]
    return choices, scores, audit


def panels(arrays, meta, backend, policy):
    snapshot = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    choices, scores, audit = choices_all(arrays, meta, MONTHS, backend, policy)
    for n in range(MONTHS+1):
        short, ss, _ = choices_all(arrays, meta, n, backend, policy)
        for name in NAMES:
            require(short[name] == choices[name][:n] and ss[name].tobytes() == scores[name][:n].tobytes(), 'All-prefix parity')
    unavailable = dict(arrays, common_available=np.zeros_like(arrays['common_available']),
        mean_return_prediction=np.full_like(arrays['mean_return_prediction'], np.nan),
        loss_prediction=np.full_like(arrays['loss_prediction'], np.nan),
        mean_return_rank=np.zeros_like(arrays['mean_return_rank']),
        negative_loss_rank=np.zeros_like(arrays['negative_loss_rank']))
    cc, sc, _ = choices_all(unavailable, meta, MONTHS, backend, policy)
    for name in NAMES[1:]:
        require(cc[name] == choices[NAMES[0]] and sc[name].tobytes() == scores[NAMES[0]].tobytes(), 'Missing whole-month fallback')
    require(snapshot == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Input mutation')
    return {name: backend.make_orders(choices[name], scores[name], meta) for name in NAMES}, scores, audit


class Backend(base.Local67Backend):
    price_parity = prior.Backend.price_parity

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
        require(calendar_record == read(self.runner.inputs['calendar']), 'Frozen143 session axis changed')
        built = model.build(px, sessions, training_meta['training'], meta, cached['train_plain'],
            cached['event_plain'], cached['common_good'], cached['train_eligible'], arrays['eligible'])
        require(set(built['arrays']) == set(EXTRA) and set(built['labels']) == set(LABELS), 'Model output schema')
        require(np.array_equal(built['labels']['ends'], np.asarray(cached['ends'], dtype='datetime64[D]')),
                'Original training calendar ends changed')
        require(not (built['labels']['good'] & ~cached['common_good']).any(), 'Endpoint cohort expands original common good')
        validate_forecasts(built['arrays'], arrays['baseline'].shape)
        return built['arrays'], built['labels'], built['label_audit'], built['model_audit'], identity, calendar_record

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


def validate_forecasts(forecast, shape):
    require(set(forecast) == set(EXTRA), 'Exactly five endpoint model outputs')
    mask = np.asarray(forecast['common_available'])
    require(mask.shape == shape and mask.dtype.kind == 'b', 'Invalid common query mask')
    for key in EXTRA[:-1]:
        a = np.asarray(forecast[key])
        require(a.shape == shape and a.dtype.kind in 'fiu', 'Invalid forecast shape/type '+key)
        require(not np.isinf(a).any(), 'Infinite endpoint forecast')
        if key.endswith('_prediction'):
            require(np.array_equal(np.isfinite(a), mask), 'Finite exactly where model available')
        else:
            require(np.isfinite(a).all() and (np.abs(a) <= .5).all()
                    and (a[~mask] == 0.).all(), 'Invalid centered query rank')


class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha, backend_factory=Backend):
        self.root = Path(root).resolve(); self.plan_path = Path(plan_path).resolve(); self.plan_sha = plan_sha
        require(sha(self.plan_path) == plan_sha, 'PLAN hash mismatch')
        self.plan = read(self.plan_path); p = self.plan
        require(p.get('round') == 155 and p.get('status') == 'REGISTERED' and p.get('arms') == list(NAMES), 'Wrong round')
        require(p.get('primary_challenger') == NAMES[2] and p.get('tolerances') == TOLERANCES, 'Selection contract mismatch')
        require(p['runtime'] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Runtime mismatch')
        self.pins = self.verify(p['hashes'])
        require(set(p['inputs']) == set(INPUTS), 'Input schema')
        self.inputs = {k: self.bound_path(v['path']) for k, v in p['inputs'].items()}
        require(all(sha(self.inputs[k]) == v['sha256'] for k, v in p['inputs'].items()), 'Input digest')
        validate_proposal(read(self.inputs['proposal']))
        require(sha(self.inputs['protected_manifest']) == PROTECTED_SHA, 'Immutable protection policy mismatch')
        require(self.inputs['design'] == self.root/DESIGN_PATH and sha(self.inputs['design']) == DESIGN_SHA,
                'Immutable endpoint design mismatch')
        require(self.inputs['calendar'] == self.root/'outputs/monthly_paired_target143/CALENDAR.json'
                and sha(self.inputs['calendar']) == CALENDAR_SHA, 'Immutable calendar mismatch')
        require(sha(self.bound_path(self.root/PREVIOUS_COMPLETION)) == PREVIOUS_COMPLETION_SHA,
                'Final154 predecessor mismatch')
        require(sha(self.bound_path(self.root/SOURCE_PLAN)) == SOURCE_PLAN_SHA, 'Frozen143 source plan mismatch')
        require(all(self.pins.get(k) == v for k, v in source_files(self.root).items()), 'Implementation not bound')
        for section in ('raw_source_hashes', 'independent_prepare_source_hashes', 'independent_outcome_source_hashes'):
            require(all(self.pins.get(k) == v for k, v in self.verify(p[section]).items()), 'Unbound '+section)
        self.backend_factory = backend_factory
        self.out = self.path(p['output_directory'])
        require(self.out == self.root/OUT and self.plan_path == self.out/'PLAN.json', 'Output path')

    def prepare(self):
        self.refresh(); require(not any((self.out/x).exists() for x in (*CACHE_FILES, 'CACHE.json')), 'Prepare already started')
        backend = self.backend_factory(self.root, self); meta, arrays, px, op = backend.load()
        require(len(arrays) == 27 and meta == read(self.inputs['meta']), 'Original metadata/27 arrays')
        base.validate_meta(meta)
        original_signature = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
        parity = backend.price_parity(meta, px, op, self.plan['raw_source_hashes'])
        extra, target_arrays, label_audit, model_audit, identity, calendar = backend.build(meta, arrays, px, op)
        require(original_signature == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()},
                'Original arrays changed while building endpoint forecasts')
        require(set(extra) == set(EXTRA), 'New array schema')
        arrays = dict(arrays, **extra)
        orders, scores, audit = panels(arrays, meta, backend, read(self.inputs['policy']))
        require(orders[NAMES[0]] == read(self.inputs['original_orders']), 'Baseline changed')
        require(set(target_arrays) == set(LABELS), 'New label schema')
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
            row['endpoint_target'] = months[i]
        comparisons = {label: backend.old130.attribution(periods, results, new=new, old=old) for label, new, old in
            (('loss_vs_incumbent', NAMES[2], NAMES[0]), ('return_vs_incumbent', NAMES[1], NAMES[0]), ('loss_vs_return', NAMES[2], NAMES[1]))}
        by_date = {row['date']: row['endpoint_target'] for row in periods}
        for comparison in comparisons.values():
            for change in comparison['changes']: change['endpoint_target'] = by_date[change['date']]
        self.write('RESULTS.json', results); self.write('MONTH_COMPARISON.json', periods); self.write('ATTRIBUTION.json', dict(comparisons=comparisons))
        fits = read(self.out/'MODEL_AUDIT.json')['new_pooled_model_fits']
        require(type(fits) is int and fits >= 0 and fits % 2 == 0, 'Shared paired model fit count')
        self.write('VALIDATION.json', dict(primary=frozen['selection']['primary'], selection=frozen['selection'],
            original_arrays_unchanged=True, independent_outcome_review_pending=True, no_observer_writes=True,
            no_intramonth_trades=True, pristine_oos=False, model_audit_sha256=sha(self.out/'MODEL_AUDIT.json'),
            initial_partial_query='2016-08-17_incumbent_fallback',
            final_partial_evaluation='2026-09-24_not_a_training_label_endpoint',
            inherited_common_good_restrictions=True, linear_probability_not_calibrated=True,
            historical_novelty_not_certified=True, new_model_fits=fits))


def source_files(root):
    root = Path(root)
    paths = [Path(__file__), Path(prior.__file__), root/'work/test_monthly_endpoint155_runner.py',
             root/'work/test_monthly_endpoint155_model.py',
             root/'work/prepare_monthly_endpoint155_proposal.py',
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
