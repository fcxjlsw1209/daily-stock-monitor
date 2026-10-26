"""Prospective round133: exact paired growth acceleration, no work on import.

Only prepare/select_first88/diagnose_later34 after root's frozen registration.
The frozen132 placement, source loading, price verification and cash mechanics
are reused; this module has three arms and only acceleration is selectable.
"""
from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_growth_acceleration133_factor as factor

require, sha, read, dump, module = base.require, base.sha, base.read, base.dump, base.module
source_events, normalized_hashes = base.source_events, base.normalized_hashes
validate_meta, corrected_choices = base.validate_meta, base.corrected_choices
TRAIN, MONTHS, WIDTH, BUDGET = base.TRAIN, base.MONTHS, base.WIDTH, base.BUDGET
CACHE_FILES, FROZEN_FILES = base.CACHE_FILES, base.FROZEN_FILES
PROTECTED_SHA, LOCAL_SHA, TOLERANCES = base.PROTECTED_SHA, base.LOCAL_SHA, dict(base.TOLERANCES)
TIME_CLARIFICATION = 'work/monthly_growth_acceleration133_time_clarification.json'
TIME_CLARIFICATION_SHA = '5a52f3cd91f6f2edf1ca68075bfb58bca1d5b611fd28baf863c639833987a387'
NAMES = ('incumbent_local67', 'matched_absolute_growth_control', 'adjacent_quarter_growth_acceleration')
CHECKS = tuple(x.replace('all_four_', 'all_three_') for x in base.CHECKS)
INPUTS = base.INPUTS + ('pair_plan','pairs','pair_summary','pair_completion')


def source_files(root):
    root = Path(root)
    result = base.source_files(root)
    paths = [Path(__file__), root/'work/test_monthly_growth_acceleration133_runner.py',
        root/'work/monthly_growth_acceleration133_factor.py',
        root/'work/test_monthly_growth_acceleration133_factor.py']
    paths += [root/p for p in factor.pairing.source_files()]
    result.update({str(Path(p).resolve()):sha(p) for p in paths})
    return result


def validate_pair_closure(runner):
    """Validate all preflight inputs and outputs before new arithmetic."""
    plan, completion = (read(runner.inputs[k]) for k in ('pair_plan','pair_completion'))
    require(completion.get('passed') is True and completion.get('stage') == 'metadata_time_feasibility',
            'Completed metadata/time preflight required')
    require(completion.get('source_hashes') == plan.get('source_hashes'), 'Preflight source maps differ')
    require(completion.get('summary') == read(runner.inputs['pair_summary']), 'Pair summary differs')
    for mapping in (plan.get('source_hashes'), completion.get('artifact_hashes')):
        for path, digest in runner.verify(mapping).items():
            require(runner.pins.get(path) == digest, 'Entire pair preflight closure must be PLAN-bound')
    artifacts = normalized_hashes(completion['artifact_hashes'], runner.root)
    for key in ('pair_plan','pairs','pair_summary'):
        require(artifacts.get(str(runner.inputs[key])) == sha(runner.inputs[key]), 'Pair completion input mismatch')

def build_features(events, rows, reviewed, meta, pair_rows):
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
    cells = factor.features_for_rows(events, rows, set(reviewed), pair_rows)
    absolute = np.full((MONTHS, WIDTH), np.nan)
    acceleration = absolute.copy()
    for cell in cells:
        i, j = cell['signal_index'], cell['pool_index']
        require((cell['absolute'] is None) == (cell['acceleration'] is None), 'Matched source mask')
        if cell['absolute'] is not None:
            absolute[i, j], acceleration[i, j] = cell['absolute'], cell['acceleration']
    require(np.array_equal(np.isfinite(absolute), np.isfinite(acceleration)), 'Matched source mask')
    return absolute, acceleration, cells

def choose(name, arrays, meta, n, absolute, acceleration, backend, policy):
    require(name in NAMES, 'Unknown arm')
    if name == NAMES[0]:
        return backend.baseline(arrays, meta, n, policy)
    hand = absolute if name == NAMES[1] else acceleration
    return corrected_choices(backend.pre(arrays, n), hand, arrays, meta, n,
        backend.adapter, backend.selector, policy)

def order_panels(arrays, meta, absolute, acceleration, backend, policy):
    orders, scores = {}, {}
    original = {k: (np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
                for k, v in arrays.items()}
    for name in NAMES:
        choices, score = choose(name, arrays, meta, MONTHS, absolute, acceleration, backend, policy)
        orders[name] = backend.make_orders(choices, score, meta)
        scores[name] = score
        for n in (0, 1, 36, 60, 76, 88):
            pc, ps = choose(name, arrays, meta, n, absolute, acceleration, backend, policy)
            require(pc == choices[:n] and ps.tobytes() == score[:n].tobytes(), 'Prefix isolation failed')
    for hand in (np.full(absolute.shape, np.nan),
                 np.where(np.isfinite(absolute), 0., np.nan)):
        for name in NAMES[1:]:
            cc, ss = choose(name, arrays, meta, MONTHS, hand, hand, backend, policy)
            require(backend.make_orders(cc, ss, meta) == orders[NAMES[0]] and
                    ss.tobytes() == scores[NAMES[0]].tobytes(), 'Missing/zero fallback failed')
    require(original == {k: (np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
                         for k, v in arrays.items()}, 'Original arrays mutated')
    return orders, scores

def select(rows):
    require([r['name'] for r in rows] == list(NAMES), 'Three fixed arms in order required')
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [r.get(k, {}).get(s) for r in (rows[2], rows[0]) for k, s in fields]
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and
                  not isinstance(v, (bool, np.bool_)) for v in values)
    good = False
    if numeric and np.isfinite(values).all():
        v, b = values[:5], values[5:]
        good = bool(v[0] > b[0] + 1e-12 and all(x >= y - t for x, y, t in
                    zip(v, b, (1e-12, .05, .03, .025, .025))))
    return dict(primary=NAMES[2] if good else NAMES[0], admitted=good, guards=good,
                guards_against={NAMES[0]: good})

class Local67Backend(base.Local67Backend):
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

class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha, backend_factory=Local67Backend):
        self.root = Path(root).resolve()
        self.plan_path = Path(plan_path).resolve()
        require(sha(self.plan_path) == plan_sha, 'PLAN hash mismatch')
        self.plan_sha = plan_sha
        self.plan = read(self.plan_path)
        p = self.plan
        require(p.get('round') == 133 and p.get('status') == 'REGISTERED' and
                p.get('arms') == list(NAMES) and p.get('primary_challenger') == NAMES[2], 'Not registered round133')
        require(p.get('tolerances') == TOLERANCES, 'Frozen tolerances required')
        require(p.get('runtime') == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                'Pinned runtime mismatch')
        self.pins = self.verify(p.get('hashes'))
        require(self.pins.get(str((self.root/TIME_CLARIFICATION).resolve())) == TIME_CLARIFICATION_SHA,
                'Fixed prospective current-publication clarification must be bound')
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
        require(proposal.get('prospective_round') == 133 and proposal.get('arms') == list(NAMES) and
                proposal.get('primary_challenger') == NAMES[2], 'Proposal mismatch')
        self.out = self.path(p['output_directory'])
        require(self.out == self.root/'outputs/monthly_growth_acceleration133' and
                self.plan_path == self.out/'PLAN.json', 'Dedicated new output directory with PLAN required')
        validate_pair_closure(self)
        self.backend_factory = backend_factory


    def prepare(self):
        self.refresh()
        require(not any((self.out/x).exists() for x in (*CACHE_FILES, 'CACHE.json')), 'Prepare already started')
        meta = read(self.inputs['meta'])
        events = source_events(read(self.inputs['events']))
        available = read(self.inputs['availability'])
        absolute, acceleration, cells = build_features(events, available['rows'], self.reviewed, meta, read(self.inputs['pairs'])['rows'])
        backend = self.backend_factory(self.root, self)
        m, arrays, px, op = backend.load()
        require(m == meta and len(arrays) == 27, 'Original META / 27 cached arrays required')
        parity = backend.price_parity(meta, px, op, self.plan['raw_source_hashes'])
        orders, scores = order_panels(arrays, meta, absolute, acceleration, backend, read(self.inputs['policy']))
        require(orders[NAMES[0]] == read(self.inputs['original_orders']), 'Original local67 orders changed')
        with (self.out/'DATA.npz').open('xb') as f:
            np.savez_compressed(f, **arrays, absolute_hand=absolute, acceleration_hand=acceleration)
        with (self.out/'SCORES.npz').open('xb') as f:
            np.savez_compressed(f, **scores)
        self.write('META.json', meta)
        self.write('FEATURE_AUDIT.json', dict(hand_cells=cells, source_mask_matched=True,
            original_arrays_unchanged=True, original_months=MONTHS, original_width=WIDTH,
            new_model_fits=0, returns_or_nav_computed=False, availability_summary=available.get('summary'),
            pair_summary=read(self.inputs['pair_summary']),
            source_limitations=read(self.inputs['proposal']).get('limitations', [])))
        self.write('PRETRAIN_ORDERS.json', orders)
        self.write('PRICE_PARITY.json', parity)
        self.write('CACHE.json', {name: sha(self.out/name) for name in CACHE_FILES})


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
            require(set(z.files) == set(old) | {'absolute_hand', 'acceleration_hand'}, 'Unexpected cached arrays')
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
            for label, new, old in (('acceleration_vs_incumbent', NAMES[2], NAMES[0]),
                                   ('absolute_vs_incumbent', NAMES[1], NAMES[0]),
                                   ('acceleration_vs_absolute', NAMES[2], NAMES[1]))}
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
