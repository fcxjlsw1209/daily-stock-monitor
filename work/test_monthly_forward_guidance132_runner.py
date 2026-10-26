"""Synthetic integration only: no real source, prices, ranks, or returns loaded."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

import numpy as np
import monthly_forward_guidance132_runner as r
import register_monthly_forward_guidance132 as registry
from test_monthly_forward_guidance_contract_v1 import fixture

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = r.module('test_forward132_adapter', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
POLICY = dict(budget=.05, ceiling=5, gap=.02)


def synthetic():
    event = fixture()
    event['forecast']['amount'].update(kind='closed_range', low='110', high='130',
        point=None, center=None, spread=None, unit_scale='1000000')
    for name, value in (('target_prior_year', '100'), ('current_quarter', '90'), ('current_prior_year', '100')):
        event['actuals'][name]['amount'].update(kind='point', point=value, low=None,
            high=None, center=None, spread=None, unit_scale='1000000')
    dates = [f'{2016+(i+7)//12:04d}-{(i+7)%12+1:02d}-01' for i in range(122)]
    etfs = ['ETF'+str(i) for i in range(17)]
    names = [event['symbol']] + ['EQ'+str(i) for i in range(31)] + etfs
    meta = dict(dates=dates, ends=[x[:-2]+'02' for x in dates], train_months=88,
        etfs=etfs, symbols=[names[:] for _ in dates])
    pre = np.tile(np.linspace(.12, .001, 49), (122, 1))
    arrays = dict(baseline=pre, core=np.zeros(pre.shape), core_weight=np.full(122, .5),
        eligible=np.ones(pre.shape, bool), effects=np.zeros(pre.shape),
        state_calm10=np.zeros(122, bool), state_weak_narrow=np.zeros(122, bool))
    arrays.update({'cached'+str(i): np.full(pre.shape, float(i)) for i in range(20)})
    rows = [dict(signal_index=i, pool_index=j, signal_date=day, symbol=symbol,
        event_id=event['event_id'] if j == 0 else None, comparison_available=j == 0,
        is_etf=symbol in etfs) for i, day in enumerate(dates) for j, symbol in enumerate(names)]
    return event, meta, arrays, rows


def trial_rows():
    return [dict(name=n, training=dict(cagr=.2, sharpe=1., max_drawdown=-.2),
                 early=dict(cagr=.2), recent=dict(cagr=.2)) for n in r.NAMES]


class Backend:
    adapter = ADAPTER
    def __init__(self, meta, arrays):
        self.meta, self.arrays = meta, arrays
        self.loads = self.training_calls = self.diagnostic_calls = self.episodes = 0
        self.old130 = SimpleNamespace(attribution=lambda periods, results, new, old:
            dict(new=new, old=old, months=len(periods)))

    def load(self):
        self.loads += 1
        return self.meta, self.arrays, None, None

    def pre(self, arrays, n):
        out = np.zeros(arrays['baseline'].shape)
        out[:n] = arrays['baseline'][:n]
        return out

    def selector(self, scores, good, symbols, effects, original, policy):
        self.episodes += 1
        self.last_pre = scores.copy()
        self.last_original = deepcopy(original)
        assert policy == POLICY
        choices = []
        for row, valid, names in zip(scores, good, symbols):
            order = sorted(np.flatnonzero(valid).tolist(), key=lambda j: (-row[j], names[j]))
            choices.append(order[:3] if len(order) >= 2 else [])
        return choices, scores.copy()

    def baseline(self, arrays, meta, n, policy):
        return self.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
            arrays, meta, n, pre_score=self.pre(arrays, n), effects=arrays['effects'],
            episode_policy=policy, boundary_selector=self.selector)

    @staticmethod
    def make_orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=[meta['symbols'][i][j] for j in row],
            scores=[float(scores[i, j]) for j in row]) for i, row in enumerate(choices)]

    def price_parity(self, *args):
        return dict(synthetic=True)

    def training(self, orders, meta, px):
        self.training_calls += 1
        self.training_length = len(orders[r.NAMES[0]][:r.TRAIN])
        rows = trial_rows()
        rows[3]['training']['cagr'] = .21
        return rows, dict(folds=[dict(n=n, end=end, selection=r.select(rows))
            for n, end in ((36, 60), (60, 76), (76, 88))], pristine_oos=False)

    def diagnostics(self, orders, meta, px, op, writer):
        self.diagnostic_calls += 1
        results = {name: {key: dict(metrics={}, slices={'training2016_2023': {'cagr': .21 if i == 3 else .2}})
            for key in ('same_close', 'next_open_zero', 'next_open_cost')} for i, name in enumerate(r.NAMES)}
        periods = [dict(date=day, split='early53' if i < 53 else 'recent35' if i < 88 else 'later34')
                   for i, day in enumerate(meta['dates'])]
        return results, periods


class FeatureTests(unittest.TestCase):
    def test_list_and_explicit_wrapper_inputs(self):
        events = [fixture()]
        self.assertEqual(r.source_events(events), events)
        self.assertEqual(r.source_events({'events':events}), events)
        for value in ({}, None, ['bad']):
            with self.subTest(value=value), self.assertRaises(ValueError): r.source_events(value)

    def test_exact_matched_values_and_full_pool(self):
        e, m, a, rows = synthetic()
        f, g, cells = r.build_features([e], rows, [e['event_id']], m)
        self.assertEqual(len(cells), 5978)
        self.assertAlmostEqual(f[0, 0], 1/22)
        self.assertAlmostEqual(g[0, 0], -1/38)
        self.assertTrue(np.array_equal(np.isfinite(f), np.isfinite(g)))
        self.assertTrue(np.isnan(f[:, 1:]).all())

    def test_no_review_and_missing_remain_missing(self):
        e, m, a, rows = synthetic()
        f, g, _ = r.build_features([e], rows, [], m)
        self.assertTrue(np.isnan(f).all() and np.isnan(g).all())
        e['actuals']['target_prior_year'] = None
        f, g, _ = r.build_features([e], rows, [e['event_id']], m)
        self.assertTrue(np.isnan(f).all() and np.isnan(g).all())

    def test_axis_and_fund_errors(self):
        e, m, a, rows = synthetic()
        for change in ('order', 'missing', 'fund'):
            bad = deepcopy(rows)
            if change == 'order': bad[0], bad[1] = bad[1], bad[0]
            if change == 'missing': bad.pop()
            if change == 'fund': bad[0]['is_etf'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                r.build_features([e], bad, [e['event_id']], m)

    def test_barrier_row_does_not_recover_old_value(self):
        e, m, a, rows = synthetic()
        rows[0]['comparison_available'] = False
        f, g, _ = r.build_features([e], rows, [e['event_id']], m)
        self.assertTrue(np.isnan(f[0, 0]) and np.isnan(g[0, 0]))


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.e, self.m, self.a, _ = synthetic()
        self.backend = Backend(self.m, self.a)
        self.h = np.full((122, 49), np.nan)

    def call(self, hand=None, n=1):
        return r.corrected_choices(self.backend.pre(self.a, n), self.h if hand is None else hand,
            self.a, self.m, n, ADAPTER, self.backend.selector, POLICY)

    def test_continuous_formula_once_and_original_support(self):
        self.h[0, 0] = .123
        before = self.a['baseline'][0, 0]
        self.call()
        self.assertEqual(self.backend.episodes, 1)
        self.assertAlmostEqual(self.backend.last_pre[0, 0]-before, .025*.123)
        self.assertEqual(self.backend.last_original['baseline'][0, 0], before)

    def test_clip_and_delta_bound(self):
        self.a['baseline'][0, 0] = .24
        self.h[0, 0] = .49
        self.call()
        self.assertAlmostEqual(self.backend.last_pre[0, 0], .24+.025*.02)

    def test_zero_and_missing_exact(self):
        _, base = self.backend.baseline(self.a, self.m, 1, POLICY)
        self.assertEqual(self.call()[1].tobytes(), base.tobytes())
        self.h[0, :32] = 0.
        self.assertEqual(self.call()[1].tobytes(), base.tobytes())

    def test_ineligible_not_corrected(self):
        self.a['eligible'][0, 0] = False
        self.h[0, 0] = .3
        choices, _ = self.call()
        self.assertNotIn(0, choices[0])
        self.assertEqual(self.backend.last_pre[0, 0], self.a['baseline'][0, 0])

    def test_bad_numbers_and_fund_rejected(self):
        for v in (True, '0.1', np.inf, .5001):
            h = self.h.astype(object); h[0, 0] = v
            with self.subTest(v=v), self.assertRaises(ValueError): self.call(h)
        self.h[0, 32] = .1
        with self.assertRaises(ValueError): self.call()

    def test_future_poison_not_read_by_prefix(self):
        self.h[0, 0] = .1
        expected = self.call()
        self.h = self.h.astype(object); self.h[1:] = 'future'
        for key in ('baseline', 'core', 'effects'):
            self.a[key] = self.a[key].astype(object); self.a[key][1:] = 'future'
        actual = self.call()
        self.assertEqual(expected[0], actual[0]); self.assertEqual(expected[1].tobytes(), actual[1].tobytes())

    def test_all_arm_prefix_zero_and_original_arrays(self):
        f = self.h.copy(); g = self.h.copy(); f[:, 0] = .1; g[:, 0] = -.1
        orders, scores = r.order_panels(self.a, self.m, f, g, self.backend, POLICY)
        self.assertEqual(list(orders), list(r.NAMES))
        self.assertEqual(orders[r.NAMES[0]], orders[r.NAMES[1]])
        self.assertNotEqual(scores[r.NAMES[0]].tobytes(), scores[r.NAMES[3]].tobytes())


class SelectionTests(unittest.TestCase):
    def test_only_forward_is_promotable(self):
        rows = trial_rows(); rows[2]['training']['cagr'] = 9.
        self.assertEqual(r.select(rows)['primary'], r.NAMES[0])
        rows[3]['training']['cagr'] = .21
        self.assertEqual(r.select(rows)['primary'], r.NAMES[3])

    def test_strict_cagr_and_all_risk_guards(self):
        for group, key, value in (('training', 'cagr', .2+1e-12), ('training', 'sharpe', .94),
            ('training', 'max_drawdown', -.24), ('early', 'cagr', .174), ('recent', 'cagr', .174)):
            rows = trial_rows(); rows[3]['training']['cagr'] = .21; rows[3][group][key] = value
            with self.subTest(group=group, key=key): self.assertFalse(r.select(rows)['admitted'])

    def test_undefined_metric_falls_back(self):
        for value in (None, True, '1.0', float('nan'), float('inf')):
            rows = trial_rows(); rows[3]['training']['cagr'] = value
            with self.subTest(value=value): self.assertFalse(r.select(rows)['admitted'])


class DefaultHookTests(unittest.TestCase):
    def test_raw_paths_normalized_without_changing_hashes(self):
        self.assertEqual(r.normalized_hashes({'x':'a'}, ROOT),
                         r.normalized_hashes({str(ROOT/'x'):'a'}, ROOT))
        with self.assertRaises(ValueError):
            r.normalized_hashes({'x':'a',str(ROOT/'x'):'b'}, ROOT)

    def test_training_uses_original_helpers_and_only_88_months(self):
        _, meta, arrays, _ = synthetic()
        b = r.Local67Backend.__new__(r.Local67Backend)
        observations = []
        def paths(px, m, n):
            observations.append(('paths', n))
            return 'blocks', None, np.arange(89), 'valid'
        def fast(choices, blocks, valid):
            observations.append(('choices', len(choices)))
            return np.full(88, .01)
        def measure(rr, m, bounds, n=None):
            observations.append(('measure', n))
            return dict(training=dict(cagr=.2, sharpe=1., max_drawdown=-.2),
                        early=dict(cagr=.2), recent=dict(cagr=.2))
        b.local = SimpleNamespace(rp=SimpleNamespace(paths=paths, fast_returns=fast,
            stats_rr=lambda rr, begin, end:dict(count=len(rr))), old=SimpleNamespace(session=SimpleNamespace(measure=measure)))
        orders = {name:[dict(symbols=[meta['symbols'][i][0]]) for i in range(122)] for name in r.NAMES}
        rows, folds = b.training(orders, meta, None)
        self.assertEqual(observations[0], ('paths', 88))
        self.assertEqual([v for k,v in observations if k == 'choices'], [88]*4)
        self.assertEqual([(x['n'],x['end']) for x in folds['folds']], [(36,60),(60,76),(76,88)])
        self.assertEqual(folds['metrics']['count'], 52)

    def test_diagnostics_calls_exact_12_original_cost_paths(self):
        _, meta, _, _ = synthetic()
        b = r.Local67Backend.__new__(r.Local67Backend)
        calls, saved = [], []
        eq = r.pd.Series([100.,101.], index=r.pd.to_datetime([meta['dates'][0], meta['ends'][-1]]))
        def replay(px, op, orders, **kw):
            calls.append((kw['execution'],kw['cost']))
            return dict(annual_returns=[dict(year=2016),dict(year=2026)]), eq, [], []
        b.local = SimpleNamespace(ex=SimpleNamespace(e=SimpleNamespace(engine=SimpleNamespace(replay=replay))),
            arch=SimpleNamespace(old=SimpleNamespace(prior=SimpleNamespace(old=SimpleNamespace(previous=SimpleNamespace(cuts=lambda e,m:{}))))))
        orders={name:[dict(symbols=[]) for _ in range(122)] for name in r.NAMES}
        results, periods=b.diagnostics(orders,meta,None,None,lambda name,value:saved.append(name))
        self.assertEqual(calls,[('close',0.),('open',0.),('open',.0005)]*4)
        self.assertEqual(len(saved),36)
        self.assertEqual(sum(x['split']=='later34' for x in periods),34)
        self.assertTrue(all(x['partial'] for x in results[r.NAMES[0]]['same_close']['metrics']['annual_returns']))


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.out = self.root/'outputs/monthly_forward_guidance132'; self.out.mkdir(parents=True)
        self.e, self.meta, self.arrays, rows = synthetic()
        self.backend = Backend(self.meta, self.arrays)
        paths = {}
        values = dict(proposal={'prospective_round':132, 'arms':list(r.NAMES), 'primary_challenger':r.NAMES[3]},
            events=[self.e], availability={'rows':rows}, meta=self.meta,
            sessions=self.meta['dates'], policy=POLICY, prices={'synthetic':True}, protected_manifest={},
            reviewer={'code':'synthetic'}, raw={'raw':'synthetic'})
        ch, sc = self.backend.baseline(self.arrays, self.meta, 122, POLICY)
        values['original_orders'] = self.backend.make_orders(ch, sc, self.meta)
        for key, value in values.items():
            paths[key] = self.root/(key+'.json'); r.dump(paths[key], value)
        r.dump(self.root/'admission.json', dict(status='PASS', semantic_review_passed=True,
            availability_review_passed=True, calendar_review_passed=True, reviewed_event_ids=[self.e['event_id']],
            events_sha256=r.sha(paths['events']), availability_sha256=r.sha(paths['availability']),
            source_hashes={str(paths['raw']):r.sha(paths['raw'])}))
        paths['admission'] = self.root/'admission.json'
        pins = {str(p):r.sha(p) for p in paths.values()}
        self.plan = dict(round=132, status='REGISTERED', arms=list(r.NAMES), primary_challenger=r.NAMES[3],
            output_directory=str(self.out), runtime=dict(python=r.sys.version, numpy=np.__version__, pandas=r.pd.__version__),
            tolerances=r.TOLERANCES, inputs={key:dict(path=str(paths[key]), sha256=r.sha(paths[key])) for key in r.INPUTS},
            hashes=pins, raw_source_hashes={str(paths['raw']):r.sha(paths['raw'])},
            independent_prepare_source_hashes={str(paths['reviewer']):r.sha(paths['reviewer'])})
        self.plan_path = self.out/'PLAN.json'; r.dump(self.plan_path, self.plan)
        self.code_patch = patch.object(r, 'source_files', return_value={str(paths['reviewer']):r.sha(paths['reviewer'])})
        self.protect_patch = patch.object(r, 'PROTECTED_SHA', r.sha(paths['protected_manifest']))
        self.code_patch.start(); self.protect_patch.start()
        self.addCleanup(self.code_patch.stop); self.addCleanup(self.protect_patch.stop)
        self.runner = self.make_runner()

    def make_runner(self):
        return r.Runner(self.root, self.plan_path, r.sha(self.plan_path), lambda root, runner:self.backend)

    def review(self):
        cache = self.runner.cache()
        artifacts = dict(cache)
        artifacts.update({str(self.out/x):r.sha(self.out/x) for x in ('PLAN.json','CACHE.json')})
        result = dict(passed=True, stage='prepare', producer_modules_imported=False, returns_or_nav_computed=False,
            monthly_label_models_fit=0, source_hashes=self.plan['independent_prepare_source_hashes'],
            artifact_hashes=artifacts, plan_sha256=r.sha(self.plan_path), cache_sha256=r.sha(self.out/'CACHE.json'),
            orders_sha256=r.sha(self.out/'PRETRAIN_ORDERS.json'))
        result.update({k:True for k in r.CHECKS})
        return result

    def test_synthetic_three_phases_and_no_repeat(self):
        self.runner.prepare()
        self.assertEqual(self.backend.training_calls, 0)
        with self.assertRaises(ValueError): self.runner.prepare()
        r.dump(self.out/'INDEPENDENT_PREPARE.json', self.review())
        self.runner.select_first88()
        self.assertEqual(self.backend.training_length, 88)
        self.assertEqual(self.backend.diagnostic_calls, 0)
        self.runner.diagnose_later34()
        self.assertEqual(self.backend.diagnostic_calls, 1)
        self.assertEqual(len(r.read(self.out/'RESULTS.json')), 4)
        self.assertEqual(sum(x['split']=='later34' for x in r.read(self.out/'MONTH_COMPARISON.json')), 34)
        with self.assertRaises(ValueError): self.runner.select_first88()

    def test_no_review_no_return_hook(self):
        self.runner.prepare()
        with self.assertRaises(FileNotFoundError): self.runner.select_first88()
        self.assertEqual(self.backend.training_calls, 0)

    def test_wrong_or_empty_review_no_return_hook(self):
        self.runner.prepare()
        original = self.review()
        for field, value in (('stage','outcome'), ('source_hashes',{}), ('artifact_hashes',{}),
                             (r.CHECKS[0],False), ('returns_or_nav_computed',True), ('monthly_label_models_fit',True)):
            bad = deepcopy(original); bad[field] = value
            path = self.out/'INDEPENDENT_PREPARE.json'; path.write_text(json.dumps(bad))
            with self.subTest(field=field), self.assertRaises(ValueError): self.runner.select_first88()
        self.assertEqual(self.backend.training_calls, 0)

    def test_tampered_cache_blocks_before_return_hook(self):
        self.runner.prepare(); r.dump(self.out/'INDEPENDENT_PREPARE.json', self.review())
        (self.out/'META.json').write_text('{}')
        with self.assertRaises(ValueError): self.runner.select_first88()
        self.assertEqual(self.backend.training_calls, 0)

    def test_changed_plan_blocks_prepare(self):
        self.plan_path.write_text('{}')
        with self.assertRaises(ValueError): self.runner.prepare()
        self.assertEqual(self.backend.loads, 0)

    def test_draft_plan_and_unadmitted_source_rejected(self):
        self.plan['status']='DRAFT'; self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError): self.make_runner()
        self.plan['status']='REGISTERED'
        p = self.root/'admission.json'; bad = r.read(p); bad['semantic_review_passed']=False; p.write_text(json.dumps(bad))
        self.plan['inputs']['admission']['sha256']=r.sha(p); self.plan['hashes'][str(p)]=r.sha(p)
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError): self.make_runner()

    def test_altered_frozen_selection_blocks_diagnostics(self):
        self.runner.prepare(); r.dump(self.out/'INDEPENDENT_PREPARE.json', self.review()); self.runner.select_first88()
        p=self.out/'FROZEN.json'; bad=r.read(p); bad['selection']['primary']=r.NAMES[2]; p.write_text(json.dumps(bad))
        with self.assertRaises(ValueError): self.runner.diagnose_later34()
        self.assertEqual(self.backend.diagnostic_calls, 0)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.local = self.root/'outputs/monthly_local_linear70'
        self.local.mkdir(parents=True)
        def put(path, value):
            path=self.root/path; path.parent.mkdir(parents=True,exist_ok=True); r.dump(path,value); return path
        self.raw={str(put('raw/'+str(i),{})):None for i in range(64)}
        self.raw={path:r.sha(path) for path in self.raw}
        for name in ('study.py','DESIGN.json','DESIGN.sha256','GRID.json','DATA.npz','META.json',
                     'INPUT_HASHES.json','INCUMBENT_POLICY.json','local_linear_ORDERS.json','model.py','calendar.py'):
            put('outputs/monthly_local_linear70/'+name,{})
        put('outputs/monthly_local_linear70/PLAN.json',dict(raw_source_hashes=self.raw))
        put('outputs/monthly_local_linear70/CACHE_HASHES.json',{'DATA.npz':r.sha(self.local/'DATA.npz')})
        proposal=put('work/monthly_forward_guidance132_proposal.json',{})
        self.reviewer=put('work/monthly_forward_guidance132_prepare_review.py',{})
        put('work/test_monthly_forward_guidance_factor132.py',{})
        events=put('outputs/monthly_forward_guidance_preflight_v1/candidate_join/JOINED_EVENTS.json',[])
        avail=put('outputs/monthly_forward_guidance_preflight_v1/availability/AVAILABILITY.json',{})
        put('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json',{})
        put('outputs/monthly_2016_extension/PRICES.npz',{})
        protected=put('work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',{})
        self.admission=put(registry.DEFAULT_ADMISSION,dict(status='PASS',semantic_review_passed=True,
            availability_review_passed=True,calendar_review_passed=True,reviewed_event_ids=['one'],
            events_sha256=r.sha(events),availability_sha256=r.sha(avail),source_hashes={str(events):r.sha(events)}))
        self.calls=[]
        def forbidden(): raise AssertionError('Array/model/return hook called during registration')
        fake=SimpleNamespace(P=self.local,registered=lambda:self.calls.append('registered'),
            verify_source=lambda:self.calls.append('verify_source'),load=forbidden,prepare=forbidden,
            implementations=lambda:[self.local/'model.py'],source_files=lambda:[],
            calendar_provenance=lambda:{'source_hashes':{str(self.local/'calendar.py'):r.sha(self.local/'calendar.py')}})
        oracle=SimpleNamespace(source_files=lambda root:[self.reviewer])
        patches=[patch.object(r,'module',side_effect=lambda name,path:oracle if 'review' in name else fake),
                 patch.object(r,'source_files',return_value={str(self.reviewer):r.sha(self.reviewer)}),
                 patch.object(registry,'PROPOSAL_SHA',r.sha(proposal)),
                 patch.object(r,'LOCAL_SHA',r.sha(self.local/'study.py')),
                 patch.object(r,'PROTECTED_SHA',r.sha(protected))]
        for p in patches: p.start(); self.addCleanup(p.stop)

    def build(self):
        return registry.build_plan(self.root,self.admission,r.sha(self.admission),self.reviewer,r.sha(self.reviewer))

    def test_build_only_hashes_and_no_write(self):
        plan=self.build()
        self.assertEqual(len(plan['raw_source_hashes']),64)
        self.assertEqual(plan['arms'],list(r.NAMES)); self.assertEqual(set(plan['inputs']),set(r.INPUTS))
        self.assertEqual(self.calls,['registered','verify_source'])
        self.assertFalse((self.root/'outputs/monthly_forward_guidance132').exists())
        self.assertEqual(plan['actual_feature_calculations_before_registration'],0)
        self.assertEqual(plan['actual_returns_before_registration'],0)

    def test_save_is_exclusive(self):
        plan=self.build(); result=registry.save_plan(self.root,plan); before=Path(result['path']).read_bytes()
        with self.assertRaises(FileExistsError):registry.save_plan(self.root,plan)
        self.assertEqual(Path(result['path']).read_bytes(),before)
        with self.assertRaises(ValueError):self.build()

    def test_wrong_admission_or_reviewer_hash_stops_before_old_hooks(self):
        for bad in ('admission','reviewer'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                registry.build_plan(self.root,self.admission,'0'*64 if bad=='admission' else r.sha(self.admission),
                    self.reviewer,'0'*64 if bad=='reviewer' else r.sha(self.reviewer))
        self.assertEqual(self.calls,[])

    def test_missing_exact_input_pin_rejected(self):
        obj=r.read(self.admission); del obj['events_sha256']; self.admission.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):self.build()

    def test_outcomes_not_collected(self):
        path=self.root/'RESULTS.json'; r.dump(path,{})
        with self.assertRaises(ValueError):registry.Bindings(self.root).add(path)

    def test_cache_manifest_cannot_escape_its_directory(self):
        mapping={'../PRICES.npz':'0'*64}
        with self.assertRaises(ValueError):registry.Bindings(self.root).mapping(mapping,self.local)


if __name__ == '__main__':
    unittest.main()
