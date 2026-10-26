"""Synthetic-only tests of233 direct signed-block transforms, frozen-score wiring and governance."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('tested_sectorsignedrank233_host', Path(__file__).with_name('monthly_sector_signedrank233_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture():
    shape = (6, 5)
    values = dict(candidate=np.tile([-.0625, -.125, -.1875, -.25, -.3125], (6, 1)),
                  control=np.tile([-.3125, -.25, -.1875, -.125, -.0625], (6, 1)), available=np.ones(shape, bool))
    values['available'][2, :4] = False
    values['available'][3, :] = False
    for key in ('candidate', 'control'):
        values[key][~values['available']] = np.nan
    a = dict(baseline=np.tile([.2, .199, .198, .197, .196], (6, 1)),
             eligible=np.ones(shape, bool), effects=np.zeros(shape),
             core=np.zeros(shape), core_weight=np.full((6, 1), .52),
             state_calm10=np.zeros(6, bool), state_weak_narrow=np.zeros(6, bool))
    a['eligible'][:, -1] = False  # Still receives its raw delta even though not a buy candidate.
    m = dict(dates=['synthetic-%d' % i for i in range(6)], symbols=[list('ABCDE') for _ in range(6)])
    return a, m, values


class SyntheticStudy:
    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter = SimpleNamespace(score_and_choose=Mock(side_effect=self.choose))
        self.rp = SimpleNamespace(make_orders=self.orders)

    @staticmethod
    def configs():
        return [dict(id=i) for i in range(5)]

    @staticmethod
    def modified_pre_score(specification, a, n):
        assert specification == dict(id=4)
        return a['baseline'].copy() + .11  # Includes all already-frozen slots.

    @staticmethod
    def boundary(pre, a, m, n):
        choices = []
        for i in range(n):
            ids = sorted(np.flatnonzero(a['eligible'][i]), key=lambda j: (-pre[i, j], j))
            choices.append([m['symbols'][i][j] for j in ids[:3]] if len(ids) >= 2 else [])
        return choices, pre[:n].copy()

    def choices(self, specification, a, m, n, *, policy):
        return self.boundary(self.modified_pre_score(specification, a, n), a, m, n)

    @staticmethod
    def orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=symbols, scores=scores[i].tolist()) for i, symbols in enumerate(choices)]

    def choose(self, base, weight, extra, a, m, n, *, pre_score, effects, episode_policy, boundary_selector):
        assert base == 'incumbent66' and weight == 0.
        assert extra.shape == a['baseline'].shape and np.isnan(extra).all()
        assert effects is a['effects'] and boundary_selector is self.ep.select_boundary
        return boundary_selector(pre_score, a, m, n)


class DirectMappingAndWiringTests(unittest.TestCase):
    def test_direct_mapping_includes_ineligible_cells_and_exact_bounds(self):
        a, m, values = fixture()
        values['candidate'][1] = [-.75, -.0625, 0., .0625, .75]
        transforms, deltas = h.scale_and_delta(values, 6)
        expected = values['candidate'][1].copy()
        np.testing.assert_array_equal(transforms[h.NAMES[1]][1], expected)
        np.testing.assert_array_equal(deltas[h.NAMES[1]][1], .025*expected)
        self.assertFalse(a['eligible'][1, -1])
        self.assertGreater(deltas[h.NAMES[1]][1, -1], 0.)
        self.assertTrue(np.all(deltas[h.NAMES[1]][0] == 0.))
        self.assertTrue(all(np.max(np.abs(x)) <= h.DOSE*h.MAX_RAW for x in deltas.values()))

    def test_missing_cells_neutral_but_singleton_not_discarded(self):
        _, _, values = fixture()
        transforms, deltas = h.scale_and_delta(values, 6)
        for name in h.NAMES[1:]:
            np.testing.assert_array_equal(transforms[name][2, :4], np.zeros(4))
            np.testing.assert_array_equal(deltas[name][3], np.zeros(5))
            self.assertLess(deltas[name][2, 4], 0.)

    def test_no_cross_sectional_scaling_or_rank(self):
        _, _, values = fixture(); values['candidate'][1] = [-.0625, -.125, -.125, -.125, -.1875]
        transforms, delta = h.scale_and_delta(values, 2)
        np.testing.assert_array_equal(transforms[h.NAMES[1]][1], values['candidate'][1])
        old = delta[h.NAMES[1]][1, 1]
        values['candidate'][1, 0] = -.5
        _, changed = h.scale_and_delta(values, 2)
        self.assertEqual(old, changed[h.NAMES[1]][1, 1])
        self.assertLess(old, .001)

    def test_common_mask_and_invalid_numerics_enforced(self):
        for kind in ('mask', 'infinity', 'shape', 'dtype'):
            _, _, values = fixture()
            if kind == 'mask': values['candidate'][1, 0] = np.nan
            if kind == 'infinity': values['control'][1, 0] = np.inf
            if kind == 'shape': values['control'] = values['control'][:, :-1]
            if kind == 'dtype': values['available'] = values['available'].astype(int)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                h.scale_and_delta(values, 6)

    def test_prefix_does_not_consume_future_features(self):
        _, _, values = fixture(); fullr, fulld = h.scale_and_delta(values, 6)
        for n in range(7):
            poisoned = copy.deepcopy(values)
            for key in ('candidate', 'control'): poisoned[key][n:] = np.inf
            ranks, deltas = h.scale_and_delta(poisoned, n)
            for arm in h.NAMES[1:]:
                np.testing.assert_array_equal(ranks[arm], fullr[arm][:n])
                np.testing.assert_array_equal(deltas[arm], fulld[arm][:n])

    def test_prefix_does_not_convert_future_object_poison(self):
        _, _, values = fixture(); expected, _ = h.scale_and_delta(values, 2)
        for key in ('candidate', 'control'):
            values[key] = values[key].astype(object)
            values[key][2:] = 'future-not-numeric'
        actual, _ = h.scale_and_delta(values, 2)
        for arm in actual: np.testing.assert_array_equal(actual[arm], expected[arm])

    def test_additive_small_slot_and_exactly_one_episode(self):
        a, m, values = fixture(); olda, oldv = copy.deepcopy(a), copy.deepcopy(values)
        study = SyntheticStudy(); policy = dict(synthetic=True)
        original = study.modified_pre_score(dict(id=4), a, 6)
        with patch.object(h, 'read', return_value=policy) as reader:
            orders, ranks, deltas, scores = h.decisions(study, a, m, values, 6)
        reader.assert_called_once_with(h.LOCAL / 'INCUMBENT_POLICY.json')
        self.assertEqual(study.adapter.score_and_choose.call_count, 2)
        self.assertEqual(study.ep.select_boundary.call_count, 2)
        for arm, call in zip(h.NAMES[1:], study.adapter.score_and_choose.call_args_list):
            expected = original.copy(); use = deltas[arm] != 0.
            expected[use] += deltas[arm][use]
            np.testing.assert_array_equal(call.kwargs['pre_score'], expected)
            self.assertIs(call.kwargs['episode_policy'], policy)
            self.assertEqual(orders[arm][0], orders[h.NAMES[0]][0])
        for key in a: np.testing.assert_array_equal(a[key], olda[key])
        for key in values: np.testing.assert_array_equal(values[key], oldv[key])

    def test_every_decision_prefix_and_no_models_called(self):
        a, m, values = fixture()
        with patch.object(h, 'read', return_value={'synthetic': True}):
            allorders, _, _, _ = h.decisions(SyntheticStudy(), a, m, values, 6)
            for n in range(7):
                orders, _, _, _ = h.decisions(SyntheticStudy(), a, m, values, n)
                for name in h.NAMES: self.assertEqual(orders[name], allorders[name][:n])

    def test_prefix_validation_rejects_bool_fraction_and_out_of_range(self):
        _, _, values = fixture()
        for n in (True, False, 1.5, np.int64(1), -1, 7):
            with self.subTest(n=n), self.assertRaises(ValueError):
                h.scale_and_delta(values, n)
        transforms, deltas = h.scale_and_delta(values, 0)
        for result in (transforms, deltas):
            self.assertTrue(all(array.shape == (0, 5) for array in result.values()))


def selection_fixture():
    metrics = {n: {p: {e: dict(cagr=.5, sharpe=1.5, max_drawdown=-.3) for e in ('training', 'early', 'recent')}
                  for p in ('same_close_zero', 'next_open_zero')} for n in h.NAMES}
    orders = {n: [dict(symbols=['A', 'B', 'C']) for _ in range(122)] for n in h.NAMES}
    for path in metrics[h.NAMES[1]]: metrics[h.NAMES[1]][path]['training']['cagr'] += .01
    for i in (1, 2, 53): orders[h.NAMES[1]][i]['symbols'] = ['A', 'B', 'D']
    return metrics, orders


class SelectionAndBindingTests(unittest.TestCase):
    def test_later34_and_extra_metrics_never_choose_arm(self):
        metrics, orders = selection_fixture(); expected = h.select_first88(metrics, orders)
        self.assertEqual(expected['primary'], h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['full122'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
            for i in range(88, 122): orders[name][i]['symbols'] = [name]
        self.assertEqual(h.select_first88(metrics, orders), expected)

    def test_sign_control_cannot_promote_and_must_be_beaten(self):
        for path in ('same_close_zero', 'next_open_zero'):
            metrics, orders = selection_fixture(); metrics[h.NAMES[2]][path]['training']['cagr'] = 2.
            result = h.select_first88(metrics, orders)
            self.assertEqual(result['primary'], h.NAMES[0]); self.assertFalse(result['admitted'])

    def test_exact_gain_era_risk_and_changed_set_guards(self):
        for path, era, key, value in [('same_close_zero', 'training', 'cagr', .504),
                ('next_open_zero', 'training', 'cagr', .49), ('same_close_zero', 'early', 'cagr', .474),
                ('same_close_zero', 'recent', 'cagr', .474), ('same_close_zero', 'training', 'sharpe', 1.449),
                ('same_close_zero', 'training', 'max_drawdown', -.331)]:
            metrics, orders = selection_fixture(); metrics[h.NAMES[1]][path][era][key] = value
            self.assertFalse(h.select_first88(metrics, orders)['admitted'])
        metrics, orders = selection_fixture()
        for i in (1, 2, 53): orders[h.NAMES[1]][i]['symbols'] = ['C', 'B', 'A']
        for i in (90, 100, 110): orders[h.NAMES[1]][i]['symbols'] = ['A', 'B', 'D']
        self.assertEqual(h.select_first88(metrics, orders)['changed_months'], [])

    def test_candidate_must_change_holdings_vs_control_in_both_eras(self):
        metrics, orders = selection_fixture()
        orders[h.NAMES[2]] = copy.deepcopy(orders[h.NAMES[1]])
        decision = h.select_first88(metrics, orders)
        self.assertFalse(decision['admitted'])
        self.assertEqual(decision['changed_months_vs_control'], [])
        for i in (1,2,3): orders[h.NAMES[2]][i]['symbols'] = ['A','E','F']
        self.assertFalse(h.select_first88(metrics, orders)['guards']['changed_support_vs_control'])
        orders[h.NAMES[2]][53]['symbols'] = ['A','E','F']
        self.assertTrue(h.select_first88(metrics, orders)['admitted'])

    def test_raw_bounds_enforced_and_equal_arms_stay_exact(self):
        _, _, values = fixture()
        values['candidate'][1] = [-.75, -.0625, 0., .0625, .75]
        values['control'] = values['candidate'].copy()
        _, delta = h.scale_and_delta(values, 2)
        np.testing.assert_array_equal(delta[h.NAMES[1]], delta[h.NAMES[2]])
        np.testing.assert_array_equal(delta[h.NAMES[1]][1], .025*values['candidate'][1])
        for wrong in (np.nextafter(.75, np.inf), -.7500001, np.finfo(float).max):
            values['candidate'][1,0] = wrong
            with self.assertRaises(ValueError): h.scale_and_delta(values,2)
        values['candidate'][0,0] = 1.
        with self.assertRaises(ValueError): h.scale_and_delta(values,1)

    def test_inherited_pin_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fake-source'; path.write_text('synthetic')
            pins = {}; h.bind(pins, path); before = dict(pins)
            h.bind(pins, path, h.sha(path)); self.assertEqual(pins, before)
            path.write_text('changed')
            with self.assertRaises(ValueError): h.bind(pins, path)
            self.assertEqual(pins, before)

    def test_rejected_plan_cannot_load_source_or_compute_features(self):
        with patch.object(h, 'gate', side_effect=ValueError('unregistered')), patch.object(h, 'local') as loader:
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                h.prepare('bad-plan-sha')
        loader.assert_not_called()

    def test_completed_predecessor_and_counters_required(self):
        state = dict(completed=[dict(trials=1, directory='old')] * 227 + [dict(trials=974, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)


class SourceBindingTests(unittest.TestCase):
    def test_sector_manifest_sources_are_exact_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            mapper = root / 'mapping.py'; mapper.write_text('synthetic mapping')
            sectors = root / 'sectors.csv'; sectors.write_text('synthetic sectors')
            manifests = [root / ('manifest%d.json' % n) for n in range(3)]
            entries = []
            raw = {str(sectors): h.sha(sectors)}
            for symbol in h.SECTOR_ETFS:
                path = root / (symbol + '.csv'); path.write_text('synthetic ' + symbol)
                digest = h.sha(path); raw[str(path)] = digest
                entries.append(dict(symbol=symbol, file=str(path), sha256=digest))
            for i, manifest in enumerate(manifests):
                manifest.write_text(json.dumps(entries if i == 0 else []))
                raw[str(manifest)] = h.sha(manifest)
            with patch.multiple(h, ROOT=root, MAPPING_PATH=mapper, MAPPING_SHA=h.sha(mapper),
                                SECTORS_PATH=sectors, ETF_MANIFESTS=tuple(manifests)):
                binding = h.sector_source_binding(raw)
                self.assertEqual(set(binding['sources']), set(h.SECTOR_ETFS))
                self.assertEqual(binding['own_sector_etfs'], list(h.SECTOR_ETFS))
                self.assertIn('XLC', binding['own_sector_etfs'])
                self.assertIn('XLRE', binding['own_sector_etfs'])
                missing = dict(raw); missing.pop(str(sectors))
                with self.assertRaises(ValueError): h.sector_source_binding(missing)
                manifests[1].write_text(json.dumps([entries[0]])); raw[str(manifests[1])] = h.sha(manifests[1])
                with self.assertRaises(ValueError): h.sector_source_binding(raw)

    def test_original_source_panel_and_mapping_are_used(self):
        callback = Mock()
        study = SimpleNamespace(ex=SimpleNamespace(t=SimpleNamespace(strict=SimpleNamespace(
            __file__=str(h.MAPPING_PATH), benchmark_map=callback))))
        adj = pd.DataFrame(columns=list(h.SECTOR_ETFS) + ['A'])
        sectors = pd.DataFrame(); source = dict(adj=adj, sectors=sectors)
        meta = dict(etfs=['E%d' % n for n in range(17)])
        with patch.object(h, 'sha', return_value=h.MAPPING_SHA):
            panel, mapper = h.sector_inputs(study, source, meta)
            self.assertIs(panel['adj'], adj); self.assertIs(panel['sectors'], sectors)
            self.assertIs(mapper, callback); callback.assert_not_called()
            with self.assertRaises(ValueError): h.sector_inputs(study, dict(adj=adj), meta)
            with self.assertRaises(ValueError): h.sector_inputs(study, dict(adj=adj[['A']], sectors=sectors), meta)
            study.ex.t.strict.__file__ = str(h.ROOT / 'work/replacement.py')
            with self.assertRaises(ValueError): h.sector_inputs(study, source, meta)


    def test_source_prefix_truncates_prices_and_classifications_and_zero(self):
        dates=pd.bdate_range('2024-01-01',periods=8)
        panel={'adj':pd.DataFrame({'A':range(8)},index=dates),
               'sectors':pd.DataFrame({'A':['Technology']*8},index=dates)}
        meta={'dates':[str(d.date()) for d in dates[[2,5]]]}
        panel['adj']=panel['adj'].astype(object);panel['adj'].iloc[6:,0]='future-poison'
        panel['sectors'].iloc[6:,0]='future-classification'
        before={k:v.copy(deep=True) for k,v in panel.items()}
        for n,length in [(0,0),(1,3),(2,6)]:
            small=h.feature_prefix_panel(panel,meta,n)
            for key in panel:
                pd.testing.assert_frame_equal(small[key],panel[key].iloc[:length])
                self.assertIsNot(small[key],panel[key])
        for n in (True,False,-1,3,1.5):
            with self.assertRaises(ValueError):h.feature_prefix_panel(panel,meta,n)
        for key in panel:pd.testing.assert_frame_equal(before[key],panel[key])


    def test_real_adapter_and_original_boundary_with_column_weights(self):
        # Compile the exact original boundary function without importing any driver or prices.
        path = h.ROOT / 'outputs/monthly_daily_episodes/episodes.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'select_boundary')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        adapter = h.module('sectorsignedrank233_test_original_adapter', h.ROOT / 'outputs/monthly_path70_research/incumbent_adapter.py')
        a, m, values = fixture()
        a['state_calm10'][2] = True
        a['core_weight'][2, 0] = .50
        a['effects'][1, 3] = .2
        before = {key: value.tobytes() for key, value in a.items()}
        study = SyntheticStudy()
        study.adapter = adapter
        study.ep = SimpleNamespace(select_boundary=Mock(side_effect=namespace['select_boundary']))
        policy = dict(budget=.05, ceiling=5, gap=.02)
        def choices(specification, arrays, meta, n, *, policy):
            return adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
                arrays, meta, n, pre_score=study.modified_pre_score(specification, arrays, n),
                effects=arrays['effects'], episode_policy=policy, boundary_selector=namespace['select_boundary'])
        study.choices = choices
        with patch.object(h, 'read', return_value=policy):
            orders, _, deltas, scores = h.decisions(study, a, m, values, 6)
        self.assertEqual(study.ep.select_boundary.call_count, 2)
        for arm, call in zip(h.NAMES[1:], study.ep.select_boundary.call_args_list):
            score, eligible, symbols, effects, original, passed_policy = call.args
            np.testing.assert_array_equal(score, study.modified_pre_score(dict(id=4), a, 6) + deltas[arm])
            np.testing.assert_array_equal(original['core'], a['core'])
            np.testing.assert_array_equal(original['baseline'], a['baseline'])
            np.testing.assert_array_equal(original['core_weight'], np.broadcast_to(a['core_weight'], (6, 5)))
            np.testing.assert_array_equal(eligible, a['eligible'])
            self.assertEqual(orders[arm][0], orders[h.NAMES[0]][0])
        self.assertEqual(before, {key: value.tobytes() for key, value in a.items()})


class AuditReceiptTests(unittest.TestCase):

    def test_frozen_training_matches_independent_slices(self):
        training, _ = selection_fixture()
        audit = dict(paths={a: {p: dict(slices={full: copy.deepcopy(s[short])
                    for short, full in [('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')]})
                    for p, s in paths.items()} for a, paths in training.items()})
        h.verify_training_against_audit(training, audit)
        for arm in h.NAMES:
            for path in training[arm]:
                for segment in training[arm][path]:
                    bad = copy.deepcopy(training); bad[arm][path][segment]['cagr'] += .01
                    with self.assertRaises(ValueError): h.verify_training_against_audit(bad, audit)


    def test_nine_paths_five_slices_exact_receipt_and_source(self):
        one = dict(passed=True, monthly_executions=122, slices={s: {} for s in ('full122', 'first88', 'early53', 'recent35', 'later34')})
        receipt = dict(passed=True, independent_accounts=True, independent_of_production_model_and_engine=True,
                       checked_paths=9, checked_slices=45, terminal_liquidation=False, accounting_source_sha256='fake',
                       paths={a: {p[0]: copy.deepcopy(one) for p in h.ev.PATHS} for a in h.NAMES})
        with patch.object(h, 'sha', return_value='fake'):
            h.require_outcome_audit(receipt)
            for change in ('path', 'months', 'slices', 'pass', 'sha'):
                bad = copy.deepcopy(receipt); path = bad['paths'][h.NAMES[0]]['same_close_zero']
                if change == 'path': bad['paths'][h.NAMES[0]].pop('next_open_zero')
                if change == 'months': path['monthly_executions'] = 121
                if change == 'slices': path['slices'].pop('later34')
                if change == 'pass': path['passed'] = False
                if change == 'sha': bad['accounting_source_sha256'] = 'wrong'
                with self.subTest(change=change), self.assertRaises(ValueError): h.require_outcome_audit(bad)


def predecessor_fixture(root):
    prior = root/'outputs/monthly_volume_slope232'
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path.resolve()),sha256=h.sha(path))
    plan=put(prior/'PLAN.json',{'input_sha256':{}})
    result=put(prior/'results/COMPLETION.json',dict(status='COMPLETE',plan_sha256=plan['sha256'],paths=9,slices=45))
    closed=put(prior/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',plan_sha256=plan['sha256'],
        result_completion_sha256=result['sha256'],protected_files=21444,observers_modified=False))
    put(prior/'RESEARCH_REVIEW.json',dict(passed=True))
    context=root/'work/context.json'
    evidence=put(root/'work/evidence.json',{'synthetic':True})
    put(context,dict(evidence_sha256={'work/evidence.json':evidence['sha256']}))
    return prior,{n:h.sha(prior/n) for n in h.PRIOR_EXPECTED},dict(completed=[dict(completion=closed)]),context


class PredecessorBindingTests(unittest.TestCase):
    def test_predecessor_completion_and_result_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();prior,expected,state,context=predecessor_fixture(root)
            with patch.multiple(h,ROOT=root,PRIOR=prior,PRIOR_EXPECTED=expected,CONTEXT=context):
                binding=h.predecessor_binding(state)
                self.assertIn(str(prior/'results/COMPLETION.json'),binding['input_sha256'])
                self.assertIn(str(root/'work/evidence.json'),binding['input_sha256'])
                for key,value in [('path',str(root/'wrong')),('sha256','wrong')]:
                    wrong=copy.deepcopy(state);wrong['completed'][0]['completion'][key]=value
                    with self.assertRaises(ValueError):h.predecessor_binding(wrong)

    def test_predecessor_and_context_evidence_cannot_change(self):
        for relative in ('outputs/monthly_volume_slope232/PLAN.json','outputs/monthly_volume_slope232/results/COMPLETION.json','work/evidence.json'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();prior,expected,state,context=predecessor_fixture(root)
                (root/relative).write_text('changed')
                with patch.multiple(h,ROOT=root,PRIOR=prior,PRIOR_EXPECTED=expected,CONTEXT=context):
                    with self.assertRaises(ValueError):h.predecessor_binding(state)

    def test_all_phase_gates_recheck_registered_predecessor_binding(self):
        plan = dict(contract=h.contract(), runtime=h.ev.runtime(), input_sha256={'x':'y'},
                    prior_central_sha256='sha', predecessor_binding={'registered':True},
                    raw_source_hashes={'raw':'sha'},sector_source_binding={'sector':True},
                    price_source_binding={'price':True},numerical_code_review={'review':True})
        state = dict(registered_queue=[dict(directory=h.OUT.name, plan_sha256='sha')])
        def reader(path): return plan if path == h.OUT / 'PLAN.json' else state
        with patch.object(h, 'contract', return_value=plan['contract']), patch.object(h, 'sha', return_value='sha'), patch.object(h, 'read', side_effect=reader), \
             patch.object(h, 'verify'), patch.object(h, 'state_check'), \
             patch.object(h, 'sector_source_binding',return_value={'sector':True}), \
             patch.object(h, 'price_source_binding',return_value={'price':True}), \
             patch.object(h, 'code_review_binding',return_value={'review':True}), \
             patch.object(h, 'predecessor_binding', return_value={'registered':True}) as predecessor:
            self.assertEqual(h.gate('sha'), plan)
            predecessor.assert_called_once_with(state)
            predecessor.return_value = {'registered':False}
            with self.assertRaises(ValueError): h.gate('sha')

    def test_original_column_weight_shape_does_not_enter_new_delta_api(self):
        import monthly_sector_signedrank233_independent as independent
        original, _, values = fixture()
        self.assertEqual(original['core_weight'].shape, (6, 1))
        _, deltas = h.scale_and_delta(values, 6)
        check = independent.verify_deltas(values, deltas, dose=h.DOSE)
        h.require_delta_audit(check, deltas)
        wrong = copy.deepcopy(deltas); wrong[h.NAMES[1]][1, 0] += .001
        with self.assertRaises(ValueError): independent.verify_deltas(values, wrong, dose=h.DOSE)



class PureApiIntegrationTests(unittest.TestCase):
    def test_real_production_and_independent_interfaces_on_synthetic_panels(self):
        import monthly_sector_signedrank233_kernel as kernel
        import monthly_sector_signedrank233_independent as independent
        import test_monthly_sector_signedrank233_kernel as fixtures
        panel, meta, calendar = fixtures.fixture()
        values, audit = kernel.build_features(panel, meta, calendar, fixtures.mapper)
        check = independent.verify_features(panel, meta, calendar, fixtures.mapper, values, audit)
        for key in h.FEATURE_FLAGS: self.assertIs(check[key], True, key)
        self.assertEqual(h.estimate_counts(check), h.estimate_counts(audit))
        transforms, deltas = h.scale_and_delta(values, len(meta['dates']))
        h.require_delta_audit(independent.verify_deltas(values, deltas, dose=h.DOSE), deltas)
        for n in range(len(meta['dates'])+1):
            prefix = h.feature_prefix_panel(panel, meta, n)
            small, small_audit = kernel.build_features(prefix, meta, calendar, fixtures.mapper, limit=n)
            self.assertEqual(small_audit['event_history'],audit['event_history'][:n])
            for key in values: np.testing.assert_array_equal(small[key],values[key][:n])
            tt, dd = h.scale_and_delta(small,n)
            h.require_delta_audit(independent.verify_deltas(small,dd,dose=h.DOSE),dd)
            for arm in h.NAMES[1:]:
                np.testing.assert_array_equal(tt[arm],transforms[arm][:n])
                np.testing.assert_array_equal(dd[arm],deltas[arm][:n])

    def test_exact_maximum_product_is_retained_not_clipped(self):
        values=dict(candidate=np.full((2,2),.75),control=np.full((2,2),-.75),available=np.ones((2,2),bool))
        _,deltas=h.scale_and_delta(values,2)
        actual=h.DOSE*h.MAX_RAW
        self.assertGreater(actual,h.MAX_DELTA)
        self.assertEqual(deltas[h.NAMES[1]][1,0],actual)
        self.assertEqual(deltas[h.NAMES[2]][1,0],-actual)
        np.testing.assert_array_equal(deltas[h.NAMES[1]][0],[0.,0.])

    def test_actual_predecessor_context_metadata_only_binding(self):
        # Hash JSON/code evidence only: no arrays, prices, labels, fits or outcome calculations.
        state=h.read(h.CENTRAL)
        h.state_check(state)
        self.assertEqual(state['registered_queue'],[])
        binding=h.predecessor_binding(state)
        self.assertEqual(binding['directory'],'monthly_volume_slope232')
        for name,digest in h.PRIOR_EXPECTED.items():
            self.assertEqual(binding['input_sha256'][str(h.PRIOR/name)],digest)
        self.assertEqual(h.sha(h.CONTEXT),h.CONTEXT_SHA)
        self.assertEqual(h.read(h.CONTEXT)['arms'],list(h.NAMES))
        self.assertEqual(len(h.PREPARED_FILES),7)
        prior=h.read(h.PRIOR/'PLAN.json')
        self.assertEqual(prior['runtime'],h.ev.runtime())
        helper=h.ROOT/'work/monthly_sector_transition189_independent.py'
        self.assertEqual(prior['input_sha256'][str(helper)],
                         'e83305d9f784ef9aa5ae0554cd1deb558d927b4182010af599b977efa69618dc')
        self.assertEqual(h.sha(helper),prior['input_sha256'][str(helper)])


class NewAuditReceiptTests(unittest.TestCase):
    @staticmethod
    def feature_receipt(available=5978):
        counts = dict(events=122, symbols_per_event=49, checked_cells=5978,
            available_cells=available, unavailable_cells=5978-available,
            source_price_observations=506*available, relative_sign_observations=12*available,
            magnitude_rank_observations=12*available, magnitude_pair_comparisons=66*available,
            signed_rank_estimates=available, sign_balance_estimates=available,
            constant_magnitude_copies=0, predictive_model_fits=0)
        return dict(**{key: True for key in h.FEATURE_FLAGS}, **counts, counts=counts.copy(),
            computes_returns=False, portfolio_returns_calculated=False,
            producer_modules_imported=False, numerical_tolerance=0.,
            mapping_source_sha256=h.MAPPING_SHA)

    def test_feature_flags_counts_and_provenance_fail_closed(self):
        values = dict(available=np.ones((122,49),bool),candidate=np.zeros((122,49)),control=np.zeros((122,49)))
        check = self.feature_receipt()
        h.require_feature_audit(check,values)
        for key in h.FEATURE_FLAGS:
            with self.subTest(flag=key), self.assertRaises(ValueError):
                h.require_feature_audit(dict(check,**{key:False}),values)
        for key, value in [('events',121),('checked_cells',5978.),('available_cells',True),
                ('source_price_observations',506*5978-1),('magnitude_pair_comparisons',66*5978-1),
                ('magnitude_rank_observations',12*5978-1),('signed_rank_estimates',5977),
                ('predictive_model_fits',1),('constant_magnitude_copies',5979),
                ('mapping_source_sha256','changed'),('computes_returns',True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.require_feature_audit(dict(check,**{key:value}),values)

    def test_delta_flags_no_cross_rank_no_new_models_and_initial_zero(self):
        import monthly_sector_signedrank233_independent as independent
        _,_,values=fixture()
        _,deltas=h.scale_and_delta(values,6)
        check=independent.verify_deltas(values,deltas,dose=h.DOSE)
        h.require_delta_audit(check,deltas)
        for key in independent.DELTA_FLAGS:
            with self.subTest(flag=key), self.assertRaises(ValueError):
                h.require_delta_audit(dict(check,**{key:False}),deltas)
        wrong=copy.deepcopy(deltas);wrong[h.NAMES[1]][0,0]=.001
        with self.assertRaises(ValueError): independent.verify_deltas(values,wrong)

    def test_selection_requires_full_independent_scores(self):
        import monthly_sector_signedrank233_independent as independent
        check=dict(**{key:True for key in independent.SELECTION_FLAGS},
            ordered_choices_checked=366,full_scores_supplied=True,numerical_tolerance=0.,
            original_source_parity_tolerance=1e-12,maximum_absolute_delta=.01875,
            producer_modules_imported=False,computes_returns=False)
        h.require_selection_audit(check,122)
        for key in independent.SELECTION_FLAGS:
            with self.subTest(flag=key), self.assertRaises(ValueError):
                h.require_selection_audit(dict(check,**{key:False}),122)
        for key,value in [('full_scores_supplied',False),('ordered_choices_checked',365),
                          ('numerical_tolerance',1e-5),('computes_returns',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                h.require_selection_audit(dict(check,**{key:value}),122)


class ControlAndPreparedArtifactTests(unittest.TestCase):
    def test_control192_compares_old_candidate_and_orders_exactly(self):
        _, meta, values = fixture()
        orders={h.NAMES[2]:[dict(symbols=['A','B','C']) for _ in meta['dates']]}
        with tempfile.TemporaryDirectory() as directory:
            old=Path(directory)/'prepared';old.mkdir()
            (old/'META.json').write_text(json.dumps(meta))
            (old/'ORDERS.json').write_text(json.dumps({'sector_relative_recurrence':orders[h.NAMES[2]]}))
            np.savez(old/'FEATURES.npz',candidate=values['control'],control=values['candidate'],available=values['available'])
            with patch.object(h,'CONTROL192',Path(directory)),patch.object(h,'control192_binding',return_value={'prepared_completion_sha256':'fixed'}):
                check=h.control192_parity(meta,values,orders)
                self.assertTrue(check['control_values_exact']);self.assertTrue(check['control_orders_exact'])
                for kind in ('value','mask','metadata','orders'):
                    m,v,o=copy.deepcopy(meta),copy.deepcopy(values),copy.deepcopy(orders)
                    if kind=='value':v['control'][1,0]+=.00001
                    if kind=='mask':v['available'][1,0]=False
                    if kind=='metadata':m['dates'][0]='changed'
                    if kind=='orders':o[h.NAMES[2]][0]['symbols']=['D','B','C']
                    with self.subTest(kind=kind),self.assertRaises(ValueError):h.control192_parity(m,v,o)

    def test_prepared_receipts_require_prefixes_control_and_zero_cost_contract(self):
        import monthly_sector_signedrank233_independent as independent
        n,w=122,49
        meta=dict(dates=['d%d'%i for i in range(n)],ends=['e%d'%i for i in range(n)],
                  symbols=[['S%d'%j for j in range(w)] for _ in range(n)])
        values=dict(candidate=np.zeros((n,w)),control=np.zeros((n,w)),available=np.ones((n,w),bool))
        deltas={a:np.zeros((n,w)) for a in h.RAW}
        def feature(rows):
            counts=dict(events=rows,symbols_per_event=w,checked_cells=rows*w,available_cells=rows*w,
                unavailable_cells=0,source_price_observations=506*rows*w,relative_sign_observations=12*rows*w,
                magnitude_rank_observations=12*rows*w,magnitude_pair_comparisons=66*rows*w,
                signed_rank_estimates=rows*w,sign_balance_estimates=rows*w,constant_magnitude_copies=0,predictive_model_fits=0)
            return dict(**{k:True for k in h.FEATURE_FLAGS},**counts,counts=counts.copy(),
                numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,
                mapping_source_sha256=h.MAPPING_SHA)
        def selected(rows):
            return dict(**{k:True for k in independent.SELECTION_FLAGS},full_scores_supplied=True,
                ordered_choices_checked=3*rows,numerical_tolerance=0.,original_source_parity_tolerance=1e-12,
                maximum_absolute_delta=h.MAX_DELTA,producer_modules_imported=False,computes_returns=False)
        f=feature(n);counts=f['counts']
        control=dict(passed=True,metadata_exact=True,available_exact=True,control_values_exact=True,
                     control_orders_exact=True,numerical_tolerance=0.,checked_cells=n*w,available_cells=n*w)
        common=dict(original_array_count=27,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,
            new_model_fits=0,new_predictive_model_fits=0,model_counts=counts.copy(),**counts)
        c=dict(common,artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES},new_returns_computed=False)
        prefixes=[]
        for rows in h.FEATURE_PREFIXES:
            vv={k:v[:rows] for k,v in values.items()};dd={k:v[:rows] for k,v in deltas.items()}
            fc=feature(rows)
            prefixes.append(dict(prefix=rows,source_rows={'adj':rows,'sectors':rows},feature=fc,
                delta=independent.verify_deltas(vv,dd),selection=selected(rows),counts=fc['counts']))
        evidence=dict(common,feature=f,delta=independent.verify_deltas(values,deltas),selection=selected(n),
            original_orders_exact=True,original_arrays_unchanged=True,original_core_preserved=True,
            single_original_episode=True,feature_prefix_sources_truncated=True,selection_uses_zero_cost_paths_only=True,
            additional_max_score_budget=h.MAX_DELTA,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),
            physical_prefix_evidence=prefixes,control192=control)
        audit=dict(counts,event_history=[dict(date=meta['dates'][i],symbols=meta['symbols'][i]) for i in range(n)],
            price_count=253,skip_sessions=42,return_count=252,block_count=12,block_sessions=21,
            neutral_pseudo_blocks=4,candidate_denominator=208,control_denominator=16,expected_twice_rank_sum=156,
            common_availability=True,within_cell_ranking_performed=True,portfolio_returns_calculated=False,
            cross_sectional_ranking_performed=False,trading_eligibility_modified=False)
        parity=dict(calendar_sha256='calendar',original_arrays_unchanged=True,original_orders_exact=True,strict_pit=False,
            original_array_fingerprints={str(i):{} for i in range(27)},control192=control,
            execution_prices={k:True for k in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact')})
        with patch.object(h,'sha',return_value='calendar'):
            h.verify_prepared_evidence(c,values,deltas,evidence,meta=meta,parity=parity,feature_audit=audit)
            for kind in ('manifest','prefixes','physical','zero_cost','core','control','counts','denominator','array_count','return_scope'):
                cc,ee,pp,aa=map(copy.deepcopy,(c,evidence,parity,audit))
                if kind=='manifest':cc['artifacts_sha256'].pop('FEATURES.npz')
                if kind=='prefixes':ee['prefixes']=122
                if kind=='physical':ee['physical_prefix_evidence'][0]['source_rows']['adj']=1
                if kind=='zero_cost':ee['selection_uses_zero_cost_paths_only']=False
                if kind=='core':ee['original_core_preserved']=False
                if kind=='control':pp['control192']['control_values_exact']=False
                if kind=='counts':cc['relative_sign_observations']-=1
                if kind=='denominator':aa['candidate_denominator']=207
                if kind=='array_count':cc['original_array_count']=26
                if kind=='return_scope':cc['new_returns_computed']=True
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    h.verify_prepared_evidence(cc,values,deltas,ee,meta=meta,parity=pp,feature_audit=aa)


class ZeroCostGovernanceTests(unittest.TestCase):
    def test_cost_path_cannot_change_training_choice(self):
        metrics, orders = selection_fixture()
        expected = h.select_first88(metrics, orders)
        for arm in h.NAMES:
            metrics[arm]['next_open_cost'] = {'training': {'cagr': -1. if arm == h.NAMES[1] else 100.}}
        self.assertEqual(h.select_first88(metrics, orders), expected)

    def test_candidate_allows_signed_magnitude_grid(self):
        _, _, values = fixture()
        values['candidate'][1] = np.array([-155, -97, 0, 13, 156]) / 208.
        transforms, delta = h.scale_and_delta(values, 2)
        np.testing.assert_array_equal(transforms[h.NAMES[1]][1], values['candidate'][1])
        np.testing.assert_array_equal(delta[h.NAMES[1]][1], .025 * values['candidate'][1])

    def test_context_and_original_score_budget_are_explicit(self):
        contract = h.contract()
        self.assertEqual(contract['prospective_context_sha256'], h.CONTEXT_SHA)
        self.assertEqual(contract['dose'], .025)
        self.assertEqual(contract['max_absolute_raw'], .75)
        self.assertEqual(contract['control_nonpromotable'], h.NAMES[2])
        context = contract['context']
        self.assertEqual(context['formation']['parameters']['doubled_rank_denominator'], 208)
        self.assertEqual(context['formation']['parameters']['control_denominator'], 16)
        self.assertFalse(context['formation']['parameter_search'])
        self.assertIn('costpathdiagnostic only', context['evaluation']['gates'])
        self.assertIn('additional bounded auxiliary adjustment', context['integration']['formula'])
        self.assertIn(h.CONTEXT, h.code_files())
        with patch.object(h, 'sha', return_value='changed'):
            with self.assertRaises(ValueError): h.contract()

    def test_mutable_central_is_not_a_persistent_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            central = root/'CENTRAL.json'; central.write_text('{}')
            with patch.multiple(h, ROOT=root, CENTRAL=central):
                with self.assertRaises(ValueError): h.verify({str(central): h.sha(central)})


if __name__ == '__main__':
    if sys.argv[1:] == ['--all-synthetic']:
        # Deliberately bypass host.tests(): this dry run publishes no receipt.
        modules=[h.module('dry_sectorsignedrank233_'+str(i),h.ROOT/'work'/name)
                 for i,name in enumerate(('test_monthly_sector_signedrank233_host.py',
                    'test_monthly_sector_signedrank233_kernel.py',
                    'test_monthly_sector_signedrank233_independent.py'))]
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
        result=unittest.TextTestRunner(verbosity=1).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main()
