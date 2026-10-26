"""Synthetic-only host wiring and receipt gates; no actual cached inputs or fits."""
from pathlib import Path
from types import SimpleNamespace
import copy
import ast
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


P = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tested_increment_calibration224_host', P/'monthly_increment_calibration224_host.py')
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def rank(raw, use):
    result = np.zeros(raw.shape, float)
    for i in range(len(raw)):
        values = pd.Series(raw[i, use[i]])
        result[i, use[i]] = values.rank(method='average', pct=True).to_numpy()-.5
    return result


def fixture():
    shape = (6, 6)
    source = np.tile(np.array([0., 2., 1., 5., 3., 4.]), (shape[0], 1))
    old = np.tile(np.array([1., 0., 5., 3., 2., 4.]), (shape[0], 1))
    use = np.ones(shape, bool); use[:, -1] = False
    a = dict(baseline=np.tile([.25, .02, -.1, .13, -.05, .07], (6, 1)),
        eligible=use, forecast_available=np.ones(shape, bool), model_available=np.ones(shape, bool),
        raw_source_log=source, raw_local_linear=old, raw_local_constant=old.copy(),
        rank_log_excess=rank(source, use),
        state_weak_narrow=np.array([True, True, False, False, False, True]),
        state_calm10=np.array([False, True, True, False, True, False]),
        extension_available=np.ones(shape, bool), effects=np.zeros((6, 6, 2)))
    a['extension_available'][2, 1] = False
    a['extension_available'][4] = False
    f = dict(raw=dict(candidate=np.tile(np.arange(6., 0., -1.), (6, 1)).reshape(-1),
                      control=((source+old)/2.).reshape(-1)), available=np.ones(shape, bool).reshape(-1))
    meta = dict(dates=['synthetic-%d' % i for i in range(6)], symbols=[list('ABCDEF') for _ in range(6)])
    return a, meta, f


class SyntheticStudy:
    """A deterministic ranker/selector double with observable adapter calls."""
    ARMS = ('local_constant', 'local_linear')

    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        frozen=h.module('increment_calibration224_synthetic_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
        self.adapter = SimpleNamespace(score_and_choose=Mock(side_effect=self.choose_adapter),_state=frozen._state,_matrix=frozen._matrix)
        self.rp = SimpleNamespace(make_orders=self.make_orders)

    @staticmethod
    def configs():
        return [dict(id=i) for i in range(5)]

    @staticmethod
    def original_rank_inputs(a, n):
        return a['rank_log_excess'][:n].copy(), a['forecast_available'][:n] & a['eligible'][:n]

    def ranked_forecasts(self, values, available, a, n):
        use = self.original_rank_inputs(a, n)[1]
        result = {key:a['rank_log_excess'][:n].copy() for key in self.ARMS}
        ready = np.array([bool(use[i].any() and available[i, use[i]].all() and
            all(np.isfinite(values[key][i, use[i]]).all() for key in self.ARMS)) for i in range(n)], bool)
        for key in self.ARMS:
            if np.isinf(values[key][:n]).any():
                raise FloatingPointError('Infinite synthetic forecast')
            for i in np.flatnonzero(ready):
                result[key][i, use[i]] = pd.Series(values[key][i, use[i]]).rank(
                    method='average', pct=True).to_numpy()-.5
        return result, ready

    def modified_pre_score(self, spec, a, n):
        assert spec == dict(id=4)
        old = rank(a['raw_local_linear'], a['eligible'] & a['forecast_available'])
        coef = .25*(.05*a['state_weak_narrow'][:, None]+.01*
            (a['state_calm10'] & ~a['state_weak_narrow'])[:, None]*a['extension_available'])
        # Existing global-model contribution is deliberately large and distinct.
        return a['baseline']+.07*a['rank_log_excess']+coef*(old-a['rank_log_excess'])

    @staticmethod
    def boundary(pre, a, meta, n):
        result = []
        for i in range(n):
            ids = np.flatnonzero(a['eligible'][i])
            ids = sorted(ids, key=lambda j:(-pre[i, j], j))
            chosen = ids[:3] if len(ids) >= 2 else []
            result.append([meta['symbols'][i][j] for j in chosen])
        return result, pre[:n].copy()

    def choices(self, spec, a, meta, n, *, policy):
        return self.boundary(self.modified_pre_score(spec, a, n), a, meta, n)

    @staticmethod
    def make_orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=ss, scores=scores[i].tolist())
                for i, ss in enumerate(choices)]

    def choose_adapter(self, base, weight, residual, a, meta, n, *, pre_score,
                       effects, episode_policy, boundary_selector):
        assert base == 'incumbent66' and weight == 0.
        assert residual.shape == a['baseline'].shape and np.isnan(residual).all()
        assert effects is a['effects']
        assert boundary_selector is self.ep.select_boundary
        return boundary_selector(pre_score, a, meta, n)


class RankSlotTests(unittest.TestCase):
    def test_exact_regime_coefficients_and_partial_first_event(self):
        a, meta, forecast = fixture(); s = SyntheticStudy()
        ranks, deltas = h.rank_and_delta(s, a, forecast, 6)
        use = a['eligible'] & a['forecast_available']; old = rank(a['raw_local_linear'], use)
        coefficient = np.zeros(a['baseline'].shape)
        coefficient[1, :] = .0125
        coefficient[2, :] = .0025; coefficient[2, 1] = 0.
        coefficient[5, :] = .0125
        for name in h.NAMES[1:]:
            want_rank = rank(forecast['raw'][h.RAW[name]].reshape(a['baseline'].shape), use)
            np.testing.assert_array_equal(ranks[name], want_rank)
            want_delta = coefficient*(want_rank-old); want_delta[~use] = 0.
            np.testing.assert_array_equal(deltas[name], want_delta)
            np.testing.assert_array_equal(deltas[name][0], np.zeros(6))
            np.testing.assert_array_equal(deltas[name][3:5], np.zeros((2, 6)))
        self.assertTrue(np.any(deltas[h.NAMES[1]][1] != 0.))
        self.assertTrue(np.any(deltas[h.NAMES[1]][2] != 0.))

    def test_ineligible_and_unavailable_rows_do_not_gain_delta(self):
        a, meta, forecast = fixture(); s = SyntheticStudy()
        forecast['available'].reshape(6, 6)[1, 0] = False
        forecast['raw']['candidate'].reshape(6, 6)[5, 1] = np.nan
        _, deltas = h.rank_and_delta(s, a, forecast, 6)
        for name in h.NAMES[1:]:
            np.testing.assert_array_equal(deltas[name][1], np.zeros(6))
            np.testing.assert_array_equal(deltas[name][:, -1], np.zeros(6))
        for name in h.NAMES[1:]:np.testing.assert_array_equal(deltas[name][5], np.zeros(6))

    def test_identical_old_raw_preserves_every_pre_score_and_order(self):
        a, meta, forecast = fixture(); s = SyntheticStudy()
        for key in forecast['raw']:
            forecast['raw'][key] = a['raw_local_linear'].reshape(-1).copy()
        before = s.modified_pre_score(s.configs()[4], a, 6)
        policy = dict(synthetic=True)
        with patch.object(h, 'read', return_value=policy):
            orders, ranks, deltas = h.decisions(s, a, meta, forecast, 6)
        for name in h.NAMES[1:]:
            np.testing.assert_array_equal(deltas[name], np.zeros((6, 6)))
            self.assertEqual(orders[name], orders[h.NAMES[0]])
        for call in s.adapter.score_and_choose.call_args_list:
            self.assertEqual(call.kwargs['pre_score'].tobytes(), before.tobytes())

    def test_replaces_local_quarter_preserves_global_and_runs_episode_once(self):
        a, meta, forecast = fixture(); s = SyntheticStudy(); policy = dict(synthetic=True)
        original_pre = s.modified_pre_score(s.configs()[4], a, 6)
        with patch.object(h, 'read', return_value=policy) as reader:
            orders, ranks, deltas = h.decisions(s, a, meta, forecast, 6)
        reader.assert_called_once_with(h.LOCAL/'INCUMBENT_POLICY.json')
        self.assertEqual(s.adapter.score_and_choose.call_count, 2)
        self.assertEqual(s.ep.select_boundary.call_count, 2)
        for name, call in zip(h.NAMES[1:], s.adapter.score_and_choose.call_args_list):
            expected = original_pre.copy(); changed = deltas[name] != 0.
            expected[changed] += deltas[name][changed]
            self.assertEqual(call.kwargs['pre_score'].tobytes(), expected.tobytes())
            np.testing.assert_array_equal(call.kwargs['effects'], a['effects'])
            self.assertIs(call.kwargs['episode_policy'], policy)
            # A whole-model replacement would move the large .07 global slot;
            # this test instead verifies only the existing local-quarter delta.
            np.testing.assert_array_equal(call.kwargs['pre_score'][0], original_pre[0])
        self.assertEqual(set(orders), set(h.NAMES))

    def test_every_small_prefix_and_poisoned_future_and_no_input_mutation(self):
        a, meta, forecast = fixture(); before_a = copy.deepcopy(a); before_f = copy.deepcopy(forecast)
        with patch.object(h, 'read', return_value={'synthetic':True}):
            all_orders, all_ranks, all_deltas = h.decisions(SyntheticStudy(), a, meta, forecast, 6)
            for n in range(7):
                orders, ranks, deltas = h.decisions(SyntheticStudy(), a, meta, forecast, n)
                for name in h.NAMES:
                    self.assertEqual(orders[name], all_orders[name][:n])
                for name in h.NAMES[1:]:
                    np.testing.assert_array_equal(ranks[name], all_ranks[name][:n])
                    np.testing.assert_array_equal(deltas[name], all_deltas[name][:n])
            for n in (1, 3, 5):
                poison = copy.deepcopy(forecast)
                for values in poison['raw'].values():
                    values.reshape(6, 6)[n:] = np.inf
                orders, ranks, deltas = h.decisions(SyntheticStudy(), a, meta, poison, n)
                for name in h.NAMES:
                    self.assertEqual(orders[name], all_orders[name][:n])
        for key, value in a.items():
            np.testing.assert_array_equal(value, before_a[key])
        for key, value in forecast['raw'].items():
            np.testing.assert_array_equal(value, before_f['raw'][key])
        np.testing.assert_array_equal(forecast['available'], before_f['available'])


def selection_fixture():
    metric = {era:dict(cagr=.5, sharpe=1.5, max_drawdown=-.3) for era in ('training','early','recent')}
    metrics = {name:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')}
               for name in h.NAMES}
    orders = {name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in h.NAMES}
    for path in metrics[h.NAMES[1]]:
        metrics[h.NAMES[1]][path]['training']['cagr'] += .01
    for i in (1, 2, 53):
        orders[h.NAMES[1]][i]['symbols'] = ['A','B','D']
    return metrics, orders



class HistoricalUniverseAndSourceTests(unittest.TestCase):
    def test_original_history_mask_uses_original_forecast_and_eligible_only(self):
        a,_,_=fixture();a['forecast_available'][1,0]=False;a['eligible'][1,1]=False
        a['model_available'][1,2]=False;a['local_available']=np.zeros(a['eligible'].shape,bool)
        expected=a['forecast_available']&a['eligible']
        np.testing.assert_array_equal(h.original_history_use(a,6),expected)
        self.assertTrue(h.original_history_use(a,2)[1,2])
        for n in (0,1,3):
            aa=copy.deepcopy(a)
            for key in ('forecast_available','eligible'):
                aa[key]=aa[key].astype(object);aa[key][n:]='future poison'
            np.testing.assert_array_equal(h.original_history_use(aa,n),expected[:n])
    def test_exact_cached_local_linear_source_cannot_be_substituted_with_constant(self):
        a,_,_=fixture();a['raw_local_constant']+=.3;a['local_available']=np.ones(a['eligible'].shape,bool)
        source={key:np.zeros(a['baseline'].size) for key in h.SOURCE_KEYS}
        for key in ('source_log','local_linear','local_constant'):source['raw_'+key]=a['raw_'+key].reshape(-1).copy()
        source['raw_control']=source['raw_local_constant'].copy()
        source['available']=a['model_available'].reshape(-1).copy();source['local_available']=a['local_available'].reshape(-1).copy()
        self.assertEqual(h.require_cached_source(source,a)['error_prediction_source'],'raw_local_linear')
        bad=copy.deepcopy(source);bad['raw_local_linear']=source['raw_control'].copy()
        with self.assertRaises(AssertionError):h.require_cached_source(bad,a)
        bad=copy.deepcopy(source);bad['available']=bad['available'].astype(int)
        with self.assertRaises(ValueError):h.require_cached_source(bad,a)
        bad=copy.deepcopy(source);bad['raw_local_linear']=bad['raw_local_linear'][:-1]
        with self.assertRaises(ValueError):h.require_cached_source(bad,a)

class BindingAndSelectionTests(unittest.TestCase):
    def test_bind_adds_or_accepts_exact_digest_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic-source'; path.write_bytes(b'synthetic only\n')
            pins = {}; h.bind(pins, path)
            self.assertEqual(pins, {str(path):h.sha(path)})
            h.bind(pins, path)
            old = dict(pins); path.write_bytes(b'changed synthetic\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting inherited'):
                h.bind(pins, path)
            self.assertEqual(pins, old)

    def test_selection_ignores_later_orders_and_extra_report_only_metrics(self):
        metrics, orders = selection_fixture()
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['later34'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
            metrics[name]['full122'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
            for i in range(88, 122):
                orders[name][i]['symbols'] = [name+str(i)]
        self.assertEqual(h.select_first88(metrics, orders), result)

    def test_pooled_increment_control_cannot_be_promoted_and_candidate_clears_both_references(self):
        metrics, orders = selection_fixture()
        metrics[h.NAMES[2]]['same_close_zero']['training']['cagr'] = 2.
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['close_vs_pooled_increment_control'])
        metrics, orders = selection_fixture()
        metrics[h.NAMES[2]]['next_open_cost']['training']['cagr'] = 2.
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['open_vs_pooled_increment_control'])

    def test_first88_changed_set_support_cannot_come_from_later34(self):
        metrics, orders = selection_fixture()
        for i in (1, 2, 53):
            orders[h.NAMES[1]][i]['symbols'] = ['C','B','A']
        for i in (90, 100, 110):
            orders[h.NAMES[1]][i]['symbols'] = ['A','B','D']
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['changed_months'], [])
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['changed_support'])




class PrefixAndContractTests(unittest.TestCase):
    def test_paired_missing_returns_incumbent_ranks_on_both_arms(self):
        a,m,f=fixture();s=SyntheticStudy();f['raw']['candidate'][7]=np.nan
        ranks,deltas=h.rank_and_delta(s,a,f,6);old=rank(a['raw_local_linear'],a['eligible']&a['forecast_available'])
        for arm in h.RAW:
            np.testing.assert_array_equal(ranks[arm][1],old[1]);np.testing.assert_array_equal(deltas[arm][1],np.zeros(6))
    def test_future_object_state_and_forecasts_are_not_consumed(self):
        a,m,f=fixture();expected=h.rank_and_delta(SyntheticStudy(),a,f,6)
        for n in range(7):
            aa,ff=copy.deepcopy(a),copy.deepcopy(f)
            for key in ('eligible','forecast_available','model_available','extension_available','state_calm10','state_weak_narrow'):
                aa[key]=aa[key].astype(object);aa[key][n:]='future poison'
            for key in ff['raw']:
                ff['raw'][key]=ff['raw'][key].astype(object);ff['raw'][key][n*6:]='future poison'
            got=h.rank_and_delta(SyntheticStudy(),aa,ff,n)
            for i in (0,1):
                for arm in h.RAW:np.testing.assert_array_equal(got[i][arm],expected[i][arm][:n])
    def test_nonboolean_consumed_states_fail(self):
        for key in ('eligible','forecast_available','model_available','extension_available','state_calm10','state_weak_narrow'):
            a,m,f=fixture();a[key]=a[key].astype(object);a[key][1]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.rank_and_delta(SyntheticStudy(),a,f,2)
    def test_physical_prefix_removes_future_labels_and_preserves_all_prior_queries(self):
        dates=np.array(['2018-11-30']*2+['2018-12-31']*2+['2019-11-29']*2,dtype='datetime64[D]')
        ends=np.array(['2018-12-31']*2+['2019-01-31']*2+['2019-12-31']*2,dtype='datetime64[D]')
        tm=dict(dates=['2018-11-30','2018-12-31','2019-11-29'],ends=['2018-12-31','2019-01-31','2019-12-31'],symbols=[['A','B']]*3,feature_names=['f']*10)
        m=dict(dates=['2019-01-31','2019-02-28','2020-01-31'],ends=['2019-02-28','2019-03-29','2020-02-28'],symbols=[['A','B']]*3)
        train=dict(dates=dates,ends=ends,offsets=np.array([0,2,4,6]),train_plain=np.zeros((6,10),object),common_good=np.ones(6,bool))
        train.update({'target_'+key:np.zeros(6,object) for key in h.TARGETS})
        source={key:np.arange(6,dtype=object) for key in h.SOURCE_KEYS};eligible=np.ones((3,2),object)
        train['train_plain'][2:]='future poison'
        for key in h.TARGETS:train['target_'+key][2:]='future poison'
        for key in source:source[key][4:]='future poison'
        eligible[2:]='future poison'
        got=h.prefix_inputs(train,tm,m,source,eligible,2)
        self.assertEqual(got[0]['dates'].shape,(2,));self.assertEqual(got[0]['offsets'].tolist(),[0,2])
        self.assertEqual(got[1]['dates'],tm['dates'][:1]);self.assertEqual(got[2]['dates'],m['dates'][:2])
        for key in source:self.assertEqual(got[3][key].tolist(),[0,1,2,3])
        self.assertTrue(np.isfinite(got[0]['train_plain'].astype(float)).all())
        empty=h.prefix_inputs(train,tm,m,source,eligible,0)
        self.assertEqual(empty[0]['dates'].shape,(0,));self.assertEqual(empty[0]['offsets'].tolist(),[0])
        self.assertEqual(empty[1]['dates'],[]);self.assertEqual(empty[2]['dates'],[])
        self.assertEqual(empty[4].shape,(0,2));self.assertTrue(all(len(v)==0 for v in empty[3].values()))
        bad=copy.deepcopy(train);bad['offsets'][1]=3
        with self.assertRaisesRegex(ValueError,'offsets'):h.prefix_inputs(bad,tm,m,source,eligible,1)
        bad=copy.deepcopy(train);bad['dates'][0]=np.datetime64('2018-11-29')
        with self.assertRaisesRegex(ValueError,'flat date identity'):h.prefix_inputs(bad,tm,m,source,eligible,1)
    def test_candidate_must_change_vs_control_in_both_training_eras(self):
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        selection=h.select_first88(metrics,orders)
        self.assertFalse(selection['admitted']);self.assertFalse(selection['guards']['changed_support_vs_control'])
    def test_exact_38_results_manifest(self):
        good={name:'synthetic' for name in h.RESULT_FILES};self.assertEqual(len(good),38);h.require_result_manifest(good)
        for kind in ('missing','extra','completion'):
            bad=dict(good)
            if kind=='missing':bad.pop(next(iter(bad)))
            else:bad['COMPLETION.json' if kind=='completion' else 'EXTRA.json']='bad'
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.require_result_manifest(bad)
    def test_only_exact_registered_external_runtime_allowed(self):
        runtime,digest=next(iter(h.EXTERNAL_RUNTIME.items()))
        with patch.object(h,'sha',return_value=digest),patch.object(Path,'is_file',return_value=True):
            h.verify({runtime:digest})
            for path,value in ((runtime,'wrong'),('/Library/Developer/CommandLineTools/other.py',digest),('/tmp/other.py',digest)):
                with self.subTest(path=path,value=value),self.assertRaises(ValueError):h.verify({path:value})
    def test_mutable_central_cannot_be_persistent_pin(self):
        with patch.object(h,'sha',return_value='synthetic'):
            with self.assertRaisesRegex(ValueError,'Mutable central'):h.verify({str(h.CENTRAL):'synthetic'})
    def test_test_receipt_is_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt=Path(directory)/'receipt.json';receipt.write_text('keep')
            empty=SimpleNamespace()
            with patch.object(h,'TEST_RECEIPT',receipt),patch.object(h,'module',return_value=empty):
                with self.assertRaisesRegex(ValueError,'write-once'):h.tests()
            self.assertEqual(receipt.read_text(),'keep')
    def test_central_compare_and_swap_rejects_concurrent_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';path.write_bytes(b'changed')
            with patch.object(h,'CENTRAL',path):
                with self.assertRaisesRegex(ValueError,'Concurrent central'):h.commit_state(b'prior',{},'test')
            self.assertEqual(path.read_bytes(),b'changed')


class ReceiptTests(unittest.TestCase):
    @staticmethod
    def model_receipt(args=None):
        import monthly_increment_calibration224_model as model
        import monthly_increment_calibration224_independent as oracle
        from test_monthly_increment_calibration224_model import fixture as model_fixture
        args=model_fixture() if args is None else args
        forecast=model.build_forecasts(*args)
        return oracle.check_forecasts(*args,forecast),forecast

    @staticmethod
    def accept(receipt,forecast):
        # Fixed external fixture axes prevent corrupt receipt/audit fields
        # from supplying their own expected coverage.
        h.require_model_audit(receipt,forecast,expected_rows=60,expected_years=3)

    def test_model_flags_are_exact_boolean_and_counts_bound(self):
        receipt,forecast=self.model_receipt();self.accept(receipt,forecast)
        self.assertEqual(len(h.MODEL_FLAGS),23);self.assertEqual(len(h.MODEL_COUNT_FIELDS),19)
        for key in h.MODEL_FLAGS:
            for value in (False,1,None):
                bad=copy.deepcopy(receipt);bad[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.accept(bad,forecast)
        for key in h.MODEL_COUNT_FIELDS:
            bad=copy.deepcopy(receipt);bad['counts'][key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):self.accept(bad,forecast)
    def test_model_schema_counts_and_scope_not_weakened(self):
        receipt,forecast=self.model_receipt()
        for key,value in (('computes_returns',True),('producer_modules_imported',True),('atol',1e-12),('annual_cohorts',10),('query_rows',3),('new_ridge_fits',1)):
            bad=dict(receipt);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.accept(bad,forecast)
        bad=copy.deepcopy(forecast);bad['audit']['counts'].pop('parameter_estimates')
        with self.assertRaises(ValueError):self.accept(receipt,bad)
    def test_pooled_and_own_parameters_count_even_at_exact_bounds(self):
        from test_monthly_increment_calibration224_model import fixture as model_fixture
        for slope in (0.,1.,-1.,2.):
            args=model_fixture()
            args[0]['target_log_excess'][:54]=slope*args[3]['raw_local_linear'][:54]
            receipt,forecast=self.model_receipt(args)
            counts=receipt['counts']
            self.assertEqual(counts['pooled_fits'],1)
            self.assertEqual(counts['own_fits'],3)
            self.assertEqual(counts['parameter_estimates'],4)
            self.assertEqual(counts['clipped_pooled_fits'],int(slope<0 or slope>1))
            self.assertEqual(counts['clipped_own_fits'],3*int(slope<0 or slope>1))
            self.accept(receipt,forecast)
            for key in ('new_model_fits','predictive_model_fits'):
                bad=copy.deepcopy(receipt);bad[key]=0
                with self.subTest(slope=slope,key=key),self.assertRaises(ValueError):self.accept(bad,forecast)
            bad=copy.deepcopy(forecast);bad['audit']['annual'][-1]['groups'][0]['own_months']=17
            with self.assertRaises(ValueError):self.accept(receipt,bad)

    def test_status_fallback_support_and_prediction_coverage_are_enforced(self):
        receipt,forecast=self.model_receipt()
        for mutate in (
            lambda a:a['annual'][-1].update(status='cold_start'),
            lambda a:a['annual'][-1].update(clipped=True),
            lambda a:a['annual'][-1].update(alpha=.123),
            lambda a:a['annual'][-1]['groups'][0].update(applied_alpha=.123),
            lambda a:a['annual'][-1]['groups'][0].update(fallback='pooled'),
            lambda a:a['annual'][-1]['groups'][0]['history_source_row_ids'].__setitem__(0,1),
            lambda a:a['annual'][-1]['groups'][0]['query_row_ids'].__setitem__(0,55),
            lambda a:a['annual'][-1]['months'][0].update(retained=False)):
            bad=copy.deepcopy(forecast);mutate(bad['audit'])
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.accept(receipt,bad)
        bad=copy.deepcopy(forecast);bad['available'][0]=False
        with self.assertRaises(ValueError):self.accept(receipt,bad)

    def test_pool_zero_denominator_and_own_fallback_receipts_are_real(self):
        from test_monthly_increment_calibration224_model import fixture as model_fixture
        args=model_fixture();args[3]['raw_source_log'][:54]=args[3]['raw_local_linear'][:54]
        receipt,forecast=self.model_receipt(args)
        self.assertEqual(receipt['counts']['zero_denominator_years'],1)
        self.assertEqual(receipt['new_model_fits'],0)
        self.accept(receipt,forecast)
        args=model_fixture();args[4][0,0]=False
        ids=np.arange(1,54,3);args[3]['raw_source_log'][ids]=args[3]['raw_local_linear'][ids]
        receipt,forecast=self.model_receipt(args)
        self.assertEqual(receipt['counts']['zero_denominator_groups'],1)
        self.assertEqual(receipt['counts']['own_fits'],1)
        self.assertEqual(receipt['new_model_fits'],2)
        self.accept(receipt,forecast)

    def test_small_month_rows_reject_false_support_and_present_products(self):
        from test_monthly_increment_calibration224_model import fixture as model_fixture
        args=model_fixture();args[4][0,1:]=False
        receipt,forecast=self.model_receipt(args)
        self.assertGreater(receipt['counts']['small_month_rows'],0)
        self.accept(receipt,forecast)
        for key in ('numerator_sum','denominator_sum','numerator_mean','denominator_mean'):
            bad=copy.deepcopy(forecast)
            month=next(m for a in bad['audit']['annual'] for m in a['months'] if not m['retained'])
            month[key]=0.
            with self.subTest(key=key),self.assertRaises(ValueError):self.accept(receipt,bad)
        bad=copy.deepcopy(forecast)
        month=next(m for a in bad['audit']['annual'] for m in a['months'] if not m['retained'])
        month['rows'][0]['product']=0.
        with self.assertRaises(ValueError):self.accept(receipt,bad)

    def test_audit_arithmetic_rejects_matching_but_false_independent_counts(self):
        receipt,forecast=self.model_receipt()
        for field in h.MODEL_COUNT_FIELDS:
            ff=copy.deepcopy(forecast);rr=copy.deepcopy(receipt)
            ff['audit']['counts'][field]+=1;rr['counts'][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):self.accept(rr,ff)
        for field in ('source_forecasts_refitted','labels_recomputed','local_available_filtered','historical_use_explicit','current_availability_unchanged','pool_uses_all_historical_ids','monthly_equal_pool_weights'):
            ff=copy.deepcopy(forecast);ff['audit'][field]=not ff['audit'][field]
            with self.subTest(field=field),self.assertRaises(ValueError):self.accept(receipt,ff)
    def test_slot_requires_pair_flags_matching_ready_and_changes(self):
        delta={key:np.zeros((122,49)) for key in h.RAW}
        receipt=dict.fromkeys(h.SLOT_FLAGS,True)
        receipt.update(ready_rows={key:122 for key in h.RAW},changed_cells={key:0 for key in h.RAW})
        h.require_slot_audit(receipt,delta)
        for key in h.SLOT_FLAGS:
            bad=copy.deepcopy(receipt);bad[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_slot_audit(bad,delta)
        bad=copy.deepcopy(receipt);bad['ready_rows'][h.NAMES[1]]=121
        with self.assertRaises(ValueError):h.require_slot_audit(bad,delta)
    def test_source_review_rejects_bad_context_or_loss_computation_before_cache_reads(self):
        evidence={'work/fake':'pin'}
        review=dict(passed=True,context_sha256=h.CONTEXT_SHA,evidence_sha256=evidence,
            cached_forecasts_are_prior_annual_asof_replays=True,not_realtime_archival_vintages=True,
            numeric_calibration_coverage_not_yet_computed=True,numeric_own_security_support_not_yet_computed=True,source_bytes_reverified=True,
            current_and_historical_prediction_key='raw_local_linear',disagreement_prediction_key='raw_source_log',own_history_mask='originalU AND206available',uses_fitted_residuals_as_historical_errors=False,
            source_npz_decoded=False,observers_modified=False,actual_new_losses=0,actual_new_coefficients=0,actual_new_parameters=0,actual_new_calibration_coefficients=0,
            actual_new_orders=0,actual_new_returns=0,matched_formal_months=113,source_annual=[{}]*11)
        def reader(path):
            if path==h.SOURCE_REVIEW:return review
            if path==h.CONTEXT:return {'evidence_sha256':evidence}
            raise AssertionError('No cache/provenance read should precede source review rejection')
        with patch.object(h,'sha',return_value=h.SOURCE_REVIEW_SHA),patch.object(h,'read',side_effect=reader):
            for key,value in (('context_sha256','wrong'),('actual_new_losses',1),('actual_new_coefficients',True),
                              ('source_npz_decoded',True),('uses_fitted_residuals_as_historical_errors',True),
                              ('actual_new_parameters',1),('actual_new_calibration_coefficients',1),('current_and_historical_prediction_key','raw_control'),('disagreement_prediction_key','raw_control'),
                              ('own_history_mask','eligible AND available')):
                old=review[key];review[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):h.model_source_binding({})
                review[key]=old
    def test_first88_freeze_binds_all_prerequisites(self):
        metrics,orders=selection_fixture();selection=h.select_first88(metrics,orders)
        frozen=dict(plan_sha256='plan',orders_sha256='digest',training_sha256='digest',prepared_sha256='digest',selection=selection)
        def read(path):return frozen if path.name=='FROZEN.json' else metrics
        with patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=read):
            self.assertEqual(h.frozen_selection('plan',orders),frozen)
            frozen['prepared_sha256']='changed'
            with self.assertRaises(ValueError):h.frozen_selection('plan',orders)


class PhaseGateTests(unittest.TestCase):
    def test_prepare_gate_precedes_model_or_data_loading(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
            loader.assert_not_called()
    def test_prepared_manifest_is_exact_before_opening_arrays(self):
        receipt=dict(status='COMPLETE',plan_sha256='plan',artifacts_sha256={name:'digest' for name in h.PREPARED_FILES})
        for change in ('missing','extra'):
            bad=copy.deepcopy(receipt)
            if change=='missing':bad['artifacts_sha256'].pop(next(iter(bad['artifacts_sha256'])))
            else:bad['artifacts_sha256']['COMPLETION.json']='extra'
            with patch.object(h,'gate'),patch.object(h,'read',return_value=bad),patch.object(h.ev,'load_arrays') as loader:
                with self.subTest(change=change),self.assertRaises(ValueError):h.prepared('plan')
                loader.assert_not_called()
    def test_current_counters_and_write_once_protection_order(self):
        rows=[dict(trials=1,directory='old') for _ in range(221)];rows[0]['trials']=960;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
        text=Path(h.__file__).read_text();tree=ast.parse(text)
        parts={x.name:ast.get_source_segment(text,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        self.assertIn("len(state['completed']) == 222",parts['close']);self.assertIn('== 1183',parts['close'])
        self.assertLess(parts['register'].index('model_source_binding(raw)'),parts['register'].index('OUT.mkdir()'))
        self.assertLess(parts['register'].index('code_review_binding()'),parts['register'].index('OUT.mkdir()'))
        self.assertLess(parts['register'].index('len(protected)==21444'),parts['register'].index('OUT.mkdir()'))
        self.assertEqual(parts['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn("not (OUT/'prepared').exists()",parts['prepare'])
        self.assertIn("not (OUT / 'selected').exists()",parts['select'])
        self.assertIn('Selection escaped first88 cutoff',parts['select'])
        self.assertIn('require_result_manifest(',parts['close'])
        self.assertLess(parts['close'].index('require_result_manifest('),parts['close'].index('require_outcome_audit('))
    def test_code_review_requires_no_actual_fits_and_nonempty_immutable_inputs(self):
        files=[h.ROOT/'work'/('synthetic'+str(i)) for i in range(6)]
        receipt=dict(passed=True,unresolved_blockers=[],context_sha256=h.CONTEXT_SHA,code_sha256={str(p.relative_to(h.ROOT)):'digest' for p in files},input_sha256={'work/fake':'digest'})
        for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):receipt[key]=False
        with patch.object(h,'code_files',return_value=files),patch.object(h,'read',return_value=receipt),patch.object(h,'sha',return_value='digest'),patch.object(h,'bind'),patch.object(h,'verify'):
            h.code_review_binding();receipt['context_sha256']='wrong'
            with self.assertRaises(ValueError):h.code_review_binding()
            receipt['context_sha256']=h.CONTEXT_SHA;receipt['actual_model_fits_computed']=True
            with self.assertRaises(ValueError):h.code_review_binding()
            receipt['actual_model_fits_computed']=False;receipt['input_sha256']={}
            with self.assertRaises(ValueError):h.code_review_binding()


class ProducerInterfaceTests(unittest.TestCase):
    def test_real_synthetic_model_receipt_and_physical_prefix_contract(self):
        import monthly_increment_calibration224_model as model
        import monthly_increment_calibration224_independent as oracle
        from test_monthly_increment_calibration224_model import fixture as model_fixture
        args=model_fixture();n=len(args[2]['dates']);width=args[4].shape[1]
        inputs=h.prefix_inputs(*args,n);built=model.build_forecasts(*inputs)
        check=oracle.check_forecasts(*inputs,built)
        h.require_model_audit(check,built,expected_rows=n*width,expected_years=len(built['audit']['annual']))
        for prefix in (0,1,n//2,n):
            part=h.prefix_inputs(*args,prefix)
            small=model.build_forecasts(*part)
            for key in h.RAW_KEYS:np.testing.assert_array_equal(small['raw'][key],built['raw'][key][:prefix*width])
            self.assertEqual(len(part[2]['dates']),prefix)
            self.assertTrue(all(len(v)==prefix*width for v in part[3].values()))
            if prefix:
                cutoff=np.datetime64('%04d-12-31'%(int(part[2]['dates'][-1][:4])-1))
                self.assertTrue(np.all(np.asarray(part[0]['ends'],dtype='datetime64[D]')<=cutoff))
            else:self.assertEqual(len(part[0]['dates']),0)
        before=model.build_forecasts(*args)
        for key in h.RAW_KEYS:np.testing.assert_array_equal(built['raw'][key],before['raw'][key])
    def test_source_only_binding_is_in_actual_prospective_contract(self):
        self.assertEqual(h.MODEL_PREFIXES,tuple(range(123)))
        self.assertEqual(h.NAMES,('incumbent_local67','own_increment_calibration','pooled_increment_control'))
        self.assertEqual(h.PRIOR.name,'monthly_risk_reliability223')
        self.assertEqual(h.SOURCE.name,'monthly_identity_transfer206')
        self.assertEqual(len(h.PREPARED_FILES),7)
        self.assertEqual(len(h.RESULT_FILES),38)
        self.assertEqual(h.contract()['max_absolute_delta'],.0125)
        self.assertEqual(h.contract()['model_fit_accounting'],h.read(h.CONTEXT)['formation']['fit_accounting'])


if __name__=='__main__':unittest.main()
