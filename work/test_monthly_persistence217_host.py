"""Synthetic host wiring checks; no market arrays or model fitting."""
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
_spec = importlib.util.spec_from_file_location('tested_persistence217_host', P/'monthly_persistence217_host.py')
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
    f = dict(raw=dict(confirmed=np.tile(np.arange(6., 0., -1.), (6, 1)).reshape(-1),
                      half=((source+old)/2.).reshape(-1)), available=np.ones(shape, bool).reshape(-1))
    meta = dict(dates=['synthetic-%d' % i for i in range(6)], symbols=[list('ABCDEF') for _ in range(6)])
    return a, meta, f


class SyntheticStudy:
    """A deterministic ranker/selector double with observable adapter calls."""
    ARMS = ('local_constant', 'local_linear')

    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        frozen=h.module('persistence217_synthetic_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
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
        forecast['raw']['confirmed'].reshape(6, 6)[5, 1] = np.nan
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

    def test_half_control_cannot_be_promoted_and_candidate_clears_both_references(self):
        metrics, orders = selection_fixture()
        metrics[h.NAMES[2]]['same_close_zero']['training']['cagr'] = 2.
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['close_vs_half_control'])
        metrics, orders = selection_fixture()
        metrics[h.NAMES[2]]['next_open_cost']['training']['cagr'] = 2.
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['open_vs_half_control'])

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
        a,m,f=fixture();s=SyntheticStudy();f['raw']['confirmed'][7]=np.nan
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
    def test_physical_prefix_removes_future_queries_without_consuming_labels(self):
        m=dict(dates=['2019-01-31','2019-02-28','2020-01-31'],ends=['2019-02-28','2019-03-29','2020-02-28'],symbols=[['A','B']]*3)
        source={key:np.arange(6,dtype=object) for key in h.SOURCE_KEYS};use=np.ones((3,2),object)
        for key in source:source[key][4:]='future poison'
        use[2:]='future poison'
        got=h.prefix_inputs(m,source,use,2)
        self.assertEqual(got[0]['dates'],m['dates'][:2])
        for key in source:self.assertEqual(got[1][key].tolist(),[0,1,2,3])
        np.testing.assert_array_equal(got[2],np.ones((2,2),bool))
        empty=h.prefix_inputs(m,source,use,0)
        self.assertEqual(empty[0]['dates'],[]);self.assertEqual(empty[2].shape,(0,2))
        self.assertTrue(all(len(v)==0 for v in empty[1].values()))
        bad=dict(source);bad['available']=bad['available'][:-1]
        with self.assertRaises(ValueError):h.prefix_inputs(m,bad,use,1)
    def test_calendar_rejects_nonformal_signals_except_initial_partial(self):
        cal={'sessions':['2016-08-17','2016-08-30','2016-08-31','2016-09-29','2016-09-30']}
        with patch.object(h,'sha',return_value='digest'):
            self.assertEqual(h.verify_calendar_dates({'dates':['2016-08-17','2016-08-31','2016-09-30']},cal)['formal_events'],2)
            for dates in (['2016-08-17','2016-08-30'],['2016-08-31'],['2016-08-17','2016-09-30','2016-08-31']):
                with self.subTest(dates=dates),self.assertRaises(ValueError):h.verify_calendar_dates({'dates':dates},cal)
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
    def model_receipt():
        counts={key:0 for key in h.MODEL_COUNT_FIELDS}
        counts.update(query_rows=5978,cached_forecast_rows=2,unsupported_rows=2)
        available=np.zeros(5978,bool);available[:2]=True
        raw={key:np.where(available,0.,np.nan) for key in h.RAW_KEYS}
        records=[dict(row=i,date='2016-08-17',symbol=s,prior_row=None,status='unsupported',reason='missing_previous_formal_month',current_difference=None,previous_difference=None,cross_year=False) for i,s in enumerate(('A','B'))]
        zero={k:0 for k in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits','new_parameters')}
        audit=dict(method='previous_formal_month_direction_confirmation',counts=counts,records=records,**zero)
        check=dict.fromkeys(h.MODEL_FLAGS,True);check.update(query_rows=5978,counts=copy.deepcopy(counts),computes_returns=False,producer_modules_imported=False,atol=0,**zero)
        return check,dict(raw=raw,available=available,audit=audit)
    def test_every_independent_flag_is_required(self):
        check,f=self.model_receipt();h.require_model_audit(check,f)
        for key in h.MODEL_FLAGS:
            bad=copy.deepcopy(check);bad[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,f)
    def test_model_scope_counts_and_zero_fits_required(self):
        check,f=self.model_receipt()
        for key,value in [('computes_returns',True),('producer_modules_imported',True),('atol',1e-12),('query_rows',1)]+[(k,1) for k in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits','new_parameters')]:
            bad=copy.deepcopy(check);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,f)
        for key in h.MODEL_COUNT_FIELDS:
            bad=copy.deepcopy(f);c=copy.deepcopy(check);bad['audit']['counts'][key]+=1;c['counts'][key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(c,bad)
        bad=copy.deepcopy(f);bad['audit']['records'][0]['cross_year']=True
        with self.assertRaises(ValueError):h.require_model_audit(check,bad)
    def test_slot_requires_pair_flags_matching_ready_and_changes(self):
        delta={key:np.zeros((122,49)) for key in h.RAW};receipt=dict.fromkeys(h.SLOT_FLAGS,True)
        receipt.update(ready_rows={key:122 for key in h.RAW},changed_cells={key:0 for key in h.RAW})
        h.require_slot_audit(receipt,delta)
        for key in h.SLOT_FLAGS:
            bad=copy.deepcopy(receipt);bad[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_slot_audit(bad,delta)
        bad=copy.deepcopy(receipt);bad['ready_rows'][h.NAMES[1]]=121
        with self.assertRaises(ValueError):h.require_slot_audit(bad,delta)
    def test_source_review_rejects_scope_changes_before_cache_reads(self):
        evidence={'work/fake':'pin'}
        r=dict(passed=True,context_sha256=h.CONTEXT_SHA,evidence_sha256=evidence,source_bytes_reverified=True,prior_annual_asof_forecasts=True,not_realtime_archival_vintages=True,original_query_identity_verified=True,formal_calendar_dates_verified=True,initial_partial_excluded=True,source_npz_decoded=False,observers_modified=False,strict_pit=False,actual_new_transforms=0,actual_new_model_fits=0,actual_new_parameters=0,actual_new_orders=0,actual_new_returns=0,current_and_previous_prediction_keys=['raw_local_linear','raw_source_log'],query_months=122,width=49)
        def reader(path):
            if path==h.SOURCE_REVIEW:return r
            if path==h.CONTEXT:return {'evidence_sha256':evidence}
            raise AssertionError('Cache reads before source review rejection')
        mutations=[('context_sha256','wrong'),('source_bytes_reverified',False),('formal_calendar_dates_verified',False),('source_npz_decoded',True),('strict_pit',True),('actual_new_model_fits',True),('actual_new_transforms',1),('actual_new_returns',1),('current_and_previous_prediction_keys',['raw_control']),('width',48)]
        with patch.object(h,'sha',return_value=h.SOURCE_REVIEW_SHA),patch.object(h,'read',side_effect=reader):
            for key,value in mutations:
                old=r[key];r[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):h.model_source_binding({})
                r[key]=old
    def test_first88_freeze_binds_all_prerequisites(self):
        metrics,orders=selection_fixture();selection=h.select_first88(metrics,orders)
        frozen=dict(plan_sha256='plan',orders_sha256='digest',training_sha256='digest',prepared_sha256='digest',selection=selection)
        def reader(path):return frozen if path.name=='FROZEN.json' else metrics
        with patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=reader):
            self.assertEqual(h.frozen_selection('plan',orders),frozen);frozen['prepared_sha256']='changed'
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
        rows=[dict(trials=1,directory='old') for _ in range(214)];rows[0]['trials']=946;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
        text=Path(h.__file__).read_text();tree=ast.parse(text)
        parts={x.name:ast.get_source_segment(text,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        self.assertIn("len(state['completed']) == 215",parts['close']);self.assertIn('== 1162',parts['close'])
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
    def test_independent_real_synthetic_interface_and_prefixes(self):
        import monthly_persistence217_model as model
        import monthly_persistence217_independent as oracle
        m=dict(dates=['2018-11-30','2018-12-31','2019-01-31'],ends=['2018-12-31','2019-01-31','2019-02-28'],symbols=[['A','B'],['B','A'],['A','B']])
        source=dict(raw_local_linear=np.array([.1,-.2,-.3,.2,-.1,-.2]),raw_source_log=np.zeros(6),available=np.ones(6,bool));u=np.ones((3,2),bool)
        args=h.prefix_inputs(m,source,u,3);f=model.build_forecasts(*args)
        check=oracle.check_forecasts(*args,f);h.require_model_audit(check,f,expected_rows=6)
        for n in range(4):
            part=h.prefix_inputs(m,source,u,n);small=model.build_forecasts(*part)
            for key in h.RAW_KEYS:np.testing.assert_array_equal(small['raw'][key],f['raw'][key][:n*2])
            self.assertEqual(len(part[0]['dates']),n)
            self.assertTrue(all(len(v)==n*2 for v in part[1].values()))


if __name__=='__main__':unittest.main()
