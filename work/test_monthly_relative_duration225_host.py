"""Synthetic-only round225 host tests. No actual225 labels, fits or returns."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import monthly_relative_duration225_host as h


def values_from(candidate,control,mask):
    candidate=np.asarray(candidate,float).copy();control=np.asarray(control,float).copy();mask=np.asarray(mask,bool).copy()
    result=dict(candidate_prediction=candidate,control_prediction=control,common_available=mask)
    for key in ('candidate','control'):
        result[key+'_prediction'][~mask]=np.nan
        rank=np.zeros(mask.shape)
        for i,row in enumerate(mask):
            count=int(row.sum())
            if count>1:rank[i,row]=(pd.Series(-result[key+'_prediction'][i,row]).rank(method='average').to_numpy()-1.)/(count-1.)-.5
        result[key+'_rank']=rank
    return result


def fixture():
    shape=(6,5);mask=np.ones(shape,bool);mask[0]=False;mask[:,-1]=False;mask[2]=False;mask[2,0]=True;mask[3]=False
    values=values_from(np.tile([.1,.2,.3,.4,.5],(6,1)),np.tile([.5,.4,.3,.2,.1],(6,1)),mask)
    a=dict(baseline=np.tile([.2,.199,.198,.197,.196],(6,1)),eligible=np.ones(shape,bool),
        effects=np.zeros(shape),core=np.zeros(shape),core_weight=np.full((6,1),.52),
        state_calm10=np.zeros(6,bool),state_weak_narrow=np.zeros(6,bool))
    a['eligible'][:,-1]=False
    m=dict(dates=['2016-08-17']+['synthetic-%d'%i for i in range(1,6)],symbols=[list('ABCDE') for _ in range(6)])
    return a,m,values


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

class RankAndWiringTests(unittest.TestCase):
    def test_both_negative_prediction_ranks_exact_budget(self):
        a,m,v=fixture();r,d=h.scale_and_delta(v,6,a['eligible'])
        np.testing.assert_allclose(r[h.NAMES[1]][1],[.5,1/6,-1/6,-.5,0.],rtol=0,atol=1e-16)
        np.testing.assert_allclose(r[h.NAMES[2]][1],[-.5,-1/6,1/6,.5,0.],rtol=0,atol=1e-16)
        for arm in d:np.testing.assert_array_equal(d[arm],.01*r[arm]);self.assertLessEqual(np.abs(d[arm]).max(),.005)
        for i in (0,2,3):self.assertTrue(all(np.all(x[i]==0) for x in d.values()))

    def test_constant_predictions_ties_and_missing(self):
        a,m,v=fixture();v['candidate_prediction'][1,:4]=.2
        v=values_from(v['candidate_prediction'],v['control_prediction'],v['common_available'])
        ranks,deltas=h.scale_and_delta(v,6,a['eligible']);self.assertTrue((deltas[h.NAMES[1]][1]==0).all())
        v['candidate_prediction'][1,:4]=[.2,.2,.4,.6]
        v=values_from(v['candidate_prediction'],v['control_prediction'],v['common_available'])
        ranks,_=h.scale_and_delta(v,6);self.assertEqual(ranks[h.NAMES[1]][1,0],ranks[h.NAMES[1]][1,1])

    def test_ridge_predictions_are_unclipped_and_not_cross_arm_ordered(self):
        a,m,v=fixture()
        v['candidate_prediction'][1,:4]=[-2.,3.,1.5,-.25]
        v['control_prediction'][1,:4]=[-3.,2.,.75,-.75]
        v=values_from(v['candidate_prediction'],v['control_prediction'],v['common_available'])
        rank,delta=h.scale_and_delta(v,6,a['eligible'])
        np.testing.assert_array_equal(rank[h.NAMES[1]],v['candidate_rank'])
        np.testing.assert_array_equal(rank[h.NAMES[2]],v['control_rank'])
        self.assertTrue(np.all(v['candidate_prediction'][1,:4]>v['control_prediction'][1,:4]))
        self.assertLessEqual(np.abs(delta[h.NAMES[1]]).max(),.005)

    def test_bad_cache_or_mask_and_initial_refused(self):
        a,m,base=fixture()
        for change in ('rank','inf','mask','dtype','ineligible','initial'):
            v=copy.deepcopy(base)
            if change=='rank':v['candidate_rank'][1,0]=-.5
            if change=='inf':v['control_prediction'][1,0]=np.inf
            if change=='mask':v['control_prediction'][1,0]=np.nan
            if change=='dtype':v['common_available']=v['common_available'].astype(int)
            if change in ('ineligible','initial'):
                i,j=(1,4) if change=='ineligible' else (0,0)
                v['common_available'][i,j]=True;v['candidate_prediction'][i,j]=v['control_prediction'][i,j]=.3
                v=values_from(v['candidate_prediction'],v['control_prediction'],v['common_available'])
            with self.subTest(change=change),self.assertRaises(ValueError):h.scale_and_delta(v,6,a['eligible'])

    def test_future_poison_not_converted_or_ranked(self):
        a,m,v=fixture();want,_=h.scale_and_delta(v,2)
        for key in h.EXTRA[:-1]:v[key]=v[key].astype(object);v[key][2:]='unobserved'
        actual,_=h.scale_and_delta(v,2)
        for key in actual:np.testing.assert_array_equal(actual[key],want[key])


    def test_additive_small_slot_and_exactly_one_episode(self):
            a, m, values = fixture(); olda, oldv = copy.deepcopy(a), copy.deepcopy(values)
            study = SyntheticStudy(); policy = dict(synthetic=True)
            original = study.modified_pre_score(dict(id=4), a, 6)
            with patch.object(h, 'read', return_value=policy) as reader:
                orders, ranks, deltas = h.decisions(study, a, m, values, 6)
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
                allorders, _, _ = h.decisions(SyntheticStudy(), a, m, values, 6)
                for n in range(7):
                    orders, _, _ = h.decisions(SyntheticStudy(), a, m, values, n)
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
                  for p in ('same_close_zero', 'next_open_cost')} for n in h.NAMES}
    orders = {n: [dict(symbols=['A', 'B', 'C']) for _ in range(122)] for n in h.NAMES}
    for path in metrics[h.NAMES[1]]: metrics[h.NAMES[1]][path]['training']['cagr'] += .01
    for i in (1, 2, 53): orders[h.NAMES[1]][i]['symbols'] = ['A', 'B', 'D']
    return metrics, orders

class SelectionAndBindingTests(unittest.TestCase):

    def test_exact_arm_and_first88_guard_names(self):
        self.assertEqual(h.NAMES,('incumbent_local67','relative_lag_run_ridge','relative_lag_incidence_ridge'))
        result=h.select_first88(*selection_fixture())
        self.assertEqual(set(result['guards']),{'close_vs_incumbent','open_vs_incumbent',
            'close_vs_relative_incidence','open_vs_relative_incidence','changed_support','changed_support_vs_control'})
        self.assertIs(result['first88_only'],True);self.assertIs(result['later34_used'],False)

    def test_later34_and_extra_metrics_never_choose_arm(self):
            metrics, orders = selection_fixture(); expected = h.select_first88(metrics, orders)
            self.assertEqual(expected['primary'], h.NAMES[1])
            for name in h.NAMES:
                metrics[name]['full122'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
                for i in range(88, 122): orders[name][i]['symbols'] = [name]
            self.assertEqual(h.select_first88(metrics, orders), expected)

    def test_relative_incidence_control_cannot_promote_and_must_be_beaten(self):
            for path in ('same_close_zero', 'next_open_cost'):
                metrics, orders = selection_fixture(); metrics[h.NAMES[2]][path]['training']['cagr'] = 2.
                result = h.select_first88(metrics, orders)
                self.assertEqual(result['primary'], h.NAMES[0]); self.assertFalse(result['admitted'])

    def test_exact_gain_era_risk_and_changed_set_guards(self):
            for path, era, key, value in [('same_close_zero', 'training', 'cagr', .504),
                    ('next_open_cost', 'training', 'cagr', .49), ('same_close_zero', 'early', 'cagr', .474),
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
            state = dict(completed=[dict(trials=1, directory='old')] * 221 + [dict(trials=962, directory=h.PRIOR.name)],
                         best_training_selected_candidate=dict(directory='monthly_local_linear70'))
            h.state_check(state)
            state['completed'][-1] = dict(trials=8, directory='wrong')
            with self.assertRaises(ValueError): h.state_check(state)

class SourceAndFitTests(unittest.TestCase):
    def test_contract_labels_models_no_zero_fit_claim(self):
        c=h.contract();self.assertEqual(c['dose'],.01);self.assertEqual(c['max_absolute_delta'],.005)
        m=c['measurement']
        self.assertEqual(m['candidate_target'],'longest_consecutive_cumulative_relative_lag_days/L')
        self.assertEqual(m['control_target'],'total_cumulative_relative_lag_days/L')
        for key in ('exact_rational_binary_float_comparison','integer_count_ratios_rounded_once',
                    'complete_official_path','candidate_label_bounded_by_control',
                    'forecasts_unclipped_and_unordered','equality_breaks_run'):
            self.assertIs(m[key],True)
        self.assertEqual(m['minimum_mature_months'],18);self.assertEqual(m['half_life_months'],12.)
        self.assertEqual(m['alpha'],10.);self.assertEqual(m['predictors'],'original_plain10')
        self.assertEqual(c['new_model_fits'],'dynamic_main_paired_annual_fits')
        self.assertIn(h.CONTEXT,h.code_files());self.assertEqual(len(h.PREPARED_FILES),13)
        self.assertIn('COHORT_PARITY.json',h.PREPARED_FILES)
        self.assertNotIn('CONTROL_PARITY.json',h.PREPARED_FILES)
        self.assertEqual(len(h.BUILD_FLAGS),17);self.assertEqual(len(h.PATH_COUNT_FIELDS),6)
        with patch.object(h,'sha',return_value='wrong'):
            with self.assertRaises(ValueError):h.contract()

    def test_prefix_inputs_physically_exclude_future_training_and_query_values(self):
        px=pd.DataFrame({'A':[1.,2.,3.,'future']},index=pd.date_range('2020-01-01',periods=4))
        m=dict(dates=['2020-01-01','2020-01-03'],ends=['2020-02-01','2020-02-03'],symbols=[['A'],['A']],etfs=[])
        tm=dict(dates=['2019-10-31','2019-11-29','2019-12-31'],ends=['2019-11-29','2019-12-31','2020-01-31'],symbols=[['A']]*3)
        e=np.ones((2,1),bool);x=np.ones((3,10),object);x[-1]='future poison'
        cached=dict(train_plain=x,event_plain=np.ones((2,10)),common_good=np.ones(3,bool),train_eligible=np.ones(3,bool))
        for n in (0,1,2):
            args=h.prefix_inputs(px,'calendar',tm,m,cached,e,n)
            self.assertEqual(len(args[3]['dates']),n);self.assertEqual(args[5].shape,(n,10))
            self.assertEqual(len(args[0]),0 if n==0 else 1 if n==1 else 3)
            self.assertEqual(args[1],'calendar');self.assertEqual(args[2]['dates'],tm['dates'][:0 if n==0 else 2])
            self.assertEqual(args[4].shape,(0 if n==0 else 2,10));self.assertEqual(args[6].shape,(0 if n==0 else 2,))
            np.testing.assert_array_equal(args[4],np.ones((0 if n==0 else 2,10)))
        self.assertEqual(len(m['dates']),2);self.assertEqual(len(tm['dates']),3)
        with self.assertRaises(ValueError):h.prefix_inputs(px,[],tm,m,cached,e,True)

    def test_actual_paired_fit_accounting(self):
        row=dict(year=2020,status='fit',model_fits=2,training_months=18,prediction_rows=3,shared_rows_weights_scalers=True)
        a=dict(new_pooled_model_fits=2,local_model_fits=0,annual=[row,dict(year=2021,status='cold_start',model_fits=0)])
        self.assertEqual(h.fit_count(a),2)
        for change in ('total','odd','bool','annual','cohort','duplicate'):
            b=copy.deepcopy(a)
            if change=='total':b['new_pooled_model_fits']=0
            if change=='odd':b['new_pooled_model_fits']=1
            if change=='bool':b['new_pooled_model_fits']=True
            if change=='annual':b['annual'][0]['model_fits']=0
            if change=='cohort':b['annual'][0]['training_months']=17
            if change=='duplicate':b['annual'][1]['year']=2020
            with self.subTest(change=change),self.assertRaises(ValueError):h.fit_count(b)

    def test_frozen155_full_metadata_closure_and_solver_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();local=root/'outputs/local';local.mkdir(parents=True);ep=root/'endpoint.json';cal=root/'calendar.json';numeric=root/'numeric.py'
            raw={str(root/f'raw{i}'):str(i) for i in range(64)};calendar={'source_hashes':{}}
            wiring=dict(meta=local/'META.json',policy=local/'INCUMBENT_POLICY.json',original_orders=local/'local_linear_ORDERS.json',prices=root/'outputs/monthly_2016_extension/PRICES.npz',train_inputs=root/'outputs/monthly_target_definition70/TRAIN_INPUTS.npz',training_meta=root/'outputs/monthly_earnings_factor_research/META.json',calendar=cal)
            hashes=dict(raw);hashes.update({str(p):'synthetic' for p in wiring.values()});hashes[str(numeric)]='solver'
            old=dict(runtime=dict(python=h.sys.version,numpy=np.__version__,pandas=pd.__version__),round=155,status='REGISTERED',hashes=hashes,raw_source_hashes=raw,independent_prepare_source_hashes={},independent_outcome_source_hashes={},calendar=calendar,inputs={k:dict(path=str(p),sha256=hashes[str(p)]) for k,p in wiring.items()})
            ep.write_text(json.dumps(old));(local/'PLAN.json').write_text(json.dumps(dict(raw_source_hashes=raw,calendar=calendar)))
            with patch.multiple(h,ROOT=root,LOCAL=local,ENDPOINT_PLAN=ep,ENDPOINT_PLAN_SHA=h.sha(ep),NUMERIC_PATH=numeric,NUMERIC_SHA='solver',CALENDAR=cal):
                result=h.endpoint_source_binding();self.assertEqual(result['input_sha256'],hashes)
                with patch.object(h,'NUMERIC_SHA', 'wrong'):
                    with self.assertRaises(ValueError):h.endpoint_source_binding()
                # Full build identity is mandatory; a matching short version is insufficient.
                original_runtime=copy.deepcopy(old['runtime'])
                for wrong_build in ('3.9.6',h.sys.version+' changed-build'):
                    old['runtime']['python']=wrong_build;ep.write_text(json.dumps(old))
                    with patch.object(h,'ENDPOINT_PLAN_SHA',h.sha(ep)):
                        with self.assertRaisesRegex(ValueError,'Frozen155 contract/runtime'):h.endpoint_source_binding()
                old['runtime']=original_runtime
                old['inputs']['train_inputs']['path']=str(root/'other');ep.write_text(json.dumps(old))
                with patch.object(h,'ENDPOINT_PLAN_SHA',h.sha(ep)):
                    with self.assertRaises(ValueError):h.endpoint_source_binding()


    def test_actual_frozen155_metadata_binding_without_arrays(self):
        # Read frozen metadata only. All numerical loaders are forbidden in this check.
        with patch.object(h,'load_npz',side_effect=AssertionError('No actual arrays')), \
             patch.object(h,'local',side_effect=AssertionError('No actual strategy')):
            binding=h.endpoint_source_binding()
        old=h.read(h.ENDPOINT_PLAN);original=h.read(h.LOCAL/'PLAN.json');calendar=h.read(h.CALENDAR)
        self.assertEqual(old['runtime'],dict(python=h.sys.version,numpy=np.__version__,pandas=pd.__version__))
        self.assertEqual(binding['input_sha256'],old['hashes'])
        self.assertEqual(binding['calendar'],original['calendar'])
        self.assertEqual(binding['calendar'],calendar['provenance'])
        self.assertEqual(len(old['raw_source_hashes']),64)
        for key,ref in binding['inputs'].items():
            self.assertEqual(binding['input_sha256'][ref['path']],ref['sha256'],key)
        paths={p.name for p in h.code_files()}
        self.assertIn('monthly_relative_duration225_host.py',paths)
        self.assertIn('test_monthly_relative_duration225_host.py',paths)
        self.assertNotIn('monthly_relative_tail190_host_v2.py',paths)
        self.assertNotIn('test_monthly_relative_tail190_host_v2.py',paths)
        self.assertEqual(h.TEST_RECEIPT.name,'monthly_relative_duration225_TESTS.json')
        self.assertEqual(h.OUT.name,'monthly_relative_duration225')

    def test_real_adapter_and_original_boundary_with_column_weights(self):
            # Compile the exact original boundary function without importing any driver or prices.
            path = h.ROOT / 'outputs/monthly_daily_episodes/episodes.py'
            tree = ast.parse(path.read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'select_boundary')
            namespace = {'np': np}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
            adapter = h.module('relativeduration225_test_original_adapter', h.ROOT / 'outputs/monthly_path70_research/incumbent_adapter.py')
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
                orders, _, deltas = h.decisions(study, a, m, values, 6)
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


def cohort_receipt():
    return dict(passed=True,prior_plan_sha256=h.COHORT_PLAN_SHA,
        prior_prepared_completion_sha256=h.PRIOR_PREPARED_SHA,
        label_keys_checked=list(h.COHORT_LABEL_KEYS),arrays_dtype_shape_bytes_exact=True,
        old_targets_loaded=False,old_forecasts_loaded=False,old_orders_loaded=False,old_returns_read=False)


def build_fixture(n=122,width=49):
    mask=np.zeros((n,width),bool);v=values_from(np.zeros(mask.shape),np.zeros(mask.shape),mask)
    labels={k:np.zeros(2) for k in h.LABEL_KEYS};labels['dates']=np.array(['2019-01-31','2019-01-31'],dtype='datetime64[D]');labels['good']=np.ones(2,bool)
    fit=dict(new_pooled_model_fits=2,local_model_fits=0,annual=[dict(year=2020,status='fit',model_fits=2,training_months=18,prediction_rows=2,shared_rows_weights_scalers=True)])
    b=dict(arrays=v,labels=labels,label_audit=dict(source_path_rows=2,source_price_observations=84,target_session_observations=40,missing_path_rows=0,lag_session_observations=12,longest_run_session_observations=8),model_audit=fit,rank_audit={})
    check=dict({k:True for k in h.BUILD_FLAGS},events=n,symbols_per_event=width,training_rows=2,good_training_rows=2,available_cells=0,new_pooled_model_fits=2,independent_model_fits=2,computes_portfolio_returns=False,producer_modules_imported=False,source_path_rows=2,source_price_observations=84,target_session_observations=40,missing_path_rows=0,lag_session_observations=12,longest_run_session_observations=8)
    return b,check


class AuditReceiptTests(unittest.TestCase):
    def test_independent_build_receipt_requires_all_fields_and_nonzero_main_fits(self):
        b,c=build_fixture();self.assertEqual(h.require_build_audit(c,b),2)
        for key in h.BUILD_FLAGS:
            with self.assertRaises(ValueError):h.require_build_audit(dict(c,**{key:False}),b)
        for key,value in [('events',121),('training_rows',1),('available_cells',1),('independent_model_fits',0),('new_pooled_model_fits',0),('computes_portfolio_returns',True)]:
            with self.assertRaises(ValueError):h.require_build_audit(dict(c,**{key:value}),b)
        b['model_audit']=dict(new_pooled_model_fits=0,local_model_fits=0,annual=[]);c.update(new_pooled_model_fits=0,independent_model_fits=0)
        with self.assertRaises(ValueError):h.require_build_audit(c,b)
        self.assertEqual(h.require_build_audit(c,b,require_fits=False),0)

    def test_all_six_path_counts_are_exact_integer_and_independently_matched(self):
        b,c=build_fixture()
        expected=('source_path_rows','source_price_observations','target_session_observations',
                  'missing_path_rows','lag_session_observations','longest_run_session_observations')
        self.assertEqual(h.PATH_COUNT_FIELDS,expected)
        for key in expected:
            for value in (True,-1,c[key]+1):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    h.require_build_audit(dict(c,**{key:value}),b)
            wrong=dict(c);wrong.pop(key)
            with self.assertRaises(ValueError):h.require_build_audit(wrong,b)

    def test_prepared_receipts_separate_main_and_diagnostic_fits(self):
        b,check=build_fixture();prefix=[dict(prefix=n,producer_model_fits=0 if n<2 else 2,independent_model_fits=0 if n<2 else 2) for n in h.FEATURE_PREFIXES]
        common=dict(new_model_fits=2,new_predictive_model_fits=2,new_pooled_model_fits=2,independent_diagnostic_model_fits=2,prefix_diagnostic_model_fits=6,prefix_independent_diagnostic_model_fits=6,prefix_model_fit_counts=prefix,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,**{key:b['label_audit'][key] for key in h.PATH_COUNT_FIELDS})
        c=dict(artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES},new_returns_computed=False,original_array_count=27,**common)
        audit=dict(model=check,delta={},selection={},cohort_parity=cohort_receipt(),source_cohort_exact=True,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),prefix_prices_truncated=True,prefix_training_inputs_truncated=True,original_orders_exact=True,original_arrays_unchanged=True,original_training_arrays_unchanged=True,original_array_count=27,**common)
        with patch.object(h,'require_delta_audit'),patch.object(h,'require_selection_audit'):
            h.verify_prepared_evidence(c,b,{},audit)
            for key,value in [('new_model_fits',0),('prefix_diagnostic_model_fits',2),('prefix_model_fit_counts',[]),('new_predictive_model_fits',True),('new_returns_computed',True)]:
                with self.assertRaises(ValueError):h.verify_prepared_evidence(dict(c,**{key:value}),b,{},audit)
            for key in ('original_arrays_unchanged','original_training_arrays_unchanged','prefix_prices_truncated','prefix_training_inputs_truncated'):
                with self.assertRaises(ValueError):h.verify_prepared_evidence(c,b,{},dict(audit,**{key:False}))
            for name in h.PREPARED_FILES:
                wrong=copy.deepcopy(c);wrong['artifacts_sha256'].pop(name)
                with self.assertRaises(ValueError):h.verify_prepared_evidence(wrong,b,{},audit)

    def test_delta_and_selection_receipts_fail_closed(self):
        _,_,v=fixture();_,delta=h.scale_and_delta(v,6)
        check=dict(passed=True,initial_delta_zero=True,common_available_only=True,common_query_population_checked=True,negative_prediction_ranks_checked=True,average_ties_checked=True,no_new_eligibility_filter=True,missing_exact_zero=True,dose=.01,max_absolute_correction=.005,changed_cells={h.RAW[k]:int(np.count_nonzero(x)) for k,x in delta.items()})
        h.require_delta_audit(check,delta)
        for key in ('initial_delta_zero','common_available_only','common_query_population_checked','negative_prediction_ranks_checked','average_ties_checked','no_new_eligibility_filter','missing_exact_zero'):
            with self.assertRaises(ValueError):h.require_delta_audit(dict(check,**{key:False}),delta)
        check=dict(passed=True,scores_and_choices_checked=True,one_original_episode_checked=True,initial_partial_fallback_checked=True,ordered_choices_checked=366)
        h.require_selection_audit(check)
        for key in ('passed','scores_and_choices_checked','one_original_episode_checked','initial_partial_fallback_checked'):
            with self.assertRaises(ValueError):h.require_selection_audit(dict(check,**{key:False}))


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
    prior = root/'outputs/monthly_increment_calibration224'
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path.resolve()),sha256=h.sha(path))
    plan=put(prior/'PLAN.json',{'input_sha256':{}})
    result=put(prior/'results/COMPLETION.json',dict(status='COMPLETE',plan_sha256=plan['sha256'],paths=9,slices=45))
    closed=put(prior/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',plan_sha256=plan['sha256'],
        result_completion_sha256=result['sha256'],protected_files=21444,observers_modified=False))
    put(prior/'RESEARCH_REVIEW.json',{'passed':True})
    context=root/'work/context.json'
    evidence=put(root/'work/evidence.json',{'synthetic':True})
    put(context,dict(evidence_sha256={'work/evidence.json':evidence['sha256']}))
    return prior,{n:h.sha(prior/n) for n in h.PRIOR_EXPECTED},dict(completed=[dict(completion=closed)]),context

class PredecessorBindingTests(unittest.TestCase):

    def test_predecessor_completion_and_result_are_bound(self):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();prior,expected,state,context=predecessor_fixture(root)
                with patch.multiple(h,ROOT=root,PRIOR=prior,PRIOR_EXPECTED=expected,CONTEXT=context), patch.object(h,'previous_cohort_binding',return_value=dict(input_sha256={})):
                    binding=h.predecessor_binding(state)
                    self.assertIn(str(prior/'results/COMPLETION.json'),binding['input_sha256'])
                    self.assertIn(str(root/'work/evidence.json'),binding['input_sha256'])
                    for key,value in [('path',str(root/'wrong')),('sha256','wrong')]:
                        wrong=copy.deepcopy(state);wrong['completed'][0]['completion'][key]=value
                        with self.assertRaises(ValueError):h.predecessor_binding(wrong)

    def test_predecessor_and_context_evidence_cannot_change(self):
            for relative in ('outputs/monthly_increment_calibration224/PLAN.json','outputs/monthly_increment_calibration224/results/COMPLETION.json','work/evidence.json'):
                with self.subTest(relative=relative),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory).resolve();prior,expected,state,context=predecessor_fixture(root)
                    (root/relative).write_text('changed')
                    with patch.multiple(h,ROOT=root,PRIOR=prior,PRIOR_EXPECTED=expected,CONTEXT=context), patch.object(h,'previous_cohort_binding',return_value=dict(input_sha256={})):
                        with self.assertRaises(ValueError):h.predecessor_binding(state)

    def test_all_phase_gates_recheck_registered_predecessor_binding(self):
            plan = dict(contract=h.contract(), runtime=h.ev.runtime(), input_sha256={'x':'y'},
                        prior_central_sha256='sha', predecessor_binding={'registered':True},
                        raw_source_hashes={'raw':'sha'},endpoint_source_binding={'endpoint':True},source_review_binding={'source':True},numerical_code_review={'review':True})
            state = dict(registered_queue=[dict(directory=h.OUT.name, plan_sha256='sha')])
            def reader(path): return plan if path == h.OUT / 'PLAN.json' else state
            with patch.object(h, 'contract', return_value=plan['contract']), patch.object(h, 'sha', return_value='sha'), patch.object(h, 'read', side_effect=reader), \
                 patch.object(h, 'verify'), patch.object(h, 'state_check'), \
                 patch.object(h, 'endpoint_source_binding',return_value={'endpoint':True}), patch.object(h,'source_review_binding',return_value={'source':True}),patch.object(h,'code_review_binding',return_value={'review':True}), \
                 patch.object(h, 'predecessor_binding', return_value={'registered':True}) as predecessor:
                self.assertEqual(h.gate('sha'), plan)
                predecessor.assert_called_once_with(state)
                predecessor.return_value = {'registered':False}
                with self.assertRaises(ValueError): h.gate('sha')

    def test_real_build_receipt_and_truncated_prefix_api(self):
        import monthly_relative_duration225_model as model
        import monthly_relative_duration225_independent as independent
        from test_monthly_relative_duration225_model import build_fixture as model_fixture
        args=model_fixture();built=model.build(*args)
        check=independent.verify_build(*args,built)
        self.assertGreater(h.require_build_audit(check,built,shape=(5,3)),0)
        cached=dict(train_plain=args[4],event_plain=args[5],common_good=args[6],train_eligible=args[7])
        for n in (0,1,2,5):
            part=h.prefix_inputs(args[0],args[1],args[2],args[3],cached,args[8],n)
            small=model.build(*part);audit=independent.verify_build(*part,small)
            h.require_build_audit(audit,small,shape=(n,3),require_fits=False)
            for key in h.EXTRA:np.testing.assert_array_equal(small['arrays'][key],built['arrays'][key][:n])
            _,d=h.scale_and_delta(small['arrays'],n,args[8][:n])
            h.require_delta_audit(independent.verify_deltas(small['arrays'],{h.RAW[k]:x for k,x in d.items()},dose=.01),d)

    def test_real_independent_delta_api_without_weight_shape_dependency(self):
        import monthly_relative_duration225_independent as independent
        original,_,v=fixture();self.assertEqual(original['core_weight'].shape,(6,1))
        _,d=h.scale_and_delta(v,6);check=independent.verify_deltas(v,{h.RAW[k]:x for k,x in d.items()},dose=.01);h.require_delta_audit(check,d)
        wrong=copy.deepcopy(d);wrong[h.NAMES[1]][1,0]+=.001
        with self.assertRaises((ValueError,AssertionError)):independent.verify_deltas(v,{h.RAW[k]:x for k,x in wrong.items()},dose=.01)


class CohortReuseTests(unittest.TestCase):
    @staticmethod
    def prior_fixture(root):
        prior=root/'outputs/monthly_relative_path209';dest=prior/'prepared';dest.mkdir(parents=True)
        (prior/'PLAN.json').write_text(json.dumps({'synthetic':True}));plan_sha=h.sha(prior/'PLAN.json')
        labels=dict(dates=np.array(['2019-01-31','2019-01-31'],dtype='datetime64[D]'),
            ends=np.array(['2019-02-28','2019-02-28'],dtype='datetime64[D]'),symbols=np.array(['A','B']),
            base_good=np.array([True,True]),good=np.array([True,True]))
        # Poison old numerical targets/forecasts/orders. Only five identity/mask arrays may be read.
        np.savez_compressed(dest/'LABELS.npz',**labels,candidate_target=np.array([object()],object),
                            control_target=np.array([object()],object))
        np.savez_compressed(dest/'FEATURES.npz',forbidden=np.array([object()],object))
        (dest/'ORDERS.json').write_text('not JSON: old orders must not be decoded')
        for name in h.PRIOR_PREPARED_FILES-{'LABELS.npz','FEATURES.npz','ORDERS.json'}:(dest/name).write_text('{}')
        completion=dict(status='COMPLETE',plan_sha256=plan_sha,new_returns_computed=False,
            original_array_count=27,artifacts_sha256={name:h.sha(dest/name) for name in h.PRIOR_PREPARED_FILES})
        (dest/'COMPLETION.json').write_text(json.dumps(completion))
        built=dict(labels=dict(labels,candidate_target=np.array([.5,1.]),control_target=np.array([1.,1.])))
        return prior,h.sha(dest/'COMPLETION.json'),plan_sha,built

    def test_registered_209_metadata_binder_without_numerical_loading(self):
        # This reads registered literal metadata and hashes bytes, never real NPZ values.
        with patch.object(h.np,'load',side_effect=AssertionError('No actual NPZ load')), \
             patch.object(h,'local',side_effect=AssertionError('No actual strategy')):
            reference=h.previous_cohort_binding()
        self.assertEqual(len(reference['input_sha256']),15)
        self.assertEqual(len(h.PRIOR_PREPARED_FILES),13);self.assertEqual(len(h.PREPARED_FILES),13)
        self.assertEqual(reference['prior_prepared_completion_sha256'],h.PRIOR_PREPARED_SHA)
        self.assertEqual(reference['prior_plan_sha256'],h.COHORT_PLAN_SHA)

    def test_exact_five_identity_masks_only_never_old_numeric_targets_or_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            prior,digest,plan_sha,built=self.prior_fixture(Path(directory))
            with patch.multiple(h,COHORT_REFERENCE=prior,PRIOR_PREPARED_SHA=digest,COHORT_PLAN_SHA=plan_sha):
                binding=h.previous_cohort_binding();self.assertEqual(len(binding['input_sha256']),15)
                receipt=h.cohort_parity(built);self.assertEqual(receipt,cohort_receipt());h.require_cohort_parity(receipt)
                for key in h.COHORT_LABEL_KEYS:
                    wrong=copy.deepcopy(built);a=wrong['labels'][key]
                    if a.dtype.kind=='b':a.flat[-1]=not a.flat[-1]
                    elif a.dtype.kind=='M':a.flat[-1]+=np.timedelta64(1,'D')
                    else:a.flat[-1]='Z'
                    with self.subTest(key=key),self.assertRaises(ValueError):h.cohort_parity(wrong)
                # New labels deliberately differ; no numeric-target equality is required.
                built['labels']['candidate_target']=np.array([0.,0.])
                built['labels']['control_target']=np.array([.1,.2])
                self.assertEqual(h.cohort_parity(built),receipt)

    def test_cohort_byte_equality_requires_dtype_shape_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            prior,digest,plan_sha,built=self.prior_fixture(Path(directory))
            with patch.multiple(h,COHORT_REFERENCE=prior,PRIOR_PREPARED_SHA=digest,COHORT_PLAN_SHA=plan_sha):
                for mode in ('dtype','shape','order'):
                    wrong=copy.deepcopy(built)
                    if mode=='dtype':wrong['labels']['dates']=wrong['labels']['dates'].astype('datetime64[ns]')
                    if mode=='shape':wrong['labels']['good']=wrong['labels']['good'].reshape(1,2)
                    if mode=='order':wrong['labels']['symbols']=wrong['labels']['symbols'][::-1]
                    with self.subTest(mode=mode),self.assertRaises(ValueError):h.cohort_parity(wrong)

    def test_cohort_reference_artifact_digest_and_schema_fail_closed(self):
        for mode in ('digest','plan','scope','missing','extra'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                prior,digest,plan_sha,built=self.prior_fixture(Path(directory));dest=prior/'prepared'
                if mode=='digest':(dest/'FEATURES.npz').write_bytes(b'corrupt')
                else:
                    c=json.loads((dest/'COMPLETION.json').read_text())
                    if mode=='plan':c['plan_sha256']='wrong'
                    if mode=='scope':c['new_returns_computed']=True
                    if mode=='missing':c['artifacts_sha256'].pop('MODEL_AUDIT.json')
                    if mode=='extra':c['artifacts_sha256']['EXTRA.json']='extra'
                    (dest/'COMPLETION.json').write_text(json.dumps(c));digest=h.sha(dest/'COMPLETION.json')
                with patch.multiple(h,COHORT_REFERENCE=prior,PRIOR_PREPARED_SHA=digest,COHORT_PLAN_SHA=plan_sha):
                    with self.assertRaises(ValueError):h.previous_cohort_binding()

    def test_cohort_receipt_every_field_required_and_prepared_rechecks_it(self):
        c=cohort_receipt();h.require_cohort_parity(c)
        for key in c:
            wrong=copy.deepcopy(c);wrong.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_cohort_parity(wrong)
        for key in ('old_targets_loaded','old_forecasts_loaded','old_orders_loaded','old_returns_read'):
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_cohort_parity(dict(c,**{key:True}))
        self.assertIn('COHORT_PARITY.json',h.PREPARED_FILES);self.assertNotIn('CONTROL_PARITY.json',h.PREPARED_FILES)
        tree=ast.parse(Path(h.__file__).read_text())
        prepared=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prepared')
        self.assertTrue(any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='cohort_parity' for x in ast.walk(prepared)))


class ModernLifecycleTests(unittest.TestCase):
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

class SourceAdmissionTests(unittest.TestCase):
    TRUE_FLAGS=('source_admission_metadata_only','metadata_axes_verified','calendar_clock_verified',
        'raw_source_hashes_verified','raw_source_map_matches_original','endpoint155_wiring_verified',
        'prior209_complete_path_certificate_verified','prior209_prepared_manifest_verified',
        'annual_cutoff_calendar_verified','canonical_identity_scope_reviewed','both225_targets_new')
    FALSE_FLAGS=('source_code_imported','npz_decoded','actual_source_coverage_recomputed',
        'actual_new_support_computed','actual_features_computed','actual_labels_computed',
        'actual_model_fits_computed','actual_orders_computed','actual_returns_computed',
        'strategy_registered','observers_modified','central_modified','strict_point_in_time',
        'old_numeric_label_parity_claimed','old_control_prediction_parity_claimed','old_control_order_parity_claimed')

    def receipt(self):
        raw={str(h.ROOT/f'work/synthetic_raw{i}'):'digest' for i in range(64)}
        coverage=dict(training_rows=8820,training_months=180,admitted_good_rows=6808,
            base_rows=6808,path_complete_base_rows=6808,prior209_certificate=True,fresh225_support=False,
            prepared_completion_sha256=h.PRIOR_PREPARED_SHA,labels_archive_sha256='digest',
            final_good_mask_sha256='a'*64)
        receipt=dict(passed=True,source_only=True,context_sha256=h.CONTEXT_SHA,
            raw_source_hashes=raw,source_sha256={'work/fake':'digest'},calendar_sha256='digest',
            post_registration_required_cohort_parity=list(h.COHORT_LABEL_KEYS),
            training_path_coverage=coverage,coverage_sha256='coverage',
            implementation=dict(path='work/source_helper.py',sha256='helper'))
        receipt.update({key:True for key in self.TRUE_FLAGS})
        receipt.update({key:False for key in self.FALSE_FLAGS})
        return raw,receipt

    @staticmethod
    def source_sha(path):
        return h.SOURCE_REVIEW_SHA if Path(path)==h.SOURCE_REVIEW else 'digest'

    def test_source_review_proofs_no_actual_work_and_no_old_numeric_parity(self):
        raw,receipt=self.receipt()
        with patch.object(h,'read',return_value=receipt),patch.object(h,'sha',side_effect=self.source_sha), \
             patch.object(h,'bind') as binder,patch.object(h,'verify'), \
             patch.object(h.np,'load',side_effect=AssertionError('No numeric source read')):
            result=h.source_review_binding(raw)
            self.assertEqual(result['training_path_coverage'],receipt['training_path_coverage'])
            self.assertIs(result['strict_pit'],False)
            calls=[call.args[1:] for call in binder.call_args_list]
            self.assertIn((h.ROOT/'work/after224_relative_duration_coverage.json','coverage'),calls)
            self.assertIn((h.ROOT/'work/source_helper.py','helper'),calls)
            for key in ('passed','source_only')+self.TRUE_FLAGS:
                for value in (False,1,None):
                    receipt[key]=value
                    with self.subTest(key=key,value=value),self.assertRaises(ValueError):h.source_review_binding(raw)
                receipt[key]=True
            for key in self.FALSE_FLAGS:
                for value in (True,0,None):
                    receipt[key]=value
                    with self.subTest(key=key,value=value),self.assertRaises(ValueError):h.source_review_binding(raw)
                receipt[key]=False

    def test_source_certificate_scope_clock_digest_and_identity_are_required(self):
        raw,receipt=self.receipt()
        patches=[('context_sha256','wrong'),('raw_source_hashes',{}),('calendar_sha256','wrong'),
            ('post_registration_required_cohort_parity',['good']),('source_sha256',{})]
        coverage_changes=[('prior209_certificate',False),('fresh225_support',True),
            ('training_rows',8819),('training_months',179),('admitted_good_rows',True),
            ('base_rows',6807),('path_complete_base_rows',6807),
            ('prepared_completion_sha256','wrong'),('labels_archive_sha256','wrong'),
            ('final_good_mask_sha256','short')]
        with patch.object(h,'sha',side_effect=self.source_sha),patch.object(h,'bind'),patch.object(h,'verify'):
            for key,value in patches:
                wrong=copy.deepcopy(receipt);wrong[key]=value
                with patch.object(h,'read',return_value=wrong),self.subTest(key=key),self.assertRaises(ValueError):
                    h.source_review_binding(raw)
            for key,value in coverage_changes:
                wrong=copy.deepcopy(receipt);wrong['training_path_coverage'][key]=value
                with patch.object(h,'read',return_value=wrong),self.subTest(key=key),self.assertRaises(ValueError):
                    h.source_review_binding(raw)
            with patch.object(h,'read',return_value=receipt),self.assertRaises(ValueError):
                h.source_review_binding(dict(list(raw.items())[:-1]))

    def test_actual_built_good_mask_must_match_source_certificate_exactly(self):
        good=np.array([True,False,True],bool);built={'labels':{'good':good}}
        coverage=dict(training_rows=3,admitted_good_rows=2,
            final_good_mask_sha256=h.hashlib.sha256(good.tobytes()).hexdigest())
        self.assertTrue(h.require_source_cohort(built,coverage))
        for wrong in (np.array([True,True,False]),good.astype(int),good.reshape(1,3),good[:2],np.zeros(3,bool)):
            with self.subTest(shape=wrong.shape,dtype=wrong.dtype),self.assertRaises(ValueError):
                h.require_source_cohort({'labels':{'good':wrong}},coverage)
        # An empty source certificate also stays exactly typed and hash-bound.
        empty=np.array([],bool)
        self.assertTrue(h.require_source_cohort({'labels':{'good':empty}},dict(training_rows=0,
            admitted_good_rows=0,final_good_mask_sha256=h.hashlib.sha256(empty.tobytes()).hexdigest())))


if __name__=='__main__':unittest.main()

