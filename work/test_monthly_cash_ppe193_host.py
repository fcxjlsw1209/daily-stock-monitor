"""Synthetic score/selection/lifecycle checks; no real financial features or returns."""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('cashppe193_tested_host', Path(__file__).with_name('monthly_cash_ppe193_host.py'))
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)


def fixture():
    shape = (6, 5)
    values = dict(candidate_bounded=np.tile([-.1, -.2, 0., .3, .4], (6, 1)),
                  control_bounded=np.tile([.1, .2, .3, .4, .5], (6, 1)),
                  common_available=np.ones(shape, bool))
    values['common_available'][2, :4] = False; values['common_available'][3] = False
    for key in h.RAW.values(): values[key][~values['common_available']] = np.nan
    arrays = dict(baseline=np.tile([.2, .199, .198, .197, .196], (6,1)),
                  eligible=np.ones(shape,bool), effects=np.zeros(shape), core=np.zeros(shape),
                  core_weight=np.full((6,1),.52), state_calm10=np.zeros(6,bool), state_weak_narrow=np.zeros(6,bool))
    arrays['eligible'][:, -1] = False
    meta = dict(dates=['synthetic-%d'%i for i in range(6)], symbols=[list('ABCDE') for _ in range(6)])
    return arrays, meta, values


class SyntheticStudy:
    def __init__(self):
        self.ep=SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter=SimpleNamespace(score_and_choose=Mock(side_effect=self.choose))
        self.rp=SimpleNamespace(make_orders=self.orders)
    @staticmethod
    def configs(): return [dict(id=i) for i in range(5)]
    @staticmethod
    def modified_pre_score(specification,a,n):
        assert specification == dict(id=4)
        return a['baseline'].copy()+.11
    @staticmethod
    def boundary(pre,a,m,n):
        chosen=[]
        for i in range(n):
            js=sorted(np.flatnonzero(a['eligible'][i]),key=lambda j:(-pre[i,j],j))
            chosen.append([m['symbols'][i][j] for j in js[:3]] if len(js)>=2 else [])
        return chosen,pre[:n].copy()
    def choices(self,s,a,m,n,*,policy): return self.boundary(self.modified_pre_score(s,a,n),a,m,n)
    @staticmethod
    def orders(choices,scores,meta):
        return [dict(date=meta['dates'][i],symbols=ss,scores=scores[i].tolist()) for i,ss in enumerate(choices)]
    def choose(self,base,weight,extra,a,m,n,*,pre_score,effects,episode_policy,boundary_selector):
        assert base=='incumbent66' and weight==0 and np.isnan(extra).all()
        assert effects is a['effects'] and boundary_selector is self.ep.select_boundary
        return boundary_selector(pre_score,a,m,n)


class ScoreTests(unittest.TestCase):
    def test_declared_arm_order_and_eligible_only_initial_not_forced_zero(self):
        a,m,v=fixture(); transforms,d=h.scale_and_delta(v,a['eligible'],6)
        self.assertEqual(h.NAMES,('incumbent_local67','same_support_cashflow_control','cash_after_ppe'))
        np.testing.assert_array_equal(transforms[h.NAMES[2]][0],v['candidate_bounded'][0])
        self.assertEqual(d[h.NAMES[2]][0,0],-.001)
        self.assertEqual(d[h.NAMES[2]][0,-1],0.)
        self.assertTrue(v['common_available'][0,-1])
    def test_bounds_are_direct_no_rank_tanh_or_clip(self):
        a,m,v=fixture(); a['eligible'][:]=True; v['candidate_bounded'][1]=[-1.,-.5,0.,.5,1.]
        _,d=h.scale_and_delta(v,a['eligible'],6)
        np.testing.assert_array_equal(d[h.NAMES[2]][1],[-.01,-.005,0.,.005,.01])
        v['candidate_bounded'][1,0]=-1.0000001
        with self.assertRaises(ValueError):h.scale_and_delta(v,a['eligible'],6)
    def test_missing_and_zero_exact_fallback(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value={}):
            for mode in ('zero','missing'):
                vv=copy.deepcopy(v)
                if mode=='missing':vv['common_available'][:]=False
                for key in h.RAW.values():vv[key][:]=np.where(vv['common_available'],0.,np.nan)
                oo,_,dd=h.decisions(SyntheticStudy(),a,m,vv,6)
                self.assertTrue(all(oo[k]==oo[h.NAMES[0]] for k in h.NAMES))
                self.assertTrue(all(np.count_nonzero(x)==0 for x in dd.values()))
    def test_bad_masks_shapes_infinity_and_eligibility_rejected(self):
        for mode in ('mask','shape','inf','bool','eligibility'):
            a,m,v=fixture()
            if mode=='mask':v['candidate_bounded'][1,0]=np.nan
            if mode=='shape':v['control_bounded']=v['control_bounded'][:,:-1]
            if mode=='inf':v['control_bounded'][1,0]=np.inf
            if mode=='bool':v['common_available']=v['common_available'].astype(int)
            if mode=='eligibility':a['eligible']=a['eligible'].astype(int)
            with self.subTest(mode=mode),self.assertRaises(ValueError):h.scale_and_delta(v,a['eligible'],6)
    def test_prefix_numeric_poison_not_read(self):
        a,m,v=fixture(); rr,dd=h.scale_and_delta(v,a['eligible'],6)
        for n in range(7):
            vv=copy.deepcopy(v)
            for key in h.RAW.values():
                vv[key]=vv[key].astype(object);vv[key][n:]='future-invalid-numeric'
            r,d=h.scale_and_delta(vv,a['eligible'],n)
            for arm in h.RAW:
                np.testing.assert_array_equal(r[arm],rr[arm][:n]);np.testing.assert_array_equal(d[arm],dd[arm][:n])
    def test_invalid_prefix_rejected(self):
        a,m,v=fixture()
        for n in (True,1.5,-1,7,np.int64(1)):
            with self.subTest(n=n),self.assertRaises(ValueError):h.scale_and_delta(v,a['eligible'],n)
    def test_exactly_one_episode_preserves_original_scores_and_inputs(self):
        a,m,v=fixture(); aa,vv=copy.deepcopy(a),copy.deepcopy(v);study=SyntheticStudy()
        with patch.object(h,'read',return_value={'synthetic':True}):oo,_,d=h.decisions(study,a,m,v,6)
        self.assertEqual(study.adapter.score_and_choose.call_count,2)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            expected=study.modified_pre_score(dict(id=4),a,6);active=d[arm]!=0;expected[active]+=d[arm][active]
            np.testing.assert_array_equal(call.kwargs['pre_score'],expected)
            self.assertIs(call.kwargs['effects'],a['effects'])
        for k in a:np.testing.assert_array_equal(a[k],aa[k])
        for k in v:np.testing.assert_array_equal(v[k],vv[k])
    def test_all_order_prefixes_stable(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value={}):
            allorders,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            for n in range(7):
                orders,_,_=h.decisions(SyntheticStudy(),a,m,v,n)
                for arm in h.NAMES:self.assertEqual(orders[arm],allorders[arm][:n])


def selection_fixture():
    metrics={arm:{path:{era:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for era in ('training','early','recent')}
                  for path in ('same_close_zero','next_open_cost')} for arm in h.NAMES}
    for path in metrics[h.NAMES[2]]:metrics[h.NAMES[2]][path]['training']['cagr']=.501
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in h.NAMES}
    return metrics,orders


class SelectionTests(unittest.TestCase):
    def test_strict_gain_not_old_point_five_or_changed_count_gate(self):
        metrics,orders=selection_fixture();out=h.select_first88(metrics,orders)
        self.assertTrue(out['admitted']);self.assertEqual(out['primary'],h.NAMES[2]);self.assertEqual(out['changed_months'],[])
    def test_close_must_strictly_beat_both(self):
        for ref in h.NAMES[:2]:
            metrics,orders=selection_fixture();metrics[ref]['same_close_zero']['training']['cagr']=.501
            self.assertFalse(h.select_first88(metrics,orders)['admitted'])
    def test_open_incumbent_equality_allowed_control_equality_rejected(self):
        metrics,orders=selection_fixture();metrics[h.NAMES[0]]['next_open_cost']['training']['cagr']=.501
        self.assertTrue(h.select_first88(metrics,orders)['admitted'])
        metrics[h.NAMES[1]]['next_open_cost']['training']['cagr']=.501
        self.assertFalse(h.select_first88(metrics,orders)['admitted'])
    def test_guard_nonfinite_and_bool_rejected(self):
        metrics,orders=selection_fixture()
        for bad in (None,True,float('nan'),float('inf')):
            mm=copy.deepcopy(metrics);mm[h.NAMES[2]]['same_close_zero']['training']['cagr']=bad
            self.assertFalse(h.select_first88(mm,orders)['admitted'])
    def test_risk_and_era_guards_for_both_paths(self):
        for path in ('same_close_zero','next_open_cost'):
            for era,key,bad in (('training','sharpe',1.449),('training','max_drawdown',-.331),('early','cagr',.474),('recent','cagr',.474)):
                mm,oo=selection_fixture();mm[h.NAMES[2]][path][era][key]=bad
                self.assertFalse(h.select_first88(mm,oo)['admitted'])
    def test_control_cannot_promote_later_never_selects(self):
        mm,oo=selection_fixture();expected=h.select_first88(mm,oo)
        for arm in h.NAMES:
            mm[arm]['later34']={'cagr':1000.};oo[arm][88:]=[dict(symbols=[arm]) for _ in range(34)]
        self.assertEqual(h.select_first88(mm,oo),expected)
        mm[h.NAMES[1]]['same_close_zero']['training']['cagr']=2.
        self.assertEqual(h.select_first88(mm,oo)['primary'],h.NAMES[0])
    def test_walk_prefix_split_and_cutoff(self):
        mm,oo=selection_fixture()
        for n in (36,60,76,88):self.assertEqual(h.select_first88(mm,oo,n)['training_months'],n)
        for n in (1,89,True):
            with self.assertRaises(ValueError):h.select_first88(mm,oo,n)
    def test_exact_original_guard_code_pinned(self):
        self.assertEqual(h.sha(h.GUARD_PATH),h.GUARD_SHA)
        with patch.object(h,'sha',return_value='bad'):
            with self.assertRaises(ValueError):h.guard({}, {})
    def test_walkforward_training_uses_only_each_declared_prefix(self):
        dates=pd.date_range('2016-08-31',periods=123,freq='BME')
        meta=dict(dates=[str(d.date()) for d in dates[:-1]],ends=[str(d.date()) for d in dates[1:]])
        _,orders=selection_fixture();calls=[]
        class Engine:
            @staticmethod
            def replay(px,op,items,*,cost,execution,start,end):
                calls.append((len(items),str(end.date())))
                idx=pd.date_range(start,end,freq='B');eq=pd.Series(100000.+np.arange(len(idx)),index=idx)
                return None,eq,None,None
            @staticmethod
            def stats(eq):return dict(cagr=.5,sharpe=1.5,max_drawdown=-.3)
        metrics,curves=h.training_curves(Engine,None,None,orders,meta,88)
        walk=h.walkforward_training(Engine,None,None,orders,meta,curves)
        self.assertEqual([f['n'] for f in walk['folds']],[36,60,76])
        self.assertEqual([f['end'] for f in walk['folds']],[60,76,88])
        self.assertFalse(walk['later34_used']);self.assertFalse(walk['used_for_primary_selection'])
        self.assertEqual([n for n,end in calls],[88]*6+[36]*6+[60]*6+[76]*6)
        for n,end in calls:self.assertEqual(end,meta['ends'][n-1])
        self.assertTrue(all(f['selection']['primary']==h.NAMES[0] for f in walk['folds']))
    def test_training_rejects_engine_output_after_cutoff(self):
        dates=pd.date_range('2020-01-31',periods=4,freq='BME');meta=dict(dates=[str(d.date()) for d in dates[:-1]],ends=[str(d.date()) for d in dates[1:]])
        _,orders=selection_fixture()
        class Engine:
            @staticmethod
            def replay(px,op,items,*,cost,execution,start,end):
                return None,pd.Series([100000.,100001.],index=[start,end+pd.Timedelta(days=1)]),None,None
        with self.assertRaises(ValueError):h.training_curves(Engine,None,None,orders,meta,2)


class SourceAndLifecycleTests(unittest.TestCase):
    def test_source_prefix_retains_held_latest_excludes_future_and_initial_empty(self):
        meta=dict(dates=['2026-01-30','2026-02-27'],ends=['2026-02-27','2026-03-31'],symbols=[['A'],['A']],etfs=['ETF'])
        dataset=dict(events=[dict(available_from='2026-01-01',held=False),dict(available_from='2026-01-29',held=True),dict(available_from='2026-03-01',amount=object())],method='synthetic')
        sm,sd=h.feature_prefix_inputs(meta,dataset,1)
        self.assertEqual(len(sm['dates']),1);self.assertEqual(len(sd['events']),2);self.assertTrue(sd['events'][-1]['held'])
        self.assertEqual(h.feature_prefix_inputs(meta,dataset,0)[1]['events'],[])
        self.assertEqual(len(dataset['events']),3)
    def test_actual_metadata_binding_no_npz_or_financial_factor(self):
        with patch.object(h.np,'load',side_effect=AssertionError('No numerical cache load')):
            source=h.financial_source_binding()
        self.assertEqual(source['events'],44);self.assertEqual(source['admitted_events'],24)
        self.assertFalse(source['old_shared_pdf_runtime_revalidated'])
        self.assertTrue(all(str(h.ROOT) in key for key in source['input_sha256']))
        self.assertFalse(any('/Applications/' in key for key in source['input_sha256']))
    def test_source_root_mutation_rejected(self):
        with patch.object(h,'SOURCE_EXPECTED',{}):
            with self.assertRaises(ValueError):h.financial_source_binding()
        bad=dict(h.SOURCE_EXPECTED);bad['DATASET.json']='0'*64
        with patch.object(h,'SOURCE_EXPECTED',bad):
            with self.assertRaises(ValueError):h.financial_source_binding()
    def test_normalized_source_review_digest_is_required(self):
        with patch.object(h,'SOURCE_REVIEW_SHA','0'*64):
            with self.assertRaises(ValueError):h.financial_source_binding()
    def test_prepare_gate_precedes_every_real_loader(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaises(ValueError):h.prepare('wrong')
            loader.assert_not_called()
    def test_write_once_register_checks_before_central(self):
        with tempfile.TemporaryDirectory() as t,patch.object(h,'OUT',Path(t)),patch.object(h,'read') as reader:
            with self.assertRaises(ValueError):h.register()
            reader.assert_not_called()
    def test_contract_no_sector_or_initial_gate_and_concept_exact(self):
        c=h.contract();self.assertEqual(c['arms'],list(h.NAMES));self.assertEqual(c['candidate'],h.NAMES[2])
        self.assertEqual(c['dose'],.01);self.assertEqual(c['execution_paths'],9)
        self.assertEqual(c['walkforward_prefixes'],[[36,60],[60,76],[76,88]])
        self.assertNotIn('source_window',c);self.assertIn('No additional first-row',c['scoring'])
    def test_state_counter_predecessor_guard(self):
        rows=[dict(trials=1,directory='older') for _ in range(190)];rows[0]['trials']=898;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state);state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
    def test_synthetic_factor_oracle_prefix_and_delta_integration(self):
        import capex193_factor as factor
        import monthly_cash_ppe193_independent as independent
        import test_capex193_factor as f
        meta,dataset=f.fixture();built=factor.build_features(meta,dataset)
        checked=independent.verify_features(meta,dataset,built);self.assertTrue(checked['passed'])
        eligibility=np.ones((3,49),bool);eligibility[:,0]=False
        _,deltas=h.scale_and_delta(built['arrays'],eligibility,3)
        check=independent.verify_deltas(built['arrays'],deltas,eligibility,dose=.01)
        h.require_delta_audit(check,deltas)
        for n in range(4):
            sm,sd=h.feature_prefix_inputs(meta,dataset,n);small=factor.build_features(sm,sd)
            for key in built['arrays']:np.testing.assert_array_equal(small['arrays'][key],built['arrays'][key][:n])
            self.assertEqual(small['audit']['cells'],built['audit']['cells'][:49*n])
    def test_actual_oracle_receipt_schema_full_synthetic_axes(self):
        import capex193_factor as factor
        import monthly_cash_ppe193_independent as independent
        import test_capex193_factor as f
        dates=[str(d.date()) for d in pd.date_range('2026-02-02',periods=122,freq='B')]
        meta,dataset=f.fixture(dates);built=factor.build_features(meta,dataset)
        checked=independent.verify_features(meta,dataset,built)
        h.require_feature_audit(checked,built['arrays'])
        self.assertEqual(checked['width'],49);self.assertEqual(checked['new_model_fits'],0)
    def test_required_feature_and_delta_flags_fail_closed(self):
        values={key:np.zeros((122,49)) for key in h.RAW.values()};values['common_available']=np.ones((122,49),bool)
        check={key:True for key in h.FEATURE_FLAGS}
        check.update(events=122,width=49,checked_cells=5978,available_cells=5978,unavailable_cells=0,computes_returns=False,new_model_fits=0)
        h.require_feature_audit(check,values)
        for key in h.FEATURE_FLAGS:
            bad=dict(check);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_feature_audit(bad,values)
    def test_accounting_requires_every_nine_path_and_fortyfive_slice(self):
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
                   checked_paths=9,checked_slices=45,terminal_liquidation=False,
                   accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
                   paths={arm:{p[0]:dict(passed=True,monthly_executions=122,slices={s:{} for s in ('full122','first88','early53','recent35','later34')}) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(check)
        check['paths'][h.NAMES[1]].pop('next_open_zero')
        with self.assertRaises(ValueError):h.require_outcome_audit(check)


if __name__=='__main__':unittest.main(verbosity=2)
