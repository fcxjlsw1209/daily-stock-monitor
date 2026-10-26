"""Synthetic-only CMF/state budget, single-episode wiring and lifecycle checks."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import monthly_cmf_state200_host as h

POLICY=dict(budget=.05,ceiling=5,gap=.02)


def fixture(weight_shape='column'):
    windows=np.array([np.linspace(.001,.025,252),np.linspace(.001,.005,252),
        np.linspace(.025,.002,252),np.r_[np.full(50,.002),np.full(202,.01)],
        np.full(252,.008),np.linspace(.030,.010,252)])
    sig=windows[:,-1];median=np.median(windows,axis=1);state=np.where(sig>median,1.,-1.)
    calm=sig*np.sqrt(252)<.10
    gate=np.array([0. if s==med else (2*sum(row<s)+sum(row==s))/252-1 for row,s,med in zip(windows,sig,median)])
    coefficient=np.where(calm,.075,.075*(1-np.where(state>0,.10,0)/.48))
    values=dict(cmf_rank=np.tile([-.3,-.1,0.,.2,.5],(6,1)),state=state,
        gate_candidate=gate,gate_control=.5*state,coefficient=coefficient,
        calm=calm,sigma=sig,median=median,sigma_window=windows)
    weights=np.where(calm,.5,.52);core=weights[:,None]*np.tile([.4,.35,.3,.28,.26],(6,1))
    support=np.tile([.10,.08,.05,.03,.01],(6,1));shape=core.shape
    a=dict(baseline=core+(1-weights[:,None])*support,core=core,core_weight=weights[:,None],
        eligible=np.ones(shape,bool),effects=np.zeros(shape),state_calm10=calm.copy(),
        state_weak_narrow=np.array([False,False,True,False,True,False]),
        extension_available=np.ones(shape,bool),forecast_available=np.ones(shape,bool),rank_log_excess=np.full(shape,.1))
    a['eligible'][:,-1]=False;a['extension_available'][1,0]=False;a['rank_log_excess'][1,1]=np.nan
    if weight_shape=='vector':a['core_weight']=weights.copy()
    if weight_shape=='full':a['core_weight']=np.broadcast_to(weights[:,None],shape).copy()
    meta=dict(dates=['synthetic-%d'%i for i in range(6)],symbols=[list('ABCDE') for _ in range(6)])
    return a,meta,values


def counts_fixture(finite=5970):
    return dict(finite_cmf_cells=finite,neutral_cmf_cells=5978-finite,
        spy_price_observations=122*315,volatility_rank_observations=122*252)


def feature_receipt():
    return dict({k:True for k in h.FEATURE_FLAGS},events=122,symbols_per_event=49,checked_cells=5978,
        computes_returns=False,predictive_model_fits=0,new_model_fits=0,**counts_fixture())


def feature_shapes():
    return {key:np.zeros((122,49) if key=='cmf_rank' else (122,252) if key=='sigma_window' else 122,
        dtype=bool if key=='calm' else float) for key in h.VALUE_KEYS}


def delta_receipt(deltas):
    return dict({key:True for key in h.DELTA_FLAGS},max_absolute_delta=.0375,
        changed_cells={key:int(np.count_nonzero(value)) for key,value in deltas.items()})


class SyntheticStudy:
    def __init__(self):
        self.ep=SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter=h.module('synthetic_cmf200_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
        self.rp=SimpleNamespace(make_orders=self.orders)
    @staticmethod
    def configs():return [dict(id=i) for i in range(5)]
    @staticmethod
    def modified_pre_score(specification,a,n):
        assert specification==dict(id=4)
        return a['baseline'].copy()+np.array([.011,.012,.013,.014,.015])
    @staticmethod
    def boundary(pre,eligible,symbols,effects,original,policy):
        choices=[]
        for score,good,ss in zip(pre,eligible,symbols):
            ids=sorted(np.flatnonzero(good).tolist(),key=lambda j:(-score[j],ss[j]))
            choices.append(ids[:2]+[ids[-1]] if len(ids)>=3 else ids if len(ids)==2 else [])
        return choices,pre.copy()
    def choices(self,specification,a,m,n,*,policy):
        return self.boundary(self.modified_pre_score(specification,a,n)[:n],a['eligible'][:n],m['symbols'][:n],
            a['effects'][:n],a,policy)
    @staticmethod
    def orders(choices,scores,meta):
        return [dict(date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],scores=scores[i].tolist())
            for i,ids in enumerate(choices)]


class ScoreTests(unittest.TestCase):
    def test_exact_delta_scaled_by_existing_support_budget(self):
        a,_,v=fixture();g,d=h.scale_and_delta(v,a,6);k=h.support_scale(a,6)
        for arm,key in h.RAW.items():
            expected=(v['coefficient'][:,None]*v['cmf_rank']*(v[key]-v['state'])[:,None])*k
            expected[0]=0.;np.testing.assert_array_equal(d[arm],expected)
            np.testing.assert_array_equal(g[arm],v[key]);self.assertLessEqual(np.abs(d[arm]).max(),.0375)
        self.assertFalse(a['eligible'][1,-1]);self.assertNotEqual(d[h.NAMES[1]][1,-1],0.)
    def test_median_tie_neutral_and_half_sign_distinct(self):
        a,_,v=fixture();g,d=h.scale_and_delta(v,a,6)
        self.assertEqual(g[h.NAMES[1]][3],0.);self.assertEqual(g[h.NAMES[2]][3],-.5)
        self.assertTrue(np.all(g[h.NAMES[1]]*v['state']>=0))
        self.assertTrue(np.any(d[h.NAMES[1]][3]!=d[h.NAMES[2]][3]))
    def test_missing_cmf_zero_and_first_partial_exact(self):
        a,_,v=fixture();_,d=h.scale_and_delta(v,a,6)
        for arm in d:
            np.testing.assert_array_equal(d[arm][0],np.zeros(5));np.testing.assert_array_equal(d[arm][:,2],np.zeros(6))
    def test_three_original_core_weight_shapes(self):
        expected=None
        for shape in ('vector','column','full'):
            a,_,v=fixture(shape);actual=h.scale_and_delta(v,a,6)
            if expected is None:expected=actual
            for i in (0,1):
                for arm in h.NAMES[1:]:np.testing.assert_array_equal(actual[i][arm],expected[i][arm])
    def test_corrupted_values_and_weight_state_rejected(self):
        for key in ('cmf_rank','state','gate_candidate','gate_control','coefficient','calm','sigma','median','sigma_window','weight','original_calm','extra'):
            a,_,v=fixture('full')
            if key=='weight':a['core_weight'][1,4]=.52
            elif key=='original_calm':a['state_calm10'][1]=False
            elif key=='extra':v['extra']=np.zeros(6)
            elif key=='calm':v[key]=v[key].astype(int)
            elif key=='cmf_rank':v[key][0,0]=np.inf
            elif key=='sigma_window':v[key][1,0]=np.nan
            else:v[key][1]+=.01
            with self.subTest(key=key),self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,a,6)
    def test_support_masks_finite_rank_and_range(self):
        a,_,_=fixture();k=h.support_scale(a,6)
        self.assertEqual(k[1,0],1-.025/.5);self.assertEqual(k[1,1],1-.025/.5);self.assertEqual(k[1,2],1-(.025+.01)/.5)
        for key in ('eligible','forecast_available','extension_available','rank_log_excess'):
            bad=copy.deepcopy(a);bad[key][1,2]=np.nan if key=='rank_log_excess' else False
            self.assertEqual(h.support_scale(bad,6)[1,2],1-.025/.5)
        a['state_weak_narrow'][1]=True;k=h.support_scale(a,6)
        self.assertTrue(np.all(k[1]==.85));self.assertTrue(np.all((k>=.85)&(k<=1)))
        for key in ('eligible','forecast_available','extension_available'):
            bad=copy.deepcopy(a);bad[key]=bad[key].astype(int)
            with self.assertRaises(ValueError):h.support_scale(bad,6)
    def test_prefix_validation_precedes_numeric_conversion(self):
        a,_,v=fixture();expected=h.scale_and_delta(v,a,6)
        for n in range(7):
            aa,vv=copy.deepcopy(a),copy.deepcopy(v)
            for key in vv:vv[key]=vv[key].astype(object);vv[key][n:]='future poison'
            for key in aa:aa[key]=aa[key].astype(object);aa[key][n:]='future poison'
            actual=h.scale_and_delta(vv,aa,n)
            for i in (0,1):
                for arm in h.NAMES[1:]:np.testing.assert_array_equal(actual[i][arm],expected[i][arm][:n])
        for n in (-1,7,True,1.5,np.int64(2)):
            with self.assertRaises(ValueError):h.scale_and_delta(v,a,n)
    def test_original_cmf_support_coefficient_against_frozen_pure_functions(self):
        def extract(path,names,namespace):
            tree=ast.parse(path.read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
            assert {x.name for x in nodes}==set(names)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace);return namespace
        a,_,v=fixture();a['residual']=np.full((6,5),.2);a['old_ml']=np.full((6,5),.1)
        state=extract(h.ROOT/'outputs/monthly_state_rank_research/state_rank.py',['score'],{'np':np})
        adapter=h.module('synthetic_scope200_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
        policy=[dict(key='residual',state='calm10',amount=.025),dict(key='old_ml',state='weak_narrow',amount=.05)]
        upstream=SimpleNamespace(configs=lambda:[{}]*6,modified_pre_score=lambda spec,arr,n:state['score'](policy,arr,n))
        spec=dict(id=3,base='incumbent67',scope='calm_nonweak',budget=.01)
        scope=extract(h.ROOT/'outputs/monthly_model_scope70/study.py',['modified_pre_score'],
            dict(np=np,adapter=adapter,configs=lambda:[spec],upstream=upstream))
        _,d=h.scale_and_delta(v,a,6);old=scope['modified_pre_score'](spec,a,6)
        for arm,key in h.RAW.items():
            changed=copy.deepcopy(a)
            nominal=v['coefficient'][:,None]*v['cmf_rank']*(v[key]-v['state'])[:,None]
            nominal[0]=0.;changed['baseline']+=nominal
            actual=scope['modified_pre_score'](spec,changed,6)
            np.testing.assert_allclose(actual,old+d[arm],rtol=0.,atol=2e-16)
    def test_original_cache_state_exact_and_isolated_flow_only_tolerance(self):
        _,_,v=fixture();high=v['state']>0;flow=v['cmf_rank']*v['state'][:,None]
        self.assertTrue(h.original_state_parity(v,high,flow)['highvol_exact'])
        h.original_state_parity(v,high,flow+1e-13)
        bad=high.copy();bad[1]=~bad[1]
        with self.assertRaises(AssertionError):h.original_state_parity(v,bad,flow)
        with self.assertRaises(AssertionError):h.original_state_parity(v,high,flow+2e-12)


class WiringTests(unittest.TestCase):
    def test_one_original_episode_support_and_order_not_reranked(self):
        a,m,v=fixture();before={k:x.tobytes() for k,x in a.items()};study=SyntheticStudy();pre=study.modified_pre_score(dict(id=4),a,6)
        with patch.object(h,'read',return_value=POLICY):orders,g,d=h.decisions(study,a,m,v,6)
        self.assertEqual(study.ep.select_boundary.call_count,2)
        for arm,call in zip(h.NAMES[1:],study.ep.select_boundary.call_args_list):
            actual,eligible,symbols,effects,original,policy=call.args
            expected=pre.copy();changed=d[arm]!=0.;expected[changed]+=d[arm][changed]
            np.testing.assert_array_equal(actual,expected)
            for key in ('baseline','core'):np.testing.assert_array_equal(original[key],a[key])
            np.testing.assert_array_equal(effects,a['effects'])
            np.testing.assert_array_equal(original['core_weight'],np.broadcast_to(a['core_weight'],a['core'].shape))
            np.testing.assert_array_equal((original['baseline']-original['core'])/(1-original['core_weight']),
                (a['baseline']-a['core'])/(1-a['core_weight']))
            self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])
            global3=sorted(np.flatnonzero(eligible[1]).tolist(),key=lambda j:(-actual[1,j],symbols[1][j]))[:3]
            self.assertNotEqual(orders[arm][1]['symbols'],[symbols[1][j] for j in global3])
        self.assertEqual(before,{k:x.tobytes() for k,x in a.items()})
    def test_zero_delta_does_not_rewrite_negative_zero_pre_score(self):
        a,m,v=fixture();v['cmf_rank'][:]=0.;study=SyntheticStudy()
        pre=study.modified_pre_score(dict(id=4),a,6);pre[2,1]=-0.
        study.modified_pre_score=Mock(return_value=pre)
        with patch.object(h,'read',return_value=POLICY):h.decisions(study,a,m,v,6)
        for call in study.ep.select_boundary.call_args_list:
            actual=call.args[0];self.assertTrue(np.signbit(actual[2,1]));self.assertEqual(actual.tobytes(),pre.tobytes())
    def test_all_prefixes_and_zero_cmf_exact_fallback(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value=POLICY):
            full,gg,dd=h.decisions(SyntheticStudy(),a,m,v,6)
            for n in range(7):
                small,g,d=h.decisions(SyntheticStudy(),a,m,v,n)
                for arm in h.NAMES:self.assertEqual(small[arm],full[arm][:n])
                for arm in h.NAMES[1:]:
                    np.testing.assert_array_equal(g[arm],gg[arm][:n]);np.testing.assert_array_equal(d[arm],dd[arm][:n])
            v['cmf_rank'][:]=0.;empty,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            self.assertTrue(all(empty[arm]==empty[h.NAMES[0]] for arm in h.NAMES))
    def test_real_kernel_oracle_all_source_and_delta_prefixes(self):
        import monthly_cmf_state200_kernel as kernel
        import monthly_cmf_state200_independent as independent
        from test_monthly_cmf_state200_kernel import Tests
        p,m,c=Tests().fixture();v,audit=kernel.build_features(p,m,c)
        check=independent.verify_features(p,m,c,v,audit);self.assertTrue(check['passed'])
        rows,width=v['cmf_rank'].shape;shape=(rows,width)
        a=dict(baseline=np.zeros(shape),core=np.zeros(shape),core_weight=np.where(v['calm'],.5,.52)[:,None],
            state_calm10=v['calm'].copy(),state_weak_narrow=np.zeros(rows,bool),
            extension_available=np.ones(shape,bool),forecast_available=np.ones(shape,bool),eligible=np.ones(shape,bool),rank_log_excess=np.zeros(shape))
        for n in range(rows+1):
            g,d=h.scale_and_delta(v,a,n);check=independent.verify_deltas({k:x[:n] for k,x in v.items()},a,d)
            h.require_delta_audit(check,d)
            pp=h.feature_prefix_panel(p,m,n);small,small_audit=kernel.build_features(pp,m,c,limit=n)
            for key in v:np.testing.assert_array_equal(small[key],v[key][:n])
            self.assertEqual(small_audit['event_history'],audit['event_history'][:n])


def selection_fixture():
    metrics={n:{p:{e:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for e in ('training','early','recent')}
        for p in ('same_close_zero','next_open_cost')} for n in h.NAMES}
    orders={n:[dict(symbols=['A','B','C']) for _ in range(122)] for n in h.NAMES}
    for path in metrics[h.NAMES[1]]:metrics[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return metrics,orders


class SelectionTests(unittest.TestCase):
    def test_later34_never_selects(self):
        metrics,orders=selection_fixture();expected=h.select_first88(metrics,orders)
        self.assertEqual(expected['primary'],h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['full122']={'cagr':1000.}
            for i in range(88,122):orders[name][i]['symbols']=[name]
        self.assertEqual(h.select_first88(metrics,orders),expected)
    def test_control_never_promoted_and_must_be_beaten(self):
        for path in ('same_close_zero','next_open_cost'):
            metrics,orders=selection_fixture();metrics[h.NAMES[2]][path]['training']['cagr']=2.
            result=h.select_first88(metrics,orders);self.assertEqual(result['primary'],h.NAMES[0])
    def test_all_guard_thresholds(self):
        changes=[('same_close_zero','training','cagr',.504),('next_open_cost','training','cagr',.49)]
        changes += [(p,e,k,v) for p in ('same_close_zero','next_open_cost') for e,k,v in
            [('early','cagr',.474),('recent','cagr',.474),('training','sharpe',1.449),('training','max_drawdown',-.331)]]
        for p,e,k,v in changes:
            metrics,orders=selection_fixture();metrics[h.NAMES[1]][p][e][k]=v
            with self.subTest(path=p,key=k):self.assertFalse(h.select_first88(metrics,orders)['admitted'])
    def test_changed_sets_against_both_refs_and_both_eras(self):
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        self.assertFalse(h.select_first88(metrics,orders)['admitted'])
        for i in (1,2,3):orders[h.NAMES[2]][i]['symbols']=['A','E','F']
        self.assertFalse(h.select_first88(metrics,orders)['guards']['changed_support_vs_control'])
        orders[h.NAMES[2]][53]['symbols']=['A','E','F'];self.assertTrue(h.select_first88(metrics,orders)['admitted'])
        for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['C','B','A']
        self.assertEqual(h.select_first88(metrics,orders)['changed_months'],[])


class GovernanceTests(unittest.TestCase):
    def test_context_and_predecessor_metadata_only(self):
        c=h.contract();self.assertEqual(c['execution_paths'],9);self.assertEqual(c['max_absolute_delta'],.0375)
        self.assertIn(h.CONTEXT,h.code_files());self.assertIn(h.ROOT/'work/monthly_nested_trend199_kernel.py',h.code_files())
        self.assertNotIn('horizons',c);self.assertNotIn('blend',c);self.assertEqual(len(h.PREPARED_FILES),7)
        state=h.read(h.CENTRAL);binding=h.predecessor_binding(state)
        self.assertEqual(binding['directory'],'monthly_nested_trend199')
        rows=[dict(trials=1,directory='old') for _ in range(197)];rows[0]['trials']=912;rows[-1]['directory']=h.PRIOR.name
        synthetic=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(synthetic);synthetic['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(synthetic)
    def test_source_binding_metadata_only(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'];binding=h.price_source_binding(raw)
        self.assertEqual(binding['sha256'],h.SOURCE_REVIEW_SHA)
        with patch.object(h,'SOURCE_REVIEW_SHA','0'*64),self.assertRaises(ValueError):h.price_source_binding(raw)
        bad=dict(raw);bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):h.price_source_binding(bad)
    def test_source_prefix_all_five_frames(self):
        f=pd.DataFrame({'SPY':[1.,2.,'future']},index=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06']))
        panel={key:f.copy() for key in h.PANEL_KEYS};meta=dict(etfs=[str(i) for i in range(17)],dates=['2020-01-03','2020-01-06'])
        self.assertEqual(set(h.price_inputs(panel,meta)),h.PANEL_KEYS)
        for frame in h.feature_prefix_panel(panel,meta,0).values():self.assertEqual(frame.shape,(0,1))
        for frame in h.feature_prefix_panel(panel,meta,1).values():self.assertEqual(frame.iloc[-1,0],2.)
        with self.assertRaises(ValueError):h.price_inputs({},meta)
    def test_counts_and_receipt_flags(self):
        for finite in (0,5970,5978):self.assertEqual(h.estimate_counts(counts_fixture(finite))['finite_cmf_cells'],finite)
        for key in h.ESTIMATE_FIELDS:
            bad=counts_fixture();bad[key]+=1
            with self.assertRaises(ValueError):h.estimate_counts(bad)
        v=feature_shapes();feature=feature_receipt();h.require_feature_audit(feature,v)
        for key in h.FEATURE_FLAGS+('new_model_fits','predictive_model_fits'):
            bad=dict(feature);bad.pop(key)
            with self.assertRaises(ValueError):h.require_feature_audit(bad,v)
        dd={arm:np.zeros((122,49)) for arm in h.NAMES[1:]};check=delta_receipt(dd);h.require_delta_audit(check,dd)
        for key in h.DELTA_FLAGS:
            bad=dict(check);bad.pop(key)
            with self.assertRaises(ValueError):h.require_delta_audit(bad,dd)
    def test_prepared_manifest_counts_and_invariants(self):
        base=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,
            original_array_count=27,**counts_fixture())
        dd={arm:np.zeros((122,49)) for arm in h.NAMES[1:]}
        audit=dict(base,feature=feature_receipt(),delta=delta_receipt(dd),prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),
            feature_prefix_sources_truncated=True,original_orders_exact=True,original_arrays_unchanged=True,
            original_regime_and_flow_checked=True,original_core_metadata_and_support_preserved=True,single_original_episode=True)
        completion=dict(base,new_returns_computed=False,artifacts_sha256={key:'synthetic' for key in h.PREPARED_FILES})
        h.verify_prepared_evidence(completion,feature_shapes(),dd,audit)
        for key in ('original_regime_and_flow_checked','original_core_metadata_and_support_preserved','single_original_episode','original_arrays_unchanged','feature_prefix_sources_truncated'):
            bad=copy.deepcopy(audit);bad[key]=False
            with self.assertRaises(ValueError):h.verify_prepared_evidence(completion,feature_shapes(),dd,bad)
        for key in ('new_predictive_model_fits',)+h.ESTIMATE_FIELDS:
            bad=copy.deepcopy(completion);bad[key]=None
            with self.assertRaises(ValueError):h.verify_prepared_evidence(bad,feature_shapes(),dd,audit)
    def test_nine_paths_fortyfive_slices(self):
        slices={s:{} for s in ('full122','first88','early53','recent35','later34')}
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
            checked_paths=9,checked_slices=45,terminal_liquidation=False,accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
            paths={arm:{p[0]:dict(passed=True,monthly_executions=122,slices=copy.deepcopy(slices)) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(check)
        for mutation in ('arm','path','slice','terminal'):
            bad=copy.deepcopy(check)
            if mutation=='arm':bad['paths'].pop(h.NAMES[2])
            if mutation=='path':bad['paths'][h.NAMES[1]].pop(h.ev.PATHS[2][0])
            if mutation=='slice':bad['paths'][h.NAMES[1]][h.ev.PATHS[0][0]]['slices'].pop('later34')
            if mutation=='terminal':bad['terminal_liquidation']=True
            with self.assertRaises(ValueError):h.require_outcome_audit(bad)
    def test_phase_gates_write_once_protection_and_cas(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
        loader.assert_not_called()
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        parts={x.name:ast.get_source_segment(code,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        self.assertLess(parts['register'].index('code_review_binding()'),parts['register'].index('OUT.mkdir()'))
        self.assertLess(parts['register'].index('len(protected) == 21444'),parts['register'].index('OUT.mkdir()'))
        self.assertEqual(parts['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn("not (OUT / 'prepared').exists()",parts['prepare']);self.assertIn("not (OUT / 'selected').exists()",parts['select'])
        self.assertIn('Selection escaped first88 cutoff',parts['select']);self.assertIn('== 1111',parts['close'])
    def test_changed_source_pin_and_stale_review_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic-source';p.write_text('source');pins={};h.bind(pins,p);saved=dict(pins)
            h.bind(pins,p,h.sha(p));p.write_text('changed')
            with self.assertRaises(ValueError):h.bind(pins,p)
            self.assertEqual(pins,saved)
        for receipt in ({'passed':False},{'passed':True,'unresolved_blockers':[],'code_sha256':{}}):
            with patch.object(h,'read',return_value=receipt),self.assertRaises(ValueError):h.code_review_binding()


if __name__=='__main__':unittest.main()
