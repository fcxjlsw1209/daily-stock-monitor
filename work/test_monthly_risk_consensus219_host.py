"""Synthetic-only219 half-core wiring, source guards and first88 selection."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import math
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('tested_risk_consensus219_host', Path(__file__).with_name('monthly_risk_consensus219_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
POLICY=dict(budget=.05,ceiling=5,gap=.02)


def rebuild(values):
    mask=values['available'];shape=mask.shape
    values['sample_vol']=np.where(mask,1.,0.)
    values['sharpe']=np.full(shape,np.nan)
    values['sharpe'][mask]=values['mean_return'][mask]/values['sample_vol'][mask]
    for raw,key in (('momentum','rank_momentum'),('sharpe','rank_sharpe')):
        ranked=np.full(shape,np.nan)
        for i,valid in enumerate(mask):
            ids=np.flatnonzero(valid);m=len(ids)
            for j in ids:
                row=values[raw][i,ids];x=values[raw][i,j]
                ranked[i,j]=(np.count_nonzero(row<x)+(np.count_nonzero(row==x)+1)/2)/m-.5
        values[key]=ranked
    values['candidate']=np.minimum(values['rank_momentum'],values['rank_sharpe'])
    values['control']=np.full(shape,np.nan)
    for i,j in zip(*np.where(mask)):
        x,y=float(values['rank_momentum'][i,j]),float(values['rank_sharpe'][i,j])
        values['control'][i,j]=x if x==y else math.fsum([x,y])/2.
    return values


def fixture(weight_shape='column'):
    shape=(6,5)
    values={'available':np.ones(shape,bool)}
    for key,row in zip(('momentum','mean_return'),([.02,.04,.06,.08,.10],[.10,.02,.08,.06,.04])):
        values[key]=np.tile(row,(6,1));values[key][5]=.125
    values['available'][2,:4]=False;values['available'][3]=False;values['available'][4,4]=False
    rebuild(values)
    calm=np.array([False,False,False,True,True,False]);weights=np.where(calm,.5,.52)
    core=weights[:,None]*np.tile([.40,.35,.30,.28,.26],(6,1));support=np.tile([.10,.08,.05,.03,.01],(6,1))
    a=dict(baseline=core+(1-weights[:,None])*support,core=core,core_weight=weights[:,None].copy(),
        eligible=np.ones(shape,bool),effects=np.zeros(shape),state_calm10=calm,state_weak_narrow=np.zeros(6,bool))
    if weight_shape=='vector':a['core_weight']=weights.copy()
    if weight_shape=='full':a['core_weight']=np.broadcast_to(weights[:,None],shape).copy()
    a['eligible'][:,-1]=False
    meta=dict(dates=['synthetic-%d'%i for i in range(6)],symbols=[list('ABCDE') for _ in range(6)])
    return a,meta,values


def counts_fixture(available=3,source=None):
    source=available if source is None else source
    return dict(source_available_by_event=[source]+[0]*121,available_by_event=[available]+[0]*121,
        source_available_cells=source,available_cells=available,valid_price_observations=253*source,
        simple_returns=252*source,ranked_observations=2*available,aggregated_scores=2*available)


class SyntheticStudy:
    """Real frozen adapter, but deliberately non-global fake episode choices."""
    def __init__(self):
        self.ep=SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter=SimpleNamespace(score_and_choose=Mock(side_effect=AssertionError('old hand adapter must not be called')))
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
    def test_original_percentile_and_half_core_formula_including_ineligible(self):
        a,_,v=fixture();targets,deltas=h.scale_and_delta(v,a,6)
        expected=np.minimum(np.arange(1,6)/5-.5,np.array([5,1,4,3,2])/5-.5)
        np.testing.assert_array_equal(targets[h.NAMES[1]][1],expected)
        np.testing.assert_array_equal(deltas[h.NAMES[1]][1],.5*(.52*expected-a['core'][1]))
        self.assertFalse(a['eligible'][1,-1]);self.assertNotEqual(deltas[h.NAMES[1]][1,-1],0.)
        self.assertTrue(np.isnan(targets[h.NAMES[1]][0]).all())
        np.testing.assert_array_equal(deltas[h.NAMES[1]][0],np.zeros(5))
    def test_final_targets_are_passed_directly_and_calm_is_suppressed(self):
        a,_,v=fixture();targets,deltas=h.scale_and_delta(v,a,6)
        self.assertTrue(np.isfinite(v['candidate'][0]).all())
        self.assertTrue(np.isfinite(v['candidate'][4,:4]).all())
        for arm,key in h.RAW.items():
            expected=v[key].copy();expected[a['state_calm10']]=np.nan;expected[0]=np.nan
            np.testing.assert_array_equal(targets[arm],expected)
            self.assertTrue(np.isnan(targets[arm][4]).all())
            np.testing.assert_array_equal(deltas[arm][a['state_calm10']],np.zeros((2,5)))
        self.assertFalse(np.array_equal(targets[h.NAMES[1]][1],targets[h.NAMES[2]][1]))
    def test_all_ties_keep_positive_original_rank_and_exact_equal_copy(self):
        a,_,v=fixture();targets,deltas=h.scale_and_delta(v,a,6)
        same=3/5-.5
        for arm in h.NAMES[1:]:
            np.testing.assert_array_equal(targets[arm][5],np.full(5,same))
            self.assertTrue(np.any(deltas[arm][5]!=0.))
        np.testing.assert_array_equal(targets[h.NAMES[1]][5],targets[h.NAMES[2]][5])
    def test_singleton_positive_half_and_empty_no_change(self):
        a,_,v=fixture();targets,deltas=h.scale_and_delta(v,a,6)
        for arm in h.NAMES[1:]:
            self.assertEqual(targets[arm][2,4],.5)
            self.assertEqual(deltas[arm][2,4],.5*(.52*.5-a['core'][2,4]))
            self.assertTrue(np.isnan(targets[arm][3]).all())
            np.testing.assert_array_equal(deltas[arm][3],np.zeros(5))
    def test_finite_zero_target_replaces_core_instead_of_missing(self):
        a,_,v=fixture();v['available'][1,4]=False
        for key in ('momentum','mean_return'):v[key][1,:4]=[1.,2.,3.,4.]
        rebuild(v);targets,deltas=h.scale_and_delta(v,a,6)
        for arm in h.NAMES[1:]:
            self.assertEqual(targets[arm][1,1],0.);self.assertEqual(deltas[arm][1,1],-.5*a['core'][1,1])
            self.assertNotEqual(deltas[arm][1,1],0.);self.assertTrue(np.isnan(targets[arm][1,4]))
            self.assertEqual(deltas[arm][1,4],0.)
    def test_three_legal_weight_shapes_are_identical(self):
        expected=None
        for shape in ('vector','column','full'):
            a,_,v=fixture(shape);actual=h.scale_and_delta(v,a,6)
            if expected is None:expected=actual
            for i in (0,1):
                for arm in h.NAMES[1:]:np.testing.assert_array_equal(actual[i][arm],expected[i][arm])
    def test_bad_weight_state_and_core_bounds_are_rejected(self):
        for kind in ('badweight','lastcell','state','shape','coreinf','corebound'):
            a,_,v=fixture('full')
            if kind=='badweight':a['core_weight'][1]=.51
            if kind=='lastcell':a['core_weight'][1,-1]=.50
            if kind=='state':a['state_calm10']=a['state_calm10'].astype(int)
            if kind=='shape':a['core_weight']=np.ones((6,2))*.52
            if kind=='coreinf':a['core'][1,0]=np.inf
            if kind=='corebound':a['core'][1,0]=.27
            with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,a,6)
    def test_corrupt_ratios_ranks_masks_and_shapes_fail(self):
        for kind in ('rankwrong','rankinf','rankmissing','rawnan','rawinf','rawmissing','mask','shape','extra'):
            a,_,v=fixture()
            if kind=='rankwrong':v['candidate'][1,0]=0.
            if kind=='rankinf':v['control'][1,0]=np.inf
            if kind=='rankmissing':v['candidate'][3,0]=0.
            if kind=='rawnan':v['momentum'][1,0]=np.nan
            if kind=='rawinf':v['mean_return'][1,0]=np.inf
            if kind=='rawmissing':v['sharpe'][3,0]=0.
            if kind=='mask':v['available']=v['available'].astype(int)
            if kind=='shape':v['control']=v['control'][:,:-1]
            if kind=='extra':v['unused']=np.zeros((6,5))
            with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,a,6)
    def test_first_row_is_validated_before_neutralization(self):
        a,_,v=fixture();v['candidate'][0,0]=np.inf
        with self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,a,1)
    def test_future_numeric_features_and_core_not_inspected(self):
        a,_,v=fixture();expected=h.scale_and_delta(v,a,6)
        for n in range(7):
            aa,vv=copy.deepcopy(a),copy.deepcopy(v)
            for key in vv:
                if key!='available':vv[key]=vv[key].astype(object);vv[key][n:]='future poison'
            for key in ('core','core_weight'):
                aa[key]=aa[key].astype(object);aa[key][n:]='future poison'
            aa['state_calm10']=aa['state_calm10'].astype(object)
            aa['state_calm10'][n:]='future poison'
            actual=h.scale_and_delta(vv,aa,n)
            for i in (0,1):
                for arm in h.NAMES[1:]:np.testing.assert_array_equal(actual[i][arm],expected[i][arm][:n])
    def test_invalid_prefix_and_prefix_zero_shape(self):
        a,_,v=fixture()
        for n in (-1,7,True,1.5,np.int64(2)):
            with self.assertRaises(ValueError):h.scale_and_delta(v,a,n)
        targets,deltas=h.scale_and_delta(v,a,0)
        for arm in h.NAMES[1:]:self.assertEqual(targets[arm].shape,deltas[arm].shape);self.assertEqual(targets[arm].shape,(0,5))


    def test_zero_sample_volatility_keeps_source_descriptors_but_no_targets(self):
        a,_,v=fixture();row=3
        self.assertTrue(np.isfinite(v['momentum'][row]).all())
        self.assertTrue(np.isfinite(v['mean_return'][row]).all())
        np.testing.assert_array_equal(v['sample_vol'][row],np.zeros(5))
        targets,deltas=h.scale_and_delta(v,a,6)
        for arm in h.NAMES[1:]:
            self.assertTrue(np.isnan(targets[arm][row]).all());np.testing.assert_array_equal(deltas[arm][row],np.zeros(5))
        bad=copy.deepcopy(v);bad['sample_vol'][1,0]=0.
        with self.assertRaises(ValueError):h.scale_and_delta(bad,a,6)

    def test_minimum_is_not_second_ranked_or_recentered(self):
        a,_,v=fixture();t,_=h.scale_and_delta(v,a,6)
        self.assertLess(float(np.sum(t[h.NAMES[1]][1])),float(np.sum(t[h.NAMES[2]][1])))
        self.assertEqual(t[h.NAMES[1]][1,0],t[h.NAMES[1]][1,1])
        bad=copy.deepcopy(v);bad['candidate'][1]=np.arange(1,6)/5-.5
        with self.assertRaises(AssertionError):h.scale_and_delta(bad,a,6)


class RealAdapterTests(unittest.TestCase):
    def test_real_adapter_preserves_original_support_once_and_returned_order(self):
        a,m,v=fixture();aa,vv=copy.deepcopy(a),copy.deepcopy(v);study=SyntheticStudy()
        adapter=h.core_adapter();pre=study.modified_pre_score(dict(id=4),a,6)
        with patch.object(h,'read',return_value=POLICY),patch.object(h,'core_adapter',return_value=adapter):
            orders,targets,deltas=h.decisions(study,a,m,v,6)
        self.assertEqual(study.ep.select_boundary.call_count,2)
        study.adapter.score_and_choose.assert_not_called()
        for arm,call in zip(h.NAMES[1:],study.ep.select_boundary.call_args_list):
            actual,eligible,symbols,effects,original,policy=call.args
            expected=pre.copy();mask=deltas[arm]!=0.;expected[mask]+=deltas[arm][mask]
            np.testing.assert_array_equal(actual,expected)
            for key in ('baseline','core','state_calm10','state_weak_narrow'):np.testing.assert_array_equal(original[key],a[key])
            np.testing.assert_array_equal(original['core_weight'],np.broadcast_to(a['core_weight'],a['core'].shape))
            np.testing.assert_array_equal((original['baseline']-original['core'])/(1-original['core_weight']),
                (a['baseline']-a['core'])/(1-a['core_weight']))
            np.testing.assert_array_equal(eligible,a['eligible']);np.testing.assert_array_equal(effects,a['effects'])
            choices,scores=study.boundary(actual,eligible,symbols,effects,original,policy)
            self.assertEqual(orders[arm],study.orders(choices,scores,m))
            top3=sorted(np.flatnonzero(eligible[1]).tolist(),key=lambda j:(-actual[1,j],symbols[1][j]))[:3]
            self.assertNotEqual(orders[arm][1]['symbols'],[symbols[1][j] for j in top3])
            self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])
        for key in a:np.testing.assert_array_equal(a[key],aa[key])
        for key in v:np.testing.assert_array_equal(v[key],vv[key])
    def test_every_decision_prefix_empty_and_all_missing_fallback(self):
        a,m,v=fixture();adapter=h.core_adapter()
        with patch.object(h,'read',return_value=POLICY),patch.object(h,'core_adapter',return_value=adapter):
            full,tt,dd=h.decisions(SyntheticStudy(),a,m,v,6)
            for n in range(7):
                small,targets,deltas=h.decisions(SyntheticStudy(),a,m,v,n)
                for arm in h.NAMES:self.assertEqual(small[arm],full[arm][:n])
                for arm in h.NAMES[1:]:
                    np.testing.assert_array_equal(targets[arm],tt[arm][:n]);np.testing.assert_array_equal(deltas[arm],dd[arm][:n])
            v['available'][:]=False;rebuild(v)
            empty,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            self.assertTrue(all(empty[arm]==empty[h.NAMES[0]] for arm in h.NAMES))
    def test_real_producer_and_oracle_synthetic_interfaces(self):
        import monthly_risk_consensus219_kernel as kernel
        import monthly_risk_consensus219_independent as independent
        from test_monthly_risk_consensus219_kernel import fixture as price_fixture
        panel,meta,calendar=price_fixture();values,audit=kernel.build_features(panel,meta,calendar)
        check=independent.verify_features(panel,meta,calendar,values,audit)
        self.assertTrue(check['passed'])
        self.assertEqual(h.FEATURE_FLAGS,independent.FEATURE_FLAGS)
        rows,width=values['available'].shape
        calm=np.array([False,True,False])[:rows]
        a=dict(baseline=np.zeros((rows,width)),core=np.zeros((rows,width)),core_weight=np.where(calm,.5,.52)[:,None],state_calm10=calm)
        for n in range(rows+1):
            targets,deltas=h.scale_and_delta(values,a,n)
            check=independent.verify_deltas({k:v[:n] for k,v in values.items()},a,targets,deltas,blend=h.BLEND)
            h.require_delta_audit(check,deltas)
            self.assertEqual(check['checked_events'],n)
            pp=h.feature_prefix_panel(panel,meta,n);mm=h.feature_prefix_meta(meta,n)
            prefix,small=kernel.build_features(pp,mm,calendar)
            self.assertTrue(independent.verify_features(pp,mm,calendar,prefix,small)['passed'])
            for key in values:
                if n:np.testing.assert_array_equal(prefix[key],values[key][:n])
                else:self.assertEqual(prefix[key].shape,(0,0))

    def test_complete_122x49_synthetic_feature_receipt_and_zero_risk(self):
        import monthly_risk_consensus219_kernel as kernel
        import monthly_risk_consensus219_independent as independent
        dates=pd.bdate_range('2013-01-01','2027-01-06');x=np.arange(len(dates),dtype=float)
        symbols=['S'+str(i) for i in range(49)]
        px=pd.DataFrame({s:100.+i+.01*x+2*np.sin(x*(.1+.001*i)) for i,s in enumerate(symbols)},index=dates)
        px[symbols[-1]]=100.
        monthends=[str(day.date()) for i,day in enumerate(dates[:-1]) if dates[i+1].month!=day.month and day>=pd.Timestamp('2016-08-31')][:121]
        m=dict(dates=['2016-08-17']+monthends,symbols=[symbols]*122,etfs=[])
        p={'adj':px};c=dates.strftime('%Y-%m-%d').tolist();values,audit=kernel.build_features(p,m,c)
        check=independent.verify_features(p,m,c,values,audit);h.require_feature_audit(check,values)
        self.assertEqual(check['source_available_cells'],5978);self.assertEqual(check['available_cells'],5856)
        self.assertEqual(check['simple_returns'],252*5978);self.assertEqual(check['aggregated_scores'],2*5856)
        self.assertTrue(np.isnan(values['candidate'][:,-1]).all())


def selection_fixture():
    metrics={n:{p:{e:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for e in ('training','early','recent')}
                for p in ('same_close_zero','next_open_cost')} for n in h.NAMES}
    orders={n:[dict(symbols=['A','B','C']) for _ in range(122)] for n in h.NAMES}
    for path in metrics[h.NAMES[1]]:metrics[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return metrics,orders


class SelectionTests(unittest.TestCase):
    def test_later34_and_extra_metrics_never_choose_arm(self):
        metrics,orders=selection_fixture();expected=h.select_first88(metrics,orders)
        self.assertEqual(expected['primary'],h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['full122']=dict(cagr=1000. if name!=h.NAMES[1] else -1.)
            for i in range(88,122):orders[name][i]['symbols']=[name]
        self.assertEqual(h.select_first88(metrics,orders),expected)
    def test_mean_control_never_promoted_and_must_be_beaten(self):
        for path in ('same_close_zero','next_open_cost'):
            metrics,orders=selection_fixture();metrics[h.NAMES[2]][path]['training']['cagr']=2.
            result=h.select_first88(metrics,orders)
            self.assertEqual(result['primary'],h.NAMES[0]);self.assertFalse(result['admitted'])
    def test_gain_era_risk_and_set_not_order_guards(self):
        failures=[('same_close_zero','training','cagr',.504),('next_open_cost','training','cagr',.49)]
        failures += [(path,era,key,value) for path in ('same_close_zero','next_open_cost') for era,key,value in
            [('early','cagr',.474),('recent','cagr',.474),('training','sharpe',1.449),('training','max_drawdown',-.331)]]
        for path,era,key,value in failures:
            metrics,orders=selection_fixture();metrics[h.NAMES[1]][path][era][key]=value
            with self.subTest(path=path,era=era,key=key):self.assertFalse(h.select_first88(metrics,orders)['admitted'])
        metrics,orders=selection_fixture()
        for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['C','B','A']
        for i in (90,100,110):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
        self.assertEqual(h.select_first88(metrics,orders)['changed_months'],[])
    def test_both_eras_and_three_changed_sets_against_control(self):
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        self.assertFalse(h.select_first88(metrics,orders)['admitted'])
        for i in (1,2,3):orders[h.NAMES[2]][i]['symbols']=['A','E','F']
        self.assertFalse(h.select_first88(metrics,orders)['guards']['changed_support_vs_control'])
        orders[h.NAMES[2]][53]['symbols']=['A','E','F']
        self.assertTrue(h.select_first88(metrics,orders)['admitted'])


class SourceAndLifecycleTests(unittest.TestCase):
    def test_metadata_source_binding_and_feature_boundary_flags(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'];binding=h.price_source_binding(raw)
        self.assertEqual(binding['sha256'],h.SOURCE_REVIEW_SHA)
        self.assertTrue(binding['source_admission_metadata_only'])
        with patch.object(h,'SOURCE_REVIEW_SHA','0'*64):
            with self.assertRaises(ValueError):h.price_source_binding(raw)
        receipt=h.read(h.SOURCE_REVIEW)
        for key in ('actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed','npz_decoded','actual_source_coverage_recomputed'):
            for value in (True,0,None):
                changed=copy.deepcopy(receipt);changed[key]=value
                with patch.object(h,'read',return_value=changed),self.assertRaises(ValueError):h.price_source_binding(raw)
        for key in ('source_admission_metadata_only','metadata_axes_verified','calendar_clock_verified','raw_source_hashes_verified','raw_source_map_matches_original','integration_source_verified','source_window_matches199'):
            changed=copy.deepcopy(receipt);changed[key]=False
            with patch.object(h,'read',return_value=changed),self.assertRaises(ValueError):h.price_source_binding(raw)
    def test_predecessor_metadata_and_central_counter_contract(self):
        state={'completed':[{'completion':{'path':str(h.PRIOR/'COMPLETION.json'),'sha256':h.PRIOR_EXPECTED['COMPLETION.json']}}]};binding=h.predecessor_binding(state)
        self.assertEqual(binding['directory'],'monthly_leader_gate218')
        self.assertEqual(binding['input_sha256'][str(h.PRIOR/'PLAN.json')],h.PRIOR_EXPECTED['PLAN.json'])
        rows=[dict(trials=1,directory='old') for _ in range(216)];rows[0]['trials']=950;rows[-1]['directory']=h.PRIOR.name
        synthetic=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(synthetic);synthetic['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(synthetic)
        text=Path(h.__file__).read_text();self.assertIn("len(state['completed']) == 217",text);self.assertIn("== 1168",text)
    def test_changed_inherited_pin_fails_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic-source';p.write_text('source');pins={};h.bind(pins,p);saved=dict(pins)
            h.bind(pins,p,h.sha(p));self.assertEqual(pins,saved);p.write_text('changed')
            with self.assertRaises(ValueError):h.bind(pins,p)
            self.assertEqual(pins,saved)
    def test_unregistered_prepare_never_loads_real_source(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
        loader.assert_not_called()
    def test_independent_code_review_gate_rejects_missing_or_stale(self):
        with patch.object(h,'read',return_value={'passed':False}):
            with self.assertRaises(ValueError):h.code_review_binding()
        with patch.object(h,'read',return_value={'passed':True,'unresolved_blockers':[],'code_sha256':{}}):
            with self.assertRaises(ValueError):h.code_review_binding()
    def test_source_prefix_zero_physically_empty_and_suffix_removed(self):
        prices=pd.DataFrame({'A':[1.,2.,'future poison']},index=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06']))
        panel={'adj':prices};meta={'etfs':[str(i) for i in range(17)],'dates':['2020-01-03','2020-01-06']}
        self.assertIs(h.price_inputs(panel,meta)['adj'],prices)
        self.assertEqual(h.feature_prefix_panel(panel,meta,0)['adj'].shape,(0,1))
        self.assertEqual(h.feature_prefix_panel(panel,meta,1)['adj'].iloc[-1,0],2.)
        self.assertEqual(len(h.feature_prefix_panel(panel,meta,1)['adj']),2)
        with self.assertRaises(ValueError):h.price_inputs({},meta)
    def test_fixed_contract_and_descriptive_count_arithmetic(self):
        c=h.contract();self.assertEqual(c['blend'],.5);self.assertEqual(c['max_absolute_delta'],.26)
        self.assertIn('253',c['source_window']['prices']);self.assertIn('t-294..t-42',c['source_window']['prices'])
        self.assertIn('UNKNOWN',c['context']['novelty']);self.assertIn(h.CONTEXT,h.code_files())
        self.assertIn(h.COVERAGE,h.code_files());self.assertEqual(h.sha(h.COVERAGE),h.COVERAGE_SHA)
        self.assertIn(h.ROOT/'work/monthly_nested_trend199_kernel.py',h.code_files())
        self.assertEqual(len(h.PREPARED_FILES),7);self.assertEqual(c['execution_paths'],9)
        for n in (0,1,49):
            counts=counts_fixture(n);self.assertEqual(h.estimate_counts(counts)['aggregated_scores'],2*n)
            for key in h.ESTIMATE_FIELDS:
                bad=copy.deepcopy(counts);bad[key]+=1
                with self.assertRaises(ValueError):h.estimate_counts(bad)
        self.assertEqual(h.estimate_counts(counts_fixture(0,3))['simple_returns'],252*3)
    def test_counts_require_common_support_and_integer_prefix_counts(self):
        for change in ('support','source_total','bool','width','length'):
            counts=counts_fixture()
            if change=='support':counts['available_by_event'][0]=4
            if change=='source_total':counts['source_available_cells']=2
            if change=='bool':counts['source_available_by_event'][0]=True
            if change=='width':counts['source_available_by_event'][0]=50
            if change=='length':counts['available_by_event'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):h.estimate_counts(counts)
    def test_prepared_receipt_requires_support_single_episode_counts_and_prefixes(self):
        mask=np.zeros((122,49),bool);mask[0,:3]=True
        counts=dict(counts_fixture(),unavailable_cells=5975)
        feature=dict({k:True for k in h.FEATURE_FLAGS},events=122,symbols_per_event=49,checked_cells=5978,
            computes_returns=False,producer_modules_imported=False,predictive_model_fits=0,new_model_fits=0,new_predictive_model_fits=0,numerical_comparison='exact canonical binary64 features, ranks and complete audit',numerical_tolerance=0.,**counts)
        deltas={arm:np.zeros((122,49)) for arm in h.NAMES[1:]}
        delta=dict({k:True for k in ('passed','half_core_formula_checked','original_pct_ranks_preserved','first_partial_unchanged','calm_rows_unchanged','missing_preserves_core','core_weights_checked','no_second_rank','independent_delta_rebuild')},blend=.5,max_absolute_delta=.26,changed_cells={k:0 for k in deltas})
        base=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,**{k:counts[k] for k in h.ESTIMATE_FIELDS})
        audit=dict(base,feature=feature,delta=delta,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),
            feature_prefix_sources_truncated=True,feature_prefix_metadata_truncated=True,feature_prefix_independently_verified=True,source_admission_metadata_only=True,source_support_exact=True,original_orders_exact=True,original_arrays_unchanged=True,
            original_core_metadata_and_support_preserved=True,single_original_episode=True)
        receipt=dict(base,new_returns_computed=False,artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES})
        h.verify_prepared_evidence(receipt,{'available':mask},deltas,audit)
        for key in ('original_core_metadata_and_support_preserved','single_original_episode','original_arrays_unchanged','feature_prefix_sources_truncated','feature_prefix_metadata_truncated','feature_prefix_independently_verified','source_admission_metadata_only','source_support_exact'):
            bad=copy.deepcopy(audit);bad[key]=False
            with self.assertRaises(ValueError):h.verify_prepared_evidence(receipt,{'available':mask},deltas,bad)
        for key in ('new_predictive_model_fits',)+h.ESTIMATE_FIELDS:
            bad=copy.deepcopy(receipt);bad[key]=None
            with self.assertRaises(ValueError):h.verify_prepared_evidence(bad,{'available':mask},deltas,audit)
    def test_all_feature_and_delta_review_flags_required(self):
        counts=dict(counts_fixture(),unavailable_cells=5975)
        mask=np.zeros((122,49),bool);mask[0,:3]=True
        feature=dict({k:True for k in h.FEATURE_FLAGS},events=122,symbols_per_event=49,checked_cells=5978,
            computes_returns=False,producer_modules_imported=False,predictive_model_fits=0,new_model_fits=0,new_predictive_model_fits=0,numerical_comparison='exact canonical binary64 features, ranks and complete audit',numerical_tolerance=0.,**counts)
        for key in h.FEATURE_FLAGS:
            bad=dict(feature);bad.pop(key)
            with self.assertRaises(ValueError):h.require_feature_audit(bad,{'available':mask})
        deltas={arm:np.zeros((122,49)) for arm in h.NAMES[1:]}
        keys=('passed','half_core_formula_checked','original_pct_ranks_preserved','first_partial_unchanged','calm_rows_unchanged','missing_preserves_core','core_weights_checked','no_second_rank','independent_delta_rebuild')
        delta=dict({k:True for k in keys},blend=.5,max_absolute_delta=.26,changed_cells={k:0 for k in deltas})
        for key in keys:
            bad=dict(delta);bad.pop(key)
            with self.assertRaises(ValueError):h.require_delta_audit(bad,deltas)
    def test_outcomes_require_all_three_arms_nine_paths_fortyfive_slices(self):
        slices={'full122':{},'first88':{},'early53':{},'recent35':{},'later34':{}}
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
            checked_paths=9,checked_slices=45,terminal_liquidation=False,
            accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
            paths={arm:{p[0]:dict(passed=True,monthly_executions=122,slices=copy.deepcopy(slices)) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(check)
        for mutation in ('arm','path','slice','terminal'):
            bad=copy.deepcopy(check)
            if mutation=='arm':bad['paths'].pop(h.NAMES[2])
            if mutation=='path':bad['paths'][h.NAMES[1]].pop(h.ev.PATHS[2][0])
            if mutation=='slice':bad['paths'][h.NAMES[1]][h.ev.PATHS[0][0]]['slices'].pop('later34')
            if mutation=='terminal':bad['terminal_liquidation']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):h.require_outcome_audit(bad)

    def test_source_coverage_clocks_and_incomplete_identities_match(self):
        import monthly_risk_consensus219_kernel as kernel
        from test_monthly_risk_consensus219_kernel import fixture as price_fixture
        panel,meta,calendar=price_fixture();_,audit=kernel.build_features(panel,meta,calendar)
        with patch.object(h,'read',return_value={'sessions':calendar}):self.assertTrue(h.require_source_support(audit))
        for kind in ('clock','count','identity','series'):
            bad=copy.deepcopy(audit)
            if kind=='clock':bad['event_history'][0]['price_start']='2099-01-01'
            if kind=='count':bad['event_history'][0]['source_available_count']+=1
            if kind=='identity':bad['event_history'][0]['cells'][0]['symbol']='WRONG'
            if kind=='series':bad['source_available_by_event'][0]+=1
            with patch.object(h,'read',return_value={'sessions':calendar}),self.assertRaises(ValueError):h.require_source_support(bad)

    def test_exact38_result_manifest_missing_extra_and_completion_rejected(self):
        manifest={name:'synthetic-digest' for name in h.RESULT_FILES}
        self.assertEqual(len(manifest),38);self.assertNotIn('COMPLETION.json',manifest)
        h.require_result_manifest(manifest)
        for missing in h.RESULT_FILES:
            bad=dict(manifest);bad.pop(missing)
            with self.subTest(missing=missing),self.assertRaises(ValueError):h.require_result_manifest(bad)
        for extra in ('COMPLETION.json','EXTRA.json'):
            bad=dict(manifest);bad[extra]='synthetic'
            with self.subTest(extra=extra),self.assertRaises(ValueError):h.require_result_manifest(bad)

    def test_mutable_central_rejected_from_review_pins(self):
        review=dict(passed=True,unresolved_blockers=[],context_sha256=h.CONTEXT_SHA,actual_model_fits_computed=False,code_sha256={str(p.relative_to(h.ROOT)):'fake' for p in h.code_files()[:6]},
            input_sha256={str(h.CENTRAL):'fake'},actual_features_computed=False,actual_portfolio_returns_computed=False,
            production_files_modified=False,strategy_registered_by_reviewer=False)
        with patch.object(h,'read',return_value=review),patch.object(h,'sha',return_value='fake'):
            with self.assertRaisesRegex(ValueError,'Mutable central'):h.code_review_binding()

    def test_write_once_and_protection_before_publishing_and_central_cas(self):
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        parts={x.name:ast.get_source_segment(code,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        register=parts['register'];self.assertLess(register.index('code_review_binding()'),register.index('OUT.mkdir()'))
        self.assertLess(register.index('len(protected) == 21444'),register.index('OUT.mkdir()'))
        self.assertIn('Code changed since tests',register)
        self.assertLess(register.index('Mutable central cannot enter registered input pins'),register.index('OUT.mkdir()'))
        closer=parts['close']
        self.assertLess(closer.index('require_result_manifest('),closer.index('require_outcome_audit('))
        self.assertEqual(parts['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn("not (OUT / 'prepared').exists()",parts['prepare'])
        self.assertIn("not (OUT / 'selected').exists()",parts['select'])
        self.assertIn('Selection escaped first88 cutoff',parts['select'])


class ImmutableContractTests(unittest.TestCase):
    def test_only_exact_registered_external_runtime_is_allowed(self):
        runtime,digest=next(iter(h.EXTERNAL_RUNTIME.items()))
        with patch.object(h,'sha',return_value=digest),patch.object(Path,'is_file',return_value=True):
            h.verify({runtime:digest})
            for path,value in ((runtime,'wrong'),('/Library/Developer/CommandLineTools/other.py',digest),('/tmp/other.py',digest)):
                with self.subTest(path=path,value=value),self.assertRaises(ValueError):h.verify({path:value})
    def test_review_requires_context_fit_scope_and_nonempty_pins(self):
        files=[h.ROOT/'work'/('synthetic'+str(i)) for i in range(6)]
        receipt=dict(passed=True,unresolved_blockers=[],context_sha256=h.CONTEXT_SHA,
            code_sha256={str(p.relative_to(h.ROOT)):'digest' for p in files},input_sha256={'work/fake':'digest'})
        for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer'):receipt[key]=False
        with patch.object(h,'code_files',return_value=files),patch.object(h,'read',return_value=receipt),patch.object(h,'sha',return_value='digest'),patch.object(h,'bind'),patch.object(h,'verify'):
            h.code_review_binding()
            for key,value in (('context_sha256','wrong'),('actual_model_fits_computed',True),('input_sha256',{})):
                old=receipt[key];receipt[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):h.code_review_binding()
                receipt[key]=old


if __name__=='__main__':unittest.main()

