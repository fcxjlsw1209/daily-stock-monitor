"""Synthetic-only host tests: three-dose algebra, original wiring and four-arm governance."""
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

spec=importlib.util.spec_from_file_location('tested_reallocation186_host',Path(__file__).with_name('monthly_aux_reallocation186_host.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture():
    n,width=8,5; shape=(n,width)
    calm=np.array([False,True,True,True,False,False,False,False])
    weak=np.array([False,False,False,True,False,True,False,True])
    high=np.array([False,False,False,False,False,False,True,True])
    cw=np.where(calm,.5,.52)[:,None]
    vw=np.where(calm,.2,.219375*(1.-np.where(high,.10,0.)/.48))[:,None]
    original=dict(baseline=np.tile([.2,.199,.198,.197,.196],(n,1)),core=np.zeros(shape),
        core_weight=cw,state_calm10=calm,state_weak_narrow=weak,eligible=np.ones(shape,bool),
        effects=np.zeros(shape),extension_available=np.zeros(shape,bool),forecast_available=np.ones(shape,bool),
        rank_log_excess=np.full(shape,.1))
    original['extension_available'][2]=True
    original['eligible'][:,-1]=False
    source=dict(vol=np.tile([-.3,-.1,.1,.3,.5],(n,1)),vol_weight=vw,state_highvol=high)
    values=dict(vol_rank=source['vol'].copy(),
        unit_delta=(original['baseline']-original['core'])-(1.-cw)*source['vol'])
    meta=dict(dates=['synthetic-%d'%i for i in range(n)],symbols=[list('ABCDE') for _ in range(n)])
    return original,source,meta,values


class SyntheticStudy:
    def __init__(self):
        self.ep=SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter=SimpleNamespace(score_and_choose=Mock(side_effect=self.choose))
        self.rp=SimpleNamespace(make_orders=self.orders)
    @staticmethod
    def configs():return [dict(id=i) for i in range(5)]
    @staticmethod
    def modified_pre_score(spec,a,n):
        assert spec==dict(id=4)
        return a['baseline'].copy()+.11
    @staticmethod
    def boundary(pre,a,m,n):
        choices=[]
        for i in range(n):
            ids=sorted(np.flatnonzero(a['eligible'][i]),key=lambda j:(-pre[i,j],j))
            choices.append([m['symbols'][i][j] for j in ids[:3]] if len(ids)>=2 else [])
        return choices,pre[:n].copy()
    def choices(self,spec,a,m,n,*,policy):return self.boundary(self.modified_pre_score(spec,a,n),a,m,n)
    @staticmethod
    def orders(choices,scores,meta):
        return [dict(date=meta['dates'][i],symbols=ss,scores=scores[i].tolist()) for i,ss in enumerate(choices)]
    def choose(self,base,weight,extra,a,m,n,*,pre_score,effects,episode_policy,boundary_selector):
        assert base=='incumbent66' and weight==0. and effects is a['effects']
        assert np.isnan(extra).all() and extra.shape==a['baseline'].shape
        return boundary_selector(pre_score,a,m,n)


def original_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class ReallocationAndWiringTests(unittest.TestCase):
    def test_original_auxiliary_algebra_and_three_fixed_doses(self):
        import monthly_aux_reallocation186_kernel as kernel
        a,s,_,v=fixture();shape=a['baseline'].shape
        other=np.tile([.4,.2,0.,-.2,-.4],(shape[0],1))
        cw=a['core_weight'];wv=s['vol_weight'];nonvol=1.-cw-wv
        a['baseline']=a['core']+wv*s['vol']+nonvol*other
        v=dict(vol_rank=s['vol'].copy(),unit_delta=(a['baseline']-a['core'])-(1.-cw)*s['vol'])
        dd,audit=kernel.build_deltas(v,a,s)
        self.assertEqual(set(dd),set(h.LAMBDAS))
        for arm,dose in h.LAMBDAS.items():
            expected=a['core']+(wv-dose*nonvol)*s['vol']+(1.+dose)*nonvol*other
            np.testing.assert_allclose((a['baseline']+dd[arm])[1:],expected[1:],rtol=0,atol=1e-16)
            np.testing.assert_array_equal(dd[arm][0],np.zeros(shape[1]))
            np.testing.assert_array_equal(dd[arm][1:],dose*v['unit_delta'][1:])
        self.assertFalse(a['eligible'][1,-1]);self.assertNotEqual(dd[h.NAMES[1]][1,-1],0.)

    def test_all_seven_original_state_coefficients_stay_positive(self):
        a,source,_,_=fixture();a['eligible'][:]=True
        calm=a['state_calm10'][:,None];weak=a['state_weak_narrow'][:,None]
        active=calm&~weak&a['extension_available']&a['forecast_available']&a['eligible']&np.isfinite(a['rank_log_excess'])
        effective=source['vol_weight']*(1.-(.025*calm+.05*weak+.01*active)/(1.-a['core_weight']))
        received=1.-a['core_weight']-source['vol_weight']
        after=effective-.10*received
        expected=np.array([.16,.156,.14,.1933125,.1704609375,.1430390625,.1249482421875])
        np.testing.assert_allclose(after[1:,0],expected,rtol=0,atol=1e-15)
        self.assertTrue((after>0.).all())
        self.assertAlmostEqual(float(after.min()),.1249482421875,places=15)
        self.assertAlmostEqual(float((.10*received).max()),.0306328125,places=15)
        # Compile only original pure support-subtraction functions, never loaders or prices.
        state=original_functions(h.ROOT/'outputs/monthly_state_rank_research/state_rank.py',['score'],{'np':np})
        adapter=h.module('aux186_state_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
        scope_spec=dict(id=3,base='incumbent67',scope='calm_nonweak',budget=.01)
        policy=[dict(key='residual',state='calm10',amount=.025),dict(key='old_ml',state='weak_narrow',amount=.05)]
        a['residual']=np.full(a['baseline'].shape,.2);a['old_ml']=np.full(a['baseline'].shape,.1)
        upstream=SimpleNamespace(configs=lambda:[{}]*6,modified_pre_score=lambda spec,arr,n:state['score'](policy,arr,n))
        scope=original_functions(h.ROOT/'outputs/monthly_model_scope70/study.py',['modified_pre_score'],
            {'np':np,'adapter':adapter,'configs':lambda:[scope_spec],'upstream':upstream})
        def revised(arr,vol):
            return scope['modified_pre_score'](scope_spec,arr,8)+.10*((arr['baseline']-arr['core'])-(1.-arr['core_weight'])*vol)
        old=revised(a,source['vol'])
        changed=copy.deepcopy(a);changed['baseline']+=source['vol_weight']*.125
        actual=revised(changed,source['vol']+.125)
        np.testing.assert_allclose(((actual-old)/.125)[1:],after[1:],rtol=0,atol=2e-15)

    def test_reverse_direction_with_positive_doses(self):
        import monthly_aux_reallocation186_kernel as kernel
        a,s,_,v=fixture();dd,audit=kernel.build_deltas(v,a,s)
        former_unit=(1.-a['core_weight'])*s['vol']-(a['baseline']-a['core'])
        np.testing.assert_array_equal(v['unit_delta'],-former_unit)
        for arm,dose in h.LAMBDAS.items():
            self.assertGreater(dose,0.)
            np.testing.assert_array_equal(dd[arm][1:],-dose*former_unit[1:])
        self.assertEqual(audit['lambdas'],h.LAMBDAS)

    def test_missing_original_vol_rank_zero_is_not_unavailable(self):
        import monthly_aux_reallocation186_kernel as kernel
        a,s,_,v=fixture();s['vol'][1,0]=0.
        v=dict(vol_rank=s['vol'].copy(),unit_delta=(a['baseline']-a['core'])-(1.-a['core_weight'])*s['vol'])
        dd,_=kernel.build_deltas(v,a,s)
        for arm,dose in h.LAMBDAS.items():self.assertEqual(dd[arm][1,0],dose*(a['baseline'][1,0]-a['core'][1,0]))

    def test_zero_unit_is_exactly_zero_and_finite_corruption_rejected(self):
        import monthly_aux_reallocation186_kernel as kernel
        a,s,_,v=fixture();a['baseline']=a['core']+(1.-a['core_weight'])*s['vol']
        v=dict(vol_rank=s['vol'].copy(),unit_delta=np.zeros(a['core'].shape))
        dd,_=kernel.build_deltas(v,a,s)
        self.assertTrue(all(np.count_nonzero(x)==0 for x in dd.values()))
        for key in v:
            bad=copy.deepcopy(v);bad[key][1,0]+=.01
            with self.assertRaises(ValueError):kernel.build_deltas(bad,a,s)
        for n in (True,-1,9,1.5):
            with self.assertRaises(ValueError):kernel.build_deltas(v,a,s,limit=n)

    def test_single_original_episode_and_original_arrays_unchanged(self):
        a,s,m,v=fixture();before=copy.deepcopy(a);study=SyntheticStudy()
        with patch.object(h,'read',return_value={'synthetic':True}):
            orders,audit,deltas=h.decisions(study,a,m,v,s,8)
        self.assertEqual(study.adapter.score_and_choose.call_count,3)
        self.assertEqual(study.ep.select_boundary.call_count,3)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            expected=study.modified_pre_score(dict(id=4),a,8);mask=deltas[arm]!=0.;expected[mask]+=deltas[arm][mask]
            np.testing.assert_array_equal(call.kwargs['pre_score'],expected)
            self.assertIs(call.args[3],a)
            self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])
        for key in a:np.testing.assert_array_equal(a[key],before[key])

    def test_every_prefix_zero_and_one_included(self):
        a,s,m,v=fixture()
        with patch.object(h,'read',return_value={'synthetic':True}):
            orders,audit,deltas=h.decisions(SyntheticStudy(),a,m,v,s,8)
            for n in range(9):
                oo,aa,dd=h.decisions(SyntheticStudy(),a,m,v,s,n)
                for arm in h.NAMES:self.assertEqual(oo[arm],orders[arm][:n])
                for arm in h.LAMBDAS:np.testing.assert_array_equal(dd[arm],deltas[arm][:n])
                self.assertEqual(aa['lambdas'],h.LAMBDAS)

    def test_weight_broadcasts_and_future_poison_are_prefix_safe(self):
        import monthly_aux_reallocation186_kernel as kernel
        a,s,_,v=fixture();expected,_=kernel.build_deltas(v,a,s,limit=3)
        for shape in ('vector','full'):
            aa=copy.deepcopy(a);ss=copy.deepcopy(s)
            for obj,key in ((aa,'core_weight'),(ss,'vol_weight')):
                obj[key]=obj[key][:,0] if shape=='vector' else np.broadcast_to(obj[key],a['core'].shape).copy()
            got,_=kernel.build_deltas(v,aa,ss,limit=3)
            for key in got:np.testing.assert_array_equal(got[key],expected[key])
        for key in ('vol_rank','unit_delta'):v[key]=v[key].astype(object);v[key][3:]='future poison'
        got,_=kernel.build_deltas(v,a,s,limit=3)
        for key in got:np.testing.assert_array_equal(got[key],expected[key])

    def test_real_boundary_uses_original_support(self):
        a,s,m,v=fixture();study=SyntheticStudy()
        ns=original_functions(h.ROOT/'outputs/monthly_daily_episodes/episodes.py',['select_boundary'],{'np':np})
        study.adapter=h.module('reallocation186_real_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
        study.ep=SimpleNamespace(select_boundary=Mock(side_effect=ns['select_boundary']))
        policy=dict(budget=.05,ceiling=5,gap=.02)
        def choices(spec,a,m,n,*,policy):
            return study.adapter.score_and_choose('incumbent66',0.,np.full(a['baseline'].shape,np.nan),a,m,n,
                pre_score=study.modified_pre_score(spec,a,n),effects=a['effects'],episode_policy=policy,boundary_selector=ns['select_boundary'])
        study.choices=choices
        with patch.object(h,'read',return_value=policy):orders,_,dd=h.decisions(study,a,m,v,s,8)
        self.assertEqual(study.ep.select_boundary.call_count,3)
        for call in study.ep.select_boundary.call_args_list:
            original=call.args[4]
            for key in ('baseline','core'):np.testing.assert_array_equal(original[key],a[key])
            np.testing.assert_array_equal(original['core_weight'],np.broadcast_to(a['core_weight'],a['core'].shape))
            np.testing.assert_array_equal(call.args[1],a['eligible'])


class SourceAndContractTests(unittest.TestCase):
    def test_context_and_local_evidence_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();evidence=root/'source';evidence.write_text('source')
            context=root/'context.json';context.write_text(json.dumps(dict(local_evidence_sha256={'source':h.sha(evidence)})))
            with patch.multiple(h,ROOT=root,CONTEXT=context,CONTEXT_SHA=h.sha(context)):
                pins=h.context_binding();self.assertEqual(set(pins),{str(evidence),str(context)})
                evidence.write_text('changed')
                with self.assertRaises(ValueError):h.context_binding()

    def test_original_quality_arrays_exact_and_shapes_not_squeezed(self):
        a,s,_,_=fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'quality.npz'
            np.savez(path,**s,**{key:a[key] for key in ('baseline','core','core_weight','state_calm10')})
            with patch.multiple(h,QUALITY=path,QUALITY_SHA=h.sha(path)):
                got,audit=h.quality_inputs(a)
                self.assertTrue(audit['original_quality_arrays_exact']);self.assertEqual(set(got),set(s))
                for key in s:np.testing.assert_array_equal(got[key],s[key])
                bad=copy.deepcopy(a);bad['core_weight']=bad['core_weight'][:,0]
                with self.assertRaises(AssertionError):h.quality_inputs(bad)
                bad=copy.deepcopy(a);bad['baseline'][1,1]+=.001
                with self.assertRaises(AssertionError):h.quality_inputs(bad)
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):h.quality_inputs(a)

    def test_fixed_contract_and_seven_artifact_set(self):
        c=h.contract();self.assertEqual(list(h.LAMBDAS.values()),[.025,.05,.10])
        self.assertEqual(c['candidates'],list(h.LAMBDAS));self.assertEqual(len(h.PREPARED_FILES),7)
        self.assertTrue(c['parameter_search']);self.assertNotIn('no_parameter_search',c)
        self.assertEqual(c['registered_trials'],4);self.assertEqual(c['execution_paths'],12);self.assertEqual(c['metric_slices'],60)
        for text in ('unit_delta=(B-C)-(1-k)*V','No new availability mask'):self.assertIn(text,c['measure'])
        for text in ('only nonzero','exactly one original episode','First partial'):self.assertIn(text,c['scoring'])
        self.assertIn('not a novel factor',c['novelty']);self.assertIn('after185 failure',c['novelty']);self.assertIn('negative lambda in185',c['scoring']);self.assertIn(h.CONTEXT,h.code_files())
        self.assertFalse(any('numerics' in x.name for x in h.code_files()))


def feature_receipt():
    return dict(**{k:True for k in h.FEATURE_FLAGS},events=122,symbols_per_event=49,checked_cells=5978,
        source_vol_finite_cells=5970,source_vol_missing_neutral_cells=8,volatility_rank_reconstructions=122,
        predictive_model_fits=0,computes_returns=False)


def delta_receipt():
    return dict(**{k:True for k in h.DELTA_FLAGS},lambdas=h.LAMBDAS,predictive_model_fits=0,episode_applications=0,
        changed_cells={arm:0 for arm in h.LAMBDAS})


class AuditTests(unittest.TestCase):
    def test_feature_and_delta_evidence_fail_closed(self):
        values=dict(vol_rank=np.zeros((122,49)),unit_delta=np.zeros((122,49)));check=feature_receipt();h.require_feature_audit(check,values)
        for key in check:
            bad=dict(check);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_feature_audit(bad,values)
        for key,value in (('events',True),('source_vol_finite_cells',5977),('predictive_model_fits',False),('volatility_rank_reconstructions',123),('source_vol_missing_neutral_cells',9)):
            with self.assertRaises(ValueError):h.require_feature_audit(dict(check,**{key:value}),values)
        dd={k:np.zeros((122,49)) for k in h.LAMBDAS};check=delta_receipt();h.require_delta_audit(check,dd)
        for key in check:
            bad=dict(check);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_delta_audit(bad,dd)
        dd[h.NAMES[1]][1,0]=.001
        with self.assertRaises(ValueError):h.require_delta_audit(check,dd)

    def test_real_independent_delta_full_receipt_and_tamper(self):
        import monthly_aux_reallocation186_kernel as kernel
        import monthly_aux_reallocation186_independent as independent
        a,s,_,v=fixture();dd,audit=kernel.build_deltas(v,a,s)
        check=independent.verify_deltas(v,a,s,dd,audit)
        h.require_delta_audit(check,dd)
        bad=copy.deepcopy(dd);bad[h.NAMES[1]][1,0]+=.001
        with self.assertRaises(ValueError):independent.verify_deltas(v,a,s,bad,audit)

    def test_prepared_manifest_and_all_invariants(self):
        counts=h.estimate_counts(feature_receipt())
        common=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,**counts)
        c=dict(common,artifacts_sha256={n:'fake' for n in h.PREPARED_FILES},new_returns_computed=False)
        audit=dict(common,feature=feature_receipt(),delta=delta_receipt(),prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),
            original_orders_exact=True,original_arrays_unchanged=True,original_quality_arrays_exact=True,quality_arrays_unchanged=True)
        values=dict(vol_rank=np.zeros((122,49)),unit_delta=np.zeros((122,49)));dd={arm:np.zeros((122,49)) for arm in h.LAMBDAS}
        h.verify_prepared_evidence(c,values,dd,audit)
        for key in h.PREPARED_FILES:
            bad=copy.deepcopy(c);bad['artifacts_sha256'].pop(key)
            with self.assertRaises(ValueError):h.verify_prepared_evidence(bad,values,dd,audit)
        for key in ('prefixes','original_orders_exact','original_arrays_unchanged','original_quality_arrays_exact','quality_arrays_unchanged','new_predictive_model_fits','volatility_rank_reconstructions'):
            bad=copy.deepcopy(audit);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.verify_prepared_evidence(c,values,dd,bad)

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

    def test_twelve_paths_five_slices_exact_receipt_and_source(self):
        one = dict(passed=True, monthly_executions=122, slices={s: {} for s in ('full122', 'first88', 'early53', 'recent35', 'later34')})
        receipt = dict(passed=True, independent_accounts=True, independent_of_production_model_and_engine=True,
                       checked_paths=12, checked_slices=60, terminal_liquidation=False, accounting_source_sha256='fake',
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


def selection_fixture():
    metrics={arm:{p:{e:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for e in ('training','early','recent')}
        for p in ('same_close_zero','next_open_cost')} for arm in h.NAMES}
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in h.NAMES}
    for arm in h.LAMBDAS:
        for path in metrics[arm]:metrics[arm][path]['training']['cagr']+=.01
        for i in (1,2,53):orders[arm][i]['symbols']=['A','B','D']
    return metrics,orders


class SelectionAndBindingTests(unittest.TestCase):
    def test_highest_passing_training_cagr_and_exact_low_lambda_tie(self):
        metrics,orders=selection_fixture();answer=h.select_first88(metrics,orders)
        self.assertEqual(answer['primary'],h.NAMES[1]);self.assertEqual(answer['selected_lambda'],.025)
        self.assertEqual(answer['admitted_candidates'],list(h.LAMBDAS))
        metrics[h.NAMES[2]]['same_close_zero']['training']['cagr']=np.nextafter(.51,1.)
        answer=h.select_first88(metrics,orders);self.assertEqual(answer['primary'],h.NAMES[2])
        metrics[h.NAMES[3]]['same_close_zero']['training']['cagr']=.9
        metrics[h.NAMES[3]]['next_open_cost']['training']['cagr']=.49
        answer=h.select_first88(metrics,orders);self.assertEqual(answer['primary'],h.NAMES[2])
        self.assertFalse(answer['per_candidate'][h.NAMES[3]]['admitted'])

    def test_later34_and_extra_metrics_never_choose_arm(self):
        metrics,orders=selection_fixture();expected=h.select_first88(metrics,orders)
        for arm in h.NAMES:
            metrics[arm]['full122']=dict(cagr=1000. if arm==h.NAMES[3] else -1.)
            for i in range(88,122):orders[arm][i]['symbols']=[arm]
        self.assertEqual(h.select_first88(metrics,orders),expected)

    def test_each_dose_must_pass_all_incumbent_risk_era_and_change_guards(self):
        cases=[('same_close_zero','training','cagr',.504),('next_open_cost','training','cagr',.49),
            ('same_close_zero','early','cagr',.474),('next_open_cost','recent','cagr',.474),
            ('same_close_zero','training','sharpe',1.449),('next_open_cost','training','max_drawdown',-.331)]
        for arm in h.LAMBDAS:
            for path,era,key,value in cases:
                metrics,orders=selection_fixture();metrics[arm][path][era][key]=value
                answer=h.select_first88(metrics,orders)
                self.assertFalse(answer['per_candidate'][arm]['admitted']);self.assertNotEqual(answer['primary'],arm)
            metrics,orders=selection_fixture()
            for i in (1,2,53):orders[arm][i]['symbols']=['C','B','A']
            for i in (90,100,110):orders[arm][i]['symbols']=['A','B','D']
            answer=h.select_first88(metrics,orders)
            self.assertEqual(answer['per_candidate'][arm]['changed_months'],[])
            self.assertFalse(answer['per_candidate'][arm]['guards']['changed_support'])

    def test_no_candidate_control_guard_and_baseline_if_none(self):
        metrics,orders=selection_fixture()
        metrics[h.NAMES[3]]['same_close_zero']['training']['cagr']=9.
        metrics[h.NAMES[3]]['next_open_cost']['training']['cagr']=-1.
        answer=h.select_first88(metrics,orders)
        self.assertEqual(answer['primary'],h.NAMES[1])
        for arm in h.LAMBDAS:
            self.assertEqual(set(answer['per_candidate'][arm]['guards']),{'close_vs_incumbent','open_vs_incumbent','changed_support'})
            metrics[arm]['same_close_zero']['training']['cagr']=.5
        answer=h.select_first88(metrics,orders)
        self.assertEqual(answer['primary'],h.NAMES[0]);self.assertFalse(answer['admitted'])
        self.assertEqual(answer['selected_lambda'],0.);self.assertEqual(answer['admitted_candidates'],[])

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
        state = dict(completed=[dict(trials=6, directory='old')] * 174 + [dict(trials=x, directory='old') for x in (2, 2, 2, 2, 2, 2, 2, 3)] + [dict(trials=4, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)


def predecessor_fixture(root):
    prior = root / 'outputs/monthly_vol_reallocation185'
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return dict(path=str(path.relative_to(root)), sha256=h.sha(path))
    old_amendment = put(root / 'outputs/old/AMENDMENT.json', dict(old=True))
    old_repair = put(root / 'outputs/old/REPAIR_COMPLETION.json', dict(old=True))
    old_code = put(root / 'work/old_repair.py', 'historical code')
    inherited = {str(root / record['path']): record['sha256'] for record in (old_amendment, old_repair, old_code)}
    before = put(prior / 'STATE_BEFORE.json', dict(active_execution_amendment=old_amendment,
        latest_execution_repair_receipt=old_repair))
    plan = put(prior / 'PLAN.json', dict(input_sha256=inherited,
        predecessor_binding=dict(input_sha256=inherited), prior_central_sha256=before['sha256']))
    result = put(prior / 'results/COMPLETION.json', dict(status='COMPLETE', plan_sha256=plan['sha256']))
    closed = put(prior / 'COMPLETION.json', dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',
        plan_sha256=plan['sha256'], result_completion_sha256=result['sha256'],
        protected_files=21444, observers_modified=False))
    expected = {name: h.sha(prior / name) for name in h.PRIOR_EXPECTED}
    state = dict(completed=[dict(completion=closed)], active_execution_amendment=old_amendment,
        latest_execution_repair_receipt=old_repair)
    return prior, expected, state


class PredecessorTests(unittest.TestCase):
    def test_closed_predecessor_without_new_repair_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); prior, expected, state = predecessor_fixture(root)
            with patch.multiple(h, ROOT=root, PRIOR=prior, PRIOR_EXPECTED=expected):
                binding = h.predecessor_binding(state)
                self.assertFalse((prior / 'AMENDMENT_01.json').exists())
                self.assertFalse((prior / 'REPAIR_COMPLETION_01.json').exists())
                self.assertFalse(binding['predecessor_requires_new_repair'])
                self.assertEqual(binding['inherited_evidence_count'], 3)
                self.assertIn(str(root / 'work/old_repair.py'), binding['input_sha256'])
                for field in ('active_execution_amendment', 'latest_execution_repair_receipt'):
                    wrong = copy.deepcopy(state); wrong.pop(field)
                    with self.assertRaises(ValueError): h.predecessor_binding(wrong)
                    wrong = copy.deepcopy(state); wrong[field]['sha256'] = 'wrong'
                    with self.assertRaises(ValueError): h.predecessor_binding(wrong)

    def test_inherited_historical_code_and_evidence_are_digest_bound(self):
        for relative in ('work/old_repair.py', 'outputs/old/AMENDMENT.json',
                         'outputs/old/REPAIR_COMPLETION.json', 'outputs/monthly_vol_reallocation185/STATE_BEFORE.json'):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve(); prior, expected, state = predecessor_fixture(root)
                (root / relative).write_text('changed')
                with patch.multiple(h, ROOT=root, PRIOR=prior, PRIOR_EXPECTED=expected):
                    with self.assertRaises(ValueError): h.predecessor_binding(state)

    def test_all_phase_gates_recheck_registered_predecessor_binding(self):
        plan = dict(contract=h.contract(), runtime=h.ev.runtime(), input_sha256={'x':'y'},
                    prior_central_sha256='sha', predecessor_binding={'registered':True},raw_source_hashes={'raw':'sha'},quality_source_binding={'quality':True})
        state = dict(registered_queue=[dict(directory=h.OUT.name, plan_sha256='sha')])
        def reader(path): return plan if path == h.OUT / 'PLAN.json' else state
        with patch.object(h, 'sha', return_value='sha'), patch.object(h, 'read', side_effect=reader), \
             patch.object(h, 'verify'), patch.object(h, 'state_check'), patch.object(h, 'context_binding', return_value={}), \
             patch.object(h, 'quality_source_binding',return_value={'quality':True}), \
             patch.object(h, 'predecessor_binding', return_value={'registered':True}) as predecessor:
            self.assertEqual(h.gate('sha'), plan)
            predecessor.assert_called_once_with(state)
            predecessor.return_value = {'registered':False}
            with self.assertRaises(ValueError): h.gate('sha')



if __name__=='__main__':unittest.main()
