"""Synthetic lifecycle tests: source gates before new ranking or orders."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import monthly_peer_path252_host_v2 as h


def score_fixture():
    base=np.array([[.4,.3,.2,.195,.18]])
    a=dict(baseline=base,core=np.zeros_like(base),core_weight=np.array([.5]),eligible=np.array([[True]*5]),effects=np.zeros_like(base),state_calm10=np.array([False]),state_weak_narrow=np.array([False]))
    a['core']=base-.5*np.array([[.2,.1,0.,-.1,-.2]])
    m=dict(dates=['2016-08-31'],ends=['2016-09-30'],symbols=[['A','B','C','D','QQQ']],etfs=['QQQ'])
    v=dict(mature=np.array([[True]*4+[False]]),supported=np.array([[True]*4+[False]]),path_target=np.array([[.1,.2,.3,.4,np.nan]]),endpoint_target=np.array([[.1,.2,.3,.4,np.nan]]))
    real=h.old.local()
    study=SimpleNamespace(modified_pre_score=lambda spec,arrays,n:arrays['baseline'][:n].copy(),configs=lambda:[{}]*5,adapter=real.adapter,ep=real.ep,rp=real.rp)
    return study,a,m,v


def training_fixture():
    ref=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    metric={a:{p:copy.deepcopy(ref) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for path in metric[h.ARMS[1]]:metric[h.ARMS[1]][path]['training']['cagr']=.52
    orders={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):orders[h.ARMS[1]][i]['symbols']=['A','B','D']
    return metric,orders


def receipt_fixture():
    ind=SimpleNamespace(ORIGINAL_SOURCE_FLAGS=('passed',),INCUMBENT_GATE_FLAGS=('passed',),SOURCE_FLAGS=('passed',),FEATURE_FLAGS=('passed',),SELECTION_FLAGS=('passed',))
    common=dict(passed=True,producer_modules_imported=False,computes_returns=False)
    values={k:np.ones((122,2),dtype=bool if k in h.KEYS[:2] else float)* (1 if k in h.KEYS[:2] else .25) for k in h.KEYS}
    records=[dict(query_index=i,initial_partial=False) for i in range(122) for j in range(2)]
    def counts(n):return dict(events=n,stock_cells=2*n,mature_cells=2*n,supported_cells=2*n,price_observations=422*n,new_model_fits=0)
    def original(n):return dict(**common,checked_events=n,episode_calls=0,new_target_ranks_computed=False,candidate_selection_computed=False,C_source_atol=1e-12,C_source_rtol=0.,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.)
    def incumbent(n):return dict(**common,checked_events=n,ordered_choices_checked=n,numerical_tolerance=0.,candidate_selection_computed=False)
    def source(n):return dict(**common,counts=counts(n),events=n,checked_events=n,numerical_tolerance=0.,full_source_report_supplied=True,episode_calls=0,new_target_ranks_computed=False,candidate_selection_computed=False,full_features_supplied=True,rank_tolerance=0.,exact_discrete=True)
    def selection(n):return dict(**common,checked_events=n,ordered_choices_checked=3*n,full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.)
    def evidence(n):return dict(original_source=original(n),daily_source=source(n),incumbent=incumbent(n),feature=source(n),selection=selection(n))
    e=dict(**evidence(122),physical_prefixes=[dict(prefix=n,**evidence(n),counts=counts(n),exact=True) for n in h.PREFIXES],cached_prefixes=[dict(prefix=n,incumbent=incumbent(n),exact=True) for n in range(123)],all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,source_inventory_preserved=True,new_model_fits=0)
    audit=dict(counts=counts(122));completion=dict(counts=counts(122),new_returns_computed=False,new_model_fits=0);source=dict(records=records,counts=counts(122))
    return e,values,audit,completion,source,ind


class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(len(h.KEYS),4);self.assertEqual(len(h.PREP_FILES),9);self.assertEqual(len(h.RESULT_FILES),38)
        self.assertEqual(h.ARMS[2],'endpoint_peer_relative_control')
    def test_training_admission(self):self.assertTrue(h.select_first88(*training_fixture())['admitted'])
    def test_later34_irrelevant(self):
        m,o=training_fixture();before=h.select_first88(m,o)
        for arm in o:
            for i in range(88,122):o[arm][i]['symbols']=['X','Y','Z']
        self.assertEqual(before,h.select_first88(m,o))
    def test_endpoint_is_never_promoted(self):
        m,o=training_fixture();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.9
        self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[0])
    def test_needs_candidate_improvement(self):
        m,o=training_fixture();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.504
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_open_and_risk_guards(self):
        for path,group,key,value in [('next_open_zero','training','cagr',.49),('same_close_zero','training','sharpe',1.6),('same_close_zero','training','max_drawdown',-.34),('same_close_zero','early','cagr',.37),('same_close_zero','recent','cagr',.57)]:
            m,o=training_fixture();m[h.ARMS[1]][path][group][key]=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_changes_in_both_training_parts(self):
        m,o=training_fixture();o[h.ARMS[1]][1]['symbols']=['A','B','C']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_undefined_experimental_sharpe_rejects(self):
        for arm in h.ARMS[1:]:
            m,o=training_fixture();m[arm]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
            r=h.select_first88(m,o);self.assertFalse(r['admitted']);self.assertEqual(r['primary'],h.ARMS[0])
    def test_undefined_incumbent_or_malformed_metric_hardfails(self):
        for arm,value in [(h.ARMS[0],None),(h.ARMS[1],np.nan),(h.ARMS[2],True)]:
            m,o=training_fixture();m[arm]['same_close_zero']['training'].update(sharpe=value,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
            with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_three_fixed_arms(self):
        m,o=training_fixture();m.pop(h.ARMS[2])
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_no_support_exact_noop(self):
        s,a,m,v=score_fixture();v['supported'][:]=False;v['path_target'][:]=np.nan;v['endpoint_target'][:]=np.nan
        before=h.old.array_snapshot(a);o,sc,pr=h.decisions(s,a,m,v,1)
        self.assertEqual(before,h.old.array_snapshot(a))
        for arm in h.ARMS:
            self.assertEqual(o[arm],o[h.ARMS[0]]);np.testing.assert_array_equal(sc[arm],sc[h.ARMS[0]]);np.testing.assert_array_equal(pr[arm],pr[h.ARMS[0]])
    def test_zero_rank_not_missing(self):
        s,a,m,v=score_fixture();v['path_target'][0,:4]=0.
        o,sc,pr=h.decisions(s,a,m,v,1);support=(a['baseline']-a['core'])/(1-a['core_weight'][:,None]);want=a['baseline'].copy();d=.05*(v['path_target']-support);mask=np.isfinite(d)&(d!=0);want[mask]+=d[mask]
        np.testing.assert_array_equal(pr[h.ARMS[1]],want);self.assertEqual(pr[h.ARMS[1]][0,4],pr[h.ARMS[0]][0,4])
    def test_original_eligibility_preserved(self):
        s,a,m,v=score_fixture();a['eligible'][0,0]=False;o,sc,pr=h.decisions(s,a,m,v,1)
        self.assertTrue(all('A' not in o[x][0]['symbols'] for x in h.ARMS));self.assertFalse(a['eligible'][0,0])
    def test_bad_feature_masks_and_values(self):
        for change in ('nonbool','unsupported_finite','supported_nan','infinite','overbound','missingkey','unsupported_maturity'):
            s,a,m,v=score_fixture()
            if change=='nonbool':v['mature']=v['mature'].astype(float)
            elif change=='unsupported_finite':v['path_target'][0,4]=0.
            elif change=='supported_nan':v['path_target'][0,0]=np.nan
            elif change=='infinite':v['path_target'][0,0]=np.inf
            elif change=='overbound':v['path_target'][0,0]=.501
            elif change=='missingkey':del v['path_target']
            else:v['mature'][0,0]=False
            with self.assertRaises(ValueError):h.decisions(s,a,m,v,1)
    def test_incumbent_gate_aborts_before_candidate_episode(self):
        s,a,m,v=score_fixture();calls=[];original=s.ep.select_boundary
        def counted(*args):calls.append('episode');return original(*args)
        def veto(*args):calls.append('veto');raise ValueError('veto')
        with patch.object(s.ep,'select_boundary',side_effect=counted):
            with self.assertRaises(ValueError):h.decisions(s,a,m,v,1,incumbent_validator=veto)
        self.assertEqual(calls,['episode','veto'])
    def test_original_source_gate_aborts_before_new_source(self):
        kernel=SimpleNamespace(build_source=lambda *a: self.fail('newsource called'))
        ind=SimpleNamespace(verify_original_source=lambda *a,**kw: {'bad':True},ORIGINAL_SOURCE_FLAGS=('passed',))
        with self.assertRaises(ValueError):h.prepare_one(None,None,None,None,None,None,0,kernel,ind)
    def test_empty_prefix_decisions(self):
        study,a,m,v=score_fixture();a={k:x[:0].copy() for k,x in a.items()};v={k:x[:0].copy() for k,x in v.items()}
        m={**m,'dates':[],'ends':[],'symbols':[]};orders,scores,pres=h.decisions(study,a,m,v,0)
        for arm in h.ARMS:self.assertEqual(orders[arm],[]);self.assertEqual(scores[arm].shape,(0,5));self.assertEqual(pres[arm].shape,(0,5))
    def test_daily_source_veto_precedes_incumbent_and_features(self):
        e,v,a,c,source,ind=receipt_fixture();trace=[]
        ind.verify_original_source=lambda *a,**kw:e['physical_prefixes'][0]['original_source']
        ind.verify_source=lambda *a,**kw:dict(e['physical_prefixes'][0]['daily_source'],passed=False)
        kernel=SimpleNamespace(build_source=lambda *a:trace.append('source') or {'counts':e['physical_prefixes'][0]['counts']},build_features=lambda *a:self.fail('newfeatures called'))
        with patch.object(h,'incumbent_decision',side_effect=AssertionError('incumbent called')):
            with self.assertRaises(ValueError):h.prepare_one(None,None,None,None,None,None,0,kernel,ind)
        self.assertEqual(trace,['source'])
    def test_incumbent_veto_precedes_new_features(self):
        e,v,a,c,source,ind=receipt_fixture();z=e['physical_prefixes'][0]
        ind.verify_original_source=lambda *a,**kw:z['original_source'];ind.verify_source=lambda *a,**kw:z['daily_source'];ind.verify_incumbent=lambda *a,**kw:dict(z['incumbent'],passed=False)
        kernel=SimpleNamespace(build_source=lambda *a:{'counts':z['counts']},build_features=lambda *a:self.fail('newfeatures called'))
        with patch.object(h,'incumbent_decision',return_value=([],np.empty((0,5)),np.empty((0,5)))):
            with self.assertRaises(ValueError):h.prepare_one(None,None,None,None,None,None,0,kernel,ind,expected_incumbent_orders=[])
    def test_bad_frozen_orders_precede_new_features(self):
        e,v,a,c,source,ind=receipt_fixture();z=e['physical_prefixes'][0];calls=[]
        ind.verify_original_source=lambda *a,**kw:z['original_source'];ind.verify_source=lambda *a,**kw:z['daily_source']
        ind.verify_incumbent=lambda *a,**kw:calls.append('independent_inc') or z['incumbent']
        kernel=SimpleNamespace(build_source=lambda *a:{'counts':z['counts']},build_features=lambda *a:self.fail('newfeatures called'))
        with patch.object(h,'incumbent_decision',return_value=([],np.empty((0,5)),np.empty((0,5)))):
            with self.assertRaisesRegex(ValueError,'frozen incumbent'):
                h.prepare_one(None,None,None,None,None,None,0,kernel,ind,expected_incumbent_orders=[{'wrong':True}])
        self.assertEqual(calls,[])
    def test_bad_frozen_orders_precede_candidate_episode(self):
        study,a,m,v=score_fixture();calls=[];old_episode=study.ep.select_boundary
        def counted(*args):calls.append('episode');return old_episode(*args)
        with patch.object(study.ep,'select_boundary',side_effect=counted):
            with self.assertRaisesRegex(ValueError,'frozen incumbent'):
                h.decisions(study,a,m,v,1,expected_incumbent_orders=[])
        self.assertEqual(calls,['episode'])
    def test_source_prefix_never_deepcopies_future_contents(self):
        class Poison:
            def __deepcopy__(self,memo):raise AssertionError('Future content read')
        row=dict(query_index=0,history_status='observed_full_window',initial_partial=False,structural_peer_support=True,deferred_gap_reasons=[],gap_reasons=[])
        source=dict(schema='fixed',events=[{'query_index':0},Poison()],records=[row,{'query_index':1,'opaque':Poison()}],counts={'future':Poison()},gaps=[Poison()])
        r=h.source_prefix(source,1);self.assertEqual(r['records'],[row]);self.assertEqual(r['counts']['stock_cells'],1)
        r=h.source_prefix(source,0);self.assertEqual(r['records'],[]);self.assertEqual(r['events'],[])
        source['records'][-1]['query_index']=True
        with self.assertRaises(ValueError):h.source_prefix(source,0)
    def test_valid_preparation_receipt(self):h.preparation_check(*receipt_fixture())
    def test_missing_prefixes(self):
        for key in ('physical_prefixes','cached_prefixes'):
            args=receipt_fixture();args[0][key].pop()
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_false_gates_rejected(self):
        for gate in ('original_source','daily_source','incumbent','feature','selection'):
            for prefix in (False,True):
                args=receipt_fixture();item=args[0]['physical_prefixes'][2] if prefix else args[0];item[gate]['passed']=False
                with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_count_and_coverage_changes_rejected(self):
        for field in h.COUNTS:
            args=receipt_fixture();args[0]['physical_prefixes'][2]['daily_source']['counts'][field]+=1
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_original_source_scope_not_weakened(self):
        for field,value in [('episode_calls',1),('new_target_ranks_computed',True),('C_source_atol',1e-8),('continuous_rtol',1e-8),('rank_tolerance',1e-10),('producer_modules_imported',True),('computes_returns',True)]:
            args=receipt_fixture();args[0]['original_source'][field]=value
            with self.assertRaises(ValueError):h.preparation_check(*args)
    def test_registration_gate(self):
        state=dict(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='h')])
        with patch.object(h,'initial_state'):
            h.registration_check(state,'h')
            state['registered_queue']=[]
            with self.assertRaises(ValueError):h.registration_check(state,'h')

class RevisionTests(unittest.TestCase):
    def test_registration_needs_exact_failed_pending_queue(self):
        good=dict(current=h.V1.name,registered_queue=[dict(directory=h.V1.name,plan_sha256=h.V1_PLAN_SHA)])
        with patch.object(h,'initial_state'):
            h.registration_start_check(good)
            for q in ([],[dict(directory=h.V1.name,plan_sha256='wrong')],[dict(directory=h.OUT.name,plan_sha256=h.V1_PLAN_SHA)]):
                with self.assertRaises(ValueError):h.registration_start_check(dict(current=h.V1.name,registered_queue=q))
    def test_source_revision_does_not_reselect_strategy(self):
        fixed=('arms','hypothesis','causal_scope','original_control','exact_math','scoring','budget','selection','execution','validation')
        old={key:{'synthetic':key} for key in fixed};current=copy.deepcopy(old)
        current['source_revision']=dict(version=2,source_only=True,prior_new_features=0,prior_new_orders=0,prior_new_returns=0,strategy_trial_count=3,
            prior_plan_sha256=h.V1_PLAN_SHA,listing_policy=dict(path=str(h.LISTING_POLICY.relative_to(h.ROOT)),sha256='policy'))
        with patch.object(h,'sha',return_value='policy'):
            h.strategy_revision_check(current,old)
            for key in fixed:
                broken=copy.deepcopy(current);broken[key]={'altered':True}
                with self.assertRaises(ValueError):h.strategy_revision_check(broken,old)
            broken=copy.deepcopy(current);broken['source_revision']['prior_new_returns']=1
            with self.assertRaises(ValueError):h.strategy_revision_check(broken,old)

if __name__=='__main__':unittest.main()
