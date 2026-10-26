"""Synthetic232 host contracts; no historical inputs, phases or performance runs."""
from pathlib import Path
import ast
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_volume_slope232_host as h


def selection_fixture():
    metric={era:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for era in ('training','early','recent')}
    scores={arm:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_zero')} for arm in h.NAMES}
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in h.NAMES}
    for path in scores[h.NAMES[1]]:scores[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return scores,orders


def state_fixture():
    rows=[dict(directory='prior',trials=5) for _ in range(227)]
    rows[0]['trials']+=63;rows[-1]['directory']=h.PRIOR.name
    return dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))


class SelectionGateTests(unittest.TestCase):
    def test_valid_fixed_zero_cost_pair_admits_candidate(self):
        m,o=selection_fixture();r=h.select_first88(m,o)
        self.assertTrue(r['admitted']);self.assertEqual(r['primary'],h.NAMES[1])
        self.assertEqual(r['changed_months'],[1,2,53]);self.assertEqual(r['changed_months_vs_control'],[1,2,53])
        self.assertTrue(r['first88_only']);self.assertFalse(r['later34_used'])

    def test_later34_and_cost_path_cannot_affect_selection(self):
        m,o=selection_fixture();expected=h.select_first88(m,o)
        for arm in h.NAMES:m[arm]['next_open_cost']=object()
        for i in range(88,122):o[h.NAMES[1]][i]['symbols']=['future_poison']
        self.assertEqual(h.select_first88(m,o),expected)
        self.assertEqual(h.select_first88(m,{k:v[:88] for k,v in o.items()}),expected)

    def test_next_open_zero_failure_cannot_be_rescued_by_cost_result(self):
        m,o=selection_fixture();m[h.NAMES[1]]['next_open_zero']['training']['cagr']=.49
        m[h.NAMES[1]]['next_open_cost']={'training':{'cagr':1e100}}
        r=h.select_first88(m,o);self.assertFalse(r['admitted']);self.assertEqual(r['primary'],h.NAMES[0])

    def test_each_reference_and_each_risk_or_subperiod_gate_matters(self):
        for ref in (h.NAMES[0],h.NAMES[2]):
            for path in ('same_close_zero','next_open_zero'):
                for era,field,value in (('training','cagr',.52),('training','sharpe',1.551),
                    ('training','max_drawdown',-.269),('early','cagr',.526),('recent','cagr',.526)):
                    m,o=selection_fixture();m[ref][path][era][field]=value
                    with self.subTest(ref=ref,path=path,era=era,field=field):self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_same_close_gain_is_half_percentage_point_not_five_percent(self):
        m,o=selection_fixture();m[h.NAMES[1]]['same_close_zero']['training']['cagr']=.505
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.NAMES[1]]['same_close_zero']['training']['cagr']=.505-2e-12
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_control_winner_is_never_promoted(self):
        m,o=selection_fixture()
        for path in m[h.NAMES[2]]:m[h.NAMES[2]][path]['training']['cagr']=.8
        self.assertEqual(h.select_first88(m,o)['primary'],h.NAMES[0])

    def test_changed_sets_must_span_both_eras_against_each_reference(self):
        for ref in (h.NAMES[0],h.NAMES[2]):
            for changed in ((1,2),(1,2,3),(53,54,55)):
                m,o=selection_fixture();o[ref]=copy.deepcopy(o[h.NAMES[1]])
                for i in changed:o[ref][i]['symbols']=['A','B','Q']
                with self.subTest(ref=ref,changed=changed):self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_only_holding_set_changes_count_not_reordering(self):
        m,o=selection_fixture()
        for i in (1,2,53):o[h.NAMES[1]][i]['symbols']=['C','B','A']
        r=h.select_first88(m,o);self.assertFalse(r['admitted']);self.assertEqual(r['changed_months'],[])

    def test_nonfinite_selected_metric_never_satisfies_guard(self):
        for value in (np.nan,-np.inf):
            m,o=selection_fixture();m[h.NAMES[1]]['same_close_zero']['training']['cagr']=value
            with self.subTest(value=value):self.assertFalse(h.select_first88(m,o)['admitted'])


class ManifestAndPhaseTests(unittest.TestCase):
    def test_result_manifest_has_exact_38_artifacts(self):
        files={k:'synthetic' for k in h.RESULT_FILES};self.assertEqual(len(files),38);h.require_result_manifest(files)
        for kind in ('missing','extra','wrongcase'):
            bad=dict(files)
            if kind=='missing':bad.pop(next(iter(bad)))
            elif kind=='extra':bad['COMPLETION.json']='extra'
            else:key=next(iter(bad));bad[key.lower()]=bad.pop(key)
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.require_result_manifest(bad)

    def test_bind_conflict_and_mutable_or_external_pins_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic';p.write_bytes(b'one');pins={};h.bind(pins,p);before=dict(pins);p.write_bytes(b'two')
            with self.assertRaises(ValueError):h.bind(pins,p)
            self.assertEqual(pins,before)
        for pins in ({str(h.CENTRAL):'wrong'},{'/tmp/unregistered.py':'wrong'},{}):
            with self.subTest(pins=pins),self.assertRaises(ValueError):h.verify(pins)

    def test_register_is_write_once_before_source_reads(self):
        with tempfile.TemporaryDirectory() as d,patch.object(h,'OUT',Path(d)),patch.object(h,'read') as read:
            with self.assertRaises(ValueError):h.register()
            read.assert_not_called()

    def test_formal_tests_receipt_guard_prevents_recursive_test_run(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';p.write_text('{}')
            with patch.object(h,'TEST_RECEIPT',p),patch.object(h,'module') as module:
                with self.assertRaises(ValueError):h.tests()
                module.assert_not_called()

    def test_state_predecessor_counts_and_incumbent_exact(self):
        h.state_check(state_fixture())
        for kind in ('count','trial','latest','incumbent'):
            s=state_fixture()
            if kind=='count':s['completed'].pop()
            elif kind=='trial':s['completed'][0]['trials']+=1
            elif kind=='latest':s['completed'][-1]['directory']='older'
            else:s['best_training_selected_candidate']['directory']='other'
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.state_check(s)

    def test_failed_plan_precedes_all_actual_loading(self):
        with patch.object(h,'gate',side_effect=ValueError('synthetic invalid PLAN')),patch.object(h,'local') as load,patch.object(h.ev,'prices') as prices:
            for name in ('prepare','select','diagnose','close'):
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'synthetic invalid PLAN'):getattr(h,name)('bad')
            load.assert_not_called();prices.assert_not_called()


def fixture():
    dates=pd.bdate_range('2014-01-01','2016-11-02');t=np.arange(len(dates),dtype=float)
    panel={'adj':pd.DataFrame({s:40.*np.exp((j+1)*.0001*t+.01*np.sin(t/(13.+j))) for j,s in enumerate('ABCDE')},index=dates),
        'volume':pd.DataFrame({s:1000.+((t*(j+3))%71) for j,s in enumerate('ABCDE')},index=dates)}
    m=dict(dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'],symbols=[list('ABCDE') for _ in range(4)],etfs=[])
    calm=np.array([False,False,True,True]);cw=np.where(calm,.5,.52)[:,None]
    support=np.tile(np.linspace(-.1,.1,5),(4,1));core=cw*np.tile(np.linspace(.45,.2,5),(4,1))
    zero=np.zeros((4,5));yes=np.ones((4,5),bool)
    a=dict(baseline=core+(1-cw)*support,core=core,core_weight=cw,eligible=yes.copy(),
        state_calm10=calm,state_weak_narrow=np.array([False,True,False,True]),pre_score=np.tile([.205,.204,.203,.202,.201],(4,1)),
        effects=np.tile([0.,0.,-.5,.5,0.],(4,1)),forecast_available=yes.copy(),extension_available=yes.copy(),
        old_ml4=zero.copy(),rank_log_excess=zero.copy(),model_available=yes.copy(),local_available=np.zeros((4,5),bool),
        raw_source_log=zero.copy(),raw_local_constant=zero.copy(),raw_local_linear=zero.copy())
    for name in ('simple_excess','log_excess','vol_scaled','loss_utility'):
        a['raw_'+name]=zero.copy();a.setdefault('rank_'+name,zero.copy())
    a['rank_local_constant']=zero.copy();a['rank_local_linear']=zero.copy();a['paired_month_ready']=np.ones(4,bool)
    assert len(a)==27
    return panel,m,[str(d.date()) for d in dates],a


def values_fixture():
    available=np.ones((4,5),bool);available[0]=False
    candidate=np.tile([3.,1.,2.,2.,4.],(4,1));control=np.tile([1.,2.,3.,4.,5.],(4,1))
    candidate[0]=np.nan;control[0]=np.nan
    return dict(candidate=candidate,control=control,available=available)


class StubStudy:
    """Records exact host wiring; its toy selector is not the research oracle."""
    def __init__(self):
        self.adapter=self;self.ep=self;self.rp=self;self.calls=[]
    def configs(self):return [None]*4+[{'id':4}]
    def modified_pre_score(self,spec,a,n):
        if spec!={'id':4}:raise AssertionError('Wrong incumbent config')
        return a['pre_score'][:n].copy()
    def choices(self,spec,a,m,n,policy=None):
        pre=self.modified_pre_score(spec,a,n)
        return self.score_and_choose('incumbent66',0.,np.full(pre.shape,np.nan),a,m,n,
            pre_score=pre,effects=a['effects'],episode_policy=policy,boundary_selector=self.select_boundary)
    def score_and_choose(self,base,budget,factor,a,m,n,*,pre_score,effects,episode_policy,boundary_selector):
        if base!='incumbent66' or budget!=0. or not np.isnan(factor).all():raise AssertionError('Unexpected extra hand budget')
        if boundary_selector!=self.select_boundary:raise AssertionError('Original boundary missing')
        self.calls.append(dict(pre=pre_score.copy(),effects=effects.copy(),eligible=a['eligible'].copy(),
            baseline=a['baseline'].copy(),core=a['core'].copy(),policy=copy.deepcopy(episode_policy)))
        return boundary_selector(pre_score,effects,a['eligible'],m,n)
    def select_boundary(self,pre,effects,eligible,m,n):
        scores=pre.copy()+.001*effects
        choices=[]
        for i in range(n):
            js=sorted(np.flatnonzero(eligible[i]).tolist(),key=lambda j:(-scores[i,j],m['symbols'][i][j]))
            choices.append(js[:3] if len(js)>=2 else [])
        return choices,scores
    def make_orders(self,choices,scores,m):
        return [dict(date=m['dates'][i],end=m['ends'][i],symbols=[m['symbols'][i][j] for j in js],scores=[float(scores[i,j]) for j in js]) for i,js in enumerate(choices)]


class IntegrationTests(unittest.TestCase):
    def test_three_arms_each_call_one_original_episode_and_preserve_27arrays(self):
        p,m,c,a=fixture();v=values_fixture();before=h.array_snapshot(a);before_v=h.array_snapshot(v)
        study=StubStudy();policy={'budget':.05,'ceiling':5,'gap':.02}
        with patch.object(h,'read',return_value=policy):o,s,q=h.decisions(m,a,v,4,study)
        self.assertEqual(len(study.calls),3);self.assertEqual(h.array_snapshot(a),before);self.assertEqual(h.array_snapshot(v),before_v)
        ranks,deltas=h.scale_and_delta(v,a['eligible'],4)
        for i,arm in enumerate(h.NAMES):
            call=study.calls[i];expect=a['pre_score'].copy()
            if i:use=deltas[arm]!=0.;expect[use]+=deltas[arm][use]
            np.testing.assert_array_equal(call['pre'],expect)
            for key in ('effects','eligible','baseline','core'):np.testing.assert_array_equal(call[key],a[key])
            self.assertEqual(call['policy'],policy)
            self.assertEqual(o[arm][0],o[h.NAMES[0]][0])
        self.assertEqual(set(q),set(h.NAMES))

    def test_real_frozen_selector_and_independent_selection_agree(self):
        import monthly_volume_slope232_independent as oracle
        _,m,_,a=fixture();v=values_fixture();before=h.array_snapshot(a)
        o,s,q=h.decisions(m,a,v,4);check=oracle.verify_selection(a,m,v,s,q)
        h.require_selection_audit(check,4)
        self.assertEqual(before,h.array_snapshot(a));self.assertEqual(set(o),set(h.NAMES))

    def test_exact_rank_ties_bounds_and_common_domain(self):
        _,_,_,a=fixture();v=values_fixture();v['available'][2,4]=False
        v['candidate'][2,4]=v['control'][2,4]=np.nan
        before=h.array_snapshot(a);r,d=h.scale_and_delta(v,a['eligible'],4)
        np.testing.assert_array_equal(r[h.NAMES[1]][1],[.25,-.5,-.125,-.125,.5])
        for arm in h.NAMES[1:]:
            self.assertTrue((np.abs(r[arm])<=.5).all());self.assertTrue((np.abs(d[arm])<=.005).all())
            self.assertFalse(d[arm][0].any());self.assertFalse(d[arm][~v['available']].any())
        self.assertEqual(before,h.array_snapshot(a))

    def test_constant_unavailable_singleton_have_exact_zero_delta(self):
        _,m,_,a=fixture();v=values_fixture()
        for key in ('candidate','control'):
            v[key][1]=3.;v[key][2]=np.nan;v[key][3]=np.nan;v[key][3,0]=7.
        v['available'][2]=False;v['available'][3]=False;v['available'][3,0]=True
        _,d=h.scale_and_delta(v,a['eligible'],4)
        for delta in d.values():self.assertFalse(delta.any())
        with patch.object(h,'read',return_value={}):o,s,q=h.decisions(m,a,v,4,StubStudy())
        for arm in h.NAMES[1:]:self.assertEqual(o[arm],o[h.NAMES[0]])

    def test_missing_feature_does_not_remove_eligible_security(self):
        _,m,_,a=fixture();v=values_fixture();v['available'][1:]=False
        v['candidate'][1:]=v['control'][1:]=np.nan
        a['eligible'][2]=[True,True,False,False,False];a['eligible'][3]=[True,False,False,False,False]
        st=StubStudy()
        with patch.object(h,'read',return_value={}):o,s,q=h.decisions(m,a,v,4,st)
        for arm in h.NAMES:self.assertEqual(len(q[arm][1]),3);self.assertEqual(len(q[arm][2]),2);self.assertEqual(q[arm][3],[])
        for call in st.calls:np.testing.assert_array_equal(call['eligible'],a['eligible'])

    def test_all_logical_prefixes_and_future_numeric_poison(self):
        _,m,_,a=fixture();v=values_fixture()
        with patch.object(h,'read',return_value={}):full=h.decisions(m,a,v,4,StubStudy())
        for n in range(5):
            aa=copy.deepcopy(a);vv=copy.deepcopy(v)
            for value in aa.values():
                if value.dtype.kind=='f':value[n:]=np.inf
            vv['candidate'][n:]=np.inf;vv['control'][n:]=np.inf
            with patch.object(h,'read',return_value={}):o,s,q=h.decisions(m,aa,vv,n,StubStudy())
            self.assertEqual(o,{k:x[:n] for k,x in full[0].items()});self.assertEqual(q,{k:x[:n] for k,x in full[2].items()})
            for arm in h.NAMES:np.testing.assert_array_equal(s[arm],full[1][arm][:n])

    def test_physical_price_volume_and_array_prefixes(self):
        p,m,c,a=fixture();before=h.array_snapshot(a)
        for n in range(5):
            pp,mm,cc,aa=h.prefix_inputs(p,m,c,a,n)
            self.assertEqual(len(mm['dates']),n);self.assertEqual(set(pp),{'adj','volume'});self.assertEqual(len(aa),27)
            self.assertTrue(all(len(z)==n for z in aa.values()))
            for frame in pp.values():
                if n:self.assertLessEqual(frame.index.max(),pd.Timestamp(m['dates'][n-1]))
                else:self.assertEqual(len(frame),0)
            if n:aa['baseline'][0]=99.
        self.assertEqual(h.array_snapshot(a),before)

    def test_invalid_prefix_boolean_and_raw_domain_rejected(self):
        _,m,_,a=fixture();v=values_fixture()
        for n in (True,-1,1.5,5):
            with self.subTest(n=n),self.assertRaises(ValueError):h.decisions(m,a,v,n,StubStudy())
        for kind in ('eligible_int','available_int','outside','finite_mask','infinity','float32'):
            aa=copy.deepcopy(a);vv=copy.deepcopy(v)
            if kind=='eligible_int':aa['eligible']=aa['eligible'].astype(int)
            elif kind=='available_int':vv['available']=vv['available'].astype(int)
            elif kind=='outside':aa['eligible'][1,0]=False
            elif kind=='finite_mask':vv['candidate'][1,0]=np.nan
            elif kind=='infinity':vv['candidate'][1,0]=np.inf
            else:vv['control']=vv['control'].astype(np.float32)
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.scale_and_delta(vv,aa['eligible'],4)


class EvidenceTests(unittest.TestCase):
    def test_real_synthetic_kernel_oracle_receipt_and_tampered_fields(self):
        import monthly_volume_slope232_kernel as kernel
        import monthly_volume_slope232_independent as oracle
        p,m,c,a=fixture();v,au=kernel.build_features(p,m,c,a['eligible']);ck=oracle.verify_features(p,m,c,a['eligible'],v,au)
        h.require_model_audit(ck,v,au,4)
        for key in oracle.FEATURE_FLAGS:
            bad=copy.deepcopy(ck);bad[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,v,au,4)
        for key in oracle.COUNT_FIELDS:
            bad=copy.deepcopy(ck);bad['counts'][key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,v,au,4)
        for key,value in [('query_rows',5),('numerical_tolerance',1e-12),('computes_returns',True),('producer_modules_imported',True),('new_model_fits',False),('predictive_model_fits',1)]:
            bad=copy.deepcopy(ck);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,v,au,4)
        for key in ('source_price_observations','source_volume_observations','descriptive_time_slope_estimates','unweighted_slope_solves','weighted_slope_solves'):
            bad=copy.deepcopy(au);bad[key]+=1
            with self.subTest(count=key),self.assertRaises(ValueError):h.require_model_audit(ck,v,bad,4)

    def test_selection_receipt_flags_bounds_and_scope(self):
        import monthly_volume_slope232_independent as oracle
        _,m,_,a=fixture();v=values_fixture();_,s,q=h.decisions(m,a,v,4)
        ck=oracle.verify_selection(a,m,v,s,q);h.require_selection_audit(ck,4)
        for key in oracle.SELECTION_FLAGS:
            bad=copy.deepcopy(ck);bad[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(bad,4)
        for key,value in [('ordered_choices_checked',11),('numerical_tolerance',1e-12),('original_source_parity_tolerance',1e-6),
            ('computes_returns',True),('producer_modules_imported',True),('max_absolute_delta',.00501)]:
            bad=copy.deepcopy(ck);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(bad,4)

    def test_source_binding_enforces_declared_proxy_not_certified_units(self):
        raw={str(h.ROOT/'synthetic'/str(i)):'r'+str(i) for i in range(64)}
        review=dict(passed=True,source_only=True,metadata_only=True,conditional_proxy_admission=True,
            volume_units_certified=False,strict_pit=False,source_npz_decoded=False,actual_support_computed=False,
            actual_features_computed=False,actual_returns_computed=False,observers_modified=False,
            context_sha256=h.CONTEXT_SHA,coverage_sha256=h.COVERAGE_SHA,raw_source_count=64,
            raw_source_hashes=raw,input_sha256={str(h.ROOT/'synthetic/source'):'x'})
        def call(r):
            with patch.object(h,'sha',return_value=h.SOURCE_REVIEW_SHA),patch.object(h,'read',return_value=r),patch.object(h,'bind') as bind,patch.object(h,'verify') as verify:
                answer=h.price_source_binding(raw);bind.assert_called_once();verify.assert_called_once();return answer
        result=call(review);self.assertTrue(result['conditional_proxy_admission']);self.assertFalse(result['volume_units_certified']);self.assertFalse(result['strict_pit'])
        for key in ('passed','source_only','metadata_only','conditional_proxy_admission','volume_units_certified','strict_pit',
            'source_npz_decoded','actual_support_computed','actual_features_computed','actual_returns_computed','observers_modified'):
            bad=copy.deepcopy(review);bad[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):call(bad)
        for key in ('context_sha256','coverage_sha256'):
            bad=copy.deepcopy(review);bad[key]='wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):call(bad)
        bad=copy.deepcopy(review);bad['raw_source_hashes'].pop(next(iter(raw)))
        with self.assertRaises(ValueError):call(bad)

    def test_fixed_contract_has_no_new_fits_or_cost_selection(self):
        c=h.contract();self.assertEqual(c['period'],['2016-08-17','2026-09-24']);self.assertEqual(c['dose'],.01)
        self.assertEqual(c['selection_paths'],['same_close_zero','next_open_zero']);self.assertTrue(c['next_open_cost_diagnostic_only'])
        self.assertEqual(c['registered_trials'],3);self.assertFalse(c['volume_units_certified']);self.assertFalse(c['strict_pit'])
        for source in (Path(h.__file__),h.ROOT/'work/monthly_volume_slope232_kernel.py'):
            tree=ast.parse(source.read_text());calls=[x.func.attr for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
            self.assertNotIn('fit',calls);self.assertNotIn('predict',calls)


class PreparedReceiptTests(unittest.TestCase):
    @staticmethod
    def receipt_fixture():
        import monthly_volume_slope232_independent as oracle
        dates=['2016-08-17']+[str(x.date()) for x in pd.date_range('2016-08-31',periods=121,freq='ME')]
        meta=dict(dates=dates,ends=dates[1:]+['2026-09-30'],symbols=[[f'S{j}' for j in range(49)] for _ in dates])
        values=dict(candidate=np.full((122,49),np.nan),control=np.full((122,49),np.nan),available=np.zeros((122,49),bool))
        counts={k:0 for k in oracle.COUNT_FIELDS};counts.update(events=122,symbols_per_event=49,checked_cells=5978,
            unavailable_cells=5978,initial_partial_cells=49,ineligible_cells=5929)
        audit=dict(counts,common_availability=True,trading_eligibility_modified=False,predictive_model_fits=0,new_model_fits=0,
            portfolio_returns_calculated=False,window_price_count=253,window_volume_count=253,span_sessions=252,skip_sessions=42,
            source_start_offset=-294,source_end_offset=-42,event_history=[dict(date=d,symbols=ss) for d,ss in zip(dates,meta['symbols'])])
        def model_check(n):
            c={k:0 for k in oracle.COUNT_FIELDS};c.update(events=n,symbols_per_event=49,checked_cells=n*49,
                unavailable_cells=n*49,initial_partial_cells=49 if n else 0,ineligible_cells=max(n-1,0)*49)
            return dict(**{k:True for k in oracle.FEATURE_FLAGS},counts={k:c[k] for k in oracle.COUNT_FIELDS},query_rows=n,numerical_tolerance=0.,
                producer_modules_imported=False,computes_returns=False,new_model_fits=0,predictive_model_fits=0)
        def selection(n):return dict(**{k:True for k in oracle.SELECTION_FLAGS},producer_modules_imported=False,computes_returns=False,
            numerical_tolerance=0.,original_source_parity_tolerance=1e-12,max_absolute_delta=0.,ordered_choices_checked=3*n)
        physical=[dict(prefix=n,source_rows=dict(adj=1000 if n else 0,volume=1000 if n else 0),
            feature=model_check(n),selection=selection(n),counts=dict(model_check(n)['counts'],events=n)) for n in h.MODEL_PREFIXES]
        evidence=dict(model=model_check(122),selection=selection(122),original_orders_exact=True,original_arrays_unchanged=True,
            original_core_preserved=True,single_original_episode=True,selection_uses_zero_cost_paths_only=True,
            original_array_count=27,additional_max_score_budget=.005,prefixes=123,model_prefixes=list(h.MODEL_PREFIXES),
            physical_prefix_evidence=physical,model_counts=h.model_counts(audit))
        completion=dict(artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES},original_array_count=27,
            model_counts=h.model_counts(audit),new_returns_computed=False,new_model_fits=0,predictive_model_fits=0,model_fit_accounting='synthetic accounting')
        parity=dict(calendar_sha256='synthetic calendar',original_arrays_unchanged=True,original_orders_exact=True,
            volume_units_certified=False,strict_pit=False,original_array_fingerprints={str(i):'synthetic bytes' for i in range(27)},
            execution_prices={k:True for k in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact')})
        return completion,meta,evidence,parity,values,audit

    def test_prepared_exact_manifest_prefix_and_invariant_gate(self):
        args=self.receipt_fixture()
        import monthly_volume_slope232_independent as oracle
        self.assertEqual(args[2]['model']['counts'],{k:args[5][k] for k in oracle.COUNT_FIELDS})
        def check(x):
            with patch.object(h,'sha',return_value='synthetic calendar'),patch.object(h,'contract',return_value={'model_fit_accounting':'synthetic accounting'}):
                h.verify_prepared_evidence(*x)
        check(args)
        mutations=(
            lambda c,m,e,p,v,a:c['artifacts_sha256'].pop('FEATURES.npz'),
            lambda c,m,e,p,v,a:c['artifacts_sha256'].update(unexpected='bad'),
            lambda c,m,e,p,v,a:e.update(prefixes=122),
            lambda c,m,e,p,v,a:e['physical_prefix_evidence'].pop(),
            lambda c,m,e,p,v,a:e['physical_prefix_evidence'][0]['source_rows'].update(adj=1),
            lambda c,m,e,p,v,a:e['physical_prefix_evidence'][1]['source_rows'].pop('volume'),
            lambda c,m,e,p,v,a:e['physical_prefix_evidence'][2]['counts'].update(events=88),
            lambda c,m,e,p,v,a:e.update(single_original_episode=1),
            lambda c,m,e,p,v,a:e.update(selection_uses_zero_cost_paths_only=False),
            lambda c,m,e,p,v,a:p.update(volume_units_certified=True),
            lambda c,m,e,p,v,a:p.update(strict_pit=True),
            lambda c,m,e,p,v,a:p['original_array_fingerprints'].pop('0'),
            lambda c,m,e,p,v,a:c.update(new_returns_computed=True),
            lambda c,m,e,p,v,a:a['event_history'][1].update(date='wrong'),
            lambda c,m,e,p,v,a:m['symbols'][1].__setitem__(1,'S0'),
        )
        for i,mutate in enumerate(mutations):
            bad=copy.deepcopy(args);mutate(*bad)
            with self.subTest(mutation=i),self.assertRaises(ValueError):check(bad)


if __name__=='__main__':unittest.main(verbosity=2)
