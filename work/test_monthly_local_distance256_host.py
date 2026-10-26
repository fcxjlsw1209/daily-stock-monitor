"""Synthetic phase/selection checks only; no actual research arrays or returns."""
import copy,unittest
from unittest.mock import patch
import numpy as np
import monthly_local_distance256_host as h

def fixture():
    base={'training':dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),'early':dict(cagr=.4),'recent':dict(cagr=.6)}
    m={a:{p:copy.deepcopy(base) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.51
    m[h.ARMS[1]]['next_open_zero']['training']['cagr']=.5
    o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['A','B','D']
    return m,o

def proof(kind,n):
    import monthly_local_distance256_independent as ind
    flags={'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}[kind]
    r=dict(**{k:True for k in flags},checked_events=n,producer_modules_imported=False,computes_returns=False)
    fitted=int(n>0)
    pc=dict(target_fits=4*fitted,log_refits=fitted,local_evaluations=fitted,local_solves=fitted,actual_regression_solves=6*fitted)
    sc=dict(pooled_log_solves=fitted,local_linear_solves=fitted,total_regression_solves=2*fitted)
    if kind in ('control','models'):
        r.update(full_control_supplied=True,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True)
    if kind=='control':
        r.update(candidate_labels_computed=False,candidate_models_computed=False,episode_calls=0,producer_equivalent_fit_counts=pc,independent_solve_counts=sc)
    elif kind=='models':
        c=dict.fromkeys(ind.MODEL_COUNT_FIELDS,0)
        c.update(training_rows=fitted,valid_training_rows=fitted,candidate_local_evaluations=fitted,candidate_local_solves=fitted,candidate_kernel_evaluations=fitted,candidate_changed_weight_evaluations=fitted)
        for key in ('target_fits','log_refits','local_evaluations','local_solves'):c['control_'+key]=pc[key]
        c['actual_regression_solves']=7*fitted
        r.update(full_bundle_supplied=True,full_labels_supplied=True,full_candidate_supplied=True,independent_numeric_system='augmented_least_squares_SVD',counts=c,producer_equivalent_counts=copy.deepcopy(c),independent_solve_counts=dict(control=copy.deepcopy(sc),candidate=dict(pooled_log_solves=0,local_linear_solves=fitted,total_regression_solves=fitted),total_regression_solves=3*fitted))
    else:
        r.update(full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,ordered_choices_checked=n if kind=='incumbent' else 2*n)
        if kind=='incumbent':r['candidate_selection_computed']=False
        else:
            c=dict.fromkeys(ind.COUNT_FIELDS,0);c.update(events=n,symbols_per_event=49,initial_partial_events=fitted,formal_events=n-fitted,incumbent_episode_calls=1,candidate_episode_calls=1)
            r.update(full_bundle_supplied=True,full_audit_supplied=True,counts=c)
    return r


def model_audit(counts):
    result=dict(schema='monthly_local_distance256_model_audit_v1',counts=copy.deepcopy(counts),mass_relative_tolerance=float(16*np.finfo(float).eps),residuals_are_oof=False,computes_returns=False)
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','candidate_pooled_copied','candidate_local_constant_copied','candidate_local_linear_only','equal_scale_original_weights_exact','residuals_are_mature_in_sample'):result[key]=True
    return result


def label_audit(counts):
    import monthly_local_distance256_independent as ind
    return dict(schema='monthly_local_distance256_labels_preserved_v1',rows=counts['training_rows'],valid_rows=counts['valid_training_rows'],target_keys=list(ind.TARGETS),target_sha256={k:'a'*64 for k in ind.TARGETS},common_good_sha256='b'*64,labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)


def evidence():
    e={k:proof(k,122) for k in ('control','models','incumbent','selection')}
    e.update(main_model_counts=model_audit(e['models']['counts']),all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)
    e['cached_prefixes']=[dict(prefix=n,exact=True,incumbent=proof('incumbent',n)) for n in range(123)]
    e['physical_prefixes']=[]
    for n in h.PREFIXES:
        v=dict(prefix=n,exact=True,model_recomputed=n!=122)
        for k in ('control','models','incumbent','selection'):v['model' if k=='models' else k]=proof(k,n)
        v['model_counts']=model_audit(v['model']['counts'])
        if n==122:v['reuses_main_verified_full']=True
        e['physical_prefixes'].append(v)
    e['fit_execution_accounting']=h.execution_accounting(e)
    ma=dict(model=copy.deepcopy(e['main_model_counts']),labels=label_audit(e['models']['counts']))
    da=dict(schema='monthly_local_distance256_decisions_v1',events=[{} for _ in range(122)],counts=copy.deepcopy(e['selection']['counts']),incumbent_gate=copy.deepcopy(e['incumbent']),computes_returns=False)
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):da[key]=True
    return e,ma,da


def sources():
    plan=dict(raw_source_hashes={'source':'a'},input_sha256={str(h.REBUILD/name):name+' hash' for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json')})
    r=dict(raw_source_hashes=copy.deepcopy(plan['raw_source_hashes']),fresh_training_array_reconstruction=False,cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True)
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:r[key]=plan['input_sha256'][str(h.REBUILD/name)]
    return r,plan

class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','diag_mass_local67'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(10,26,6))
        self.assertEqual(h.PRIOR.name,'monthly_asset_target255')
    def test_training_only(self):
        m,o=fixture();r=h.select_first88(m,o);self.assertTrue(r['admitted'])
        for arm in o:
            for i in range(88,122):o[arm][i]['symbols']=['Q','X','Y']
        self.assertEqual(r,h.select_first88(m,o))
    def test_close_open_failures(self):
        for path,value in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=fixture();m[h.ARMS[1]][path]['training']['cagr']=value
            self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[0])
    def test_risk_segments(self):
        for path in ('same_close_zero','next_open_zero'):
            for group,key,value in [('training','sharpe',1.64),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)]:
                m,o=fixture();m[h.ARMS[1]][path][group][key]=value
                self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_roundoff_training_only(self):
        m,o=fixture();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.505-5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.505-2e-12
        self.assertFalse(h.select_first88(m,o)['admitted'])
        for group,key,value in [('training','sharpe',1.65-5e-13),('early','cagr',.375-5e-13)]:
            m,o=fixture();m[h.ARMS[1]]['same_close_zero'][group][key]=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_support(self):
        for ids in ((54,55),(1,2,3),(53,54,55)):
            m,o=fixture();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[0]])
            for i in ids:o[h.ARMS[1]][i]['symbols']=['A','B','D']
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_same_holdings_rank_not_support(self):
        m,o=fixture()
        for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['C','B','A']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_required_metrics(self):
        for arm in h.ARMS:
            for value in (True,'0.7',float('nan'),float('inf')):
                m,o=fixture();m[arm]['same_close_zero']['training']['cagr']=value
                with self.assertRaises(ValueError):h.select_first88(m,o)
        for extra in ('later34','next_open_cost'):
            m,o=fixture();m[h.ARMS[1]][extra]={}
            with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_undefined_sharpe(self):
        for arm in h.ARMS:
            m,o=fixture();m[arm]['same_close_zero']['training'].update(sharpe=None,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd',daily_return_sd=0.)
            if arm==h.ARMS[0]:
                with self.assertRaises(ValueError):h.select_first88(m,o)
            else:
                r=h.select_first88(m,o);self.assertFalse(r['admitted']);self.assertTrue(r['guards']['next_open'])
    def test_short_orders(self):
        m,o=fixture();o[h.ARMS[1]]=o[h.ARMS[1]][:87]
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_counter_queue(self):
        s=dict(completed=[dict(directory='old',trials=5) for _ in range(250)]+[dict(directory=h.PRIOR.name,trials=15)])
        h.initial_state(s);s.update(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='x')]);h.registration_check(s,'x')
        for key,value in [('current','bad'),('registered_queue',[])]:
            d=copy.deepcopy(s);d[key]=value
            with self.assertRaises(ValueError):h.registration_check(d,'x')
        s['completed'][-1]['trials']=14
        with self.assertRaises(ValueError):h.initial_state(s)
    def test_predecessor_digest(self):
        s=dict(completed=[dict(completion=dict(path=str(h.PRIOR/'COMPLETION.json'),sha256='x'))])
        with patch.object(h,'sha',return_value='x'):
            h.predecessor_binding(s);s['completed'][-1]['completion']['sha256']='bad'
            with self.assertRaises(ValueError):h.predecessor_binding(s)
    def test_receipt_fail_closed(self):
        r=dict(proof('models',1),good=True)
        h.receipt(r,1,('good',),'models')
        for key,value in [('passed',False),('checked_events',2),('good',False),('producer_modules_imported',True),('computes_returns',True)]:
            with self.assertRaises(ValueError):h.receipt(dict(r,**{key:value}),1,('good',),'models')
    def test_control_gate_once_before_candidate(self):
        class Oracle:
            CONTROL_FLAGS=('good',)
            @staticmethod
            def verify_control(*args,**kwargs):return dict(proof('control',0),good=True)
        callback,r=h.model_callbacks(Oracle,{}, {},{},0);callback({});self.assertEqual(set(r),{'control'})
        with self.assertRaises(ValueError):callback({})
    def test_control_early_compute_rejected(self):
        class Oracle:
            CONTROL_FLAGS=()
            @staticmethod
            def verify_control(*args,**kwargs):return dict(proof('control',0),candidate_labels_computed=True)
        callback,_=h.model_callbacks(Oracle,{}, {},{},0)
        with self.assertRaises(ValueError):callback({})
    def test_prefix_full_scores(self):
        o={a:[{'symbols':['A','B','C']}] for a in h.ARMS};s={a:np.array([[1.,2.,3.]]) for a in h.ARMS}
        wanted=(o,s,s,{})
        h.exact_prefix(wanted,copy.deepcopy(wanted),1)
        got=copy.deepcopy(wanted);got[1][h.ARMS[1]][0,2]=4.
        with self.assertRaises(AssertionError):h.exact_prefix(wanted,got,1)
    def test_bundle_prefix_includes_labels(self):
        f=dict(width=1,training_row_ids=np.array([1,2]),label_values=np.array([np.nan,.2]))
        for role in ('control','candidate'):f[role]=dict(raw={'source_log':np.array([.1,.2])},available=np.array([True,True]),local_available=np.array([True,True]))
        small=copy.deepcopy(f);small.update(training_row_ids=np.array([1]),label_values=np.array([np.nan]))
        for role in ('control','candidate'):small[role]=dict(raw={'source_log':np.array([.1])},available=np.array([True]),local_available=np.array([True]))
        h.bundle_parity(f,small,1);small['label_values'][0]=0.
        with self.assertRaises(AssertionError):h.bundle_parity(f,small,1)
    def test_finite_sd_evidence(self):
        class Engine:
            @staticmethod
            def stats(e):return dict(cagr=0.,sharpe=None,max_drawdown=0.)
        ix=h.pd.date_range('2000-01-01',periods=123);eq=h.pd.Series(100000.,index=ix)
        m=dict(dates=[str(d.date()) for d in ix[:122]],ends=[str(d.date()) for d in ix[1:]])
        r=h.train_metrics(Engine,eq,m);self.assertEqual(r['training']['daily_return_sd'],0.);self.assertEqual(r['training']['sharpe_status'],'undefined_sharpe')

    def test_full_receipt_fields(self):
        for kind in ('control','models','incumbent','selection'):
            for n in (0,1,122):
                r=proof(kind,n);h.receipt(r,n,(),kind)
                for key in [k for k in r if k.startswith('full_')]:
                    with self.subTest(kind=kind,n=n,key=key):
                        bad=copy.deepcopy(r);del bad[key]
                        with self.assertRaises(ValueError):h.receipt(bad,n,(),kind)
                for key in [k for k in r if 'tolerance' in k or k in ('continuous_atol','continuous_rtol')]:
                    for value in (False,True,float('nan'),1e-12):
                        bad=copy.deepcopy(r);bad[key]=value
                        with self.assertRaises(ValueError):h.receipt(bad,n,(),kind)
                bad=copy.deepcopy(r);bad['checked_events']=bool(n)
                with self.assertRaises(ValueError):h.receipt(bad,n,(),kind)
        with self.assertRaises(ValueError):h.receipt(proof('models',1),1,(),'unknown')
    def test_receipt_choice_scope(self):
        for kind in ('incumbent','selection'):
            for value in (True,0,3):
                r=proof(kind,1);r['ordered_choices_checked']=value
                with self.assertRaises(ValueError):h.receipt(r,1,(),kind)
        r=proof('incumbent',1);r['candidate_selection_computed']=True
        with self.assertRaises(ValueError):h.receipt(r,1,(),'incumbent')
        r=proof('selection',1);r['counts']['symbols_per_event']=6
        h.receipt(r,1,(),'selection',width=6)
        with self.assertRaises(ValueError):h.receipt(r,1,(),'selection',width=49)
        r=proof('control',1);r['episode_calls']=1
        with self.assertRaises(ValueError):h.receipt(r,1,(),'control')
    def test_model_count_relations(self):
        for key,value in [('control_target_fits',3),('actual_regression_solves',13),('control_local_solves',2),('training_rows',True),('training_rows',0),('candidate_pooled_fits',1),('candidate_log_refits',1),('candidate_local_evaluations',2),('candidate_equal_scale_evaluations',1)]:
            r=proof('models',1);r['counts'][key]=value;r['producer_equivalent_counts']=copy.deepcopy(r['counts'])
            with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
    def test_model_oracle_counts(self):
        r=proof('models',1);r['independent_solve_counts']['total_regression_solves']+=1
        with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
        r=proof('control',1);r['independent_solve_counts'].update(pooled_log_solves=2,total_regression_solves=3)
        with self.assertRaises(ValueError):h.receipt(r,1,(),'control')
    def test_empty_prefix_no_training_or_solves(self):
        for kind in ('control','models'):
            r=proof(kind,1);r['checked_events']=0
            with self.assertRaises(ValueError):h.receipt(r,0,(),kind)
        r=proof('models',0);r['counts']['training_rows']=1;r['producer_equivalent_counts']=copy.deepcopy(r['counts'])
        with self.assertRaises(ValueError):h.receipt(r,0,(),'models')
        h.receipt(proof('selection',0),0,(),'selection')
    def test_preparation_cross_bindings(self):
        e,ma,da=evidence();h.preparation_evidence(e,ma,da)
        self.assertEqual(e['fit_execution_accounting']['totals']['total_regression_solves'],48)
        self.assertEqual(e['fit_execution_accounting']['model_invocations'],5)
        for target in ('main_model_counts','models','physical','decisions','incumbent_gate'):
            ee,mm,dd=evidence()
            if target=='main_model_counts':ee[target]['counts']['training_rows']+=1
            elif target=='models':ee[target]['counts']['training_rows']+=1
            elif target=='physical':ee['physical_prefixes'][2]['model_counts']['counts']['training_rows']+=1
            elif target=='decisions':dd['counts']['events']-=1
            else:dd['incumbent_gate']['ordered_choices_checked']-=1
            with self.assertRaises(ValueError):h.preparation_evidence(ee,mm,dd)
    def test_cached_middle_receipt_not_just_exact_flag(self):
        e,ma,da=evidence();del e['cached_prefixes'][61]['incumbent']['full_scores_supplied']
        with self.assertRaises(ValueError):h.preparation_evidence(e,ma,da)
        e,ma,da=evidence();e['cached_prefixes'][1]['prefix']=True
        with self.assertRaises(ValueError):h.preparation_evidence(e,ma,da)
        e,ma,da=evidence();del e['cached_prefixes'][61]
        with self.assertRaises(ValueError):h.preparation_evidence(e,ma,da)
    def test_full_prefix_reuse_never_recounted(self):
        for key,value in [('model_recomputed',True),('reuses_main_verified_full',False)]:
            e,ma,da=evidence();e['physical_prefixes'][-1][key]=value
            with self.assertRaises(ValueError):h.preparation_evidence(e,ma,da)
        e,ma,da=evidence();e['physical_prefixes'][-1]['incumbent']['extra']='not full main receipt'
        with self.assertRaises(ValueError):h.preparation_evidence(e,ma,da)
        e,ma,da=evidence();e['fit_execution_accounting']['totals']['total_regression_solves']+=18
        with self.assertRaises(ValueError):h.preparation_evidence(e,ma,da)
    def test_source_provenance_checks(self):
        r,p=sources();h.source_evidence(r,p)
        for key in ('raw_source_hashes','original_training_hash','original_identity_hash','saved169_parity_hash'):
            rr,pp=sources();rr[key]='bad'
            with self.assertRaises(ValueError):h.source_evidence(rr,pp)
        for key in ('fresh_training_array_reconstruction','original_arrays_unchanged','original_training_unchanged','original_orders_exact'):
            rr,pp=sources();rr[key]=not rr[key]
            with self.assertRaises(ValueError):h.source_evidence(rr,pp)

    def test_unchanged_labels_and_local_only_model_proof(self):
        e,ma,da=evidence();h.model_preservation(ma['model'],ma['labels'],122)
        for key in ('labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed'):
            bad=copy.deepcopy(ma);bad['labels'][key]=not bad['labels'][key]
            with self.assertRaises(ValueError):h.model_preservation(bad['model'],bad['labels'],122)
        for key in ('candidate_pooled_copied','candidate_local_constant_copied','candidate_local_linear_only','equal_scale_original_weights_exact','residuals_are_mature_in_sample'):
            bad=copy.deepcopy(ma);bad['model'][key]=False
            with self.assertRaises(ValueError):h.model_preservation(bad['model'],bad['labels'],122)
        for key,value in [('rows',True),('valid_rows',0),('target_keys',['log_excess']),('common_good_sha256','bad')]:
            bad=copy.deepcopy(ma);bad['labels'][key]=value
            with self.assertRaises(ValueError):h.model_preservation(bad['model'],bad['labels'],122)
        for value in (0.,1e-10,True):
            bad=copy.deepcopy(ma);bad['model']['mass_relative_tolerance']=value
            with self.assertRaises(ValueError):h.model_preservation(bad['model'],bad['labels'],122)
    def test_candidate_no_pooled_or_log_solves(self):
        for key in ('candidate_pooled_fits','candidate_log_refits'):
            r=proof('models',1);r['counts'][key]=1;r['producer_equivalent_counts']=copy.deepcopy(r['counts'])
            with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
        r=proof('models',1);r['independent_solve_counts']['candidate']['pooled_log_solves']=1
        r['independent_solve_counts']['candidate']['total_regression_solves']=2;r['independent_solve_counts']['total_regression_solves']=4
        with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
    def test_equal_scale_and_geometry_counts(self):
        r=proof('models',1);r['counts']['candidate_equal_scale_evaluations']=1;r['counts']['candidate_changed_weight_evaluations']=0;r['producer_equivalent_counts']=copy.deepcopy(r['counts'])
        h.receipt(r,1,(),'models')
        for key,value in [('candidate_kernel_evaluations',0),('candidate_changed_weight_evaluations',1),('candidate_local_solves',2)]:
            bad=copy.deepcopy(r);bad['counts'][key]=value;bad['producer_equivalent_counts']=copy.deepcopy(bad['counts'])
            with self.assertRaises(ValueError):h.receipt(bad,1,(),'models')
    def test_preserved_raw_and_availability_bytes(self):
        c=proof('models',1)['counts'];b=dict(audit=model_audit(c),label_preservation=label_audit(c))
        b['control']=dict(raw=dict(source_log=np.array([.1,np.nan]),local_constant=np.array([.2,np.nan])),available=np.array([True,False]),local_available=np.array([True,False]));b['candidate']=copy.deepcopy(b['control'])
        h.bundle_preservation(b,1)
        for key in ('source_log','local_constant'):
            bad=copy.deepcopy(b);bad['candidate']['raw'][key][0]=.3
            with self.assertRaises(ValueError):h.bundle_preservation(bad,1)
        bad=copy.deepcopy(b);bad['candidate']['available'][0]=False
        with self.assertRaises(ValueError):h.bundle_preservation(bad,1)

if __name__=='__main__':unittest.main(verbosity=2)
