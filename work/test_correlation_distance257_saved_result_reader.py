"""Offline257 reader rejection tests; invented metadata only, no actual artifacts."""
import copy
from datetime import date,timedelta
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import correlation_distance257_saved_result_reader as reader

def fixture():
    paths={};exports={};cert={}
    for path in reader.PATHS:
        fee=1.0 if path=='next_open_cost' else 0.0
        metrics=dict(cagr=.2,sharpe=1.1,max_drawdown=-.1,fees_paid=fee,final_equity=200000.0)
        slices={s:dict(cagr=.2,sharpe=1.1,max_drawdown=-.1) for s in reader.SLICES}
        slices['full122']=dict(metrics)
        paths[path]=dict(metrics=metrics,slices=slices)
        holdings=[dict(signal_date=(date(2020,1,1)+timedelta(days=i)).isoformat(),execution_date=(date(2020,1,2)+timedelta(days=i)).isoformat(),symbols=['A','B','C'],scores=[3.,2.,1.],fee=fee if i==0 else 0.) for i in range(122)]
        exports[path]=dict(HOLDINGS=holdings,TRADES=[dict(date='2020-01-02',symbol='A',cost=fee,dollars=100.)],DAILY=[dict(date='2020-01-02',equity=100000.)],DAILY_POSITIONS=[dict(date='2020-01-02',cash=0.,units={'A':1.})])
        cert[path]=dict(passed=True,fees_paid=fee)
    return paths,copy.deepcopy(paths),exports,copy.deepcopy(exports),cert,copy.deepcopy(cert)

def reconcile_current(args):
    total=reader.ordered_period_fees(args[3]['next_open_cost']['HOLDINGS'])
    args[1]['next_open_cost']['metrics']['fees_paid']=total
    args[1]['next_open_cost']['slices']['full122']['fees_paid']=total

class SavedComparisonTests(unittest.TestCase):
    def check(self,args):return reader.compare_incumbent_saved(*args)
    def reject(self,args,message):
        with self.assertRaisesRegex(ValueError,message):self.check(args)
    def test_identical_all_saved_values_pass(self):
        result=self.check(fixture())
        self.assertTrue(result['all_incumbent_metrics_slices_exact'])
        self.assertEqual(result['period_fee_exception']['actual_differences'],[])
    def test_one_ulp_period_difference_exactly_reconciles_own_aggregates(self):
        args=fixture();args[3]['next_open_cost']['HOLDINGS'][0]['fee']=math.nextafter(1.0,math.inf);reconcile_current(args)
        result=self.check(args)
        self.assertFalse(result['all_incumbent_metrics_slices_exact'])
        self.assertTrue(result['all_other_incumbent_metrics_slices_exact'])
        self.assertEqual(result['cost_aggregate_reconciliation']['difference'],math.nextafter(1.,math.inf)-1.)
        self.assertTrue(result['cost_aggregate_reconciliation']['independent_totals_exact'])
        self.assertFalse(result['cost_aggregate_reconciliation']['aggregate_tolerance_added'])
        self.assertEqual(len(result['period_fee_exception']['actual_differences']),1)
    def test_cost_nonfee_cagr_change_rejected(self):
        args=fixture();args[1]['next_open_cost']['metrics']['cagr']=math.nextafter(.2,math.inf)
        self.reject(args,'metrics exact')
    def test_nonfull_slice_fee_not_exempt(self):
        args=fixture();args[1]['next_open_cost']['slices']['first88']['fees_paid']=1.
        self.reject(args,'slices exact')
    def test_full122_nonfee_change_rejected(self):
        args=fixture();args[1]['next_open_cost']['slices']['full122']['sharpe']=math.nextafter(1.1,math.inf)
        self.reject(args,'slices exact')
    def test_zero_path_fee_change_rejected(self):
        args=fixture();args[3]['same_close_zero']['HOLDINGS'][0]['fee']=1e-16
        self.reject(args,'Only inherited cost period fee exception')
    def test_even_equal_both_round_zero_fee_contamination_rejected(self):
        args=fixture()
        for i in (2,3):args[i]['next_open_zero']['TRADES'][0]['cost']=1e-16
        self.reject(args,'every zero-fee field exact0')
    def test_unexplained_aggregate_change_rejected_despite_tiny_delta(self):
        args=fixture();args[1]['next_open_cost']['metrics']['fees_paid']=math.nextafter(1.,math.inf)
        self.reject(args,'metrics fees_paid exact chronological')
    def test_full122_aggregate_must_independently_match(self):
        args=fixture();args[1]['next_open_cost']['slices']['full122']['fees_paid']=math.nextafter(1.,math.inf)
        self.reject(args,'slices.full122 fees_paid exact chronological')
    def test_previous_aggregate_also_reconstructed(self):
        args=fixture();args[0]['next_open_cost']['metrics']['fees_paid']=math.nextafter(1.,math.inf)
        self.reject(args,'previous/metrics fees_paid exact chronological')
    def test_independent_certificate_one_ulp_difference_rejected(self):
        args=fixture();args[5]['next_open_cost']['fees_paid']=math.nextafter(1.,math.inf)
        self.reject(args,'Independent incumbent cost fee totals exact')
    def test_period_fee_above_inherited_bound_rejected(self):
        args=fixture();args[3]['next_open_cost']['HOLDINGS'][1]['fee']=math.nextafter(1e-10,math.inf);reconcile_current(args)
        self.reject(args,'Only inherited cost period fee exception')
    def test_cost_holdings_nonfee_change_rejected(self):
        args=fixture();args[3]['next_open_cost']['HOLDINGS'][0]['scores'][0]=math.nextafter(3.,math.inf)
        self.reject(args,'Holding nonfee fields exact')
    def test_cost_nav_change_rejected(self):
        args=fixture();args[3]['next_open_cost']['DAILY'][0]['equity']=math.nextafter(100000.,math.inf)
        self.reject(args,'Incumbent saved DAILY exact')
    def test_cost_trade_change_rejected(self):
        args=fixture();args[3]['next_open_cost']['TRADES'][0]['dollars']=math.nextafter(100.,math.inf)
        self.reject(args,'Incumbent saved TRADES exact')
    def test_fee_accumulation_matches_engine_not_fsum(self):
        args=fixture()
        for i in (2,3):
            rows=args[i]['next_open_cost']['HOLDINGS']
            for j,fee in enumerate((1e16,1.,1.)):rows[j]['fee']=fee
        for i in (0,1):
            args[i]['next_open_cost']['metrics']['fees_paid']=1e16
            args[i]['next_open_cost']['slices']['full122']['fees_paid']=1e16
        for i in (4,5):args[i]['next_open_cost']['fees_paid']=1e16
        self.assertNotEqual(math.fsum(row['fee'] for row in args[3]['next_open_cost']['HOLDINGS']),1e16)
        self.assertEqual(self.check(args)['cost_aggregate_reconciliation']['current'],1e16)
    def test_nonchronological_fee_accumulation_rejected(self):
        args=fixture()
        for i in (2,3):
            rows=args[i]['next_open_cost']['HOLDINGS'];rows[0],rows[1]=rows[1],rows[0]
        self.reject(args,'Chronological saved fee accumulation')
    def test_nonfinite_or_negative_fee_rejected(self):
        for value in (float('inf'),float('nan'),-1.,True):
            with self.subTest(value=value):
                args=fixture();args[3]['next_open_cost']['HOLDINGS'][0]['fee']=value
                self.reject(args,'Finite nonnegative period fees')
    def test_no_input_mutation(self):
        args=fixture();before=copy.deepcopy(args);self.check(args);self.assertEqual(args,before)

def accounting_fixture():
    stats=dict(start='2020-01-01',end='2021-01-01',years=1.,final_equity=110000.,cagr=.1,sharpe=1.,max_drawdown=-.1,trading_sessions=252,total_return=.1)
    result=dict(arms={});ind=dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={});training={};documents={}
    for arm in reader.ARMS:
        result['arms'][arm]={};ind['paths'][arm]={};training[arm]={}
        for path in reader.PATHS:
            fee=1. if path=='next_open_cost' else 0.
            metrics=dict(stats,fees_paid=fee,dollar_turnover=10.,trade_rows=1)
            slices={s:copy.deepcopy(stats) for s in reader.SLICES};slices['full122']=copy.deepcopy(metrics)
            result['arms'][arm][path]=dict(metrics=metrics,slices=slices,accounting=dict(exported_fill_ledger_matches=True,terminal_liquidation=False))
            ind['paths'][arm][path]=dict(passed=True,monthly_executions=122,slices={s:copy.deepcopy(stats) for s in reader.SLICES},fees_paid=fee,dollar_turnover=10.,trade_rows=1)
            if path!='next_open_cost':training[arm][path]={s:copy.deepcopy(stats) for s in ('training','early','recent')}
            export=dict(HOLDINGS=[dict(fee=fee if i==0 else 0.) for i in range(122)],TRADES=[dict(cost=fee)],DAILY=[dict(date='2020-01-01',equity=100000.)],DAILY_POSITIONS=[dict(date='2020-01-01',cash=0.)])
            for kind,data in export.items():documents[str(reader.OUT/'results'/(arm+'_'+path+'_'+kind+'.json'))]=data
    class FakeEvidence:
        def read(self,path):return documents[str(path)]
    return FakeEvidence(),result,ind,training,documents

class FullSavedAccountingTests(unittest.TestCase):
    def test_all9paths_and_exact_zero_streams(self):
        args=accounting_fixture();result,_=reader.accounting(*args[:4]);self.assertEqual(result['independent_paths'],9);self.assertEqual(result['zero_cost_fee_fields_exact_zero'],6*123)
    def test_experimental_zero_full122_aggregate_cannot_hide_in_near(self):
        args=accounting_fixture();args[1]['arms'][reader.ARMS[2]]['same_close_zero']['slices']['full122']['fees_paid']=1e-16
        with self.assertRaisesRegex(ValueError,'full122 exact0'):reader.accounting(*args[:4])
    def test_experimental_zero_tiny_trade_fee_rejected(self):
        args=accounting_fixture();name=str(reader.OUT/'results'/(reader.ARMS[2]+'_next_open_zero_TRADES.json'));args[4][name][0]['cost']=1e-16
        with self.assertRaisesRegex(ValueError,'Every zero-fee'):reader.accounting(*args[:4])

class SourcePathCanonicalizationTests(unittest.TestCase):
    def fixture(self):return {'outputs/invented/source'+str(i)+'.csv':format(i,'064x') for i in range(64)}
    def test_absolute_and_relative64_exact_pairs_are_equivalent(self):
        rel=self.fixture();absolute={str(reader.ROOT/p):h for p,h in rel.items()}
        self.assertEqual(reader.canonical_source_map(absolute),reader.canonical_source_map(rel))
    def test_changed_digest_changes_normalized_map(self):
        a=self.fixture();b=copy.deepcopy(a);b['outputs/invented/source0.csv']='f'*64
        self.assertNotEqual(reader.canonical_source_map(a),reader.canonical_source_map(b))
    def test_full_paths_not_basename_or_suffix_match(self):
        left=self.fixture();right=self.fixture();h=right.pop('outputs/invented/source0.csv');right['outputs/another/source0.csv']=h
        self.assertNotEqual(reader.canonical_source_map(left),reader.canonical_source_map(right))
    def test_outside_root_absolute_path_rejected(self):
        data=self.fixture();h=data.pop('outputs/invented/source0.csv');data['/private/tmp/outputs/invented/source0.csv']=h
        with self.assertRaisesRegex(ValueError,'current research root'):reader.canonical_source_map(data)
    def test_raw_parent_or_cloud_path_rejected_before_resolve(self):
        for bad in ('../monthly-stock-research/outputs/a.csv','outputs/CloudStorage/../a.csv',str(reader.ROOT/'CloudStorage/a.csv'),'outputs/../a.csv'):
            data=self.fixture();h=data.pop('outputs/invented/source0.csv');data[bad]=h
            with self.subTest(path=bad),self.assertRaisesRegex(ValueError,'Forbidden raw'):reader.canonical_source_map(data)
    def test_identical_digest_alias_collision_is_rejected(self):
        data=self.fixture();data.pop('outputs/invented/source63.csv');data[str(reader.ROOT/'outputs/invented/source0.csv')]=data['outputs/invented/source0.csv']
        with self.assertRaisesRegex(ValueError,'collision'):reader.canonical_source_map(data)
    def test_missing_or_extra_entry_rejected(self):
        for count in (63,65):
            data={'outputs/invented/source'+str(i)+'.csv':format(i,'064x') for i in range(count)}
            with self.subTest(count=count),self.assertRaisesRegex(ValueError,'Exactly64'):reader.canonical_source_map(data)
    def test_bare_sha_and_string_key_required(self):
        for bad in ({'sha256':'0'*64},True,'g'*64,'0'*63):
            data=self.fixture();data['outputs/invented/source0.csv']=bad
            with self.subTest(value=bad),self.assertRaisesRegex(ValueError,'bare SHA256'):reader.canonical_source_map(data)
        for bad in ('',None,0):
            data=self.fixture();h=data.pop('outputs/invented/source0.csv');data[bad]=h
            with self.subTest(key=bad),self.assertRaisesRegex(ValueError,'Nonempty'):reader.canonical_source_map(data)
    def test_symlink_escape_rejected_after_resolve(self):
        with TemporaryDirectory() as tmp:
            oldroot=reader.ROOT;root=Path(tmp).resolve()/'repo';root.mkdir();outside=Path(tmp).resolve()/'outside';outside.mkdir();(root/'link').symlink_to(outside,target_is_directory=True)
            reader.ROOT=root
            try:
                data=self.fixture();h=data.pop('outputs/invented/source0.csv');data['link/source.csv']=h
                with self.assertRaisesRegex(ValueError,'Only local evidence'):reader.canonical_source_map(data)
            finally:reader.ROOT=oldroot
    def test_normalization_does_not_mutate_maps(self):
        data=self.fixture();before=copy.deepcopy(data);reader.canonical_source_map(data);self.assertEqual(data,before)




from types import SimpleNamespace

def proof(kind,n):
    ind=SimpleNamespace(**reader.SCHEMAS,**reader.FLAGS)
    flags={'control':ind.CONTROL_FLAGS,'diagnostic_models':ind.DIAGNOSTIC_MODEL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'diagnostic_selection':ind.DIAGNOSTIC_SELECTION_FLAGS,'selection':ind.SELECTION_FLAGS}[kind]
    r=dict(**{k:True for k in flags},checked_events=n,producer_modules_imported=False,computes_returns=False);fitted=int(n>0)
    pc=dict(target_fits=4*fitted,log_refits=fitted,local_evaluations=fitted,local_solves=fitted,actual_regression_solves=6*fitted)
    sc=dict(pooled_log_solves=fitted,local_linear_solves=fitted,total_regression_solves=2*fitted);lc=dict(pooled_log_solves=0,local_linear_solves=fitted,total_regression_solves=fitted)
    if kind in ('control','diagnostic_models','models'):r.update(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True)
    if kind=='control':r.update(full_control_supplied=True,candidate_labels_computed=False,candidate_models_computed=False,episode_calls=0,producer_equivalent_fit_counts=pc,independent_solve_counts=sc)
    elif kind=='diagnostic_models':
        c=dict.fromkeys(ind.DIAGNOSTIC_MODEL_COUNT_FIELDS,0);c.update(training_rows=fitted,valid_training_rows=fitted,candidate_local_evaluations=fitted,candidate_local_solves=fitted,candidate_kernel_evaluations=fitted,candidate_changed_weight_evaluations=fitted,actual_regression_solves=7*fitted)
        for key in ('target_fits','log_refits','local_evaluations','local_solves'):c['control_'+key]=pc[key]
        r.update(full_diagonal_supplied=True,full_source_inputs_supplied=True,correlation_moments_computed=False,correlation_models_computed=False,episode_calls=0,counts=c,producer_equivalent_counts=copy.deepcopy(c),independent_matrix_builds=0,independent_solve_counts=dict(control=copy.deepcopy(sc),diagonal=copy.deepcopy(lc),total_regression_solves=3*fitted))
    elif kind=='models':
        c=dict.fromkeys(ind.MODEL_COUNT_FIELDS,0);c.update(training_rows=fitted,valid_training_rows=fitted,correlation_matrix_builds=fitted,actual_regression_solves=8*fitted)
        for key in ('target_fits','log_refits','local_evaluations','local_solves'):c['control_'+key]=pc[key]
        for role in ('diag','candidate'):
            for key in ('local_evaluations','local_solves','kernel_evaluations','changed_weight_evaluations'):c[role+'_'+key]=fitted
        r.update(full_bundle_supplied=True,full_labels_supplied=True,full_control_supplied=True,full_diagonal_supplied=True,full_candidate_supplied=True,independent_numeric_system='augmented_least_squares_SVD',counts=c,producer_equivalent_counts=copy.deepcopy(c),independent_matrix_builds=fitted,independent_solve_counts=dict(control=copy.deepcopy(sc),diagonal=copy.deepcopy(lc),candidate=copy.deepcopy(lc),total_regression_solves=4*fitted))
    else:
        r.update(numerical_tolerance=0.,ordered_choices_checked=3*n if kind=='selection' else n)
        if kind=='diagnostic_selection':r.update(full_diagonal_scores_supplied=True,full_diagonal_pre_scores_supplied=True,correlation_selection_computed=False,episode_calls=1)
        else:r.update(full_scores_supplied=True,full_pre_scores_supplied=True)
        if kind=='incumbent':r['candidate_selection_computed']=False
        elif kind=='selection':
            c=dict.fromkeys(ind.COUNT_FIELDS,0);c.update(events=n,symbols_per_event=49,initial_partial_events=fitted,formal_events=n-fitted,incumbent_episode_calls=1,diag_episode_calls=1,candidate_episode_calls=1)
            r.update(full_bundle_supplied=True,full_audit_supplied=True,counts=c)
    return r


def model_audit(counts):
    result=dict(schema='monthly_correlation_distance257_model_audit_v1',counts=copy.deepcopy(counts),mass_relative_tolerance=reader.MASS_RTOL,residuals_are_oof=False,computes_returns=False)
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','candidate_pooled_copied','candidate_local_constant_copied','candidate_local_linear_only','diagonal_forecast_reproduced','correlation_floor_coordinates_retained','identity_correlation_reuses_diagonal','residuals_are_mature_in_sample'):result[key]=True
    return result


def label_audit(counts):
    ind=SimpleNamespace(**reader.SCHEMAS,**reader.FLAGS)
    return dict(schema='monthly_correlation_distance257_labels_preserved_v1',rows=counts['training_rows'],valid_rows=counts['valid_training_rows'],target_keys=list(ind.TARGETS),target_sha256={k:'a'*64 for k in ind.TARGETS},common_good_sha256='b'*64,labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)


def parity_receipt(saved,n,model):
    r=dict(events=n,cache_sha256=saved['input_sha256'],orders_exact=True,scores_exact=True,pre_scores_exact=True)
    if model:r['forecast_exact']=True
    return r


def evidence():
    kinds=('control','diagnostic_models','models','incumbent','diagnostic_selection','selection');e={k:proof(k,122) for k in kinds};saved={'input_sha256':{name:'a'*64 for name in reader.DIAGNOSTIC_FILES}}
    e.update(main_model_counts=model_audit(e['models']['counts']),all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True,diagnostic_cache_sha256=saved['input_sha256'],diagnostic_parity=parity_receipt(saved,122,True))
    e['cached_prefixes']=[dict(prefix=n,exact=True,incumbent=proof('incumbent',n),diagnostic_selection=proof('diagnostic_selection',n),diagnostic_parity=parity_receipt(saved,n,False)) for n in range(123)]
    e['physical_prefixes']=[]
    for n in reader.PREFIXES:
        v=dict(prefix=n,exact=True,model_recomputed=n!=122,diagnostic_parity=parity_receipt(saved,n,True))
        for k in kinds:v['model' if k=='models' else k]=proof(k,n)
        v['model_counts']=model_audit(v['model']['counts'])
        if n==122:v['reuses_main_verified_full']=True
        e['physical_prefixes'].append(v)
    e['fit_execution_accounting']=reader.execution_accounting(e)
    ma=dict(model=copy.deepcopy(e['main_model_counts']),labels=label_audit(e['models']['counts']))
    da=dict(schema='monthly_correlation_distance257_decisions_v1',events=[{} for _ in range(122)],counts=copy.deepcopy(e['selection']['counts']),incumbent_gate=copy.deepcopy(e['incumbent']),diagnostic_gate=copy.deepcopy(e['diagnostic_selection']),computes_returns=False)
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact','diagonal_decisions_reproduced'):da[key]=True
    return e,ma,da


def selection_fixture():
    train={a:{p:{g:dict(cagr=.21 if a==reader.ARMS[2] else .2,sharpe=1.,max_drawdown=-.1,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None) for g in ('training','early','recent')} for p in reader.PATHS[:2]} for a in reader.ARMS}
    changed={reader.ARMS[2]+'_vs_'+reader.ARMS[0]:dict(months=[dict(index=i) for i in (1,54,55)])}
    return train,changed


class Selection257Tests(unittest.TestCase):
    def test_good_two_arm_selection(self):
        t,c=selection_fixture();self.assertTrue(reader.saved_selection(t,c)['admitted'])
        self.assertEqual(set(reader.saved_selection(t,c)['guards']),{'same_close','next_open','support'})
    def test_only_two_changed_months_fail(self):
        t,c=selection_fixture();c[next(iter(c))]['months']=[dict(index=45),dict(index=74)]
        r=reader.saved_selection(t,c);self.assertFalse(r['admitted']);self.assertFalse(r['guards']['support'])
    def test_later_changes_do_not_rescue(self):
        t,c=selection_fixture();c[next(iter(c))]['months']=[dict(index=i) for i in range(88,122)]
        self.assertFalse(reader.saved_selection(t,c)['admitted'])
    def test_training_cagr_only_existing_epsilon(self):
        t,c=selection_fixture();t[reader.ARMS[2]]['same_close_zero']['training']['cagr']=.205-5e-13
        self.assertTrue(reader.saved_selection(t,c)['admitted'])
        t[reader.ARMS[2]]['same_close_zero']['training']['sharpe']=.95-5e-13
        self.assertFalse(reader.saved_selection(t,c)['admitted'])
    def test_early_cagr_no_epsilon(self):
        t,c=selection_fixture();t[reader.ARMS[2]]['same_close_zero']['early']['cagr']=.175-5e-13
        self.assertFalse(reader.saved_selection(t,c)['admitted'])
    def test_candidate_undefined_refuses_and_incumbent_undefined_errors(self):
        for arm in reader.ARMS:
            t,c=selection_fixture();t[arm]['same_close_zero']['training'].update(sharpe=None,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd',daily_return_sd=0.)
            if arm==reader.ARMS[0]:
                with self.assertRaises(ValueError):reader.saved_selection(t,c)
            else:self.assertEqual(reader.saved_selection(t,c)['admitted'],arm==reader.ARMS[1])
    def test_invalid_required_metrics_rejected(self):
        for v in (True,float('inf'),float('nan'),'0.7'):
            t,c=selection_fixture();t[reader.ARMS[2]]['same_close_zero']['training']['cagr']=v
            with self.assertRaises(ValueError):reader.saved_selection(t,c)


class ModelReceipt257Tests(unittest.TestCase):
    def test_all_saved_scope_and_solve_accounting(self):
        e,ma,da=evidence();reader.preparation_evidence(e,ma,da)
        a=e['fit_execution_accounting'];self.assertEqual(a['model_invocations'],5);self.assertEqual(a['totals']['total_regression_solves'],68)
        self.assertEqual(a['entries'][1]['events'],0);self.assertEqual(a['entries'][1]['total_regression_solves'],0)
    def test_standard_flags_cannot_be_bypassed_with_empty_flags(self):
        for kind in ('control','diagnostic_models','models','incumbent','diagnostic_selection','selection'):
            r=proof(kind,1);key=next(k for k in reader.FLAGS[{'control':'CONTROL_FLAGS','diagnostic_models':'DIAGNOSTIC_MODEL_FLAGS','models':'MODEL_FLAGS','incumbent':'INCUMBENT_FLAGS','diagnostic_selection':'DIAGNOSTIC_SELECTION_FLAGS','selection':'SELECTION_FLAGS'}[kind]] if k!='passed');r[key]=False
            with self.assertRaises(ValueError):reader.receipt(r,1,(),kind)
    def test_early_labels_and_candidate_selection_fail(self):
        for kind,key in (('control','candidate_labels_computed'),('control','candidate_models_computed'),('incumbent','candidate_selection_computed')):
            r=proof(kind,1);r[key]=True
            with self.assertRaises(ValueError):reader.receipt(r,1,(),kind)
    def test_missing_full_proofs_and_wrong_tolerance_fail(self):
        for kind in ('control','diagnostic_models','models','incumbent','diagnostic_selection','selection'):
            r=proof(kind,1)
            for key in [k for k in r if k.startswith('full_')]:
                bad=copy.deepcopy(r);del bad[key]
                with self.assertRaises(ValueError):reader.receipt(bad,1,(),kind)
            for key in [k for k in r if 'tolerance' in k or k in ('continuous_atol','continuous_rtol')]:
                bad=copy.deepcopy(r);bad[key]=False
                with self.assertRaises(ValueError):reader.receipt(bad,1,(),kind)
    def test_native_integer_counters_not_bool(self):
        for kind in ('control','diagnostic_models','models','incumbent','diagnostic_selection','selection'):
            r=proof(kind,1);r['checked_events']=True
            with self.assertRaises(ValueError):reader.receipt(r,1,(),kind)
    def test_producer_5fits_vs_oracle1fit_not_conflated(self):
        r=proof('control',1);r['independent_solve_counts'].update(pooled_log_solves=5,total_regression_solves=6)
        with self.assertRaises(ValueError):reader.receipt(r,1,(),'control')
    def test_model_count_inconsistent_is_rejected(self):
        for key,value in [('actual_regression_solves',13),('control_target_fits',3),('control_local_solves',2),('training_rows',True),('candidate_pooled_fits',1),('candidate_log_refits',1)]:
            r=proof('models',1);r['counts'][key]=value;r['producer_equivalent_counts']=copy.deepcopy(r['counts'])
            with self.assertRaises(ValueError):reader.receipt(r,1,(),'models')
    def test_empty_prefix_cannot_claim_fits(self):
        for kind in ('control','models'):
            r=proof(kind,1);r['checked_events']=0
            with self.assertRaises(ValueError):reader.receipt(r,0,(),kind)
    def test_middle_prefix_certificate_is_required(self):
        e,ma,da=evidence();del e['cached_prefixes'][61]['incumbent']['full_scores_supplied']
        with self.assertRaises(ValueError):reader.preparation_evidence(e,ma,da)
    def test_physical122_cannot_claim_recomputation(self):
        e,ma,da=evidence();e['physical_prefixes'][-1]['model_recomputed']=True
        with self.assertRaises(ValueError):reader.preparation_evidence(e,ma,da)
    def test_count_tamper_cannot_be_masked_by_passed(self):
        for slot in ('models','main_model_counts','physical'):
            e,ma,da=evidence()
            if slot=='models':e[slot]['counts']['training_rows']+=1
            elif slot=='main_model_counts':e[slot]['counts']['training_rows']+=1
            else:e['physical_prefixes'][2]['model_counts']['counts']['training_rows']+=1
            with self.assertRaises(ValueError):reader.preparation_evidence(e,ma,da)
    def test_total_model_count_does_not_double_count_full122(self):
        e,ma,da=evidence();e['fit_execution_accounting']['totals']['total_regression_solves']+=18
        with self.assertRaises(ValueError):reader.preparation_evidence(e,ma,da)
    def test_all_input_receipts_preserved(self):
        args=evidence();before=copy.deepcopy(args);reader.preparation_evidence(*args);self.assertEqual(args,before)


class SavedNAVImpactTests(unittest.TestCase):
    def test_descriptive_nav_growth_not_replay(self):
        ch=dict(months=[dict(index=45,date='2020-01-01',end='2020-02-01')]);ex={}
        for a,finish in zip(reader.ARMS,(110.,110.,105.)):
            ex[a,'same_close_zero']=dict(DAILY=[dict(date='2020-01-01',equity=100.),dict(date='2020-02-01',equity=finish)])
        r=reader.saved_month_impacts(ch,ex);self.assertFalse(r['new_portfolio_replay']);self.assertAlmostEqual(r['months'][0]['candidate_minus_incumbent'],-.05)
    def test_invalid_saved_nav_rejected(self):
        ch=dict(months=[dict(date='a',end='b')]);ex={(a,'same_close_zero'):dict(DAILY=[dict(date='a',equity=0.),dict(date='b',equity=1.)]) for a in reader.ARMS}
        with self.assertRaises(ValueError):reader.saved_month_impacts(ch,ex)


def model_fixture():
    c=proof('models',1)['counts'];c.update(training_rows=3,valid_training_rows=2,diag_equal_scale_evaluations=1,diag_changed_weight_evaluations=0,candidate_identity_correlation_evaluations=1,candidate_changed_weight_evaluations=0)
    model=model_audit(c);model['original_cache_parity']=dict(passed=True,events=122,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False)
    labels=label_audit(c)
    old=dict(status='fit',event_row=0,date='2016-08-17',symbol='S0',year=2016,cutoff='2015-12-31',own_rows=2,own_months=18,latest_label_end='2015-11-30',constant_correction=.1,linear_correction=.2,weight_mass=1.,effective_samples=2.)
    a=dict(feature_count=10,alpha=10.,half_life_months=12.,min_months=18,min_own_months=18,source=dict(annual=[dict(status='fit',prediction_rows=1)]),annual=[dict(status='fit',prediction_rows=1,year=2016,cutoff='2015-12-31',latest_label_end='2015-11-30')],local=[old])
    g=dict(source_scale=[1.]*10,coefficients=[.1]*10,old_distance=[0.,1.],distance=[0.,1.],original_weights=[.5,.5],unscaled_weights=[.5,.5],weights=[.5,.5],equal_scales=True,original_mass=1.,unscaled_mass=1.,mass_multiplier=1.,matched_mass=1.,effective_samples=2.,original_effective_samples=2.,mass_absolute_error=0.,maximum_absolute_weight_change=0.,weights_exact_original=True)
    b=dict(schema='monthly_local_distance256_candidate_audit_v1',feature_count=10,target='unchanged cached centered log_excess',pooled_copied=True,local_constant_copied=True,residuals='same mature in-sample pooled residuals as control',local_linear_only=True,annual=[dict(year=2016,cutoff='2015-12-31',training_row_ids=[0,1],query_row_ids=[0],source_scale=[1.]*10)],local=[dict(old,own_row_ids=[0,1],geometry=g)])
    candidate=copy.deepcopy(b);candidate['schema']='monthly_correlation_distance257_candidate_audit_v1'
    identity=[[float(j==k) for k in range(10)] for j in range(10)]
    candidate['annual'][0]['moments']=dict(source_mean=[0.]*10,raw_scale=[1.]*10,source_scale=[1.]*10,floored=[False]*10,training_weights=[.5,.5],covariance=copy.deepcopy(identity),correlation=copy.deepcopy(identity),shrinkage=copy.deepcopy(identity),cholesky=copy.deepcopy(identity),transform=copy.deepcopy(identity),trace=10.,metric=[[float(j==k)/10 for k in range(10)] for j in range(10)],identity_correlation=True)
    cg={k:copy.deepcopy(v) for k,v in g.items() if k not in ('coefficients','equal_scales')};cg.update(identity_correlation=True,diagonal_geometry=copy.deepcopy(g));candidate['local'][0]['geometry']=cg
    return dict(model=model,labels=labels,control=a,diagonal=b,candidate=candidate),dict(dates=['2016-08-17'],symbols=[['S'+str(i) for i in range(49)]])


def decision_fixture():
    dates=['2016-08-17']+[(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(121)]
    meta=dict(dates=dates,symbols=[['S'+str(i) for i in range(49)] for _ in dates]);orders={a:[dict(symbols=['S0','S1','S2']) for _ in dates] for a in reader.ARMS}
    es=[dict(query_index=i,date=d,initial_partial=i==0,weak=False,ranking_indices=[0,1,2,3],active_indices=[],diag_changed_delta_indices=[],diag_delta=[0.]*49,changed_delta_indices=[],delta=[0.]*49,incumbent_choices=[0,1,2],diag_choices=[0,1,2],candidate_choices=[0,1,2],episode_budget=.05) for i,d in enumerate(dates)]
    c={k:0 for k in reader.SCHEMAS['COUNT_FIELDS']};c.update(events=122,symbols_per_event=49,initial_partial_events=1,formal_events=121,original_eligible_cells=488,ranking_cells=488,incumbent_episode_calls=1,diag_episode_calls=1,candidate_episode_calls=1)
    ra=dict(schema='monthly_correlation_distance257_rank_delta_v1',events=122,ranking_cells=488,ready_rows=[True]*122,active_indices=[[] for _ in dates],changed_cells=0,original_ranking_population=True,no_group_ranking=True,no_own_history_filter=True,pooled_ranks_preserved=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)
    return dict(counts=c,events=es,rank_delta={a:copy.deepcopy(ra) for a in reader.ARMS[1:]}),meta,orders


class SavedModelSummaryTests(unittest.TestCase):
    def test_complete_saved_local_and_preservation_metadata(self):
        a,m=model_fixture();r=reader.model_summary(a,m);self.assertEqual(r['roles']['candidate']['saved_geometry_records_checked'],1);self.assertTrue(r['experimental_pooled_and_log_fits_zero'])
    def test_changed_label_or_missing_target_rejected(self):
        for key,value in [('labels_unchanged',False),('recentered',True),('target_keys',['log_excess'])]:
            a,m=model_fixture();a['labels'][key]=value
            with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_equal_scale_cannot_claim_changed_weights(self):
        a,m=model_fixture();a['candidate']['local'][0]['geometry']['weights'][0]=math.nextafter(.5,math.inf)
        with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_mass_error_cannot_exceed_prospective_bound(self):
        a,m=model_fixture();g=a['candidate']['local'][0]['geometry'];g['matched_mass']=1.+1e-10;g['mass_absolute_error']=g['matched_mass']-1.
        with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_candidate_preserved_constant_or_scale_invalid(self):
        a,m=model_fixture();a['candidate']['local'][0]['constant_correction']=.2
        with self.assertRaises(ValueError):reader.model_summary(a,m)
        a,m=model_fixture();a['candidate']['annual'][0]['source_scale'][0]=0.
        with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_local_maturity_or_actual_fit_count_rejected(self):
        for key,value in (('latest_label_end','2016-01-01'),('status','insufficient_own_history')):
            a,m=model_fixture();a['candidate']['local'][0][key]=value
            with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_full_decision_metadata(self):
        self.assertEqual(reader.decision_summary(*decision_fixture())['counts']['candidate_changed_holdings_events'],0)
    def test_active_nonweak_or_partial_rejected(self):
        for i in (0,1):
            a,m,o=decision_fixture();a['events'][i]['active_indices']=[0];a['rank_delta'][reader.ARMS[2]]['active_indices'][i]=[0]
            with self.assertRaises(ValueError):reader.decision_summary(a,m,o)
    def test_delta_index_and_zero_delta_order_rejected(self):
        a,m,o=decision_fixture();a['events'][3]['delta'][0]=.01
        with self.assertRaises(ValueError):reader.decision_summary(a,m,o)
        a,m,o=decision_fixture();o[reader.ARMS[2]][2]['scores']=[.1]
        with self.assertRaises(ValueError):reader.decision_summary(a,m,o)
    def test_native_count_tamper_rejected(self):
        a,m,o=decision_fixture();a['counts']['candidate_changed_delta_cells']=True
        with self.assertRaises(ValueError):reader.decision_summary(a,m,o)


class Correlation257Tests(unittest.TestCase):
    def test_diagonal_best_never_promoted(self):
        t,c=selection_fixture()
        for p in reader.PATHS[:2]:t[reader.ARMS[1]][p]['training']['cagr']=5.;t[reader.ARMS[2]][p]['training']['cagr']=.19
        self.assertEqual(reader.saved_selection(t,c)['primary'],reader.ARMS[0])
    def test_descriptive_matrix_and_early_diagonal_accounting(self):
        e,ma,da=evidence();reader.preparation_evidence(e,ma,da);a=e['fit_execution_accounting']
        self.assertFalse(a['correlation_matrix_builds_are_predictive_fits']);self.assertEqual(a['totals']['producer_correlation_matrix_builds'],4);self.assertEqual(a['totals']['independent_diagnostic_gate_solves'],12)
        a['totals']['total_regression_solves']+=a['totals']['producer_correlation_matrix_builds']
        with self.assertRaises(ValueError):reader.preparation_evidence(e,ma,da)
    def test_early_correlation_computation_rejected(self):
        for kind,key in (('diagnostic_models','correlation_models_computed'),('diagnostic_models','correlation_moments_computed'),('diagnostic_selection','correlation_selection_computed')):
            r=proof(kind,1);r[key]=True
            with self.assertRaises(ValueError):reader.receipt(r,1,(),kind)
    def test_full_diagnostic_forecast_and_selection_proofs_required(self):
        for scope in ('main','physical','cached'):
            e,ma,da=evidence();x=e if scope=='main' else e['physical_prefixes'][2] if scope=='physical' else e['cached_prefixes'][55]
            x['diagnostic_parity']['orders_exact']=False
            with self.assertRaises(ValueError):reader.preparation_evidence(e,ma,da)
        e,ma,da=evidence();e['diagnostic_parity']['forecast_exact']=False
        with self.assertRaises(ValueError):reader.preparation_evidence(e,ma,da)
    def test_six_prior_pins_cannot_be_partial_or_changed(self):
        a={k:'a'*64 for k in reader.DIAGNOSTIC_FILES};reader.diagnostic_cache_binding(a,dict(a),dict(a))
        for name in reader.DIAGNOSTIC_FILES:
            b=dict(a);b[name]='b'*64
            with self.assertRaises(ValueError):reader.diagnostic_cache_binding(a,b,a)
            b=dict(a);del b[name]
            with self.assertRaises(ValueError):reader.diagnostic_cache_binding(a,a,b)
    def test_diagonal_json_model_and_orders_exact(self):
        model,meta=model_fixture();orders={a:[{'symbols':['A'],'scores':[1.]}] for a in reader.ARMS}
        before=dict(control=copy.deepcopy(model['control']),candidate=copy.deepcopy(model['diagonal']),labels=copy.deepcopy(model['labels']));before['labels']['schema']='monthly_local_distance256_labels_preserved_v1'
        prior_orders={a:copy.deepcopy(orders[a]) for a in reader.ARMS[:2]}
        reader.diagnostic_json_parity(meta,prior_orders,before,meta,orders,model)
        for slot in ('orders','audit','labels'):
            b=copy.deepcopy(model);o=copy.deepcopy(orders)
            if slot=='orders':o[reader.ARMS[1]][0]['scores'][0]=math.nextafter(1.,math.inf)
            elif slot=='audit':b['diagonal']['local'][0]['linear_correction']=.3
            else:b['labels']['common_good_sha256']='e'*64
            with self.assertRaises(ValueError):reader.diagnostic_json_parity(meta,prior_orders,before,meta,o,b)
    def test_moment_shape_and_nonfinite_rejected(self):
        for key,value in (('trace',0.),('trace',True),('source_scale',[float('inf')]*10),('correlation',[[1.]])):
            a,m=model_fixture();a['candidate']['annual'][0]['moments'][key]=value
            with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_floor_and_exact_identity_claims_rejected(self):
        for target in ('floor','identity','shrinkage','upper','symmetry'):
            a,m=model_fixture();x=a['candidate']['annual'][0]['moments']
            if target=='floor':x['floored'][0]=True
            elif target=='identity':x['identity_correlation']=False
            elif target=='shrinkage':x['shrinkage'][0][1]=1e-16
            elif target=='upper':x['cholesky'][0][1]=1e-16
            else:x['correlation'][0][1]=1e-16
            with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_identity_geometry_must_reuse_diagonal(self):
        a,m=model_fixture();g=a['candidate']['local'][0]['geometry'];g['distance'][1]=math.nextafter(1.,math.inf)
        with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_diagonal_and_candidate_count_separation(self):
        for field in ('diag_changed_weight_evaluations','candidate_identity_correlation_evaluations','correlation_matrix_builds'):
            a,m=model_fixture();a['model']['counts'][field]+=1
            with self.assertRaises(ValueError):reader.model_summary(a,m)
    def test_six_zero_streams_all_have_strict_trade_fees(self):
        for arm in reader.ARMS:
            for path in reader.PATHS[:2]:
                args=accounting_fixture();args[4][str(reader.OUT/'results'/(arm+'_'+path+'_TRADES.json'))][0]['cost']=1e-16
                with self.assertRaisesRegex(ValueError,'Every zero-fee'):reader.accounting(*args[:4])


if __name__=='__main__':unittest.main(verbosity=2)

