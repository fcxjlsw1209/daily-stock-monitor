"""Offline254 reader rejection tests; invented metadata only, no actual artifacts."""
import copy
from datetime import date,timedelta
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import calendar_memory254_saved_result_reader as reader

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

def selection_fixture():
    train={}
    for a in reader.ARMS:
        train[a]={}
        for p in reader.PATHS[:2]:
            c=.21 if a==reader.ARMS[1] else .20
            train[a][p]={k:dict(cagr=c,sharpe=1.,max_drawdown=-.1,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None) for k in ('training','early','recent')}
    changed={reader.ARMS[1]+'_vs_'+ref:dict(months=[dict(index=i) for i in (0,1,60)]) for ref in (reader.ARMS[0],reader.ARMS[2])}
    return train,changed

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
        args=accounting_fixture();args[1]['arms'][reader.ARMS[1]]['same_close_zero']['slices']['full122']['fees_paid']=1e-16
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



class SelectionTests(unittest.TestCase):
    def test_calendar_passes_and_fixed21_never_primary(self):
        train,changed=selection_fixture();p=reader.saved_selection(train,changed)
        self.assertTrue(p['admitted']);self.assertEqual(p['primary'],reader.ARMS[1])
        self.assertEqual(p['diagnostic_control_not_selectable'],reader.ARMS[2])
    def test_control_outperforms_each_cagr_or_close_sharpe_rejects(self):
        for path,key in (('same_close_zero','cagr'),('next_open_zero','cagr'),('same_close_zero','sharpe')):
            tr,ch=selection_fixture();tr[reader.ARMS[2]][path]['training'][key]=9.
            with self.subTest(path=path,key=key):
                p=reader.saved_selection(tr,ch);self.assertFalse(p['guards']['calendar_vs_fixed21']);self.assertEqual(p['primary'],reader.ARMS[0])
    def test_comparator_close_undefined_rejects_but_open_allowed(self):
        for path,want in (('same_close_zero',False),('next_open_zero',True)):
            tr,ch=selection_fixture();tr[reader.ARMS[2]][path]['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
            self.assertIs(reader.saved_selection(tr,ch)['admitted'],want)
    def test_candidate_undefined_either_zero_rejects(self):
        for path in reader.PATHS[:2]:
            tr,ch=selection_fixture();tr[reader.ARMS[1]][path]['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
            self.assertFalse(reader.saved_selection(tr,ch)['admitted'])
    def test_incumbent_undefined_and_illegal_metric_hardfail(self):
        tr,ch=selection_fixture();tr[reader.ARMS[0]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        with self.assertRaises(ValueError):reader.saved_selection(tr,ch)
        for v in (True,float('nan'),float('inf')):
            tr,ch=selection_fixture();tr[reader.ARMS[2]]['next_open_zero']['training']['cagr']=v
            with self.assertRaises(ValueError):reader.saved_selection(tr,ch)
    def test_only_cagr_gets_inherited_tolerance(self):
        tr,ch=selection_fixture();tr[reader.ARMS[2]]['same_close_zero']['training']['cagr']=.21+5e-13
        self.assertTrue(reader.saved_selection(tr,ch)['admitted'])
        tr[reader.ARMS[2]]['same_close_zero']['training']['sharpe']=math.nextafter(1.,math.inf)
        self.assertFalse(reader.saved_selection(tr,ch)['admitted'])
    def test_later_changes_do_not_rescue_support(self):
        tr,ch=selection_fixture();ch[reader.ARMS[1]+'_vs_'+reader.ARMS[0]]['months']=[dict(index=i) for i in (88,89,121)]
        self.assertFalse(reader.saved_selection(tr,ch)['guards']['support'])
    def test_incumbent_risk_and_segment_guards_retained(self):
        for path,group,key,value in (('same_close_zero','training','sharpe',.949),('next_open_zero','training','max_drawdown',-.131),('same_close_zero','early','cagr',.1749),('next_open_zero','recent','cagr',.1749)):
            tr,ch=selection_fixture();tr[reader.ARMS[1]][path][group][key]=value
            self.assertFalse(reader.saved_selection(tr,ch)['admitted'])


def memory_fixture():
    schemas,flags=reader.schema_flags();names=['A%02d'%i for i in range(49)]
    dates=['2016-08-17']+[(date(2016,8,17)+timedelta(days=i)).isoformat() for i in range(1,122)]
    meta=dict(dates=dates,ends=dates[1:]+['2017-12-31'],symbols=[names[:] for _ in dates])
    sc=dict(zip(schemas['SOURCE_COUNT_FIELDS'],(122,49,49,100,90,85,80,20,1,121,122*49,5000,2500,0,0)))
    source=dict(schema='monthly_calendar_memory254_features_v1',counts=sc,anchor_rule='official_monthends_original_good_positive_signal_no_lockout',common_maturity_rule='maximum_of_actual_exits',numeric_cutoff=dates[-1],candidate_selection_computed=False,portfolio_returns_computed=False)
    counts={k:0 for k in schemas['COUNT_FIELDS']}
    counts.update(events=122,symbols_per_event=49,initial_partial_events=1,formal_events=121,original_eligible_cells=610,query_signal_cells=100,query_finite_cells=500,original_episode_calls=1,candidate_episode_calls=1,control_episode_calls=1,anchors=100,mature_common_anchors=80)
    events=[];orders={a:[] for a in reader.ARMS}
    for i,d in enumerate(dates):
        e=dict(query_index=i,date=d,initial_partial=i==0,original_eligible_indices=list(range(5)),pre_order=list(range(5)),boundary_candidates=[2,3,4],episode_budget=.05,incumbent_choices=[0,1,2],candidate_choices=[0,1,2],control_choices=[0,1,2])
        events.append(e)
        for a in reader.ARMS:orders[a].append(dict(date=d,end=meta['ends'][i],signal_date=d,symbols=names[:3],scores=[.3,.2,.1]))
    audit=dict(schema='monthly_calendar_memory254_decisions_v1',counts=counts,events=events)
    return schemas,flags,source,audit,meta,orders


def source_receipt(flags,source,n):
    x=dict(**{k:True for k in flags['SOURCE_FLAGS']},checked_events=n,source_audit=copy.deepcopy(source),counts=copy.deepcopy(source['counts']),
      full_source_audit_supplied=True,episode_calls=0,memory_effects_computed=False,candidate_selection_computed=False,producer_modules_imported=False,computes_returns=False,continuous_atol=2e-10,continuous_rtol=2e-10,exact_discrete=True,source_data_sha256='1'*64)
    x['counts']['queries']=n;x['source_audit']['counts']['queries']=n
    return x


def inc_receipt(flags,n):
    r={k:True for k in flags['INCUMBENT_GATE_FLAGS']}
    r.update(checked_events=n,ordered_choices_checked=n,numerical_tolerance=0.,full_scores_supplied=True,full_pre_scores_supplied=True,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
    return r


def selection_receipt(flags,counts,n):
    r={k:True for k in flags['SELECTION_FLAGS']}
    r.update(checked_events=n,ordered_choices_checked=3*n,numerical_tolerance=0.,full_scores_supplied=True,full_pre_scores_supplied=True,full_effects_supplied=True,full_audit_supplied=True,producer_modules_imported=False,computes_returns=False,counts=copy.deepcopy(counts))
    r['counts']['events']=n;return r


def prepare_fixture():
    schemas,flags,source,audit,meta,orders=memory_fixture()
    full=source_receipt(flags,source,122)
    gates=dict(source=full,incumbent=inc_receipt(flags,122))
    e=dict(all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_input_unchanged=True,original_full49_pre_preserved=True,new_predictive_model_fits=0,pre_candidate_gates=gates,selection=selection_receipt(flags,audit['counts'],122),all123_cached_control_gates=[],physical_prefixes=[])
    for n in range(123):
        s=dict(passed=True,checked_events=n,cached_source_reference=True,full_source_checked_events=122,no_new_source_rebuild_claim=True,full_source_receipt_sha256=reader.digest_record(full),full_source_data_sha256=full['source_data_sha256'])
        e['all123_cached_control_gates'].append(dict(prefix=n,exact=True,gates=dict(source=s,incumbent=inc_receipt(flags,n))))
    for n in reader.PREFIXES:
        check=selection_receipt(flags,audit['counts'],n)
        e['physical_prefixes'].append(dict(prefix=n,exact=True,gates=dict(source=source_receipt(flags,source,n),incumbent=inc_receipt(flags,n)),selection=check,counts=copy.deepcopy(check['counts'])))
    pc=dict(new_model_fits=0,new_predictive_model_fits=0,new_returns_computed=False,events=122,descriptive_counts=copy.deepcopy(audit['counts']))
    parity=dict(source_data_sha256=full['source_data_sha256'],source_audit_sha256=reader.digest_record(source))
    return e,pc,audit,source,parity,flags


class SourceAndDecisionTests(unittest.TestCase):
    def test_exact_schema_source_counts_and_saved_choices(self):
        s,f,src,au,m,o=memory_fixture()
        self.assertEqual(reader.source_summary(src,m,s)['counts']['anchors'],100)
        self.assertEqual(reader.decision_summary(au,src,m,o,s)['counts']['events'],122)
        self.assertEqual((len(reader.PREP),len(reader.RESULT)),(10,38))
    def test_source_count_or_maturity_rule_tamper_rejected(self):
        for field in ('common','count','scope','rule'):
            s,f,src,au,m,o=memory_fixture()
            if field=='common':src['counts']['common_valid_anchors']=91
            elif field=='count':src['counts']['query_cells']-=1
            elif field=='scope':src['portfolio_returns_computed']=True
            else:src['common_maturity_rule']='shorter_exit'
            with self.subTest(field=field),self.assertRaises(ValueError):reader.source_summary(src,m,s)
    def test_new_source_counts_not_copied_from_previous_round(self):
        s,f,src,au,m,o=memory_fixture();src['counts']['anchors']+=1;src['counts']['common_invalid_anchors']+=1
        self.assertEqual(reader.source_summary(src,m,s)['counts']['anchors'],101)
        with self.assertRaises(ValueError):reader.decision_summary(au,src,m,o,s)
    def test_choice_outside_original_boundary_and_reordered_identity_rejects(self):
        for mutation in ('top2','rank6','identity','partial','counts'):
            s,f,src,au,m,o=memory_fixture()
            if mutation=='top2':au['events'][1]['candidate_choices']=[0,2,3]
            elif mutation=='rank6':au['events'][1]['candidate_choices']=[0,1,5]
            elif mutation=='identity':o[reader.ARMS[1]][1]['symbols'].reverse()
            elif mutation=='partial':o[reader.ARMS[1]][0]['scores'][0]+=.01
            else:au['counts']['candidate_changed_holdings_events']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):reader.decision_summary(au,src,m,o,s)
    def test_only_one_selector_each_and_no_fit(self):
        for key in ('original_episode_calls','candidate_episode_calls','control_episode_calls','new_model_fits'):
            s,f,src,au,m,o=memory_fixture();au['counts'][key]+=1
            with self.assertRaises(ValueError):reader.decision_summary(au,src,m,o,s)
    def test_complete_preparation_passes(self):
        out=reader.preparation(*prepare_fixture())
        self.assertFalse(out['source_data_digest_recomputed_from_npz'])
    def test_source_receipt_cannot_claim_prediction_or_incomplete_audit(self):
        for key,value in (('episode_calls',1),('memory_effects_computed',True),('candidate_selection_computed',True),('full_source_audit_supplied',False),('continuous_atol',1e-9),('producer_modules_imported',True)):
            args=prepare_fixture();args[0]['pre_candidate_gates']['source'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):reader.preparation(*args)
    def test_cached_full_source_digest_and_data_binding_rejects(self):
        for field in ('full_source_receipt_sha256','full_source_data_sha256','full_source_checked_events','no_new_source_rebuild_claim'):
            args=prepare_fixture();s=args[0]['all123_cached_control_gates'][53]['gates']['source'];s[field]='bad'
            with self.subTest(field=field),self.assertRaises(ValueError):reader.preparation(*args)
    def test_source_data_and_audit_parity_digests_rejects(self):
        for field in ('source_data_sha256','source_audit_sha256'):
            args=prepare_fixture();args[4][field]='f'*64
            with self.assertRaises(ValueError):reader.preparation(*args)
    def test_missing_physical_or_cached_prefix_and_old_endpoint_gate_rejects(self):
        for kind in ('physical','cached','endpoint'):
            args=prepare_fixture()
            if kind=='physical':args[0]['physical_prefixes'].pop()
            elif kind=='cached':args[0]['all123_cached_control_gates'].pop()
            else:args[0]['pre_candidate_gates']['endpoints']={'passed':True}
            with self.assertRaises(ValueError):reader.preparation(*args)
    def test_full_effects_pre_score_and_exact_scope_required(self):
        for kind,field,value in (('selection','full_effects_supplied',False),('selection','ordered_choices_checked',122),('incumbent','full_pre_scores_supplied',False),('incumbent','numerical_tolerance',1e-10)):
            args=prepare_fixture();r=args[0]['selection'] if kind=='selection' else args[0]['pre_candidate_gates']['incumbent'];r[field]=value
            with self.assertRaises(ValueError):reader.preparation(*args)


if __name__=='__main__':unittest.main(verbosity=2)

