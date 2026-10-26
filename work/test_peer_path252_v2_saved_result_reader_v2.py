"""Offline saved-evidence comparison counterexamples; no actual artifacts read."""
import copy
from datetime import date,timedelta
import math
import unittest

import peer_path252_v2_saved_result_reader_v2 as reader
from tempfile import TemporaryDirectory
from pathlib import Path
import json


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
    def test_byte_identity_required_for_zero_exports(self):
        args=fixture();results=dict(arms={reader.ARMS[0]:args[1]});previous=dict(arms={reader.ARMS[0]:args[0]})
        completion=dict(artifacts_sha256={'RESULTS.json':'result','INDEPENDENT_OUTCOME.json':'cert'})
        review=dict(passed=True,results_completion_sha256='completion',incumbent_reproduction=dict(period_fee_exception=dict(absolute_bound=1e-10,scope='ONLY next_open_cost individual HOLDINGS.fee; cumulative fields require exact own-ledger accumulation and exact cross-round independent certificates, not a tolerance')))
        class FakeEvidence:
            def __init__(self):
                self.pins={}
                self.documents={str(reader.PREVIOUS_REVIEW):review,str(reader.PRIOR/'results/COMPLETION.json'):completion,str(reader.PRIOR/'results/RESULTS.json'):previous,str(reader.PRIOR/'results/INDEPENDENT_OUTCOME.json'):dict(passed=True,accounting_source_sha256='same',paths={reader.ARMS[0]:args[4]})}
                for p in reader.PATHS:
                    for kind in reader.KINDS:
                        name=reader.ARMS[0]+'_'+p+'_'+kind+'.json';completion['artifacts_sha256'][name]='same'
                        self.documents[str(reader.PRIOR/'results'/name)]=args[2][p][kind]
                        self.pins[str((reader.OUT/'results'/name).relative_to(reader.ROOT))]='same'
                name=reader.ARMS[0]+'_same_close_zero_DAILY.json'
                self.pins[str((reader.OUT/'results'/name).relative_to(reader.ROOT))]='different-bytes'
            def read(self,path,want=None):
                self.pins[str(path.relative_to(reader.ROOT))]=want
                return self.documents[str(path)]
        ev=FakeEvidence();exports={(reader.ARMS[0],p):args[3][p] for p in reader.PATHS}
        with self.assertRaisesRegex(ValueError,'unchanged export bytes'):
            reader.previous_incumbent(ev,results,exports,dict(passed=True,accounting_source_sha256='same',paths={reader.ARMS[0]:args[5]}))





def listing_fixture(day='2026-08-31',mature=False):
    old=dict(symbol='SPCX',issuer_key='https://companiesmarketcap.com/spacex/marketcap/',not_before='2026-06-08',available_from='2026-06-09',kind='exchange_new_listing_lower_bound',evidence_id='spcx_nasdaq_new_issue')
    new=dict(symbol='SPCX',issuer_key=old['issuer_key'],not_before='2026-06-12',available_from='2026-08-05',kind='issuer_actual_first_trade',evidence_id='spcx_actual_trading_confirmation')
    policy=dict(schema='monthly_peer_path252_v2_listing_policy_v1',source_only=True,publication_date='2026-08-04',strict_pit=False,original_history_bound=old,path_bound_override=new)
    bound=new if day>=new['available_from'] else old
    dates=['2026-06-08','2026-06-11','2026-06-12','2026-06-15']
    row=dict(date=day,symbol='SPCX',issuer_key=old['issuer_key'],initial_partial=False,history_bound=copy.deepcopy(old),path_history_bound=copy.deepcopy(bound),history_status='observed_full_window' if mature else 'official_pre_inception',path_observations=[dict(date=d,status='missing' if d<bound['not_before'] else 'valid',reason='missing_value' if d<bound['not_before'] else None) for d in dates])
    return row,dates,policy


def revision_fixture():
    policy=listing_fixture()[2];oldsha=reader.FIXED['outputs/monthly_peer_path252/PLAN.json']
    fixed=('arms','hypothesis','causal_scope','original_control','exact_math','scoring','budget','selection','execution','validation')
    ctx={k:{'invented':k} for k in fixed}
    raw={'outputs/invented/source'+str(i)+'.csv':format(i,'064x') for i in range(64)}
    oldplan=dict(context=copy.deepcopy(ctx),raw_source_hashes=raw)
    src=dict(schema='monthly_peer_path252_metadata_source_contract_v2',raw_source_hashes=raw,clock_events=[],mapping_records=[],identity_cells=[],listing_bounds=[])
    oldsrc=copy.deepcopy(src)
    failure=dict(plan_sha256=oldsha,phase='prepare',terminal=dict(exit_code=1,error='post_bound_missing:SPCX/2026-06-08:missing_value'),prepared_exists=False,new_features_computed=False,new_orders_computed=False,new_returns_computed=False)
    gaps=dict(plan_sha256=oldsha,source_only=True,features_computed=False,orders_computed=False,returns_computed=False,model_fits=0,events=122,counts=dict(stock_cells=3904,price_observations=816992),gap_occurrences=4,unique_gap_count=4,raw_source_hashes=raw,unique_gaps=[dict(symbol='SPCX',date='2026-06-'+str(d).zfill(2),query_indices=[121]) for d in (8,9,10,11)])
    evidence=dict(plan_sha256=oldsha,failure_sha256=reader.FIXED['work/monthly_peer_path252_PREPARE_FAILURE.json'],gap_audit_sha256=reader.FIXED['work/monthly_peer_path252_SOURCE_GAPS.json'],new_strategy_outcomes=0,source_validation_failures=1)
    revision=dict(version=2,source_only=True,prior_new_features=0,prior_new_orders=0,prior_new_returns=0,strategy_trial_count=3,prior_source_attempts=1,prior_plan_sha256=oldsha,prior_source_failure_sha256=evidence['failure_sha256'],gap_audit_sha256=evidence['gap_audit_sha256'],listing_policy=dict(path='work/monthly_peer_path252_v2_LISTING_POLICY.json',sha256=reader.FIXED['work/monthly_peer_path252_v2_LISTING_POLICY.json']))
    ctx.update(schema='monthly_peer_path252_context_v2',source_revision=revision);src.update(source_revision=copy.deepcopy(revision),listing_policy=revision['listing_policy'])
    plan=dict(schema='monthly_peer_path252_v2',source_revision=2,prior_source_failure=evidence,raw_source_hashes=raw)
    state=dict(completed=[dict(trials=5,directory='old') for _ in range(246)]+[dict(trials=24,directory='monthly_basket_core251')],current='monthly_peer_path252',registered_queue=[dict(directory='monthly_peer_path252',plan_sha256=oldsha)],best_training_selected_candidate=dict(directory='monthly_local_linear70'))
    return plan,ctx,src,state,oldplan,oldsrc,failure,gaps,policy


class ListingEvidenceTests(unittest.TestCase):
    def test_august4_original_and_august5_actual_bound(self):
        for day,applies in [('2026-07-31',False),('2026-08-04',False),('2026-08-05',True),('2026-08-31',True)]:
            row,dates,p=listing_fixture(day)
            self.assertEqual(reader.path_evidence(row,dates,p),(row['path_history_bound'],applies))
    def test_premature_override_is_rejected(self):
        row,dates,p=listing_fixture('2026-08-04');row['path_history_bound']=p['path_bound_override']
        with self.assertRaisesRegex(ValueError,'query-visible'):reader.path_evidence(row,dates,p)
    def test_wrong_issuer_cannot_receive_override(self):
        row,dates,p=listing_fixture();row['issuer_key']='old-fund'
        with self.assertRaisesRegex(ValueError,'query-visible'):reader.path_evidence(row,dates,p)
    def test_original_bound_is_not_rewritten(self):
        row,dates,p=listing_fixture();row['history_bound']['not_before']='2026-06-12'
        with self.assertRaisesRegex(ValueError,'retain exact original'):reader.path_evidence(row,dates,p)
    def test_mature_path_cannot_borrow_warmup_missing(self):
        with self.assertRaisesRegex(ValueError,'Mature stocks'):reader.path_evidence(*listing_fixture(mature=True))
    def test_positive_pretrade_evidence_rejected(self):
        row,dates,p=listing_fixture();row['path_observations'][0].update(status='valid',reason=None)
        with self.assertRaisesRegex(ValueError,'Pre-bound absence'):reader.path_evidence(row,dates,p)
    def test_missing_posttrade_evidence_rejected(self):
        row,dates,p=listing_fixture();row['path_observations'][-1].update(status='missing',reason='missing_value')
        with self.assertRaisesRegex(ValueError,'post-bound'):reader.path_evidence(row,dates,p)
    def test_path_date_or_status_tampering_rejected(self):
        for field,value in [('date','2026-06-09'),('reason','invalid'),('status','unknown')]:
            row,dates,p=listing_fixture();row['path_observations'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):reader.path_evidence(row,dates,p)
    def test_partial_is_no_numeric_evidence(self):
        row,dates,p=listing_fixture('2016-08-17');row.update(initial_partial=True,history_bound=None,path_history_bound=None,history_status='initial_partial',path_observations=[])
        self.assertEqual(reader.path_evidence(row,[],p),(None,False))
        row['path_observations']=[dict(date='2016-01-01',status='valid',reason=None)]
        with self.assertRaises(ValueError):reader.path_evidence(row,['2016-01-01'],p)


class RevisionEvidenceTests(unittest.TestCase):
    def test_valid_revision_retains_single_three_trial_count(self):
        result=reader.revision_summary(*revision_fixture());self.assertTrue(result['three_strategy_trials_counted_once']);self.assertEqual(result['trials_before'],1254)
    def test_prior_queue_cannot_be_empty_or_other_registration(self):
        for queue in ([],[dict(directory='other',plan_sha256='bad')]):
            args=revision_fixture();args[3]['registered_queue']=queue
            with self.assertRaisesRegex(ValueError,'failed pending'):reader.revision_summary(*args)
    def test_prior_outcomes_or_missing_failure_cannot_be_erased(self):
        for key in ('prepared_exists','new_features_computed','new_orders_computed','new_returns_computed'):
            args=revision_fixture();args[6][key]=True
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'stopped before'):reader.revision_summary(*args)
    def test_changed_strategy_or_source_rejected(self):
        for index,key in ((1,'budget'),(2,'clock_events'),(2,'mapping_records')):
            args=revision_fixture();args[index][key]={'different':1}
            with self.subTest(key=key),self.assertRaises(ValueError):reader.revision_summary(*args)
    def test_wrong_gap_or_publication_clock_rejected(self):
        args=revision_fixture();args[7]['unique_gaps'][0]['date']='2026-06-12'
        with self.assertRaisesRegex(ValueError,'Four specific'):reader.revision_summary(*args)
        args=revision_fixture();args[8]['path_bound_override']['available_from']='2026-06-13'
        with self.assertRaisesRegex(ValueError,'first-trade evidence'):reader.revision_summary(*args)
    def test_trial_or_failure_digest_tampering_rejected(self):
        for key,value in [('strategy_trial_count',6),('prior_new_returns',1),('prior_source_failure_sha256','wrong')]:
            args=revision_fixture();args[1]['source_revision'][key]=value;args[2]['source_revision'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):reader.revision_summary(*args)


def selection_fixture():
    train={}
    for a in reader.ARMS:
        train[a]={}
        for p in reader.PATHS[:2]:
            c=.21 if a==reader.ARMS[1] else .20
            train[a][p]={k:dict(cagr=c,sharpe=1.,max_drawdown=-.1,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None) for k in ('training','early','recent')}
    changed={reader.ARMS[1]+'_vs_'+ref:dict(months=[dict(index=i) for i in (0,1,60)]) for ref in (reader.ARMS[0],reader.ARMS[2])}
    return train,changed


class SelectionAndSchemaTests(unittest.TestCase):
    def test_candidate_only_admission_and_undefined_control_refuses(self):
        train,changed=selection_fixture();self.assertEqual(reader.saved_selection(train,changed)['primary'],reader.ARMS[1])
        train[reader.ARMS[2]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        answer=reader.saved_selection(train,changed);self.assertFalse(answer['admitted']);self.assertEqual(answer['primary'],reader.ARMS[0])
    def test_risk_guard_does_not_borrow_cagr_epsilon(self):
        train,changed=selection_fixture();train[reader.ARMS[1]]['same_close_zero']['training']['sharpe']=math.nextafter(.95,-math.inf)
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_only_later34_changes_cannot_qualify(self):
        train,changed=selection_fixture()
        for value in changed.values():value['months']=[dict(index=i) for i in (88,99,121)]
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_literal_legacy_schema_composition_without_import(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'schema.py';p.write_text("LEGACY_SOURCE_RECORD_KEYS=('a','b')\nSOURCE_RECORD_KEYS=LEGACY_SOURCE_RECORD_KEYS+('path_history_bound',)\n")
            self.assertEqual(reader.constants(p,('LEGACY_SOURCE_RECORD_KEYS','SOURCE_RECORD_KEYS'))['SOURCE_RECORD_KEYS'],('a','b','path_history_bound'))
    def test_phase_manifest_rejects_extra_file(self):
        with TemporaryDirectory() as tmp:
            oldroot=reader.ROOT;reader.ROOT=Path(tmp).resolve()
            try:
                folder=Path(tmp)/'phase';folder.mkdir();artifact=folder/'a.json';artifact.write_text('{}')
                c=dict(status='COMPLETE',plan_sha256='plan',artifacts_sha256={'a.json':reader.digest(artifact)})
                completion=folder/'COMPLETION.json';completion.write_text(json.dumps(c))
                e=reader.Evidence();e.phase(folder,reader.digest(completion),'plan',{'a.json'})
                (folder/'unregistered.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'Unexpected'):e.phase(folder,reader.digest(completion),'plan',{'a.json'})
            finally:reader.ROOT=oldroot
    def test_canonical_rational_rejects_nonreduced_or_float(self):
        for value in ({'numerator':'2','denominator':'2'},{'numerator':'01','denominator':'2'},{'numerator':1,'denominator':'2'}):
            with self.subTest(value=value),self.assertRaises(ValueError):reader.rational_metadata(value)




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


def full_source_fixture():
    import calendar as civil
    axis=[];d=date(2015,1,1)
    while d<=date(2026,9,3):
        if d.weekday()<5:axis.append(d.isoformat())
        d+=timedelta(days=1)
    dates=['2016-08-17'];year,month=2016,8
    for _ in range(121):
        d=date(year,month,civil.monthrange(year,month)[1])
        while d.weekday()>=5:d-=timedelta(days=1)
        dates.append(d.isoformat());month+=1
        if month==13:year+=1;month=1
    names=['A'+str(i).zfill(2) for i in range(31)]+['SPCX']+['ETF'+str(i) for i in range(17)]
    etfs=names[32:];meta=dict(dates=dates,symbols=[names.copy() for _ in dates],etfs=etfs)
    schemas=reader.constants(reader.ROOT/'work/monthly_peer_path252_independent_v2.py',('COUNT_FIELDS','SOURCE_TOP_KEYS','SOURCE_EVENT_KEYS','LEGACY_SOURCE_RECORD_KEYS','SOURCE_RECORD_KEYS','EVENT_KEYS','CELL_KEYS'))
    policy=listing_fixture()[2];source=dict(mapping_records=[],mapping_record_keys=['query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence'],clock_events=[],identity_cells=[],listing_bounds=[policy['original_history_bound']])
    daily={k:False for k in schemas['SOURCE_TOP_KEYS']};daily.update(schema='monthly_peer_path252_daily_source_report_v2',status='ADMITTED',admitted=True,events=[],records=[],gaps=[],model_fits=0)
    for k in ('source_only','original_inventory_preserved','original_failed_audit_preserved','daily_paths_checked'):daily[k]=True
    audit=dict(schema='monthly_peer_path252_features_v2',events=[])
    orders={a:[dict(symbols=['A00','A01','A02']) for _ in dates] for a in reader.ARMS}
    for q,day in enumerate(dates):
        t=axis.index(day);partial=q==0;pdates=[] if partial else axis[t-252:t-41];ends=[] if partial else [pdates[0],pdates[-1]];pos=[] if partial else [t-252,t-42]
        bound=policy['original_history_bound'] if day>=policy['original_history_bound']['available_from'] else None
        warm=bound is not None and not partial and pdates[0]<bound['not_before']
        mature=[] if partial else list(range(31 if warm else 32));supported=mature.copy()
        clock=dict(query_index=q,date=day,symbols=names,query_position=t,next_official_session=axis[t+1],numeric_evaluation_required=not partial,price_start=axis[t-252],price_end=axis[t-42],source_start_position=t-252,source_end_position=t-42,price_count=211,return_count=210,stock_indices=list(range(32)))
        source['clock_events'].append(clock)
        ev=dict(query_index=q,date=day,initial_partial=partial,stock_indices=list(range(32)),source_positions=pos,source_dates=ends,mature_indices=mature,supported_indices=supported,path_dates=pdates);daily['events'].append(ev)
        fa=dict(query_index=q,date=day,source_dates=ends,source_positions=pos,stock_indices=list(range(32)),mature_indices=mature,supported_indices=supported,cells=[]);audit['events'].append(fa)
        for j,symbol in enumerate(names[:32]):
            issuer=policy['original_history_bound']['issuer_key'] if symbol=='SPCX' else 'issuer/'+symbol
            state='initial_partial' if partial else ('official_pre_inception' if j==31 and warm else 'observed_full_window')
            r={k:None for k in schemas['SOURCE_RECORD_KEYS']}
            r.update(query_index=q,column_index=j,date=day,symbol=symbol,issuer_key=issuer,original_sector='TECH',sector='TECH',mapping_origin='synthetic',mapping_repair='unchanged',mapping_evidence=None,initial_partial=partial,source_positions=pos,source_dates=ends,gap_reasons=[],history_status=state,history_bound=bound if j==31 and not partial else None,peer_indices=[i for i in mature if i!=j] if j in mature else [],peer_issuer_count=len(mature)-1 if j in mature else 0,structural_peer_support=j in supported)
            r['peer_indices'].sort(key=lambda i:(policy['original_history_bound']['issuer_key'] if i==31 else 'issuer/'+names[i],names[i],i))
            r['path_history_bound']=policy['path_bound_override'] if j==31 and day>='2026-08-05' else r['history_bound']
            r['path_observations']=[dict(date=d,status='missing' if r['path_history_bound'] and d<r['path_history_bound']['not_before'] else 'valid',reason='missing_value' if r['path_history_bound'] and d<r['path_history_bound']['not_before'] else None) for d in pdates]
            source['mapping_records'].append({k:r[k] for k in source['mapping_record_keys']});daily['records'].append(r)
            cell={k:r[k] for k in schemas['CELL_KEYS'][:7]}
            for k in ('focal_gross','daily_peer_gross','endpoint_peer_gross','path_raw','endpoint_raw'):cell[k]=dict(numerator='1',denominator='1') if j in mature else None
            cell.update(path_target=0. if j in supported else None,endpoint_target=0. if j in supported else None);fa['cells'].append(cell)
            if j==31 and bound:
                source['identity_cells'].append(dict(query_index=q,column_index=j,date=day,symbol=symbol,issuer_key=issuer,not_before=bound['not_before'],available_from=bound['available_from'],bound_visible=True,price_start=clock['price_start'],price_end=clock['price_end'],price_window_dates_before_bound=sum(d<bound['not_before'] for d in axis[t-252:t-41])))
    counts=reader.counted(daily['records'],122);daily['counts']=counts;audit['counts']=counts
    return daily,audit,meta,dict(sessions=axis),source,orders,schemas,policy


class FullSourceMetadataTest(unittest.TestCase):
    def test_full122_fake211_statuses_new_bound_and_counts(self):
        args=full_source_fixture();result=reader.source_summary(*args)
        self.assertEqual(result['counts']['price_observations'],816992)
        self.assertEqual(len(result['path_bound_overrides']),1)
        self.assertEqual(result['path_bound_overrides'][0]['query_index'],121)
        self.assertIn('2026-06-08',result['path_bound_overrides'][0]['missing_dates'])
        self.assertFalse(result['exact_reference_statistics_recomputed'])




class SourcePathCanonicalizationTests(unittest.TestCase):
    def fixture(self):return {'outputs/invented/source'+str(i)+'.csv':format(i,'064x') for i in range(64)}
    def test_absolute_and_relative64_exact_pairs_are_equivalent(self):
        rel=self.fixture();absolute={str(reader.ROOT/p):h for p,h in rel.items()}
        self.assertEqual(reader.canonical_source_map(absolute),reader.canonical_source_map(rel))
    def test_revision_chain_accepts_only_path_spelling_difference(self):
        args=revision_fixture();rel=copy.deepcopy(args[2]['raw_source_hashes']);absolute={str(reader.ROOT/p):h for p,h in rel.items()}
        for obj in (args[0],args[4],args[7]):obj['raw_source_hashes']=absolute
        self.assertTrue(reader.revision_summary(*args)['source_failure_not_erased'])
    def test_one_changed_digest_still_fails_revision(self):
        args=revision_fixture();args[0]['raw_source_hashes']=dict(args[0]['raw_source_hashes']);args[0]['raw_source_hashes']['outputs/invented/source0.csv']='f'*64
        with self.assertRaisesRegex(ValueError,'No source substitution'):reader.revision_summary(*args)
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


if __name__=='__main__':unittest.main(verbosity=2)
