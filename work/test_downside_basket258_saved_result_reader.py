"""Offline258 reader rejection tests; invented metadata only, no actual artifacts."""
import copy
from datetime import date,timedelta
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import downside_basket258_saved_result_reader as reader

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
    def test_downside_only_and_both_references(self):
        train,changed=selection_fixture();r=reader.saved_selection(train,changed)
        self.assertTrue(r['admitted']);self.assertEqual(r['primary'],reader.ARMS[1])
        train[reader.ARMS[2]]['same_close_zero']['training']['cagr']=.9
        r=reader.saved_selection(train,changed);self.assertFalse(r['admitted']);self.assertEqual(r['primary'],reader.ARMS[0])
    def test_one_control_change_but_three_inc_changes(self):
        train,changed=selection_fixture();changed[reader.ARMS[1]+'_vs_'+reader.ARMS[2]]['months']=[dict(index=4)]
        self.assertTrue(reader.saved_selection(train,changed)['admitted'])
        changed[reader.ARMS[1]+'_vs_'+reader.ARMS[0]]['months']=[dict(index=4),dict(index=60)]
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_undefined_control_vetoes(self):
        train,changed=selection_fixture();train[reader.ARMS[2]]['next_open_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_undefined_incumbent_hardfail(self):
        train,changed=selection_fixture();train[reader.ARMS[0]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        with self.assertRaises(ValueError):reader.saved_selection(train,changed)
    def test_cagr_slack_not_risk_slack(self):
        train,changed=selection_fixture();train[reader.ARMS[1]]['same_close_zero']['training']['cagr']=.205-5e-13
        self.assertTrue(reader.saved_selection(train,changed)['admitted'])
        train[reader.ARMS[1]]['same_close_zero']['training']['sharpe']=.95-5e-13
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_later_changes_cannot_support_or_rescue(self):
        train,changed=selection_fixture()
        for value in changed.values():value['months']=[dict(index=i) for i in (88,100,121)]
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_extra_cost_training_path_rejected(self):
        train,changed=selection_fixture();train[reader.ARMS[1]]['next_open_cost']={}
        with self.assertRaises(ValueError):reader.saved_selection(train,changed)


def stats_fixture():
    return dict(schema='monthly_downside_basket258_moments_v1',price_count=253,return_count=252,normalized_prices=[[1.]*253 for _ in range(3)],wealth=[1.]*253,prior_totals=[3.]*252,contributions=[[0.]*252 for _ in range(3)],portfolio_returns=[0.]*252,direct_returns=[0.]*252,downside_square_sum=0.,symmetric_square_sum=0.,upside_square_sum=0.,downside_moment=0.,symmetric_moment=0.,upside_moment=0.,negative_returns=0,zero_returns=252,positive_returns=0,zero_downside=True,zero_symmetric=True)


def features_fixture():
    schemas,_=reader.schema_flags();dates=[(date(2014,1,1)+timedelta(days=i)).isoformat() for i in range(1000)]
    names=['S%02d'%i for i in range(49)];meta=dict(dates=dates[600:722],symbols=[names.copy() for _ in range(122)])
    features=[];events=[];orders={a:[] for a in reader.ARMS};windows=[]
    for i,day in enumerate(meta['dates']):
        active=i==0;ids=[0,1,2] if active else [0,1];t=600+i;source_dates=dates[t-294:t-41]
        cells=[dict(column_index=j,symbol=names[j],source_available=True,reason='available',observations_checked=253,start_price=1.,end_price=1.,listing=None) for j in ids] if active else []
        trios=[dict(indices=ids.copy(),symbols=[names[j] for j in ids],raw_core_mean=0.,weighted_core_mean=0.,final_mean=0.,auxiliary_mean=0.,raw_core_regret=0.,final_regret=0.,feasible=True,statistics=stats_fixture())] if active else []
        f=dict(query_index=i,date=day,symbols=names.copy(),initial_partial=False,calm=True,weak=False,core_weight=.5,eligible_indices=ids.copy(),pre_scores=[0.]*49,final_scores=[0.]*49,weighted_core=[0.]*49,incumbent_choices=ids.copy(),pre_order=ids.copy(),shortlist_indices=ids.copy() if active else [],incumbent_trio=ids.copy() if active else None,reference_raw_core_mean=0. if active else None,reference_final_mean=0. if active else None,feasible_trio_indices=[0] if active else [],session_position=t,price_start=source_dates[0],price_end=source_dates[-1],source_dates=source_dates,clock_sessions=dates[t-294:t+2],source_cells=cells,trios=trios,branch='available' if active else 'fewer_than_three')
        features.append(f);events.append(dict(copy.deepcopy(f),choices={a:ids.copy() for a in reader.ARMS},winner_trio_indices={a:0 if active else None for a in reader.ARMS[1:]},tied_trio_indices={a:[0] if active else [] for a in reader.ARMS[1:]}))
        for a in reader.ARMS:orders[a].append(dict(symbols=[names[j] for j in ids],scores=[0.]*len(ids)))
        windows.append(dict(price_start=source_dates[0],price_end=source_dates[-1]))
    c=dict.fromkeys(schemas['COUNT_FIELDS'],0);c.update(events=122,symbols_per_event=49,eligible_cells=245,active_events=1,shortlist_cells=3,source_available_cells=3,source_price_observations=759,enumerated_trios=1,feasible_trios=1,trio_statistics=1,formation_return_observations=252,zero_downside_trios=1,zero_symmetric_trios=1,available_events=1,original_episode_calls=1)
    audit=dict(schema='monthly_downside_basket258_decisions_v1',counts=c,events=events,incumbent_gate={},source_gate={})
    return features,audit,meta,orders,dict(sessions=dates),dict(calendar=dict(windows=windows),listing_bounds=[]),schemas


class SavedMomentEvidenceTests(unittest.TestCase):
    def test_schema_literals_without_import(self):
        schemas,flags=reader.schema_flags();self.assertEqual(len(schemas['COUNT_FIELDS']),22);self.assertEqual(len(schemas['STAT_KEYS']),20)
        self.assertIn('canonical_moments_exact',flags['PREPARE_FLAGS'])
    def test_zero_moments_are_available_not_fallback(self):
        counts,summary=reader.feature_summary(*features_fixture());self.assertEqual(counts['zero_symmetric_trios'],1);self.assertEqual(summary['branch_counts']['available'],1)
    def test_counter_tamper(self):
        args=features_fixture();args[1]['counts']['zero_symmetric_trios']=0
        with self.assertRaises(ValueError):reader.feature_summary(*args)
    def test_direct_minimum_or_tie_indices_tamper(self):
        args=features_fixture();args[1]['events'][0]['winner_trio_indices'][reader.ARMS[1]]=None
        with self.assertRaises(ValueError):reader.feature_summary(*args)
    def test_complete_source_prelisting_contradiction(self):
        args=features_fixture();bound=dict(symbol='S00',issuer_key='issuer',not_before='2020-01-01',available_from='2014-01-01',evidence_id='proof')
        args[5]['listing_bounds']=[bound]
        listing={k:v for k,v in bound.items() if k!='symbol'};listing.update(visible=True,start_precedes_bound=True)
        args[0][0]['source_cells'][0]['listing']=listing;args[1]['events'][0]['source_cells'][0]['listing']=copy.deepcopy(listing)
        with self.assertRaisesRegex(ValueError,'prelisting'):reader.feature_summary(*args)
    def test_source_dates_cannot_shift(self):
        args=features_fixture();args[0][0]['price_end']='2099-01-01';args[1]['events'][0]['price_end']='2099-01-01'
        with self.assertRaises(ValueError):reader.feature_summary(*args)
    def test_zero_and_sign_metadata_cannot_hide_loss(self):
        keys=tuple(stats_fixture())
        s=stats_fixture();s['portfolio_returns'][0]=-.01
        with self.assertRaises(ValueError):reader.stat_summary(s,keys)
        s=stats_fixture();s['symmetric_moment']=-1e-30
        with self.assertRaises(ValueError):reader.stat_summary(s,keys)
    def test_missing_stat_array_cell_rejected(self):
        s=stats_fixture();s['contributions'][0].pop()
        with self.assertRaises(ValueError):reader.stat_summary(s,tuple(stats_fixture()))
    def test_no_input_mutation(self):
        args=features_fixture();before=copy.deepcopy(args);reader.feature_summary(*args);self.assertEqual(args,before)


def proof(kind,n,fields,flags):
    c=dict.fromkeys(fields,0);c.update(events=n,symbols_per_event=49,original_episode_calls=1,trio_statistics=n,formation_return_observations=252*n)
    fname={'source':'FEATURE_FLAGS','cached_source':'FEATURE_FLAGS','incumbent':'INCUMBENT_FLAGS','selection':'SELECTION_FLAGS','full':'PREPARE_FLAGS'}[kind]
    r=dict.fromkeys(flags[fname],True);r.update(checked_events=n,producer_modules_imported=False,computes_returns=False,original_episode_calls=2 if kind=='full' else 1,descriptive_path_evaluations=n if kind in ('source','full') else 0,cached_path_validations=n if kind in ('cached_source','selection','full') else 0)
    if kind in ('source','cached_source','full'):r.update(continuous_atol=2e-10,continuous_rtol=2e-10,nav_atol=2e-12,nav_rtol=2e-12,rank_tolerance=0.,exact_discrete=True)
    else:r['numerical_tolerance']=0.
    if kind in ('source','cached_source'):r.update(full_features_supplied=True,candidate_selection_computed=False,source_rebuilt=kind=='source',feature_counts=c)
    elif kind=='incumbent':r.update(full_scores_supplied=True,full_pre_scores_supplied=True,candidate_selection_computed=False,ordered_choices_checked=n)
    else:r.update(full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,counts=c,ordered_choices_checked=3*n,descriptive_path_returns_checked=True)
    return r


def evidence_fixture():
    schemas,flags=reader.schema_flags();fields=schemas['COUNT_FIELDS']
    def item(n,cached=False):
        p=proof('selection' if cached else 'full',n,fields,flags)
        return dict(prefix=n,gates=dict(incumbent=proof('incumbent',n,fields,flags),source=proof('cached_source' if cached else 'source',n,fields,flags)),**{'selection' if cached else 'full':p},counts=p['counts'],exact=True,**({} if cached else dict(reused_full=False,source_recomputed=True)))
    main=item(122);features=[{} for _ in range(122)]
    e=dict(main=main,cached_prefixes=[item(n,True) for n in range(123)],physical_prefixes=[item(n) for n in reader.PREFIXES[:-1]]+[dict(copy.deepcopy(main),reused_full=True,source_recomputed=False)],new_model_fits=0,predictive_model_fits=0,all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,full_features_sha256=reader.object_digest(features),full_source_certificate_sha256=reader.object_digest(main['gates']['source']))
    e['execution_accounting']=reader.execution_counts(e)
    pc=dict(new_model_fits=0,new_predictive_model_fits=0,new_returns_computed=False,events=122,descriptive_counts=main['counts'])
    a=dict(counts=main['counts'],incumbent_gate=main['gates']['incumbent'],source_gate=main['gates']['source'])
    return e,pc,a,features,flags,fields


class ReceiptAndTerminalsTests(unittest.TestCase):
    def test_complete_receipts_and_actual_count_totals(self):
        args=evidence_fixture();r=reader.preparation(*args)
        self.assertEqual(r['execution_accounting']['physical_preparations'],5);self.assertEqual(r['execution_accounting']['total_path_moment_calculations'],23565)
    def test_122_must_be_reused(self):
        args=evidence_fixture();args[0]['physical_prefixes'][-1]['source_recomputed']=True
        with self.assertRaises(ValueError):reader.preparation(*args)
    def test_missing_cached_proof(self):
        args=evidence_fixture();args[0]['cached_prefixes'].pop(53)
        with self.assertRaises(ValueError):reader.preparation(*args)
    def test_execution_count_cannot_ignore_cached_recomputations(self):
        args=evidence_fixture();args[0]['execution_accounting']['producer_cached_path_validations']=0
        with self.assertRaises(ValueError):reader.preparation(*args)
    def test_source_proof_cannot_select(self):
        args=evidence_fixture();args[0]['cached_prefixes'][1]['gates']['source']['candidate_selection_computed']=True
        with self.assertRaises(ValueError):reader.preparation(*args)
    def test_false_is_not_zero_call_count(self):
        schemas,flags=reader.schema_flags();r=proof('incumbent',0,schemas['COUNT_FIELDS'],flags);r['descriptive_path_evaluations']=False
        with self.assertRaises(ValueError):reader.receipt(r,0,'incumbent',flags,schemas['COUNT_FIELDS'])
    def test_terminal_all_three_and_no_failed_exit(self):
        hashes={p:p for p in ('prepare','select','diagnose')}
        t=dict(plan_sha256='plan',phases={p:dict(session_id=i+1,exit_code=0,chunk_id='ok',artifact_path=str(reader.OUT/f),artifact_sha256=p) for i,(p,f) in enumerate((('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')))})
        reader.terminals_check(t,'plan',hashes)
        for mode in ('missing','exit','hash','artifact'):
            x=copy.deepcopy(t)
            if mode=='missing':x['phases'].pop('diagnose')
            elif mode=='exit':x['phases']['diagnose']['exit_code']=1
            elif mode=='hash':x['phases']['diagnose']['artifact_sha256']='bad'
            else:x['phases']['diagnose']['artifact_path']=str(reader.OUT/'results/RESULTS.json')
            with self.assertRaises(ValueError):reader.terminals_check(x,'plan',hashes)

class FormalBindingTests(unittest.TestCase):
    def test_terminal_formal99_and_fixed_review_bindings(self):
        self.assertEqual(reader.EXPECTED_FORMAL_TESTS,99)
        self.assertEqual(reader.FIXED['work/monthly_downside_basket258_TESTS.json'],'51f178212acd92d8ba1d7c8db2fa1c77dfd2abc9e6fd87a540138495a733c22a')
        self.assertEqual(reader.FIXED['work/monthly_downside_basket258_CODE_REVIEW.json'],'8161189e60cc9399a4f44193599ab912ebe60bae4afb15f9dcda9283bfd70b04')

if __name__=='__main__':unittest.main(verbosity=2)
