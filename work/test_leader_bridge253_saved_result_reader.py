"""Offline253 reader rejection tests; invented metadata only, no actual artifacts."""
import copy
from datetime import date,timedelta
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import leader_bridge253_saved_result_reader as reader

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


class SelectionAndSchemaTests(unittest.TestCase):
    def test_only_bridge_can_win_even_when_control_better(self):
        train,changed=selection_fixture()
        for p in reader.PATHS[:2]:train[reader.ARMS[2]][p]['training']['cagr']=9.
        answer=reader.saved_selection(train,changed)
        self.assertEqual(answer['primary'],reader.ARMS[1]);self.assertEqual(answer['diagnostic_control_not_selectable'],reader.ARMS[2])
    def test_bad_bridge_never_promotes_strong_control(self):
        train,changed=selection_fixture()
        for p in reader.PATHS[:2]:
            train[reader.ARMS[1]][p]['training']['cagr']=.19;train[reader.ARMS[2]][p]['training']['cagr']=9.
        self.assertEqual(reader.saved_selection(train,changed)['primary'],reader.ARMS[0])
    def test_undefined_diagnostic_does_not_veto_bridge(self):
        train,changed=selection_fixture()
        train[reader.ARMS[2]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        self.assertTrue(reader.saved_selection(train,changed)['admitted'])
    def test_undefined_candidate_rejected_incumbent_undefined_hardfail(self):
        for arm in reader.ARMS[:2]:
            train,changed=selection_fixture();train[arm]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
            if arm==reader.ARMS[0]:
                with self.assertRaises(ValueError):reader.saved_selection(train,changed)
            else:self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_risk_guard_has_no_cagr_epsilon(self):
        train,changed=selection_fixture();train[reader.ARMS[1]]['same_close_zero']['training']['sharpe']=math.nextafter(.95,-math.inf)
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_first88_changes_support_both_suberas(self):
        for rows in ((1,2,3),(54,60,80),(88,99,121),(1,60)):
            train,changed=selection_fixture();changed[reader.ARMS[1]+'_vs_'+reader.ARMS[0]]['months']=[dict(index=i) for i in rows]
            self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_cagr_roundoff_only_existing_boundary(self):
        train,changed=selection_fixture();m=train[reader.ARMS[1]]['same_close_zero']['training']
        m['cagr']=.2+.005-1e-12
        self.assertTrue(reader.saved_selection(train,changed)['admitted'])
        m['cagr']=math.nextafter(m['cagr'],-math.inf)
        self.assertFalse(reader.saved_selection(train,changed)['admitted'])
    def test_malformed_diagnostic_still_hardfails(self):
        train,changed=selection_fixture();train[reader.ARMS[2]]['same_close_zero']['training']['sharpe']=float('nan')
        with self.assertRaises(ValueError):reader.saved_selection(train,changed)
    def test_literal_schema_composition_no_producer_import(self):
        schemas,ref,flags=reader.schema_flags()
        self.assertEqual((len(schemas['COUNT_FIELDS']),len(schemas['AUDIT_EVENT_KEYS'])),(22,14))
        self.assertEqual(len(ref['COUNT_FIELDS']),31);self.assertEqual(len(flags['ENDPOINT_FLAGS']),6)
        self.assertEqual(len(flags['SELECTION_FLAGS']),23)
    def test_phase_manifest_rejects_extra_and_changed_file(self):
        with TemporaryDirectory() as tmp:
            oldroot=reader.ROOT;reader.ROOT=Path(tmp).resolve()
            try:
                folder=Path(tmp)/'phase';folder.mkdir();artifact=folder/'a.json';artifact.write_text('{}')
                c=dict(status='COMPLETE',plan_sha256='plan',artifacts_sha256={'a.json':reader.digest(artifact)})
                completion=folder/'COMPLETION.json';completion.write_text(json.dumps(c));e=reader.Evidence()
                e.phase(folder,reader.digest(completion),'plan',{'a.json'});artifact.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'Digest'):e.phase(folder,reader.digest(completion),'plan',{'a.json'})
                (folder/'unregistered.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'Unexpected'):e.phase(folder,reader.digest(completion),'plan',{'a.json'})
            finally:reader.ROOT=oldroot
    def test_canonical_rational_rejects_nonreduced_or_float(self):
        for v in ({'numerator':'2','denominator':'2'},{'numerator':'01','denominator':'2'},{'numerator':1,'denominator':'2'}):
            with self.assertRaises(ValueError):reader.rational_metadata(v)


def gate_fixture(n=0,source=True):
    flags=reader.schema_flags()[2]
    sg=dict.fromkeys(flags['SOURCE_GATE_FLAGS'],True)
    sg.update(checked_events=n,episode_calls=0,new_target_ranks_computed=False,new_basket_selection_computed=False,C_source_atol=1e-12,C_source_rtol=0.,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
    ig=dict.fromkeys(flags['INCUMBENT_GATE_FLAGS'],True)
    ig.update(checked_events=n,ordered_choices_checked=n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
    ep=dict.fromkeys(flags['ENDPOINT_FLAGS'],True)
    ep.update(checked_events=n,ordered_choices_checked=2*n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
    out=dict(incumbent=ig,endpoints=ep)
    if source:out=dict(source=sg,**out)
    return out,flags


class EarlyGateTests(unittest.TestCase):
    def test_n0_and_full_all_three_gates(self):
        for n in (0,122):
            for source in (False,True):
                g,f=gate_fixture(n,source);reader.pre_gates(g,n,f,source)
    def test_missing_endpoint_or_wrong_order_count(self):
        for mutation in ('missing','count','flag','selection'):
            g,f=gate_fixture(122)
            if mutation=='missing':g.pop('endpoints')
            if mutation=='count':g['endpoints']['ordered_choices_checked']=122
            if mutation=='flag':g['endpoints']['exact_basket251_endpoint_checked']=False
            if mutation=='selection':g['endpoints']['candidate_selection_computed']=True
            with self.assertRaises(ValueError):reader.pre_gates(g,122,f,True)
    def test_source_may_not_compute_basket_or_episode(self):
        for key,value in (('episode_calls',1),('new_basket_selection_computed',True),('new_target_ranks_computed',True)):
            g,f=gate_fixture();g['source'][key]=value
            with self.assertRaises(ValueError):reader.pre_gates(g,0,f,True)
    def test_cached_gate_still_needs_two_endpoints(self):
        g,f=gate_fixture(0,False);g['source']={}
        with self.assertRaises(ValueError):reader.pre_gates(g,0,f,False)


def bridge_fixture():
    schemas=reader.schema_flags()[0];names=['S%02d'%j for j in range(49)];dates=['2016-08-17','2016-08-31'];features=[];events=[];orders={a:[] for a in reader.ARMS}
    for i,day in enumerate(dates):
        partial=i==0
        cells=[dict(measurement=dict(momentum=dict(numerator='1',denominator='2'))) for j in range(49)]
        source=dict(date=day,original_eligible_indices=list(range(8)),new_eligible_indices=list(range(8)),shortlist_indices=list(range(6)),cells=cells)
        e=dict(query_index=i,date=day,branch='initial_partial' if partial else 'restricted_episode',original_eligible_indices=list(range(8)),new_eligible_indices=None if partial else list(range(8)),shortlist_indices=None if partial else list(range(6)),selector_eligible_indices=list(range(8 if partial else 6)),pre_order=list(range(8 if partial else 6)),boundary_candidates=[2,3,4],episode_budget=.05,original_remaining_support=[.43]*49,incumbent_choices=[0,1,2],candidate_choices=[0,1,2] if partial else [0,1,3],basket_choices=[0,1,2] if partial else [1,3,5])
        features.append(source);events.append(e)
        for a,k in zip(reader.ARMS,('incumbent_choices','candidate_choices','basket_choices')):orders[a].append(dict(date=day,symbols=[names[j] for j in e[k]]))
    counts=dict.fromkeys(schemas['COUNT_FIELDS'],0)
    counts.update(events=2,symbols_per_event=49,initial_partial_events=1,formal_events=1,original_eligible_cells=16,positive_momentum_cells=8,shortlist_cells=6,restricted_eligible_cells=14,removed_eligible_cells=2,threeplus_name_events=1,candidate_changed_holdings_events=1,basket_changed_holdings_events=1,candidate_vs_basket_changed_holdings_events=1,original_episode_calls=1,candidate_episode_calls=1,diagnostic_episode_calls=0,reused_feature_events=2)
    return features,dict(schema='monthly_leader_bridge253_decisions_v1',counts=counts,events=events),dict(dates=dates,symbols=[names,names]),orders,schemas


class BridgeEvidenceTests(unittest.TestCase):
    def test_saved_domain_counts_and_choices(self):
        args=bridge_fixture();counts,summary=reader.bridge_summary(*args)
        self.assertEqual(counts,args[1]['counts']);self.assertEqual(summary['branch_counts'],{'initial_partial':1,'restricted_episode':1})
    def test_rank_six_member_rejected_even_in_top6(self):
        args=bridge_fixture();args[1]['events'][1]['candidate_choices']=[0,1,5];args[3][reader.ARMS[1]][1]['symbols']=['S00','S01','S05']
        with self.assertRaises(ValueError):reader.bridge_summary(*args)
    def test_top_two_locked_and_no_candidate_domain_expansion(self):
        for ids in ([0,3,4],[0,1,7]):
            args=bridge_fixture();args[1]['events'][1]['candidate_choices']=ids;args[3][reader.ARMS[1]][1]['symbols']=['S%02d'%j for j in ids]
            with self.assertRaises(ValueError):reader.bridge_summary(*args)
    def test_changed_counter_or_partial_is_rejected(self):
        for mutation in ('counter','partial'):
            args=bridge_fixture()
            if mutation=='counter':args[1]['counts']['candidate_episode_calls']=2
            else:args[1]['events'][0]['selector_eligible_indices']=list(range(6))
            with self.assertRaises(ValueError):reader.bridge_summary(*args)
    def test_exact_positive_momentum_predicate(self):
        args=bridge_fixture();args[0][1]['cells'][0]['measurement']['momentum']={'numerator':'0','denominator':'1'}
        with self.assertRaisesRegex(ValueError,'positive momentum'):reader.bridge_summary(*args)
    def test_reference_endpoint_diagnostic_choice_must_match_orders(self):
        args=bridge_fixture();args[3][reader.ARMS[2]][1]['symbols'][0]='S02'
        with self.assertRaisesRegex(ValueError,'membership'):reader.bridge_summary(*args)
    def test_does_not_mutate_input_metadata(self):
        args=bridge_fixture();before=copy.deepcopy(args);reader.bridge_summary(*args);self.assertEqual(args,before)


if __name__=='__main__':unittest.main(verbosity=2)
