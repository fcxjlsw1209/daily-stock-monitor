"""Generated-only tests for the independent coverage150 preparation audit."""
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
from fractions import Fraction
from pathlib import Path
import argparse
import ast
import json
import unittest
from unittest.mock import patch
import numpy as np
import monthly_financial_coverage150_prepare_review as r
import test_monthly_cash_quality146_prepare_review as fixtures


def event(key='amzn', symbol='AMZN'):
    source = dict(path='synthetic/report.html',sha256='a'*64)
    period = dict(period_start='2025-01-01',period_end='2025-12-31',days_inclusive=365)
    e = dict(event_id=key,symbol=symbol,fiscal_year=2025,release_date='2026-01-30',available_from='2026-02-02',
             periods=dict(current=period),source=source,fact_admitted=False,model_admitted=False,strategy_admitted=False,
             ni_assets_admission=dict(admitted=True,reasons=[],definition=r.DEFINITION),
             identity=dict(issuer_id='0000001234',security_id=symbol,identity_valid_from='2026-02-02',
                           identity_valid_through=str(date(2025,12,31)+timedelta(days=550)),
                           scope='source_local_report_binding_and_use_window_not_continuous_security_master',certified_daily_security_master=False),
             clock_review=dict(first_pinned_session='2026-02-02',verified=True))
    e['fields'] = {k:dict(value_decimal=v,scale='1',currency_literal='$',currency_iso=None,source=deepcopy(source),
                          available_from=e['available_from'],as_of='2024-12-31' if k==r.FIELDS[2] else '2025-12-31',
                          fact_admitted=False,displayed_amount=v,locator=dict(table=1,row=2))
                   for k,v in zip(r.FIELDS,('10','150','50'))}
    e['fields'][r.FIELDS[0]]['period']=deepcopy(period)
    return e


def context(e):
    c = dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],release_date=e['release_date'],
             available_session=e['available_from'])
    if e['symbol']=='AMZN':
        c.update(e['source'],current_end='2025-12-31',comparison_end='2024-12-31')
    else:
        c.update(source=deepcopy(e['source']),periods=dict(current=deepcopy(e['fields'][r.FIELDS[0]]['period']),
                                                        comparison=dict(period_end='2024-12-31')))
    return c


def catalog(e):
    return dict(event_id=e['event_id'],symbol=e['symbol'],publication_date=e['release_date'],available_session=e['available_from'],
                fiscal_quarter=4,clock_review=deepcopy(e['clock_review']))


def fixture():
    pool=['AAPL','AMD','MSFT','NVDA','AMZN','CSCO']+['S'+str(i) for i in range(26)]+['ETF'+str(i) for i in range(17)]
    meta=dict(dates=['2026-03-02','2026-04-01'],symbols=[pool[:],pool[:]],etfs=pool[32:],ends=['2026-03-31','2026-04-30'])
    original=np.full((2,49),np.nan)
    original[:,0]=.123
    e=event()
    return meta,original,dict(schema_version=1,method=r.METHOD,events=[e],latest_event_catalog=[catalog(e)])


def scheduled(e, release, available):
    e=deepcopy(e)
    e['release_date'],e['available_from']=release,available
    e['identity']['identity_valid_from']=available
    e['clock_review']=dict(first_pinned_session=available,verified=True)
    for cell in e['fields'].values():
        if cell is not None:cell['available_from']=available
    return e


class ArithmeticTests(unittest.TestCase):
    def test_integer_closed_form(self):
        self.assertEqual(r.bounded_earnings([(10,0),(150,0),(50,0)]),1/11)
    def test_negative_net_income(self):
        e=event();e['fields'][r.FIELDS[0]]['value_decimal']='-10'
        self.assertEqual(r.report_oracle(e,'2026-03-02')['earnings_bounded'],-1/11)
    def test_zero_net_income_is_valid(self):
        e=event();e['fields'][r.FIELDS[0]]['value_decimal']='0'
        v=r.report_oracle(e,'2026-03-02');self.assertTrue(v['available']);self.assertEqual(v['earnings_bounded'],0.)
    def test_mixed_scales(self):
        e=event();e['fields'][r.FIELDS[0]].update(value_decimal='.01',scale='1e3')
        self.assertEqual(r.report_oracle(e,'2026-03-02')['earnings_bounded'],1/11)
    def test_large_integer_low_decimal_context(self):
        n=10**150+1
        with localcontext() as c:
            c.prec=2
            self.assertEqual(r.bounded_earnings([(n,0),(n+1,0),(n+2,0)]),float(Fraction(2*n,4*n+3)))
    def test_bounded_rounding_endpoint(self):
        self.assertEqual(r.bounded_earnings([(10**400,0),(1,0),(1,0)]),1.)
    def test_nonpositive_assets(self):
        for name in r.FIELDS[1:]:
            for v in ('0','-1'):
                e=event();e['fields'][name]['value_decimal']=v
                self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'invalid_required_source_field')
    def test_nonfinite_and_invalid_decimal(self):
        for v in ('NaN','Infinity','1,234',' 10',''):
            e=event();e['fields'][r.FIELDS[0]]['value_decimal']=v
            self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
    def test_bad_scale(self):
        for v in ('0','-1','NaN'):
            e=event();e['fields'][r.FIELDS[0]]['scale']=v
            self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])


class ReportTests(unittest.TestCase):
    def test_gate_required(self):
        e=event();e.pop('ni_assets_admission')
        self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'missing_or_wrong_admission')
    def test_hold_blocks(self):
        e=event();e['ni_assets_admission'].update(admitted=False,reasons=['hold'])
        self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'latest_event_held')
    def test_missing_field(self):
        e=event();e['fields'][r.FIELDS[1]]=None
        self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
    def test_annual_duration(self):
        e=event();e['fields'][r.FIELDS[0]]['period']['period_start']='2025-10-01'
        self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'invalid_annual_period')
    def test_duration_mismatch(self):
        e=event();e['fields'][r.FIELDS[0]]['period']['days_inclusive']=364
        self.assertEqual(r.report_oracle(e,'2026-03-02')['reason'],'invalid_annual_period')
    def test_prior_assets_boundary(self):
        e=event();e['fields'][r.FIELDS[2]]['as_of']='2024-12-30'
        self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
    def test_other_source_rejected(self):
        e=event();e['fields'][r.FIELDS[1]]['source']['sha256']='b'*64
        self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
    def test_other_cell_clock_rejected(self):
        e=event();e['fields'][r.FIELDS[1]]['available_from']='2026-02-03'
        self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
    def test_currency_consistency(self):
        e=event();e['fields'][r.FIELDS[1]]['currency_iso']='CAD'
        self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
    def test_identity_end(self):
        e=event();e['identity']['identity_valid_through']='2030-01-01'
        self.assertFalse(r.report_oracle(e,'2026-03-02')['available'])
    def test_expiry_inclusive(self):
        e=event();d=date(2025,12,31)+timedelta(days=550)
        self.assertTrue(r.report_oracle(e,str(d))['available'])
        self.assertEqual(r.report_oracle(e,str(d+timedelta(days=1)))['reason'],'stale_latest_event')
    def test_not_yet_available(self):
        self.assertEqual(r.report_oracle(event(),'2026-01-30')['reason'],'invalid_available_time')
    def test_cfo_sbc_notes_do_not_gate_measure(self):
        e=event();e['cfo_notes']=['restricted_cash'];e['stock_compensation']=None
        self.assertTrue(r.report_oracle(e,'2026-03-02')['available'])


class CatalogTests(unittest.TestCase):
    def run_latest(self,extra=None,extra_row=None,signal='2026-03-02'):
        e=event();events={e['event_id']:e};rows=[catalog(e)]
        if extra:events[extra['event_id']]=extra
        if extra_row:rows.append(extra_row)
        return r.latest_oracle(rows,events,'AMZN',signal)
    def test_no_known(self):
        self.assertEqual(self.run_latest(signal='2026-01-01')['reason'],'no_known_annual_event')
    def test_unprocessed_latest_blocks_fallback(self):
        e=scheduled(event('missing'),'2026-02-20','2026-02-23')
        self.assertEqual(self.run_latest(extra_row=catalog(e))['reason'],'latest_annual_event_unprocessed')
    def test_held_latest_blocks_fallback(self):
        e=scheduled(event('held'),'2026-02-20','2026-02-23');e['ni_assets_admission']['admitted']=False
        self.assertEqual(self.run_latest(e,catalog(e))['reason'],'latest_event_held')
    def test_future_release_does_not_block(self):
        e=scheduled(event('future'),'2026-04-20','2026-04-21')
        self.assertEqual(self.run_latest(extra_row=catalog(e))['event_id'],'amzn')
    def test_unknown_past_clock_blocks(self):
        e=scheduled(event('unknown'),'2026-02-20','2026-02-23');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['reason'],'latest_annual_event_clock_unverified')
    def test_unknown_future_clock_does_not_block(self):
        e=scheduled(event('future'),'2026-04-20','2026-04-21');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['event_id'],'amzn')
    def test_unknown_same_day_not_used(self):
        e=scheduled(event('today'),'2026-03-02','2026-03-03');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['event_id'],'amzn')
    def test_latest_tie_blocks(self):
        e=event('tie')
        self.assertEqual(self.run_latest(e,catalog(e))['reason'],'ambiguous_latest_annual_event')
    def test_conflicting_duplicate_blocks(self):
        row=catalog(event());row['extra']='conflict'
        self.assertEqual(self.run_latest(extra_row=row)['reason'],'conflicting_catalog_event')
    def test_identical_duplicate_safe(self):
        self.assertTrue(self.run_latest(extra_row=catalog(event()))['available'])
    def test_clock_mismatch_rejected(self):
        row=catalog(event());row['clock_review']['first_pinned_session']='2026-02-03'
        with self.assertRaises(ValueError):self.run_latest(extra_row=row)
    def test_event_catalog_mismatch(self):
        e=scheduled(event('new'),'2026-02-20','2026-02-23');row=catalog(e);e['available_from']='2026-02-24'
        with self.assertRaises(ValueError):self.run_latest(e,row)


class FeatureTests(unittest.TestCase):
    def test_old_bytes_and_new_value(self):
        m,a,d=fixture();before=a.tobytes();v=r.features_oracle(m,a,d)
        self.assertEqual(v['arrays']['earnings_original'].tobytes(),before)
        self.assertEqual(v['arrays']['earnings_expanded'][:,0].tobytes(),a[:,0].tobytes())
        self.assertTrue((v['arrays']['earnings_expanded'][:,4]==1/11).all())
        self.assertEqual(a.tobytes(),before);self.assertEqual(len(v['cells']),98)
    def test_empty_prefix(self):
        m,a,d=fixture();m.update(dates=[],symbols=[])
        v=r.features_oracle(m,a[:0],d);self.assertEqual(v['arrays']['earnings_expanded'].shape,(0,49))
    def test_distinct_masks(self):
        m,a,d=fixture();v=r.features_oracle(m,a,d)['arrays']
        self.assertEqual(v['original_available'].sum(),2);self.assertEqual(v['expanded_available'].sum(),4)
    def test_zero_new_value_remains_available(self):
        m,a,d=fixture();d['events'][0]['fields'][r.FIELDS[0]]['value_decimal']='0'
        v=r.features_oracle(m,a,d)['arrays'];self.assertTrue(v['expanded_available'][0,4]);self.assertEqual(v['earnings_expanded'][0,4],0.)
    def test_etf_no_feature(self):
        m,a,d=fixture();v=r.features_oracle(m,a,d)
        self.assertTrue(np.isnan(v['arrays']['earnings_expanded'][:,32:]).all())
    def test_nonbatch_old_missing_preserved(self):
        m,a,d=fixture();v=r.features_oracle(m,a,d)['arrays']
        self.assertEqual(v['earnings_expanded'][:,6:].tobytes(),a[:,6:].tobytes())
    def test_old_feature_on_new_issuer_rejected(self):
        m,a,d=fixture();a[0,4]=.2
        with self.assertRaises(ValueError):r.features_oracle(m,a,d)
    def test_duplicate_source_event_rejected(self):
        m,a,d=fixture();d['events'].append(deepcopy(d['events'][0]))
        with self.assertRaises(ValueError):r.features_oracle(m,a,d)
    def test_duplicate_pool_rejected(self):
        m,a,d=fixture();m['symbols'][0][4]='AAPL'
        with self.assertRaises(ValueError):r.features_oracle(m,a,d)
    def test_features_prefix_future_pruning(self):
        m,a,d=fixture();future=scheduled(event('future'),'2026-03-20','2026-03-23')
        future['ni_assets_admission']['admitted']=False;d['events'].append(future);d['latest_event_catalog'].append(catalog(future))
        full=r.features_oracle(m,a,d)
        small=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);d['events'].pop();d['latest_event_catalog'].pop()
        prefix=r.features_oracle(small,a[:1],d)
        self.assertEqual(prefix['cells'],full['cells'][:49]);self.assertTrue(full['arrays']['expanded_available'][0,4]);self.assertFalse(full['arrays']['expanded_available'][1,4])
    def test_saved_audit_drift_rejected(self):
        m,a,d=fixture();expected=r.features_oracle(m,a,d);plan=dict(legacy149={},coverage_source={})
        saved=dict(method=r.METHOD,original_cells=2,expanded_cells=4,added_cells=2,original_input_unchanged=True,new_model_fits=0,
                   returns_computed=False,reasons=dict(r.Counter(x['reason'] for x in expected['cells'])),cells=deepcopy(expected['cells']),
                   legacy149={},coverage_source={},original149_earnings_cache_and_mask_verified=True,original149_control_scores_and_orders_byte_exact=True)
        r.verify_audit(saved,expected,plan);saved['cells'][4]['earnings_bounded']=.9
        with self.assertRaises(ValueError):r.verify_audit(saved,expected,plan)


class ScoreTests(unittest.TestCase):
    def test_independent_common_mask_parity(self):
        a,m,h=fixtures.score_fixture();s,c,d=r.score_oracle(a,m,h,h)
        old_s,old_c,_=r.legacy.score_oracle(a,m,h,h,h)
        self.assertEqual(s[r.NAMES[0]].tobytes(),old_s[r.legacy.NAMES[0]].tobytes())
        self.assertEqual(s[r.NAMES[1]].tobytes(),old_s[r.legacy.NAMES[1]].tobytes())
        self.assertEqual(c[r.NAMES[1]],old_c[r.legacy.NAMES[1]])
    def test_distinct_masks_no_collapse(self):
        a,m,h=fixtures.score_fixture();original=h.copy();original[0,0]=np.nan
        s,c,d=r.score_oracle(a,m,original,h)
        self.assertEqual(d[r.NAMES[1]][0,0],0.);self.assertEqual(d[r.NAMES[2]][0,0],.008)
    def test_zero_missing_fallback(self):
        a,m,h=fixtures.score_fixture()
        for val in (np.nan,0.):
            h[:,:4]=val;s,c,d=r.score_oracle(a,m,h,h)
            for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
    def test_prefix_does_not_inspect_future_bad_hand(self):
        a,m,h=fixtures.score_fixture();full=r.score_oracle(a,m,h,h);h[1,:]=np.inf
        s,c,d=r.score_oracle(a,m,h,h,n=1)
        self.assertEqual(s[r.NAMES[2]].tobytes(),full[0][r.NAMES[2]][:1].tobytes())
    def test_empty_score_prefix(self):
        a,m,h=fixtures.score_fixture();s,c,d=r.score_oracle(a,m,h,h,n=0)
        self.assertEqual(s[r.NAMES[2]].shape,(0,5));self.assertEqual(c[r.NAMES[2]],[])
    def test_only_eligible_nonzero(self):
        a,m,h=fixtures.score_fixture();a['eligible'][0,0]=False;s,c,d=r.score_oracle(a,m,h,h)
        self.assertEqual(d[r.NAMES[2]][0,0],0.);self.assertEqual(d[r.NAMES[2]][0,3],0.)
    def test_one_episode_each_arm(self):
        a,m,h=fixtures.score_fixture();helper=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=helper),patch.object(helper,'episode',wraps=helper.episode) as episode:
            r.score_oracle(a,m,h,h);self.assertEqual(episode.call_count,3)
    def test_reversed_order_rejected(self):
        a,m,h=fixtures.score_fixture();s,c,_=r.score_oracle(a,m,h,h);orders=r.ordered_orders(m,s,c)
        orders[r.NAMES[2]][0]['symbols'].reverse()
        with self.assertRaises(ValueError):r.verify_scores_and_orders(s,orders,m,s,c)
    def test_full_matrix_drift_rejected(self):
        a,m,h=fixtures.score_fixture();s,c,_=r.score_oracle(a,m,h,h);orders=r.ordered_orders(m,s,c);bad=deepcopy(s);bad[r.NAMES[2]][0,4]+=.1
        with self.assertRaises(AssertionError):r.verify_scores_and_orders(bad,orders,m,s,c)
    def test_original_array_bytes(self):
        original={str(i):np.array([i],dtype=np.float64) for i in range(27)};prepared=dict(original,**{k:np.array([0.]) for k in r.EXTRA})
        r.verify_original_arrays(original,prepared);prepared['0']=np.array([-0.])
        with self.assertRaises(ValueError):r.verify_original_arrays(original,prepared)


class SourceGateTests(unittest.TestCase):
    sessions=['2026-01-29','2026-01-30','2026-02-02','2026-02-03']
    def gate(self,e):return r.source_gate(e,context(e),'0000001234',self.sessions)
    def test_valid_source(self):self.assertEqual(self.gate(event()),dict(admitted=True,reasons=[],definition=r.DEFINITION))
    def test_early_calendar_held(self):
        e=event();e['clock_review']=dict(first_pinned_session=None,verified=False)
        v=r.source_gate(e,context(e),'0000001234',['2026-02-02','2026-02-03'])
        self.assertEqual(v['reasons'],['unverified_exchange_calendar'])
    def test_both_2019_holds(self):
        for symbol in r.NEW:
            e=event(symbol=symbol);e['fiscal_year']=2019;self.assertIn('2019',self.gate(e)['reasons'][0])
    def test_null_is_held_not_filled(self):
        e=event();e['fields'][r.FIELDS[2]]=None;self.assertEqual(self.gate(e)['reasons'],['missing_prior_total_assets'])
    def test_bad_identity_rejected(self):
        e=event();e['identity']['issuer_id']='wrong'
        with self.assertRaises(ValueError):self.gate(e)
    def test_bad_context_rejected(self):
        e=event();c=context(e);c['sha256']='b'*64
        with self.assertRaises(ValueError):r.source_gate(e,c,'0000001234',self.sessions)
    def test_scale_sign_held(self):
        e=event();e['fields'][r.FIELDS[0]]['scale']='-1';self.assertIn('invalid_annual_net_income',self.gate(e)['reasons'])
    def test_cfo_only_flags_do_not_block(self):
        e=event();e['notes']=['SBC','restricted cash'];self.assertTrue(self.gate(e)['admitted'])
    def test_no_source_freshness_gate(self):
        e=event();self.assertTrue(self.gate(e)['admitted']);self.assertFalse(r.report_oracle(e,'2030-01-01')['available'])
    def test_first_session_no_extrapolation(self):
        self.assertIsNone(r.first_session('2026-01-28',self.sessions));self.assertIsNone(r.first_session('2026-02-03',self.sessions))
        self.assertEqual(r.first_session('2026-01-30',self.sessions),'2026-02-02')


class IntegrationTests(unittest.TestCase):
    def test_all_prefixes_reconstructed_with_generated_49_axis(self):
        meta,original,data=fixture()
        arrays,_,_=fixtures.score_fixture()
        arrays={k:(np.tile(v,(1,10))[:,:49] if v.ndim==2 else v.copy()) for k,v in arrays.items()}
        expected=r.features_oracle(meta,original,data)
        scores,choices,_=r.score_oracle(arrays,meta,expected['arrays'][r.EXTRA[0]],expected['arrays'][r.EXTRA[1]])
        r.verify_prefixes(arrays,meta,original,data,expected,scores,choices)
    def test_149_full_122_cache_mask_and_nan_bytes(self):
        base={str(i):np.full((122,49),float(i)) for i in range(27)}
        hand=np.full((122,49),np.nan);hand[:,0]=.2
        cache=dict(base,earnings_bounded=hand,cash_flow_bounded=hand.copy(),comp_adjusted_cash_flow_bounded=hand.copy(),common_available=np.isfinite(hand))
        prepared=dict(earnings_original=hand.copy(),original_available=np.isfinite(hand))
        r.verify_reference(base,prepared,cache)
        prepared['original_available'][0,0]=False
        with self.assertRaises(ValueError):r.verify_reference(base,prepared,cache)
    def test_149_old_array_tampering_rejected(self):
        base={str(i):np.full((122,49),float(i)) for i in range(27)}
        hand=np.full((122,49),np.nan)
        cache=dict(base,earnings_bounded=hand,cash_flow_bounded=hand.copy(),comp_adjusted_cash_flow_bounded=hand.copy(),common_available=np.isfinite(hand))
        cache['1']=cache['1'].copy();cache['1'][0,0]=2.
        with self.assertRaises(ValueError):r.verify_reference(base,{},cache)
    def test_prospective_contract_and_link_mismatch(self):
        reader=fixtures.MemoryReader()
        concept=reader.add(r.CONCEPT,dict(synthetic_concept=True))
        proposal=dict(round=150,status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',arms=list(r.NAMES),primary_challenger=r.NAMES[2],dose=.01,
                      selection_contract=r.selection_contract(),measurement=r.measurement_contract(),concept=concept,
                      sources=dict(legacy149={'frozen':'old'},coverage_source={'fixed':'new'}),actual_new_factors=0,actual_new_orders=0,actual_new_returns=0)
        ref=reader.add('proposal.json',proposal)
        plan=dict(inputs=dict(proposal=ref),legacy149=proposal['sources']['legacy149'],coverage_source=proposal['sources']['coverage_source'])
        pins={key:reader.sha(key) for key in reader.documents}
        with patch.object(r,'CONCEPT_SHA',concept['sha256']):
            r.verify_proposal(reader,plan,pins)
            plan['coverage_source']={'fixed':'other'}
            with self.assertRaises(ValueError):r.verify_proposal(reader,plan,pins)
    def test_common_mask_not_imposed_across_hands(self):
        a,m,h=fixtures.score_fixture();empty=np.full_like(h,np.nan)
        scores,choices,_=r.score_oracle(a,m,empty,h)
        self.assertEqual(scores[r.NAMES[1]].tobytes(),scores[r.NAMES[0]].tobytes())
        self.assertNotEqual(scores[r.NAMES[2]].tobytes(),scores[r.NAMES[0]].tobytes())


class ContractTests(unittest.TestCase):
    def test_missing_sha_fails_before_reader(self):
        with patch.object(r.old,'Reader') as reader:
            with self.assertRaises(ValueError):r.audit('/synthetic','x',None)
            reader.assert_not_called()
    def test_binding_rejects_wrong_path_or_sha(self):
        reader=fixtures.MemoryReader();ref=reader.add('a.json',dict(a=1));pins={str(reader.path('a.json')):ref['sha256']}
        r.bound_ref(reader,ref,pins,'a.json')
        with self.assertRaises(ValueError):r.bound_ref(reader,ref,pins,'b.json')
        with self.assertRaises(ValueError):r.bound_ref(reader,dict(ref,sha256='f'*64),pins)
    def test_unregistered_closure_rejected(self):
        reader=fixtures.MemoryReader();ref=reader.add('a.json',dict(a=1))
        with self.assertRaises(ValueError):r.closure(reader,{'a.json':ref['sha256']},{})
    def test_exact_contract_typed(self):
        with self.assertRaises(ValueError):r.exact({'a':False},{'a':0})
    def test_no_current_producer_imports(self):
        source=Path(r.__file__).read_text();tree=ast.parse(source)
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any('coverage150' in n or 'admission' in n or 'source_batch' in n for n in names))
    def test_declared_interfaces(self):
        self.assertEqual(len(r.NAMES),3);self.assertEqual(len(r.EXTRA),4);self.assertEqual(len(r.CHECKS),9)
        self.assertEqual(r.selection_contract()['dose'],.01);self.assertEqual(r.measurement_contract()['old_events'],44)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if args.receipt:
        receipt=r.synthetic_receipt(result)
        with args.receipt.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
