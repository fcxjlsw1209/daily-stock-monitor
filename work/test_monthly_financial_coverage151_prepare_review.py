"""Generated-only tests for the independent coverage151 preparation audit."""
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
import monthly_financial_coverage151_prepare_review as r
import test_monthly_cash_quality146_prepare_review as fixtures


def event(key='intc', symbol='INTC'):
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
    if e['symbol']=='INTC':
        c.update(e['source'],current_end='2025-12-31',comparison_end='2024-12-31')
    else:
        c.update(source=deepcopy(e['source']),periods=dict(current=deepcopy(e['fields'][r.FIELDS[0]]['period']),
                                                        comparison=dict(period_end='2024-12-31')))
    return c


def catalog(e):
    return dict(event_id=e['event_id'],symbol=e['symbol'],publication_date=e['release_date'],available_session=e['available_from'],
                fiscal_quarter=4,clock_review=deepcopy(e['clock_review']))


def fixture():
    pool=['AAPL','AMD','MSFT','NVDA','INTC','KLAC']+['S'+str(i) for i in range(26)]+['ETF'+str(i) for i in range(17)]
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
        return r.latest_oracle(rows,events,'INTC',signal)
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
        self.assertEqual(self.run_latest(extra_row=catalog(e))['event_id'],'intc')
    def test_unknown_past_clock_blocks(self):
        e=scheduled(event('unknown'),'2026-02-20','2026-02-23');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['reason'],'latest_annual_event_clock_unverified')
    def test_unknown_future_clock_does_not_block(self):
        e=scheduled(event('future'),'2026-04-20','2026-04-21');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['event_id'],'intc')
    def test_unknown_same_day_not_used(self):
        e=scheduled(event('today'),'2026-03-02','2026-03-03');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['event_id'],'intc')
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
        m,a,d=fixture();before=a.tobytes();v=r.features_oracle(m,a,a,d)
        self.assertEqual(v['arrays']['earnings_original'].tobytes(),before)
        self.assertEqual(v['arrays']['earnings_expanded'][:,0].tobytes(),a[:,0].tobytes())
        self.assertTrue((v['arrays']['earnings_expanded'][:,4]==1/11).all())
        self.assertEqual(a.tobytes(),before);self.assertEqual(len(v['cells']),98)
    def test_empty_prefix(self):
        m,a,d=fixture();m.update(dates=[],symbols=[])
        v=r.features_oracle(m,a[:0],a[:0],d);self.assertEqual(v['arrays']['earnings_expanded'].shape,(0,49))
    def test_distinct_masks(self):
        m,a,d=fixture();v=r.features_oracle(m,a,a,d)['arrays']
        self.assertEqual(v['original_available'].sum(),2);self.assertEqual(v['expanded_available'].sum(),4)
    def test_zero_new_value_remains_available(self):
        m,a,d=fixture();d['events'][0]['fields'][r.FIELDS[0]]['value_decimal']='0'
        v=r.features_oracle(m,a,a,d)['arrays'];self.assertTrue(v['expanded_available'][0,4]);self.assertEqual(v['earnings_expanded'][0,4],0.)
    def test_etf_no_feature(self):
        m,a,d=fixture();v=r.features_oracle(m,a,a,d)
        self.assertTrue(np.isnan(v['arrays']['earnings_expanded'][:,32:]).all())
    def test_nonbatch_old_missing_preserved(self):
        m,a,d=fixture();v=r.features_oracle(m,a,a,d)['arrays']
        self.assertEqual(v['earnings_expanded'][:,6:].tobytes(),a[:,6:].tobytes())
    def test_old_feature_on_new_issuer_rejected(self):
        m,a,d=fixture();a[0,4]=.2
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,d)
    def test_duplicate_source_event_rejected(self):
        m,a,d=fixture();d['events'].append(deepcopy(d['events'][0]))
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,d)
    def test_duplicate_pool_rejected(self):
        m,a,d=fixture();m['symbols'][0][4]='AAPL'
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,d)
    def test_features_prefix_future_pruning(self):
        m,a,d=fixture();future=scheduled(event('future'),'2026-03-20','2026-03-23')
        future['ni_assets_admission']['admitted']=False;d['events'].append(future);d['latest_event_catalog'].append(catalog(future))
        full=r.features_oracle(m,a,a,d)
        small=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);d['events'].pop();d['latest_event_catalog'].pop()
        prefix=r.features_oracle(small,a[:1],a[:1],d)
        self.assertEqual(prefix['cells'],full['cells'][:49]);self.assertTrue(full['arrays']['expanded_available'][0,4]);self.assertFalse(full['arrays']['expanded_available'][1,4])
    def test_saved_audit_drift_rejected(self):
        m,a,d=fixture();expected=r.features_oracle(m,a,a,d);plan=dict(legacy149={},prior150={},coverage_sources={})
        saved=dict(method=r.METHOD,original_cells=2,batch1_cells=2,expanded_cells=4,added_cells=2,original_input_unchanged=True,batch1_input_unchanged=True,new_model_fits=0,
                   returns_computed=False,reasons=dict(r.Counter(x['reason'] for x in expected['cells'])),cells=deepcopy(expected['cells']),
                   legacy149={},prior150={},coverage_sources={},original149_earnings_cache_and_mask_verified=True,original149_control_scores_and_orders_byte_exact=True,prior150_control_scores_and_orders_byte_exact=True,prior150_batch1_earnings_cache_and_mask_verified=True)
        r.verify_audit(saved,expected,plan);saved['cells'][4]['earnings_bounded']=.9
        with self.assertRaises(ValueError):r.verify_audit(saved,expected,plan)


class ScoreTests(unittest.TestCase):
    def test_independent_common_mask_parity(self):
        a,m,h=fixtures.score_fixture();s,c,d=r.score_oracle(a,m,h,h,h)
        old_s,old_c,_=r.legacy.score_oracle(a,m,h,h,h)
        self.assertEqual(s[r.NAMES[0]].tobytes(),old_s[r.legacy.NAMES[0]].tobytes())
        self.assertEqual(s[r.NAMES[1]].tobytes(),old_s[r.legacy.NAMES[1]].tobytes())
        self.assertEqual(c[r.NAMES[1]],old_c[r.legacy.NAMES[1]])
    def test_distinct_masks_no_collapse(self):
        a,m,h=fixtures.score_fixture();original=h.copy();original[0,0]=np.nan
        s,c,d=r.score_oracle(a,m,original,h,h)
        self.assertEqual(d[r.NAMES[1]][0,0],0.);self.assertEqual(d[r.NAMES[2]][0,0],.008)
    def test_zero_missing_fallback(self):
        a,m,h=fixtures.score_fixture()
        for val in (np.nan,0.):
            h[:,:4]=val;s,c,d=r.score_oracle(a,m,h,h,h)
            for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
    def test_prefix_does_not_inspect_future_bad_hand(self):
        a,m,h=fixtures.score_fixture();full=r.score_oracle(a,m,h,h,h);h[1,:]=np.inf
        s,c,d=r.score_oracle(a,m,h,h,h,n=1)
        self.assertEqual(s[r.NAMES[2]].tobytes(),full[0][r.NAMES[2]][:1].tobytes())
    def test_empty_score_prefix(self):
        a,m,h=fixtures.score_fixture();s,c,d=r.score_oracle(a,m,h,h,h,n=0)
        self.assertEqual(s[r.NAMES[2]].shape,(0,5));self.assertEqual(c[r.NAMES[2]],[])
    def test_only_eligible_nonzero(self):
        a,m,h=fixtures.score_fixture();a['eligible'][0,0]=False;s,c,d=r.score_oracle(a,m,h,h,h)
        self.assertEqual(d[r.NAMES[2]][0,0],0.);self.assertEqual(d[r.NAMES[2]][0,3],0.)
    def test_one_episode_each_arm(self):
        a,m,h=fixtures.score_fixture();helper=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=helper),patch.object(helper,'episode',wraps=helper.episode) as episode:
            r.score_oracle(a,m,h,h,h);self.assertEqual(episode.call_count,4)
    def test_reversed_order_rejected(self):
        a,m,h=fixtures.score_fixture();s,c,_=r.score_oracle(a,m,h,h,h);orders=r.ordered_orders(m,s,c)
        orders[r.NAMES[2]][0]['symbols'].reverse()
        with self.assertRaises(ValueError):r.verify_scores_and_orders(s,orders,m,s,c)
    def test_full_matrix_drift_rejected(self):
        a,m,h=fixtures.score_fixture();s,c,_=r.score_oracle(a,m,h,h,h);orders=r.ordered_orders(m,s,c);bad=deepcopy(s);bad[r.NAMES[2]][0,4]+=.1
        with self.assertRaises(AssertionError):r.verify_scores_and_orders(bad,orders,m,s,c)
    def test_original_array_bytes(self):
        original={str(i):np.array([i],dtype=np.float64) for i in range(27)};prepared=dict(original,**{k:np.array([0.]) for k in r.EXTRA})
        r.verify_original_arrays(original,prepared);prepared['0']=np.array([-0.])
        with self.assertRaises(ValueError):r.verify_original_arrays(original,prepared)



class SourceGateTests(unittest.TestCase):
    sessions=['2026-01-29','2026-01-30','2026-02-02','2026-02-03']
    def fixture(self,symbol='INTC',year=2025):
        e=event(symbol=symbol);e['publication_date']=e.pop('release_date');e['fiscal_year']=year
        end=f'{year}-12-27' if symbol=='INTC' else f'{year}-06-30'
        start=f'{year-1}-12-29' if symbol=='INTC' else f'{year-1}-07-01'
        prior_end=str(date.fromisoformat(start)-timedelta(days=1))
        p=dict(period_start=start,period_end=end,days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1)
        e['periods']=dict(current=p,start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',start_assumption_certified=False)
        e['source']['url']='https://issuer.example/x'
        e['context_holds']=['annual_start_not_directly_quoted_requires_source_use_assumption'] if symbol=='INTC' else []
        for f,c in e['fields'].items():
            c.update(source=deepcopy(e['source']),as_of=prior_end if f==r.FIELDS[2] else end,period=deepcopy(p) if f==r.FIELDS[0] else None)
        e['identity']['identity_valid_through']=str(date.fromisoformat(end)+timedelta(days=550))
        c=dict(event_id=e['event_id'],symbol=symbol,fiscal_year=year,evidence_path=e['source']['path'],evidence_sha256=e['source']['sha256'],source_url=e['source']['url'],publication_date=e['publication_date'],available_session=e['available_from'],official_actual_period_end=end,
            context_summary=dict(comparison_assets_date=prior_end,current_annual_period=dict(start=None if symbol=='INTC' else start,end=end,candidate_start_from_prior_end_plus_one=start)))
        rr=dict(event_id=e['event_id'],clock_verified=True,boundary_assumption_verified=symbol=='INTC',assumption_id='INTC_consecutive_annual_boundaries_v1',derived_start=start,start_is_source_quoted=False)
        e['source_review']=dict(event_id=e['event_id'],metadata_verified=True,all_three_positions_reviewed=True,boundary_assumption_verified=rr['boundary_assumption_verified'],original_context_holds_preserved=True)
        return e,c,rr
    def gate(self,items,sessions=None):
        e,c,rr=items
        return r.source_gate(e,c,'0000001234',sessions or self.sessions,rr)
    def test_intc_explicit_boundary(self):self.assertTrue(self.gate(self.fixture())['admitted'])
    def test_klac_calendar_boundary(self):self.assertTrue(self.gate(self.fixture('KLAC'))['admitted'])
    def test_unverified_boundary_stays_held(self):
        f=self.fixture();f[2]['boundary_assumption_verified']=False;f[0]['source_review']['boundary_assumption_verified']=False
        self.assertIn('INTC_boundary_assumption_unverified',self.gate(f)['reasons'])
    def test_bad_boundary_daycount_held(self):
        f=self.fixture();f[0]['fields'][r.FIELDS[0]]['period']['days_inclusive']=365
        self.assertIn('invalid_current_annual_period',self.gate(f)['reasons'])
    def test_quoted_start_not_allowed(self):
        f=self.fixture();f[1]['context_summary']['current_annual_period']['start']='2024-12-29'
        self.assertIn('invalid_current_annual_period',self.gate(f)['reasons'])
    def test_unknown_context_hold_preserved(self):
        f=self.fixture();f[0]['context_holds'].append('new_uncertainty')
        self.assertIn('retained_context_hold:new_uncertainty',self.gate(f)['reasons'])
    def test_asc606_hold(self):
        f=self.fixture('KLAC',2019)
        self.assertIn('KLAC_FY2019_ASC606_transition_unresolved',self.gate(f)['reasons'])
    def test_null_does_not_fill(self):
        f=self.fixture();f[0]['fields'][r.FIELDS[2]]=None
        self.assertIn('missing_prior_total_assets',self.gate(f)['reasons'])
    def test_wrong_identity_fails(self):
        f=self.fixture();f[0]['identity']['issuer_id']='wrong'
        with self.assertRaises(ValueError):self.gate(f)
    def test_clock_is_independent(self):
        f=self.fixture();f[2]['clock_verified']=False
        with self.assertRaises(ValueError):self.gate(f)
    def test_scale_negative_held(self):
        f=self.fixture();f[0]['fields'][r.FIELDS[0]]['scale']='-1'
        self.assertIn('invalid_annual_net_income',self.gate(f)['reasons'])
    def test_no_age_gate_at_source(self):
        self.assertTrue(self.gate(self.fixture())['admitted'])
        self.assertFalse(r.report_oracle(self.fixture()[0],'2030-01-01')['available'])


class IntegrationTests(unittest.TestCase):
    def test_all_four_prefixes(self):
        meta,original,data=fixture();arrays,_,_=fixtures.score_fixture()
        arrays={k:(np.tile(v,(1,10))[:,:49] if v.ndim==2 else v.copy()) for k,v in arrays.items()}
        expected=r.features_oracle(meta,original,original,data)
        ss,cc,_=r.score_oracle(arrays,meta,original,original,expected['arrays']['earnings_expanded'])
        r.verify_prefixes(arrays,meta,original,original,data,expected,ss,cc)
    def test_batch1_cells_kept_without_recalculation(self):
        meta,a,data=fixture();meta['symbols'][0][6]=meta['symbols'][1][6]='AMZN'
        b=a.copy();b[:,6]=.456
        v=r.features_oracle(meta,a,b,data)['arrays']
        self.assertEqual(v['earnings_batch1'].tobytes(),b.tobytes())
        self.assertEqual(v['earnings_expanded'][:,6].tobytes(),b[:,6].tobytes())
    def test_batch1_unrelated_change_rejected(self):
        meta,a,data=fixture();b=a.copy();b[0,0]=.4
        with self.assertRaises(ValueError):r.features_oracle(meta,a,b,data)
    def test_batch1_cannot_precover_intc(self):
        meta,a,data=fixture();b=a.copy();b[0,4]=.4
        with self.assertRaises(ValueError):r.features_oracle(meta,a,b,data)
    def test_150_cache_bytes(self):
        base={str(i):np.full((122,49),float(i)) for i in range(27)}
        a=np.full((122,49),np.nan);a[:,0]=.2;b=a.copy();b[:,1]=.1
        cache=dict(base,earnings_original=a,earnings_expanded=b,original_available=np.isfinite(a),expanded_available=np.isfinite(b))
        prepared=dict(earnings_batch1=b.copy(),batch1_available=np.isfinite(b))
        r.verify_prior_reference(base,prepared,cache,a,np.isfinite(a))
        prepared['earnings_batch1'][0,1]=.2
        with self.assertRaises(ValueError):r.verify_prior_reference(base,prepared,cache,a,np.isfinite(a))
    def test_149_cache_bytes(self):
        base={str(i):np.full((122,49),float(i)) for i in range(27)};a=np.full((122,49),np.nan)
        cache=dict(base,earnings_bounded=a,cash_flow_bounded=a,comp_adjusted_cash_flow_bounded=a,common_available=np.isfinite(a))
        prepared=dict(earnings_original=a.copy(),original_available=np.isfinite(a))
        r.verify_reference(base,prepared,cache)
        prepared['original_available'][0,0]=True
        with self.assertRaises(ValueError):r.verify_reference(base,prepared,cache)
    def test_proposal_sources(self):
        reader=fixtures.MemoryReader();concept=reader.add(r.CONCEPT,dict(synthetic=True))
        sources=dict(legacy149={'a':1},prior150={'b':2},coverage_sources={'batch1':{},'batch2':{}})
        p=dict(round=151,status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',arms=list(r.NAMES),primary_challenger=r.NAMES[3],dose=.01,selection_contract=r.selection_contract(),measurement=r.measurement_contract(),concept=concept,sources=sources,actual_new_factors=0,actual_new_orders=0,actual_new_returns=0)
        ref=reader.add('proposal.json',p);plan=dict(inputs=dict(proposal=ref),**sources);pins={k:reader.sha(k) for k in reader.documents}
        with patch.object(r,'CONCEPT_SHA',concept['sha256']):
            r.verify_proposal(reader,plan,pins)
            plan['prior150']={}
            with self.assertRaises(ValueError):r.verify_proposal(reader,plan,pins)
    def test_missing_sha_prevents_io(self):
        with patch.object(r.old,'Reader') as reader:
            with self.assertRaises(ValueError):r.audit('/synthetic','x',None)
            reader.assert_not_called()
    def test_declared_interfaces(self):
        self.assertEqual((len(r.NAMES),len(r.EXTRA),len(r.CHECKS)),(4,6,11))
        self.assertEqual(r.measurement_contract()['source_events'],45)
        self.assertEqual(r.measurement_contract()['annual_catalog_events'],64)
    def test_no_new_producer_imports(self):
        tree=ast.parse(Path(r.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names += [a.name for a in n.names]
            if isinstance(n,ast.ImportFrom):names.append(n.module or '')
        self.assertFalse(any('151_runner' in n or '151_factor' in n or '151_selection' in n or 'admission' in n for n in names))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if args.receipt:
        if not result.wasSuccessful():raise SystemExit('Failed tests cannot issue prereview')
        if (r.ROOT/r.STUDY/'PLAN.json').exists():raise SystemExit('Already registered')
        receipt=r.synthetic_receipt(result)
        with args.receipt.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
