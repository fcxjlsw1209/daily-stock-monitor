"""Generated-only tests for the independent coverage153 preparation audit."""
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
import monthly_financial_coverage153_prepare_review as r
import test_monthly_cash_quality146_prepare_review as fixtures


def event(key='mu', symbol='MU'):
    source = dict(path='synthetic/report.html',sha256='a'*64)
    period = dict(period_start='2025-01-02',period_end='2025-12-31',days_inclusive=364)
    e = dict(event_id=key,symbol=symbol,fiscal_year=2025,release_date='2026-01-30',available_from='2026-02-02',
             periods=dict(current=period),source=source,fact_admitted=False,model_admitted=False,strategy_admitted=False,
             ni_assets_admission=dict(admitted=True,reasons=[],definition=r.DEFINITION),
             identity=dict(issuer_id='0000001234',security_id=symbol,identity_valid_from='2026-02-02',
                           identity_valid_through=str(date(2025,12,31)+timedelta(days=550)),
                           scope='source_local_report_binding_and_use_window_not_continuous_security_master',certified_daily_security_master=False),
             clock_review=dict(first_pinned_session='2026-02-02',verified=True))
    e['fields'] = {k:dict(value_decimal=v,scale='1',currency_literal='$',currency_iso=None,source=deepcopy(source),
                          available_from=e['available_from'],as_of='2025-01-01' if k==r.FIELDS[2] else '2025-12-31',
                          fact_admitted=False,displayed_amount=v,locator=dict(table=1,row=2))
                   for k,v in zip(r.FIELDS,('10','150','50'))}
    e['fields'][r.FIELDS[0]]['period']=deepcopy(period)
    return e


def context(e):
    c = dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],release_date=e['release_date'],
             available_session=e['available_from'])
    if e['symbol']=='MU':
        c.update(e['source'],current_end='2025-12-31',comparison_end='2024-12-31')
    else:
        c.update(source=deepcopy(e['source']),periods=dict(current=deepcopy(e['fields'][r.FIELDS[0]]['period']),
                                                        comparison=dict(period_end='2024-12-31')))
    return c


def catalog(e):
    return dict(event_id=e['event_id'],symbol=e['symbol'],publication_date=e['release_date'],available_session=e['available_from'],
                fiscal_quarter=4,clock_review=deepcopy(e['clock_review']))


def fixture():
    pool=['AAPL','AMD','MSFT','NVDA','MU','PEP']+['S'+str(i) for i in range(26)]+['ETF'+str(i) for i in range(17)]
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
        e=event();e['fields'][r.FIELDS[0]]['period']['days_inclusive']=365
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
        return r.latest_oracle(rows,events,'MU',signal)
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
        self.assertEqual(self.run_latest(extra_row=catalog(e))['event_id'],'mu')
    def test_unknown_past_clock_blocks(self):
        e=scheduled(event('unknown'),'2026-02-20','2026-02-23');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['reason'],'latest_annual_event_clock_unverified')
    def test_unknown_future_clock_does_not_block(self):
        e=scheduled(event('future'),'2026-04-20','2026-04-21');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['event_id'],'mu')
    def test_unknown_same_day_not_used(self):
        e=scheduled(event('today'),'2026-03-02','2026-03-03');row=catalog(e);row['clock_review']['verified']=False
        self.assertEqual(self.run_latest(extra_row=row)['event_id'],'mu')
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
        m,a,d=fixture();before=a.tobytes();v=r.features_oracle(m,a,a,a,a,d)
        self.assertEqual(v['arrays']['earnings_original'].tobytes(),before)
        self.assertEqual(v['arrays']['earnings_expanded'][:,0].tobytes(),a[:,0].tobytes())
        self.assertTrue((v['arrays']['earnings_expanded'][:,4]==1/11).all())
        self.assertEqual(a.tobytes(),before);self.assertEqual(len(v['cells']),98)
    def test_empty_prefix(self):
        m,a,d=fixture();m.update(dates=[],symbols=[])
        v=r.features_oracle(m,a[:0],a[:0],a[:0],a[:0],d);self.assertEqual(v['arrays']['earnings_expanded'].shape,(0,49))
    def test_distinct_masks(self):
        m,a,d=fixture();v=r.features_oracle(m,a,a,a,a,d)['arrays']
        self.assertEqual(v['original_available'].sum(),2);self.assertEqual(v['expanded_available'].sum(),4)
    def test_zero_new_value_remains_available(self):
        m,a,d=fixture();d['events'][0]['fields'][r.FIELDS[0]]['value_decimal']='0'
        v=r.features_oracle(m,a,a,a,a,d)['arrays'];self.assertTrue(v['expanded_available'][0,4]);self.assertEqual(v['earnings_expanded'][0,4],0.)
    def test_etf_no_feature(self):
        m,a,d=fixture();v=r.features_oracle(m,a,a,a,a,d)
        self.assertTrue(np.isnan(v['arrays']['earnings_expanded'][:,32:]).all())
    def test_nonbatch_old_missing_preserved(self):
        m,a,d=fixture();v=r.features_oracle(m,a,a,a,a,d)['arrays']
        self.assertEqual(v['earnings_expanded'][:,6:].tobytes(),a[:,6:].tobytes())
    def test_old_feature_on_new_issuer_rejected(self):
        m,a,d=fixture();a[0,4]=.2
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,a,a,d)
    def test_duplicate_source_event_rejected(self):
        m,a,d=fixture();d['events'].append(deepcopy(d['events'][0]))
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,a,a,d)
    def test_duplicate_pool_rejected(self):
        m,a,d=fixture();m['symbols'][0][4]='AAPL'
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,a,a,d)
    def test_features_prefix_future_pruning(self):
        m,a,d=fixture();future=scheduled(event('future'),'2026-03-20','2026-03-23')
        future['ni_assets_admission']['admitted']=False;d['events'].append(future);d['latest_event_catalog'].append(catalog(future))
        full=r.features_oracle(m,a,a,a,a,d)
        small=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);d['events'].pop();d['latest_event_catalog'].pop()
        prefix=r.features_oracle(small,a[:1],a[:1],a[:1],a[:1],d)
        self.assertEqual(prefix['cells'],full['cells'][:49]);self.assertTrue(full['arrays']['expanded_available'][0,4]);self.assertFalse(full['arrays']['expanded_available'][1,4])
    def test_saved_audit_drift_rejected(self):
        m,a,d=fixture();expected=r.features_oracle(m,a,a,a,a,d);plan=dict(legacy149={},prior152={},coverage_sources={})
        saved=dict(method=r.METHOD,original_cells=2,batch1_cells=2,batch2_cells=2,batch3_cells=2,expanded_cells=4,added_cells=2,original_input_unchanged=True,batch1_input_unchanged=True,batch2_input_unchanged=True,batch3_input_unchanged=True,new_model_fits=0,
                   returns_computed=False,reasons=dict(r.Counter(x['reason'] for x in expected['cells'])),cells=deepcopy(expected['cells']),
                   legacy149={},prior152={},coverage_sources={},original149_earnings_cache_and_mask_verified=True,original149_control_scores_and_orders_byte_exact=True,prior150_control_scores_and_orders_byte_exact=True,prior150_batch1_earnings_cache_and_mask_verified=True,prior151_batch2_earnings_cache_and_mask_verified=True,prior151_control_scores_and_orders_byte_exact=True,prior152_batch3_earnings_cache_and_mask_verified=True,prior152_control_scores_and_orders_byte_exact=True)
        r.verify_audit(saved,expected,plan);saved['cells'][4]['earnings_bounded']=.9
        with self.assertRaises(ValueError):r.verify_audit(saved,expected,plan)


class ScoreTests(unittest.TestCase):
    def test_independent_common_mask_parity(self):
        a,m,h=fixtures.score_fixture();s,c,d=r.score_oracle(a,m,h,h,h,h,h)
        old_s,old_c,_=r.legacy.score_oracle(a,m,h,h,h)
        self.assertEqual(s[r.NAMES[0]].tobytes(),old_s[r.legacy.NAMES[0]].tobytes())
        self.assertEqual(s[r.NAMES[1]].tobytes(),old_s[r.legacy.NAMES[1]].tobytes())
        self.assertEqual(c[r.NAMES[1]],old_c[r.legacy.NAMES[1]])
    def test_distinct_masks_no_collapse(self):
        a,m,h=fixtures.score_fixture();original=h.copy();original[0,0]=np.nan
        s,c,d=r.score_oracle(a,m,original,h,h,h,h)
        self.assertEqual(d[r.NAMES[1]][0,0],0.);self.assertEqual(d[r.NAMES[2]][0,0],.008)
    def test_zero_missing_fallback(self):
        a,m,h=fixtures.score_fixture()
        for val in (np.nan,0.):
            h[:,:4]=val;s,c,d=r.score_oracle(a,m,h,h,h,h,h)
            for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
    def test_prefix_does_not_inspect_future_bad_hand(self):
        a,m,h=fixtures.score_fixture();full=r.score_oracle(a,m,h,h,h,h,h);h[1,:]=np.inf
        s,c,d=r.score_oracle(a,m,h,h,h,h,h,n=1)
        self.assertEqual(s[r.NAMES[2]].tobytes(),full[0][r.NAMES[2]][:1].tobytes())
    def test_empty_score_prefix(self):
        a,m,h=fixtures.score_fixture();s,c,d=r.score_oracle(a,m,h,h,h,h,h,n=0)
        self.assertEqual(s[r.NAMES[2]].shape,(0,5));self.assertEqual(c[r.NAMES[2]],[])
    def test_only_eligible_nonzero(self):
        a,m,h=fixtures.score_fixture();a['eligible'][0,0]=False;s,c,d=r.score_oracle(a,m,h,h,h,h,h)
        self.assertEqual(d[r.NAMES[2]][0,0],0.);self.assertEqual(d[r.NAMES[2]][0,3],0.)
    def test_one_episode_each_arm(self):
        a,m,h=fixtures.score_fixture();helper=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=helper),patch.object(helper,'episode',wraps=helper.episode) as episode:
            r.score_oracle(a,m,h,h,h,h,h);self.assertEqual(episode.call_count,6)
    def test_reversed_order_rejected(self):
        a,m,h=fixtures.score_fixture();s,c,_=r.score_oracle(a,m,h,h,h,h,h);orders=r.ordered_orders(m,s,c)
        orders[r.NAMES[2]][0]['symbols'].reverse()
        with self.assertRaises(ValueError):r.verify_scores_and_orders(s,orders,m,s,c)
    def test_full_matrix_drift_rejected(self):
        a,m,h=fixtures.score_fixture();s,c,_=r.score_oracle(a,m,h,h,h,h,h);orders=r.ordered_orders(m,s,c);bad=deepcopy(s);bad[r.NAMES[2]][0,4]+=.1
        with self.assertRaises(AssertionError):r.verify_scores_and_orders(bad,orders,m,s,c)
    def test_original_array_bytes(self):
        original={str(i):np.array([i],dtype=np.float64) for i in range(27)};prepared=dict(original,**{k:np.array([0.]) for k in r.EXTRA})
        r.verify_original_arrays(original,prepared);prepared['0']=np.array([-0.])
        with self.assertRaises(ValueError):r.verify_original_arrays(original,prepared)



class IntegrationTests(unittest.TestCase):
    def test_all_four_prefixes(self):
        meta,original,data=fixture();arrays,_,_=fixtures.score_fixture()
        arrays={k:(np.tile(v,(1,10))[:,:49] if v.ndim==2 else v.copy()) for k,v in arrays.items()}
        expected=r.features_oracle(meta,original,original,original,original,data)
        ss,cc,_=r.score_oracle(arrays,meta,original,original,original,original,expected['arrays']['earnings_expanded'])
        r.verify_prefixes(arrays,meta,original,original,original,original,data,expected,ss,cc)
    def test_batch1_cells_kept_without_recalculation(self):
        meta,a,data=fixture();meta['symbols'][0][6]=meta['symbols'][1][6]='AMZN'
        b=a.copy();b[:,6]=.456
        v=r.features_oracle(meta,a,b,b,b,data)['arrays']
        self.assertEqual(v['earnings_batch1'].tobytes(),b.tobytes())
        self.assertEqual(v['earnings_expanded'][:,6].tobytes(),b[:,6].tobytes())
    def test_batch1_unrelated_change_rejected(self):
        meta,a,data=fixture();b=a.copy();b[0,0]=.4
        with self.assertRaises(ValueError):r.features_oracle(meta,a,b,b,b,data)
    def test_batch1_cannot_precover_intc(self):
        meta,a,data=fixture();b=a.copy();b[0,4]=.4
        with self.assertRaises(ValueError):r.features_oracle(meta,a,b,b,b,data)
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
        sources=dict(legacy149={'a':1},prior152={'b':2},coverage_sources={'batch1':{},'batch2':{},'batch3':{},'batch4':{}})
        p=dict(round=153,status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',arms=list(r.NAMES),primary_challenger=r.NAMES[5],dose=.01,selection_contract=r.selection_contract(),measurement=r.measurement_contract(),concept=concept,sources=sources,actual_new_factors=0,actual_new_orders=0,actual_new_returns=0)
        ref=reader.add('proposal.json',p);plan=dict(inputs=dict(proposal=ref),**sources);pins={k:reader.sha(k) for k in reader.documents}
        with patch.object(r,'CONCEPT_SHA',concept['sha256']):
            r.verify_proposal(reader,plan,pins)
            plan['prior152']={}
            with self.assertRaises(ValueError):r.verify_proposal(reader,plan,pins)
    def test_missing_sha_prevents_io(self):
        with patch.object(r.old,'Reader') as reader:
            with self.assertRaises(ValueError):r.audit('/synthetic','x',None)
            reader.assert_not_called()
    def test_declared_interfaces(self):
        self.assertEqual((len(r.NAMES),len(r.EXTRA),len(r.CHECKS)),(6,10,15))
        self.assertEqual(r.measurement_contract()['source_events'],64)
        self.assertEqual(r.measurement_contract()['annual_catalog_events'],119)
    def test_no_new_producer_imports(self):
        tree=ast.parse(Path(r.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names += [a.name for a in n.names]
            if isinstance(n,ast.ImportFrom):names.append(n.module or '')
        self.assertFalse(any('153_runner' in n or '153_factor' in n or '153_selection' in n or 'admission' in n for n in names))




def source_fixture(symbol='MU',year=2025):
    e=event(symbol=symbol);e['publication_date']=e.pop('release_date');e['fiscal_year']=year
    e['event_id']=symbol+'_'+e['publication_date']
    end=f'{year}-08-28' if symbol=='MU' else f'{year}-12-27'
    start=str(date.fromisoformat(end)-timedelta(days=363))
    prior_end=str(date.fromisoformat(start)-timedelta(days=1))
    per=dict(period_start=start,period_end=end,days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1)
    e['periods']=dict(current=per,comparison=dict(period_end=prior_end),start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',start_assumption_certified=False,assumption_id=r.ASSUMPTIONS[symbol])
    e['source'].update(url='https://example.invalid/report',receipt_path='synthetic/receipt.json',receipt_sha256='b'*64)
    e['context_holds']=['currency_ISO_and_first_public_version_not_certified','proposed_annual_start_boundary_assumption_not_yet_approved']
    e['accounting_context_numbers_redacted']=[]
    rawidentity=dict(issuer=r.ISSUERS[symbol],symbol=symbol,source_anchors=['consolidated statement'],certified_daily_security_master=False)
    e['identity']=dict(rawidentity,issuer_id=r.ISSUERS[symbol],security_id=symbol,identity_valid_from=e['available_from'],identity_valid_through=str(date.fromisoformat(end)+timedelta(days=550)),scope='source_local_report_binding_and_use_window_not_continuous_security_master',certified_daily_security_master=False)
    for f,c in e['fields'].items():c.update(source=deepcopy(e['source']),scale='1000000',as_of=prior_end if f==r.FIELDS[2] else end,period=deepcopy(per) if f==r.FIELDS[0] else None)
    ctx=dict(event_id=e['event_id'],symbol=symbol,fiscal_year=year,source=deepcopy(e['source']),evidence_path=e['source']['path'],evidence_sha256=e['source']['sha256'],source_url=e['source']['url'],publication_date=e['publication_date'],available_session=e['available_from'],official_actual_period_end=end,periods=dict(current_end=end,prior_end=prior_end,proposed_current_start=start),context_holds=deepcopy(e['context_holds']),accounting_context_numbers_redacted=[])
    ctx.update(source_kind='HTML_original',original_report_bytes_verified=True,identity_anchors=rawidentity['source_anchors'],available_session_verified=True,fields={f:{'scale':'1000000'} for f in r.FIELDS})
    e.update(first_public_version_verified=False,first_public_time_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,financial_calculations=0,accounting_review_flags=[],gaps=[],complete_required_fields=True)
    acc=dict(event_id=e['event_id'],source=deepcopy(e['source']),status='resolved',scope='current_annual_net_income_same_report_assets',reasons=[],basis='Comparable annual amounts.',field_checks=dict.fromkeys(r.FIELDS,True),current_prior_common_basis=True,evidence=[dict(path=e['source']['path'],sha256=e['source']['sha256'],locator=dict(kind='document_text',start=0,end=5),quote='basis')])
    rr=dict(event_id=e['event_id'],passed=True,symbol=symbol,fiscal_year=year,metadata_verified=True,identity_verified=True,issuer_id=r.ISSUERS[symbol],security_id=symbol,source=deepcopy(e['source']),retained_context_holds=deepcopy(e['context_holds']),accounting_context_numbers_redacted=[],available_from=e['available_from'],release_date=e['publication_date'],periods_verified=True,currency_verified=True,clock_verified=True,boundary_assumption_verified=True,assumption_id=r.ASSUMPTIONS[symbol],derived_start=start,start_is_source_quoted=False,accounting_review=acc,boundary_evidence={f:dict(table=1,headers=['annual'],date=prior_end if f==r.FIELDS[2] else end) for f in r.FIELDS},fields={f:dict(is_null=False,verified=True,source_value_present=True,value_and_sign_verified=True,units_verified=True,period_verified=True,source_locator_verified=True) for f in r.FIELDS})
    e['source_review']=dict(event_id=e['event_id'],metadata_verified=True,all_three_positions_reviewed=True,boundary_assumption_verified=True,accounting_review=deepcopy(acc),original_context_holds_preserved=True)
    return e,ctx,rr,rawidentity


class SourceGateTests(unittest.TestCase):
    sessions=['2026-01-29','2026-01-30','2026-02-02','2026-02-03']
    def gate(self,f):
        e,c,rr,identity=f
        return r.source_gate(e,c,self.sessions,rr,identity)
    def test_mu_source_conditionally_available(self):self.assertTrue(self.gate(source_fixture())['admitted'])
    def test_pep_source_52_week_boundary(self):self.assertTrue(self.gate(source_fixture('PEP'))['admitted'])
    def test_no_source_age_gate(self):
        f=source_fixture(year=2018);self.assertTrue(self.gate(f)['admitted']);self.assertFalse(r.report_oracle(f[0],'2026-03-02')['available'])
    def test_negative_and_zero_NI_allowed(self):
        for value in ('-10','0'):
            f=source_fixture();f[0]['fields'][r.FIELDS[0]]['value_decimal']=value
            self.assertTrue(self.gate(f)['admitted'])
    def test_positive_assets_required(self):
        for field in r.FIELDS[1:]:
            f=source_fixture();f[0]['fields'][field]['value_decimal']='0'
            self.assertIn('invalid_'+field,self.gate(f)['reasons'])
    def test_correct_source_scale_required(self):
        f=source_fixture();f[0]['fields'][r.FIELDS[0]]['scale']='1'
        self.assertIn('invalid_annual_net_income',self.gate(f)['reasons'])
    def test_uncertified_currency_cannot_change(self):
        f=source_fixture();f[0]['fields'][r.FIELDS[0]]['currency_iso']='USD'
        self.assertIn('invalid_annual_net_income',self.gate(f)['reasons'])
    def test_source_nulls_stay_null_and_held(self):
        f=source_fixture();f[0]['fields']=dict.fromkeys(r.FIELDS);f[0]['identity']['identity_valid_through']=None;f[0]['gaps']=[dict(field=k,reason='synthetic null') for k in r.FIELDS];f[0]['complete_required_fields']=False
        result=self.gate(f)
        self.assertFalse(result['admitted']);self.assertTrue(all(f[0]['fields'][k] is None for k in r.FIELDS))
        self.assertIn('invalid_current_annual_period',result['reasons'])
    def test_unverified_calendar_held(self):
        f=source_fixture();f[2]['clock_verified']=False;f[1]['available_session_verified']=False;f[0]['clock_review']=dict(first_pinned_session=None,verified=False)
        with patch.object(self,'sessions',['2027-01-01','2027-01-04']):self.assertIn('unverified_exchange_calendar',self.gate(f)['reasons'])
    def test_unknown_holds_preserved(self):
        f=source_fixture();f[0]['context_holds'].append('unexplained_restatement');f[1]['context_holds'].append('unexplained_restatement')
        self.assertIn('retained_context_hold:unexplained_restatement',self.gate(f)['reasons'])
    def test_context_hold_removal_fails(self):
        f=source_fixture();f[0]['context_holds']=[]
        with self.assertRaises(ValueError):self.gate(f)
    def test_boundary_not_verified_stays_held(self):
        f=source_fixture();f[2]['boundary_assumption_verified']=False;f[0]['source_review']['boundary_assumption_verified']=False
        self.assertIn('annual_boundary_assumption_unverified',self.gate(f)['reasons'])
    def test_quarter_duration_rejected(self):
        f=source_fixture();f[0]['fields'][r.FIELDS[0]]['period']['days_inclusive']=90
        self.assertIn('invalid_current_annual_period',self.gate(f)['reasons'])
    def test_identity_anchors_preserved(self):
        f=source_fixture();f[0]['identity']['source_anchors']=[]
        with self.assertRaises(ValueError):self.gate(f)
    def test_accounting_unresolved_not_admitted(self):
        f=source_fixture();f[2]['accounting_review']['status']='unresolved';f[0]['source_review']['accounting_review']=deepcopy(f[2]['accounting_review'])
        self.assertIn('accounting_comparability_unresolved',self.gate(f)['reasons'])
    def test_accounting_false_resolution_rejected(self):
        for key,value in (('source',{}),('field_checks',{}),('current_prior_common_basis',False),('reasons',['unresolved']),('evidence',[])):
            f=source_fixture();f[2]['accounting_review'][key]=value
            with self.assertRaises(ValueError):self.gate(f)
    def test_added_identity_removed_without_losing_original(self):
        f=source_fixture();raw=deepcopy(f[0]);raw['identity']=f[3]
        for k in ('ni_assets_admission','clock_review','source_review'):raw.pop(k)
        self.assertEqual(r.strip_added_event(f[0],raw),raw)
        f[0]['identity']['unrecognized']=True
        with self.assertRaises(ValueError):r.strip_added_event(f[0],raw)


class NewControlTests(unittest.TestCase):
    def test_imported_oracles_not_shadowed_in_audit(self):
        tree=ast.parse(Path(r.__file__).read_text());audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
        targets={node.id for node in ast.walk(audit) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}
        self.assertFalse(targets & {'prior','previous','prior152','legacy','old'})

    def test_151_exact_control(self):
        base={str(i):np.full((122,49),float(i)) for i in range(27)}
        a=np.full((122,49),np.nan);a[:,0]=.2;b=a.copy();b[:,1]=.1;c=b.copy();c[:,2]=.3
        cached=dict(base,earnings_original=a,original_available=np.isfinite(a),earnings_batch1=b,batch1_available=np.isfinite(b),earnings_expanded=c,expanded_available=np.isfinite(c))
        new=dict(earnings_batch2=c.copy(),batch2_available=np.isfinite(c))
        r.verify_prior151_reference(base,new,cached,a,np.isfinite(a),b,np.isfinite(b))
        new['earnings_batch2'][0,2]=.4
        with self.assertRaises(ValueError):r.verify_prior151_reference(base,new,cached,a,np.isfinite(a),b,np.isfinite(b))
    def test_batch2_unrelated_change_rejected(self):
        m,a,d=fixture();b=a.copy();b[0,0]=.4
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,b,b,d)
    def test_batch2_cannot_contain_new_symbol(self):
        m,a,d=fixture();b=a.copy();b[0,4]=.4
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,b,b,d)
    def test_batch2_intc_preserved(self):
        m,a,d=fixture()
        for pool in m['symbols']:pool[6]='INTC'
        b=a.copy();b[:,6]=-.25
        v=r.features_oracle(m,a,a,b,b,d)
        self.assertEqual(v['arrays']['earnings_expanded'][:,6].tobytes(),b[:,6].tobytes())
        self.assertEqual(v['cells'][6]['reason'],'preserved_batch2_coverage')
    def test_all_123_prefixes(self):
        m,a,d=fixture();base,_,_=fixtures.score_fixture()
        import pandas as pd
        days=[str(x.date()) for x in pd.bdate_range('2026-03-02',periods=123)]
        m.update(dates=days[:-1],ends=days[1:],symbols=[m['symbols'][0][:] for _ in range(122)])
        a=np.tile(a[:1],(122,1))
        arrays={k:(np.tile(v[:1],(122,10))[:,:49] if v.ndim==2 else np.tile(v[:1],122)) for k,v in base.items()}
        expected=r.features_oracle(m,a,a,a,a,d)
        ss,cc,_=r.score_oracle(arrays,m,a,a,a,a,expected['arrays']['earnings_expanded'])
        r.verify_prefixes(arrays,m,a,a,a,a,d,expected,ss,cc)




def source_package_fixture(admitted_count=5):
    import pandas as pd
    reader=fixtures.MemoryReader();sessions=[str(d.date()) for d in pd.bdate_range('2026-01-01','2027-12-31')]
    events=[];raws=[];reviews=[];contexts=[];acc=[];catalog=[];dependencies={}
    for i,symbol in enumerate(['MU']*2+['PEP']*7):
        e,c,rr,rawidentity=source_fixture(symbol);release=str(pd.Timestamp('2026-02-02')+pd.Timedelta(days=7*i))[:10]
        available=r.first_session(release,sessions);key=symbol+'_'+release
        src=dict(path='synthetic/'+key+'.html',url='https://example.invalid/'+key,receipt_path='synthetic/'+key+'.json')
        src['sha256']=reader.add(src['path'],{'text':'synthetic accounting basis'})['sha256'];src['receipt_sha256']=reader.add(src['receipt_path'],{'source':key})['sha256']
        dependencies.update({src['path']:src['sha256'],src['receipt_path']:src['receipt_sha256']})
        e.update(event_id=key,publication_date=release,available_from=available,source=deepcopy(src),clock_review=dict(first_pinned_session=available,verified=True))
        e['identity'].update(identity_valid_from=available)
        for cell in e['fields'].values():cell.update(source=deepcopy(src),available_from=available)
        c.update(event_id=key,publication_date=release,available_session=available,source=deepcopy(src),evidence_path=src['path'],evidence_sha256=src['sha256'],source_url=src['url'],fiscal_quarter=4,context_status='synthetic')
        rr.update(event_id=key,release_date=release,available_from=available,source=deepcopy(src))
        rr['accounting_review'].update(event_id=key,source=deepcopy(src),evidence=[dict(path=src['path'],sha256=src['sha256'],locator=dict(start=0,end=5),quote='basis')])
        e['source_review'].update(event_id=key,accounting_review=deepcopy(rr['accounting_review']))
        reasons=[]
        if i>=admitted_count:
            reasons=['retained_context_hold:synthetic_uncertainty'];e['context_holds'].append('synthetic_uncertainty');c['context_holds'].append('synthetic_uncertainty');rr['retained_context_holds'].append('synthetic_uncertainty')
        e['ni_assets_admission']=dict(admitted=not reasons,reasons=reasons,definition=r.DEFINITION)
        raw={k:deepcopy(v) for k,v in e.items() if k not in ('ni_assets_admission','clock_review','source_review')};raw['identity']=deepcopy(rawidentity)
        raws.append(raw);events.append(e);reviews.append(rr);contexts.append(c);acc.append(rr['accounting_review'])
        catalog.append({**{k:deepcopy(c[k]) for k in r.CATALOG_KEYS},'fiscal_quarter':4,'source_kind':'issuer_original_release_full_bytes'})
    for symbol,count in (('MU',5),('PEP',9)):
        for j in range(count):
            i=len(contexts);original=i<16
            c=dict(event_id=symbol+'_unprocessed_'+str(j),symbol=symbol,fiscal_year=2026,publication_date='2027-01-04',available_session='2027-01-05',available_session_verified=True,fiscal_quarter=4,official_actual_period_end='2026-08-27' if symbol=='MU' else '2026-12-26',source={},source_kind='HTML_original' if original else 'index_observation_not_original_report',original_report_bytes_verified=original,context_status='unprocessed',context_holds=[])
            contexts.append(c)
            catalog.append(dict(c,source_kind='issuer_original_release_full_bytes' if original else 'public_index_observation_not_original_bytes'))
    projection=[{k:c[k] for k in r.CATALOG_KEYS} for c in contexts]
    cr=reader.add(r.BATCH4_CONTEXT,dict(events=contexts));dependencies[cr['path']]=cr['sha256']
    reader.add(r.CENSUS,dict(annual_documents=deepcopy(catalog)))
    pl=reader.add(r.BATCH4+'/PLAN.json',dict(records=raws,inputs=dependencies.copy(),code={},catalog_ref=cr,catalog_ids=[x['event_id'] for x in catalog],catalog_metadata=projection))
    ex=reader.add(r.BATCH4+'/EXTRACTIONS.json',dict(events=raws,plan_sha256=pl['sha256'],catalog_ids=[x['event_id'] for x in catalog],catalog_metadata=projection))
    dependencies.update({pl['path']:pl['sha256'],ex['path']:ex['sha256']})
    ac=reader.add(r.ACCOUNTING,dict(passed=True,stage='independent_field_specific_accounting_review',plan_sha256=pl['sha256'],extractions_sha256=ex['sha256'],input_hashes=dependencies.copy(),events=acc));dependencies[ac['path']]=ac['sha256']
    policy=dict(source_scope=dict(symbols=['MU','PEP'],no_automatic_admission=True,source_context=cr),fixed_catalog_metadata=projection,required_fields=list(r.FIELDS),known_time=dict(maximum_age_days=550),periods=dict(boundary_assumptions={s:dict(enabled=True,not_a_quoted_fact=True,assumption_id=r.ASSUMPTIONS[s],days_inclusive=[364,371]) for s in ('MU','PEP')}),preservation=dict(first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False))
    refs={'policy':reader.add(r.BATCH4_REFS['policy'],policy)};dependencies[refs['policy']['path']]=refs['policy']['sha256']
    review=dict(passed=True,plan_sha256=pl['sha256'],extractions_sha256=ex['sha256'],input_hashes=dependencies.copy(),source_hashes={},event_reviews=reviews,counters=dict(events=9,published_nonnull_values_verified=27,null_fields_reviewed=0,catalog_events=23,unprocessed_catalog_events=14,original_HTML_catalog_events=16,index_catalog_events=7))
    refs['review']=reader.add(r.BATCH4_REFS['review'],review);dependencies[refs['review']['path']]=refs['review']['sha256']
    batches={};oldref=dict(immutable=True)
    for key,num,admitted,symbol in (('batch1',26,17,'AMZN'),('batch2',19,14,'INTC'),('batch3',10,5,'KO')):
        ds=dict(events=[dict(event_id=key+str(i),symbol=symbol) for i in range(num)],latest_event_catalog=[dict(event_id=key+str(i)) for i in range(32)],original_four_issuer_reference=oldref)
        paths=getattr(r,'COVERAGE_REFS' if key=='batch1' else key.upper()+'_REFS')
        batches[key]={k:reader.add(paths[k],ds if k=='dataset' else {}) for k in ('dataset','admission','manifest','coverage')}
    data=dict(schema_version=1,method=r.METHOD,source_plan_sha256=pl['sha256'],policy_sha256=refs['policy']['sha256'],independent_review_sha256=refs['review']['sha256'],events=events,latest_event_catalog=[dict(row,clock_review=dict(first_pinned_session=r.first_session(row['publication_date'],sessions),verified=True),processed_in_batch=row['event_id'] in {x['event_id'] for x in events}) for row in catalog],original_four_issuer_reference=oldref,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,scores_computed=0,ratios_computed=0,returns_computed=0,new_strategy_trials=0)
    for key,num,admitted in (('batch1',26,17),('batch2',19,14),('batch3',10,5)):
        br=batches[key];ds=reader.json(br['dataset']['path'])
        data['prior_'+key+'_reference']=dict(path=br['dataset']['path'],sha256=br['dataset']['sha256'],admission_path=br['admission']['path'],admission_sha256=br['admission']['sha256'],manifest_path=br['manifest']['path'],manifest_sha256=br['manifest']['sha256'],coverage_path=br['coverage']['path'],coverage_sha256=br['coverage']['sha256'],events=num,admitted=admitted,held=num-admitted,annual_catalog_events=32,preserved_without_modification=True,records_canonical_sha256=r.canonical(ds['events']),catalog_canonical_sha256=r.canonical(ds['latest_event_catalog']))
    counts=dict(events=9,admitted=admitted_count,held=9-admitted_count,reviewed_nonnull_fields=27,reviewed_null_fields=0,annual_catalog_events=23,unprocessed_annual_catalog_events=14,original_events=44,original_admitted=31,original_held=13,prior_batch1_events=26,prior_batch1_admitted=17,prior_batch1_held=9,prior_batch2_events=19,prior_batch2_admitted=14,prior_batch2_held=5,prior_batch3_events=10,prior_batch3_admitted=5,prior_batch3_held=5)
    admission=dict(passed=True,stage='conditional_source_admission_not_strategy_registration',source_plan_sha256=pl['sha256'],policy_sha256=refs['policy']['sha256'],independent_review_sha256=refs['review']['sha256'],input_hashes=dependencies.copy(),actual_factors_calculated=0,actual_returns_calculated=0,new_strategy_trials=0,counts=counts)
    refs.update(dataset=reader.add(r.BATCH4_REFS['dataset'],data),admission=reader.add(r.BATCH4_REFS['admission'],admission),coverage=reader.add(r.BATCH4_REFS['coverage'],dict(counts=dict(months=122,stocks=73,stock_slots=3904))))
    refs['manifest']=reader.add(r.BATCH4_REFS['manifest'],dict(status='conditional_source_package_no_strategy_registration',input_hashes=dependencies.copy(),outputs={n+'.json':refs[k]['sha256'] for k,n in (('dataset','DATASET'),('admission','ADMISSION'),('coverage','COVERAGE'))}))
    plan=dict(coverage_sources=dict(batches,batch4=refs));pins={str(reader.path(k)):reader.sha(k) for k in reader.documents}
    b1,b2,b3=(reader.json(batches[k]['dataset']['path']) for k in ('batch1','batch2','batch3'))
    combined2=dict(schema_version=1,method=r.METHOD,events=b1['events']+b2['events'],latest_event_catalog=b1['latest_event_catalog']+b2['latest_event_catalog'])
    combined3=dict(schema_version=1,method=r.METHOD,events=combined2['events']+b3['events'],latest_event_catalog=combined2['latest_event_catalog']+b3['latest_event_catalog'])
    return reader,plan,pins,sessions,combined3,combined2,b1


class PackageTests(unittest.TestCase):
    def check(self,fixture):
        reader,plan,pins,sessions,combined3,combined2,b1=fixture
        with patch.object(r.prior152,'verify_coverage',return_value=(combined3,combined2,b1)):
            return r.verify_coverage(reader,plan,pins,{},sessions)
    def test_full_metadata_source_package(self):
        combined,third,second,first=self.check(source_package_fixture())
        self.assertEqual((len(combined['events']),len(combined['latest_event_catalog'])),(64,119))
        self.assertEqual([len(x['events']) for x in (third,second,first)],[55,45,26])
    def test_admission_count_is_recomputed_not_hardcoded(self):
        for count in (0,3,9):
            combined,*_=self.check(source_package_fixture(count))
            self.assertEqual(sum(e['ni_assets_admission']['admitted'] for e in combined['events'][-9:]),count)
    def test_source_hash_tamper_rejected(self):
        f=source_package_fixture();reader=f[0];key=next(k for k in reader.documents if k.endswith('.html'));reader.documents[key]={'tampered':True}
        with self.assertRaises(ValueError):self.check(f)
    def test_catalog_removed_rejected(self):
        f=source_package_fixture();f[0].documents[str(f[0].path(r.CENSUS))]['annual_documents'].pop()
        with self.assertRaises(ValueError):self.check(f)
    def test_prior_batch3_records_change_rejected(self):
        f=source_package_fixture();f[0].documents[str(f[0].path(r.BATCH3_REFS['dataset']))]['events'].pop()
        with self.assertRaises(ValueError):self.check(f)
    def test_all_four_source_kind_aliases_only(self):
        for kind,(label,original) in {'issuer_original_release_full_bytes':('HTML_original',True),'issuer_distributed_original_prnewswire_release':('HTML_original',True),'public_index_observation_not_original_bytes':('index_observation_not_original_report',False),'public_indexed_issuer_release_excerpt':('index_observation_not_original_report',False)}.items():
            raw=dict(event_id='synthetic',source_kind=kind)
            r.verify_catalog_context(raw,dict(raw,source_kind=label,original_report_bytes_verified=original))
            with self.assertRaises(ValueError):r.verify_catalog_context(raw,dict(raw,source_kind=label,original_report_bytes_verified=not original))
        with self.assertRaises(ValueError):r.verify_catalog_context(dict(source_kind='unknown'),{})
    def test_alias_cannot_cover_source_identity_drift(self):
        with self.assertRaises(ValueError):r.verify_catalog_context(dict(source_kind='issuer_original_release_full_bytes',event_id='a'),dict(source_kind='HTML_original',event_id='b',original_report_bytes_verified=True))


class FourthControlTests(unittest.TestCase):
    def cache_fixture(self):
        base={str(i):np.full((122,49),float(i)) for i in range(27)}
        a=np.full((122,49),np.nan);a[:,0]=.2;b=a.copy();b[:,1]=.1;c=b.copy();c[:,2]=.3;d=c.copy();d[:,3]=-.2
        masks=[np.isfinite(x) for x in (a,b,c,d)]
        cache=dict(base,earnings_original=a,original_available=masks[0],earnings_batch1=b,batch1_available=masks[1],earnings_batch2=c,batch2_available=masks[2],earnings_expanded=d,expanded_available=masks[3])
        prepared=dict(earnings_batch3=d.copy(),batch3_available=masks[3].copy())
        return base,prepared,cache,a,masks[0],b,masks[1],c,masks[2]
    def test_152_exact_control(self):
        args=self.cache_fixture();r.verify_prior152_reference(*args)
        args[1]['earnings_batch3'][0,3]=-.1
        with self.assertRaises(ValueError):r.verify_prior152_reference(*args)
    def test_152_original_control_corruption(self):
        args=self.cache_fixture();args[2]['earnings_batch2']=args[2]['earnings_batch2'].copy();args[2]['earnings_batch2'][0,2]=.9
        with self.assertRaises(ValueError):r.verify_prior152_reference(*args)
    def test_152_mask_dtype_and_shape(self):
        for mode in ('dtype','shape','content'):
            args=self.cache_fixture();x=args[1]['batch3_available']
            args[1]['batch3_available']=x.astype(np.uint8) if mode=='dtype' else x.reshape(49,122) if mode=='shape' else ~x
            with self.assertRaises(ValueError):r.verify_prior152_reference(*args)
    def test_batch3_ko_preserved(self):
        m,a,d=fixture()
        for pool in m['symbols']:pool[6]='KO'
        b=a.copy();b[:,6]=-.25
        v=r.features_oracle(m,a,a,a,b,d)
        self.assertEqual(v['arrays']['earnings_expanded'][:,6].tobytes(),b[:,6].tobytes())
        self.assertEqual(v['cells'][6]['reason'],'preserved_batch3_coverage')
    def test_batch3_unrelated_change_rejected(self):
        m,a,d=fixture();b=a.copy();b[0,0]=.4
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,a,b,d)
    def test_batch3_cannot_precover_MU(self):
        m,a,d=fixture();b=a.copy();b[0,4]=.4
        with self.assertRaises(ValueError):r.features_oracle(m,a,a,a,b,d)
    def test_sixth_arm_full_scores_and_orders_checked(self):
        a,m,h=fixtures.score_fixture();scores,choices,_=r.score_oracle(a,m,h,h,h,h,h)
        orders=r.ordered_orders(m,scores,choices);orders[r.NAMES[-1]][0]['symbols'].reverse()
        with self.assertRaises(ValueError):r.verify_scores_and_orders(scores,orders,m,scores,choices)
    def test_bad_365_day_new_source_is_held(self):
        e,c,review,ident=source_fixture('PEP');p=e['fields'][r.FIELDS[0]]['period'];p['days_inclusive']=365
        self.assertIn('invalid_current_annual_period',r.source_gate(e,c,SourceGateTests.sessions,review,ident)['reasons'])
    def test_new_source_index_is_not_admitted(self):
        e,c,review,ident=source_fixture();c.update(source_kind='index_observation_not_original_report',original_report_bytes_verified=False)
        with self.assertRaises(ValueError):r.source_gate(e,c,SourceGateTests.sessions,review,ident)


class AnnualAndPriorChainTests(unittest.TestCase):
    def test_only_364_371_days_have_new_signal(self):
        for n in (330,350,364,365,366,370,371,372,400):
            e=event();end=date(2025,12,31);start=end-timedelta(days=n-1)
            p=e['fields'][r.FIELDS[0]]['period'];p.update(period_start=str(start),days_inclusive=n)
            e['fields'][r.FIELDS[2]]['as_of']=str(start-timedelta(days=1))
            answer=r.report_oracle(e,'2026-03-02')
            self.assertIs(answer['available'],n in (364,371))
            if n not in (364,371):self.assertEqual(answer['reason'],'invalid_annual_period')
    def chain(self):
        reader=fixtures.MemoryReader()
        plan=dict(legacy149={'binding':'original'},coverage_sources={k:{'synthetic':k} for k in ('batch1','batch2','batch3','batch4')},inputs={k:{'path':k,'sha256':'c'*64} for k in r.INPUTS[1:]},raw_source_hashes={'synthetic_raw':'a'*64})
        cache={name:reader.add(r.PRIOR152+'/'+name,{'synthetic':name})['sha256'] for name in r.CACHE_FILES}
        cref=reader.add(r.PRIOR152_REFS['cache'],cache)
        ancestor=dict(legacy149=plan['legacy149'],coverage_sources={k:plan['coverage_sources'][k] for k in ('batch1','batch2','batch3')},round=152,status='REGISTERED',hashes={},inputs=plan['inputs'],raw_source_hashes=plan['raw_source_hashes'],prior151={'frozen':'151-chain'})
        pref=reader.add(r.PRIOR152_REFS['plan'],ancestor)
        outcome=dict(plan_sha256=pref['sha256'],cache_sha256=cref['sha256'],passed=True,stage='evaluate',artifact_hashes={r.PRIOR152+'/'+k:v for k,v in cache.items()},orders_sha256=cache['PRETRAIN_ORDERS.json'],source_hashes={})
        oref=reader.add(r.PRIOR152_REFS['outcome'],outcome)
        done=dict(round=152,plan_sha256=pref['sha256'],independent_outcome_sha256=oref['sha256'])
        dref=reader.add(r.PRIOR152_REFS['completion'],done)
        plan['prior152']=dict(plan=pref,cache=cref,outcome=oref,completion=dref)
        pins={str(reader.path(k)):reader.sha(k) for k in reader.documents}
        return reader,plan,pins,ancestor
    def verify(self,f):
        reader,plan,pins,p=f
        with patch.object(r,'PRIOR152_PLAN',plan['prior152']['plan']['sha256']),patch.object(r,'PRIOR152_DONE',plan['prior152']['completion']['sha256']),patch.object(r.prior152,'verify_prior151',return_value={}) as nested:
            answer=r.verify_prior152(reader,plan,pins)
            self.assertEqual(answer,p)
            self.assertEqual(nested.call_args.args[1]['prior151'],p['prior151'])
            return answer
    def test_complete_prior152_chain_uses_nested151(self):self.verify(self.chain())
    def test_prior152_raw_source_drift_rejected(self):
        f=self.chain();f[1]['raw_source_hashes']={'different':'b'*64}
        with self.assertRaises(ValueError):self.verify(f)
    def test_prior152_source_package_drift_rejected(self):
        f=self.chain();f[1]['coverage_sources']['batch3']={'different':1}
        with self.assertRaises(ValueError):self.verify(f)
    def test_prior152_cache_hash_drift_rejected(self):
        f=self.chain();f[0].documents[str(f[0].path(r.PRIOR152+'/FEATURE_AUDIT.json'))]={'different':1}
        with self.assertRaises(ValueError):self.verify(f)
    def test_prior152_missing_registered_cache_pin_rejected(self):
        f=self.chain();f[2].pop(str(f[0].path(r.PRIOR152+'/DATA.npz')))
        with self.assertRaises(ValueError):self.verify(f)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if args.receipt:
        if not result.wasSuccessful():raise SystemExit('Failed tests cannot issue prereview')
        if (r.ROOT/r.STUDY/'PLAN.json').exists():raise SystemExit('Already registered')
        receipt=r.synthetic_receipt(result)
        with args.receipt.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
