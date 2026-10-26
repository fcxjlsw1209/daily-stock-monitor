"""Synthetic source-only fixtures; no actual research inputs or returns."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import peer239_source_independent as audit


def policy_fixture():
    rows=[]
    for name,(benchmark,publication,start,end) in audit.RULES.items():
        rows.append(dict(symbol=name,benchmark=benchmark,public_date=publication,
            valid_from=start,valid_until_exclusive=end,source_file='synthetic/evidence.json',
            source_ids=['synthetic_'+name]))
    return dict(schema='ownsector238_repair_source_policy_v1',missing_mapping_rules=rows,
        communication_benchmark=dict(old='XLC',new='VOX',price_field='adjClose',fill_or_splice=False))


def fixture():
    axis=pd.bdate_range('2015-01-01','2017-02-01');calendar=[str(d.date()) for d in axis]
    dates=['2016-08-17','2016-08-31','2016-09-30'];names=['A','B','C','D','E','FUND']
    meta=dict(dates=dates,symbols=[names.copy() for _ in dates],etfs=['FUND'])
    adj=pd.DataFrame(100.,index=axis,columns=names)
    sectors=pd.DataFrame('Technology',index=axis,columns=names,dtype=object)
    sectors[['D','E']]='Healthcare';sectors['FUND']='Never read ETF classification'
    keys={'A':'issuer_c','B':'issuer_a','C':'issuer_b','D':'issuer_d','E':'issuer_e','FUND':object()}
    return dict(adj=adj,sectors=sectors),meta,calendar,keys,policy_fixture()


def physical(args,n):
    panel,meta,calendar,issuers,policy=deepcopy(args)
    mm={k:(v[:n] if k in ('dates','symbols','ends') else v) for k,v in meta.items()}
    if n:
        last=meta['dates'][n-1]
        panel={k:v.loc[:last].copy() for k,v in panel.items()}
        calendar=calendar[:calendar.index(last)+2]
    else:
        panel={k:v.iloc[:0].copy() for k,v in panel.items()};calendar=[]
    return panel,mm,calendar,issuers,policy


def cell(report,i=1,j=0):
    return next(x for x in report['records'] if x['query_index']==i and x['column_index']==j)


def endpoints(args,row=1):
    t=args[2].index(args[1]['dates'][row])
    return [pd.Timestamp(args[2][t-252]),pd.Timestamp(args[2][t-42])]


def long_fixture():
    panel,meta,_,keys,policy=fixture()
    axis=pd.bdate_range('2015-01-01','2026-10-02');calendar=[str(d.date()) for d in axis]
    last={}
    for d in axis:last[str(d.to_period('M'))]=str(d.date())
    dates=['2016-08-17']+[v for k,v in sorted(last.items()) if '2016-08'<=k<='2026-08']
    names=meta['symbols'][0]
    meta=dict(dates=dates,symbols=[names.copy() for _ in dates],etfs=['FUND'])
    panel=dict(adj=pd.DataFrame(100.,index=axis,columns=names),
        sectors=pd.DataFrame('Technology',index=axis,columns=names,dtype=object))
    panel['sectors'][['D','E']]='Healthcare'
    return panel,meta,calendar,keys,policy


class IndependentSourceTests(unittest.TestCase):
    def test_known_full_stock_scope_counts_and_peer_order(self):
        args=fixture();r=audit.inspect(*args)
        self.assertEqual(r['counts'],dict(events=3,stock_cells=15,distinct_stock_symbols=5,
            partial_stock_cells=5,regular_stock_cells=10,issuer_missing_cells=0,duplicate_issuer_cells=0,
            sector_missing_cells=0,structural_supported_cells=9,structural_unavailable_cells=6,
            endpoint_checks=20,valid_endpoints=20,missing_endpoints=0,invalid_endpoints=0,gap_cells=0))
        self.assertEqual(cell(r)['peer_indices'],[1,2]);self.assertEqual(cell(r,j=1)['peer_indices'],[2,0])
        self.assertEqual(cell(r,j=3)['peer_issuer_count'],1)
        self.assertFalse(cell(r,j=3)['structural_peer_support']);self.assertTrue(r['admitted'])
        check=audit.check_sources(*args,r)
        self.assertTrue(all(check[k] is True for k in audit.FLAGS))
        self.assertFalse(check['producer_modules_imported']);self.assertFalse(check['real_source_files_loaded'])
        self.assertFalse(check['issuer_publication_evidence_checked_by_this_api'])
        self.assertEqual(check['checked_endpoints'],20)

    def test_partial_identity_and_sector_still_required_but_no_prices(self):
        args=physical(fixture(),1);args[0]['adj']=args[0]['adj'].iloc[:0]
        r=audit.inspect(*args);self.assertTrue(r['admitted'])
        self.assertEqual(r['counts']['endpoint_checks'],0)
        self.assertTrue(all(x['endpoints']==x['source_positions']==x['source_dates']==[] for x in r['records']))
        args[3]['A']=None;r=audit.inspect(*args)
        self.assertEqual(cell(r,0)['gap_reasons'],['missing_issuer'])
        args[0]['sectors'].at[pd.Timestamp(args[1]['dates'][0]),'B']=None
        self.assertIn('missing_sector',cell(audit.inspect(*args),0,1)['gap_reasons'])

    def test_all_direct_stocks_ignore_incidental_eligibility_metadata(self):
        args=fixture();r=audit.inspect(*args)
        args[1]['eligible']=np.zeros((3,6),bool)
        after=audit.inspect(*args);audit.record_check(after,r)
        self.assertTrue(all(len(x['stock_indices'])==5 for x in r['events']))

    def test_duplicate_issuer_all_representatives_and_whole_issuer_exclusion(self):
        args=fixture();args[3]['B']=args[3]['C']='same_company'
        r=audit.inspect(*args);self.assertEqual(r['counts']['duplicate_issuer_cells'],6)
        self.assertEqual(cell(r)['peer_indices'],[1,2]);self.assertEqual(cell(r)['peer_issuer_count'],1)
        self.assertEqual(cell(r,j=1)['peer_indices'],[0]);self.assertEqual(cell(r,j=2)['peer_indices'],[0])
        self.assertIn('duplicate_issuer',cell(r,j=1)['gap_reasons'])
        self.assertEqual(r['status'],'SOURCE_GAPS')

    def test_issuer_missing_vs_malformed_and_unused_issuer_poison(self):
        for value in (None,):
            args=fixture();args[3]['A']=value
            self.assertEqual(cell(audit.inspect(*args))['gap_reasons'],['missing_issuer'])
        args=fixture();del args[3]['A'];self.assertFalse(audit.inspect(*args)['admitted'])
        for value in ('', ' spaced ', 2, True, [], {}):
            args=fixture();args[3]['A']=value
            with self.subTest(value=value),self.assertRaises(ValueError):audit.inspect(*args)
        args=fixture();args[3]['FUTURE_ONLY']=['malformed'];args[3]['FUND']=['not consumed']
        self.assertTrue(audit.inspect(*args)['admitted'])

    def test_missing_sector_does_not_turn_into_structural_admission(self):
        args=fixture();day=pd.Timestamp(args[1]['dates'][1]);args[0]['sectors'].at[day,'A']=None
        r=audit.inspect(*args);row=cell(r)
        self.assertEqual(row['gap_reasons'],['missing_sector']);self.assertEqual(row['mapping_repair'],'unresolved')
        self.assertEqual(row['peer_indices'],[]);self.assertEqual(row['peer_issuer_count'],0)
        self.assertFalse(row['structural_peer_support']);self.assertFalse(r['admitted'])
        self.assertEqual(len(row['endpoints']),2)

    def test_foreign_precedence_and_xlc_label_normalization_no_etf_prices(self):
        args=fixture();args[1]['symbols']=[['TSM']+x[1:] for x in args[1]['symbols']]
        args[0]['adj']=args[0]['adj'].rename(columns={'A':'TSM'})
        args[0]['sectors']=args[0]['sectors'].rename(columns={'A':'TSM'});args[0]['sectors']['TSM']='Healthcare'
        args[3]['TSM']=args[3].pop('A');r=audit.inspect(*args)
        self.assertEqual(cell(r)['sector'],'XLK');self.assertEqual(cell(r)['mapping_origin'],'frozen_FOREIGN_override')
        args=fixture();args[0]['sectors'][['A','B','C']]='Communication Services';r=audit.inspect(*args)
        self.assertEqual(cell(r)['original_sector'],'XLC');self.assertEqual(cell(r)['sector'],'VOX')
        self.assertEqual(cell(r)['mapping_repair'],'full_period_communication_proxy')
        self.assertNotIn('VOX',args[0]['adj'].columns);self.assertNotIn('SPY',args[0]['adj'].columns)
        self.assertTrue(r['admitted'])

    def test_fixed_six_dated_repairs_and_nonmissing_precedence(self):
        rules=audit.policy_rules(policy_fixture())
        for name,(benchmark,_,start,end) in audit.RULES.items():
            self.assertEqual(audit.resolve(None,name,pd.Timestamp(start),rules),
                (benchmark,'dated_business_proxy',['synthetic_'+name]))
            before=pd.Timestamp(start)-pd.Timedelta(days=1)
            self.assertEqual(audit.resolve(None,name,before,rules),(None,'unresolved',None))
            if end:self.assertEqual(audit.resolve(None,name,pd.Timestamp(end),rules),(None,'unresolved',None))
            self.assertEqual(audit.resolve('XLP',name,pd.Timestamp(start),rules),('XLP','unchanged',None))
        self.assertEqual(audit.resolve('XLC','GOOGL',pd.Timestamp('2018-09-28'),rules),
            ('VOX','full_period_communication_proxy',None))

    def test_policy_rule_source_schema_tamper_rejected(self):
        for mutation in (
            lambda p:p.update(schema='wrong'),lambda p:p['missing_mapping_rules'].pop(),
            lambda p:p['missing_mapping_rules'][0].update(valid_from='2023-03-10'),
            lambda p:p['missing_mapping_rules'][0].update(benchmark='XLK'),
            lambda p:p['missing_mapping_rules'][0].update(source_ids=[]),
            lambda p:p['communication_benchmark'].update(fill_or_splice=True),
        ):
            args=fixture();mutation(args[4])
            with self.assertRaises(ValueError):audit.inspect(*args)

    def test_missing_and_invalid_endpoint_reasons(self):
        cases=[(None,'missing','missing_value'),(pd.NA,'missing','missing_value'),
            (float('nan'),'missing','missing_value'),(True,'invalid','invalid_boolean'),
            ('100','invalid','invalid_nonreal'),([100],'invalid','invalid_nonreal'),
            (float('inf'),'invalid','invalid_nonfinite'),(-1.,'invalid','invalid_nonpositive'),
            (0.,'invalid','invalid_nonpositive'),(10**1000,'invalid','invalid_numeric_conversion')]
        for value,status,reason in cases:
            args=fixture();args[0]['adj']=args[0]['adj'].astype(object);s,_=endpoints(args)
            args[0]['adj'].at[s,'A']=value;r=audit.inspect(*args);row=cell(r)
            self.assertEqual(row['endpoints'][0]['status'],status)
            self.assertEqual(row['endpoints'][0]['reason'],reason)
            self.assertIn('endpoint:security_start:'+reason,row['gap_reasons'])
            self.assertFalse(r['admitted'])

    def test_endpoint_rows_columns_no_fill_and_shortfall_still_checked(self):
        args=fixture();s,_=endpoints(args);args[0]['adj']=args[0]['adj'].drop(index=s)
        row=cell(audit.inspect(*args));self.assertEqual(row['endpoints'][0]['reason'],'missing_row')
        args=fixture();args[0]['adj']=args[0]['adj'].drop(columns='D')
        row=cell(audit.inspect(*args),j=3);self.assertFalse(row['structural_peer_support'])
        self.assertEqual([x['reason'] for x in row['endpoints']],['missing_column','missing_column'])

    def test_future_recent_and_nonendpoint_prices_are_not_consumed(self):
        args=fixture();base=audit.inspect(*args);used=set()
        for i in (1,2):used.update(endpoints(args,i))
        args[0]['adj']=args[0]['adj'].astype(object)
        args[0]['adj'].loc[~args[0]['adj'].index.isin(used),:]='unused poison'
        args[0]['adj']['FUND']=['ETF poison']*len(args[0]['adj'])
        args[0]['sectors']['FUND']=[['ETF unhashable'] for _ in args[0]['sectors'].index]
        audit.record_check(audit.inspect(*args),base)

    def test_mapping_exact_query_date_not_future_or_forward_fill(self):
        args=fixture();day=pd.Timestamp(args[1]['dates'][1]);args[0]['sectors']=args[0]['sectors'].drop(index=day)
        r=audit.inspect(*args);self.assertIsNone(cell(r)['sector'])
        args=fixture();day=pd.Timestamp(args[1]['dates'][1]);args[0]['sectors'].at[day,'A']='Healthcare'
        self.assertEqual(cell(audit.inspect(*args))['peer_indices'],[3,4])
        args[0]['sectors'].at[day,'A']=['malformed']
        with self.assertRaises(ValueError):audit.inspect(*args)

    def test_metadata_calendar_structures_and_named_partial(self):
        args=fixture();args[1]['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):audit.inspect(*args)
        args=physical(fixture(),2);args[2].pop()
        with self.assertRaises(ValueError):audit.inspect(*args)
        args=fixture();args[1]['symbols'][1][1]='A'
        with self.assertRaises(ValueError):audit.inspect(*args)
        args=fixture();args[0]['adj']=args[0]['adj'].iloc[::-1]
        with self.assertRaises(ValueError):audit.inspect(*args)
        args=fixture();args[2].insert(0,'2014-12-28')
        with self.assertRaises(ValueError):audit.inspect(*args)
        args=fixture();args[1]['dates'][0]='2016-08-18'
        with self.assertRaises(ValueError):audit.inspect(*args)
        for bad in (-1,4,True,1.0):
            with self.assertRaises(ValueError):audit.inspect(*fixture(),limit=bad)

    def test_zero_and_all_small_physical_logical_prefixes(self):
        args=fixture();full=audit.inspect(*args)
        for n in range(4):
            logical=audit.inspect(*args,limit=n);short=audit.inspect(*physical(args,n))
            audit.record_check(short,logical)
            self.assertEqual(short['events'],full['events'][:n])
            self.assertEqual(short['records'],[r for r in full['records'] if r['query_index']<n])
        self.assertTrue(audit.inspect(*physical(args,0))['admitted'])

    def test_future_metadata_identity_classification_poison(self):
        args=fixture();base=audit.inspect(*args,limit=2)
        args[1]['dates'][2]='not a date';args[1]['symbols'][2]=['BAD']
        args[3]['BAD']=[]
        args[0]['sectors'].loc['2016-09-30','A']=['future poison']
        audit.record_check(audit.inspect(*args,limit=2),base)
        with self.assertRaises(ValueError):audit.inspect(*args)

    def test_all_122_source_events_five_physical_prefixes(self):
        args=long_fixture();full=audit.inspect(*args)
        self.assertEqual(len(full['events']),122)
        for n in (0,1,53,88,122):
            short=audit.inspect(*physical(args,n));logical=audit.inspect(*args,limit=n)
            audit.record_check(short,logical)
            self.assertEqual(short['records'],[r for r in full['records'] if r['query_index']<n])
            self.assertEqual(short['counts']['endpoint_checks'],2*5*max(0,n-1))

    def test_receipt_rejects_record_peer_count_scope_and_flag_tamper(self):
        args=fixture();full=audit.inspect(*args)
        for mutation in (
            lambda r:r['counts'].update(stock_cells=0),lambda r:r.update(momentum_computed=True),
            lambda r:r['records'][0].update(peer_indices=[1]),
            lambda r:r['records'][0].update(peer_issuer_count=9),
            lambda r:r['events'][1].update(source_positions=[0,210]),
            lambda r:r['records'][5]['endpoints'][0].update(status='invalid'),
            lambda r:r['records'].pop(),lambda r:r.update(gaps=[r['records'][0]]),
        ):
            wrong=deepcopy(full);mutation(wrong)
            with self.assertRaises(ValueError):audit.check_sources(*args,wrong)

    def test_inputs_unmodified_and_gap_records_detached(self):
        args=fixture();args[3]['A']=None
        saved=deepcopy(args);r=audit.inspect(*args)
        for k in args[0]:pd.testing.assert_frame_equal(args[0][k],saved[0][k])
        self.assertEqual(args[1],saved[1]);self.assertEqual(args[2],saved[2]);self.assertEqual(args[4],saved[4])
        self.assertIsNone(args[3]['A']);r['gaps'][0]['gap_reasons'].append('mutated')
        self.assertNotIn('mutated',r['records'][0]['gap_reasons'])

    def test_no_producer_or_strategy_import_no_ratio_and_dependencies_pinned(self):
        text=Path(audit.__file__).read_text();tree=ast.parse(text)
        imports=[x.module for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]
        imports += [a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]
        self.assertNotIn('peer239_sources',imports);self.assertNotIn('peer239_source_host',imports)
        self.assertFalse(any(isinstance(x,ast.Div) for x in ast.walk(tree)))
        self.assertIn(Path(audit.frozen.__file__).resolve(),audit.dependency_files())
        with patch.object(audit,'FROZEN_SHA','0'*64):
            with self.assertRaises(ValueError):audit.inspect(*fixture())


class BilateralTests(unittest.TestCase):
    def test_unique_both_fixture_full_report_all_small_prefixes_and_invalid_sources(self):
        import peer239_sources as producer
        import test_peer239_sources as other
        for args in (fixture(),other.fixture()):
            full=audit.inspect(*args)
            audit.record_check(producer.inspect(*args),full)
            for n in range(len(args[1]['dates'])+1):
                logical=audit.inspect(*args,limit=n)
                audit.record_check(producer.inspect(*args,limit=n),logical)
                short=physical(args,n)
                audit.record_check(producer.inspect(*short),audit.inspect(*short))
        for value in (None,True,'100',float('inf'),0.):
            args=fixture();args[0]['adj']=args[0]['adj'].astype(object)
            args[0]['adj'].at[endpoints(args)[0],'A']=value
            audit.record_check(producer.inspect(*args),audit.inspect(*args))


if __name__=='__main__':unittest.main(verbosity=2)
