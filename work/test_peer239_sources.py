"""Offline synthetic-only coverage of issuer/sector/endpoint source admission."""
from copy import deepcopy
from datetime import date,timedelta
import json
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import peer239_sources as subject


def policy_fixture():
    rules=[('AZN','XLV','2023-03-11',None),('BUD','XLP','2016-03-15',None),
        ('CHL','VOX','2016-04-28',None),('GOOGL','XLK','2016-02-12','2018-09-24'),
        ('PDD','XLY','2020-04-25',None),('TSLA','XLY','2020-02-14',None)]
    return dict(schema='ownsector238_repair_source_policy_v1',missing_mapping_rules=[
        dict(symbol=s,benchmark=b,public_date=str(date.fromisoformat(start)-timedelta(days=1)),valid_from=start,
            valid_until_exclusive=end,source_file='synthetic_evidence.json',source_ids=['synthetic_'+s])
        for s,b,start,end in rules],communication_benchmark=dict(old='XLC',new='VOX',price_field='adjClose',fill_or_splice=False))


def fixture():
    dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31']
    axis=pd.bdate_range('2015-01-01','2017-01-03');symbols=['AAA','BBB','CCC','DDD','QQQ']
    panel=dict(adj=pd.DataFrame(100.,index=axis,columns=symbols),sectors=pd.DataFrame('Technology',index=axis,columns=symbols))
    panel['sectors']['DDD']='Healthcare'
    meta=dict(dates=dates,symbols=[symbols.copy() for _ in dates],etfs=['QQQ'])
    issuers={'AAA':'issuer-z','BBB':'issuer-a','CCC':'issuer-m','DDD':'issuer-d','QQQ':'malformed unused ETF key'}
    return panel,meta,[str(d.date()) for d in axis],issuers,policy_fixture()


def inspect(args,limit=None):
    return subject.inspect(*args,limit=limit)


def one(args,i=1,symbol='AAA'):
    return next(r for r in inspect(args)['records'] if r['query_index']==i and r['symbol']==symbol)


class SourceTests(unittest.TestCase):
    def test_complete_schema_counts_and_peer_order(self):
        report=inspect(fixture());c=report['counts']
        self.assertTrue(report['admitted']);self.assertEqual(report['status'],'ADMITTED')
        self.assertEqual(c,dict(events=4,stock_cells=16,distinct_stock_symbols=4,partial_stock_cells=4,regular_stock_cells=12,issuer_missing_cells=0,duplicate_issuer_cells=0,sector_missing_cells=0,structural_supported_cells=12,structural_unavailable_cells=4,endpoint_checks=24,valid_endpoints=24,missing_endpoints=0,invalid_endpoints=0,gap_cells=0))
        self.assertEqual(one(fixture())['peer_indices'],[1,2]);self.assertEqual(one(fixture(),symbol='BBB')['peer_indices'],[2,0])
        self.assertEqual(tuple(report['records'][0]),subject.RECORD_KEYS);self.assertEqual(tuple(report['events'][0]),subject.EVENT_KEYS)
        self.assertEqual(tuple(c),subject.COUNT_FIELDS);json.dumps(report,allow_nan=False)
        for k in ('momentum_computed','rankings_computed','orders_computed','returns_computed','strict_pit'):self.assertIs(report[k],False)
        self.assertEqual(report['model_fits'],0)

    def test_etfs_no_identity_mapping_or_price_reads(self):
        args=fixture();args[0]['adj']['QQQ']=np.nan;args[0]['sectors']['QQQ']=[[1]]*len(args[0]['sectors']);args[3]['QQQ']={}
        self.assertTrue(inspect(args)['admitted']);self.assertTrue(all(r['symbol']!='QQQ' for r in inspect(args)['records']))

    def test_all_direct_stock_cells_ignore_any_eligibility(self):
        args=fixture();args[1]['eligible']=[[False]*5]*4;args[0]['adj']['DDD']=np.nan
        r=inspect(args);self.assertEqual(r['counts']['missing_endpoints'],6);self.assertEqual(r['counts']['gap_cells'],3)
        self.assertEqual(len(r['records']),16)

    def test_partial_inspects_identity_and_mapping_without_price(self):
        args=subject.physical_prefix(*fixture(),1);args[0]['adj']=args[0]['adj'].iloc[:0]
        args[3].pop('AAA');args[0]['sectors']['BBB']='Unknown';r=inspect(args)
        self.assertEqual(r['counts']['endpoint_checks'],0);self.assertEqual(r['counts']['gap_cells'],2)
        self.assertEqual(r['events'][0]['source_positions'],[]);self.assertTrue(all(x['endpoints']==[] for x in r['records']))

    def test_first_regular_row_not_bypassed(self):
        args=list(fixture());args[1]=dict(args[1],dates=args[1]['dates'][1:],symbols=args[1]['symbols'][1:])
        r=inspect(args);self.assertEqual(r['counts']['partial_stock_cells'],0);self.assertEqual(r['counts']['endpoint_checks'],24)

    def test_foreign_override_then_panel_then_repair(self):
        args=fixture();args[1]['symbols']=[['ASML','GOOGL','CHL','BUD','QQQ'] for _ in args[1]['dates']]
        args[0]['adj'].columns=args[1]['symbols'][0];args[0]['sectors'].columns=args[1]['symbols'][0]
        args[0]['sectors']['ASML']='Healthcare';args[0]['sectors']['GOOGL']='Unknown';args[0]['sectors']['CHL']='Unknown';args[0]['sectors']['BUD']='Unknown'
        args[3].update({s:'issuer-'+s for s in args[1]['symbols'][0] if s!='QQQ'})
        rr=inspect(args)['records'][:4]
        self.assertEqual([r['sector'] for r in rr],['XLK','XLK','VOX','XLP'])
        self.assertEqual(rr[0]['mapping_origin'],'frozen_FOREIGN_override');self.assertEqual(rr[0]['mapping_repair'],'unchanged')
        self.assertEqual(rr[1]['mapping_repair'],'dated_business_proxy');self.assertEqual(rr[1]['mapping_evidence'],['synthetic_GOOGL'])

    def test_xlc_all_dates_normalizes_but_technology_does_not(self):
        args=fixture();args[0]['sectors']['AAA']='Communication Services';rr=inspect(args)['records']
        self.assertTrue(all(r['sector']=='VOX' and r['original_sector']=='XLC' and r['mapping_repair']=='full_period_communication_proxy' for r in rr if r['symbol']=='AAA'))
        self.assertTrue(all(r['sector']=='XLK' for r in rr if r['symbol'] in ('BBB','CCC')))
        self.assertNotIn('VOX',args[0]['adj'].columns)

    def test_dated_repair_boundary_no_earlier_or_later_inference(self):
        args=fixture();axis=pd.bdate_range('2014-01-01','2024-01-03');days=['2016-01-29','2016-02-29','2018-08-31','2018-09-28']
        args[0]['adj']=pd.DataFrame(100.,index=axis,columns=['GOOGL']);args[0]['sectors']=pd.DataFrame('Unknown',index=axis,columns=['GOOGL'])
        args[1]['dates']=days;args[1]['symbols']=[['GOOGL'] for _ in days];args[3]['GOOGL']='issuer-google'
        args=list(args);args[2]=[str(d.date()) for d in axis]
        self.assertEqual([r['sector'] for r in inspect(args)['records']],[None,'XLK','XLK',None])

    def test_nonmissing_mapping_never_overwritten_by_rule(self):
        args=fixture();args[1]['symbols']=[['GOOGL','BBB','CCC','DDD','QQQ'] for _ in range(4)]
        args[0]['adj']=args[0]['adj'].rename(columns={'AAA':'GOOGL'});args[0]['sectors']=args[0]['sectors'].rename(columns={'AAA':'GOOGL'})
        args[0]['sectors']['GOOGL']='Healthcare';args[3]['GOOGL']='issuer-google'
        r=one(args,symbol='GOOGL');self.assertEqual(r['sector'],'XLV');self.assertEqual(r['mapping_repair'],'unchanged')

    def test_missing_sector_no_forward_fill_and_full_gap_collection(self):
        args=fixture();args[0]['sectors']=args[0]['sectors'].drop(pd.Timestamp(args[1]['dates'][1]));r=inspect(args)
        self.assertEqual(r['counts']['sector_missing_cells'],4);self.assertEqual(r['counts']['gap_cells'],4)
        self.assertTrue(all(g['query_index']==1 and g['gap_reasons']==['missing_sector'] for g in r['gaps']))

    def test_missing_issuer_is_gap_and_no_focal_peers(self):
        args=fixture();args[3].pop('BBB');args[3]['CCC']=None;r=inspect(args)
        self.assertEqual(r['counts']['issuer_missing_cells'],8)
        self.assertEqual(one(args,symbol='BBB')['peer_indices'],[]);self.assertEqual(one(args)['peer_issuer_count'],0)

    def test_duplicate_wholeissuer_marks_all_representatives(self):
        args=fixture();args[3]['BBB']='issuer-z';r=inspect(args)
        self.assertEqual(r['counts']['duplicate_issuer_cells'],8);self.assertFalse(r['admitted'])
        a=one(args);b=one(args,symbol='BBB');c=one(args,symbol='CCC')
        self.assertEqual(a['peer_indices'],[2]);self.assertEqual(b['peer_indices'],[2]);self.assertEqual(c['peer_indices'],[0,1])
        self.assertEqual(c['peer_issuer_count'],1);self.assertFalse(c['structural_peer_support'])

    def test_duplicate_peer_issuer_count_unique_no_representative_selection(self):
        args=fixture();args[0]['sectors']['DDD']='Technology';args[3]['DDD']=args[3]['BBB']
        r=one(args);self.assertEqual(r['peer_indices'],[1,3,2]);self.assertEqual(r['peer_issuer_count'],2);self.assertTrue(r['structural_peer_support'])
        self.assertFalse(inspect(args)['admitted'])

    def test_structural_unavailable_alone_still_admitted(self):
        args=fixture();args[0]['sectors']['AAA']='Energy';r=inspect(args)
        self.assertTrue(r['admitted']);self.assertEqual(r['counts']['structural_supported_cells'],0)
        self.assertEqual(r['counts']['structural_unavailable_cells'],16)

    def test_endpoint_offsets_exact_and_no_price_values_exported(self):
        args=fixture();r=one(args);pos=args[2].index(args[1]['dates'][1])
        self.assertEqual(r['source_positions'],[pos-252,pos-42]);self.assertEqual(r['source_dates'],[args[2][pos-252],args[2][pos-42]])
        self.assertEqual([p['role'] for p in r['endpoints']],['security_start','security_end'])
        self.assertTrue(all(set(p)=={'role','symbol','date','status','reason'} for p in r['endpoints']))

    def test_all_numeric_endpoint_taxonomy(self):
        cases=[(None,'missing','missing_value'),(pd.NA,'missing','missing_value'),(float('nan'),'missing','missing_value'),
            (True,'invalid','invalid_boolean'),(np.bool_(True),'invalid','invalid_boolean'),('100','invalid','invalid_nonreal'),
            (1+0j,'invalid','invalid_nonreal'),(float('inf'),'invalid','invalid_nonfinite'),(-float('inf'),'invalid','invalid_nonfinite'),
            (0.,'invalid','invalid_nonpositive'),(-1,'invalid','invalid_nonpositive'),(10**400,'invalid','invalid_numeric_conversion')]
        for value,status,reason in cases:
            with self.subTest(value=repr(value)):
                args=fixture();args[0]['adj']=args[0]['adj'].astype(object);pos=args[2].index(args[1]['dates'][1]);dd=pd.Timestamp(args[2][pos-252])
                args[0]['adj'].at[dd,'AAA']=value;r=one(args)
                self.assertEqual((r['endpoints'][0]['status'],r['endpoints'][0]['reason']),(status,reason))
                self.assertIn('endpoint:security_start:'+reason,r['gap_reasons'])

    def test_missing_rows_columns_are_gaps_without_fill(self):
        args=fixture();args[0]['adj']=args[0]['adj'].drop(columns='AAA');r=one(args)
        self.assertEqual([p['reason'] for p in r['endpoints']],['missing_column']*2)
        args=fixture();pos=args[2].index(args[1]['dates'][1]);args[0]['adj']=args[0]['adj'].drop(pd.Timestamp(args[2][pos-42]));r=one(args)
        self.assertEqual(r['endpoints'][1]['reason'],'missing_row')

    def test_all_gap_types_retained_sorted(self):
        args=fixture();args[3].pop('AAA');args[0]['sectors']['AAA']='Unknown';args[0]['adj']['AAA']=0.
        r=one(args);self.assertEqual(r['gap_reasons'],sorted(['missing_issuer','missing_sector','endpoint:security_start:invalid_nonpositive','endpoint:security_end:invalid_nonpositive']))
        rr=inspect(args);self.assertEqual(rr['gaps'],[r for r in rr['records'] if r['gap_reasons']])

    def test_prefixes_and_source_immutability(self):
        args=fixture();before=deepcopy(args);full=inspect(args)
        for n in range(5):
            logical=inspect(args,n);physical=inspect(subject.physical_prefix(*args,n))
            self.assertEqual(logical,physical);self.assertEqual(logical['records'],[r for r in full['records'] if r['query_index']<n])
            self.assertEqual(logical['events'],full['events'][:n])
        for k in args[0]:pd.testing.assert_frame_equal(args[0][k],before[0][k])
        self.assertEqual(args[1:],before[1:])

    def test_future_meta_sources_and_unused_issuer_values_not_consumed(self):
        args=fixture();expected=inspect(args,2);cutoff=pd.Timestamp(args[1]['dates'][1]);args[0]['adj']=args[0]['adj'].astype(object)
        args[0]['adj'].loc[args[0]['adj'].index>cutoff,:]='poison';args[0]['sectors'].loc[args[0]['sectors'].index>cutoff,:]='unknown'
        args[1]['dates'][2]='bad';args[1]['symbols'][2]=['bad'];args[3]['FUTURE']={}
        self.assertEqual(inspect(args,2),expected)

    def test_current_and_skip42_prices_not_consumed(self):
        args=fixture();args=subject.physical_prefix(*args,2);expected=inspect(args);args[0]['adj']=args[0]['adj'].astype(object)
        pos=args[2].index(args[1]['dates'][1]);cutoff=pd.Timestamp(args[2][pos-42]);args[0]['adj'].loc[args[0]['adj'].index>cutoff,:]='poison'
        self.assertEqual(inspect(args),expected)

    def test_no_arithmetic_on_valid_extreme_prices(self):
        args=fixture();args[0]['adj']['AAA']=1e308;args[0]['adj']['BBB']=1e-300
        self.assertTrue(inspect(args)['admitted'])

    def test_malformed_issuer_raises_but_absent_is_gap(self):
        for value in ('',' issuer ',True,1,{},[],1.5):
            args=fixture();args[3]['AAA']=value
            with self.subTest(value=repr(value)),self.assertRaises(ValueError):inspect(args)
        args=fixture();args[3][' bad ']='issuer'
        with self.assertRaises(ValueError):inspect(args)

    def test_query_metadata_limit_and_policy_errors(self):
        for lim in (-1,True,5,1.,'1'):
            with self.subTest(limit=lim),self.assertRaises(ValueError):inspect(fixture(),lim)
        mutations=[lambda a:a[1]['dates'].__setitem__(1,'2016-8-31'),lambda a:a[1]['dates'].__setitem__(1,'2016-08-30'),
            lambda a:a[1]['symbols'][1].__setitem__(1,'AAA'),lambda a:a[1]['symbols'][1].pop(),
            lambda a:a[1]['etfs'].append('QQQ'),lambda a:a[1]['dates'].__setitem__(2,'2016-08-17'),
            lambda a:a[4]['missing_mapping_rules'][0].__setitem__('benchmark','XLK')]
        for change in mutations:
            args=fixture();change(args)
            with self.assertRaises(ValueError):inspect(args)

    def test_bad_calendar_axes_and_missing_witness_raise(self):
        for mode in ('duplicate','reversed','weekend','missing_query','no_witness','empty'):
            args=list(fixture())
            if mode=='duplicate':args[2].insert(1,args[2][0])
            elif mode=='reversed':args[2]=args[2][::-1]
            elif mode=='weekend':args[2].insert(2,'2015-01-03')
            elif mode=='missing_query':args[2].remove(args[1]['dates'][1])
            elif mode=='no_witness':args[2]=args[2][:args[2].index(args[1]['dates'][-1])+1]
            else:args[2]=[]
            with self.subTest(mode=mode),self.assertRaises(ValueError):inspect(args)

    def test_bad_frames_rejected(self):
        for mode in ('duplicate','reverse','timezone','intraday','columns','notframe'):
            args=fixture();f=args[0]['adj']
            if mode=='duplicate':f=pd.concat([f.iloc[:1],f])
            elif mode=='reverse':f=f.iloc[::-1]
            elif mode=='timezone':f=f.tz_localize('UTC')
            elif mode=='intraday':f=f.copy();f.index=f.index+pd.Timedelta(hours=1)
            elif mode=='columns':f=f.copy();f.columns=['A']*len(f.columns)
            else:f=[]
            args[0]['adj']=f
            with self.subTest(mode=mode),self.assertRaises(ValueError):inspect(args)

    def test_unhashable_sector_raises_structure_error(self):
        args=fixture();args[0]['sectors'].at[pd.Timestamp(args[1]['dates'][0]),'AAA']=['bad']
        with self.assertRaises(ValueError):inspect(args)

    def test_no_frozen_loader_or_collector_invoked(self):
        with mock.patch.object(subject.original,'previous',side_effect=AssertionError('No host permitted')),mock.patch.object(subject.repair,'parse_vox',side_effect=AssertionError('No VOX data permitted')):
            self.assertTrue(inspect(fixture())['admitted'])


if __name__=='__main__':unittest.main(verbosity=2)
