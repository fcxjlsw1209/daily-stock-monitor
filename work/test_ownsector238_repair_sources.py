from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import ownsector238_source_audit as original
import ownsector238_repair_sources as repair


def fixture():
    calendar = pd.bdate_range('2015-01-01', '2016-09-02')
    names = ['T', 'BUD', 'GOOGL', 'QQQ']
    panel = {'adj':pd.DataFrame(100., index=calendar, columns=names+['SPY','XLC','XLP','XLK']),
             'sectors':pd.DataFrame(index=calendar, columns=names)}
    panel['sectors']['T'] = 'Communication Services'
    panel['adj']['XLC'] = np.nan
    meta = {'dates':['2016-08-17','2016-08-31'], 'symbols':[names.copy(),names.copy()], 'etfs':['QQQ']}
    eligible = np.ones((2,4),dtype=bool)
    def mapper(day, symbols, sectors, etfs):
        return {s: original._mapped(pd.Timestamp(day),s,sectors,etfs) for s in symbols}
    source = original.inspect_sources(panel, meta, [str(x.date()) for x in calendar], eligible, mapper)
    policy = json.loads((Path(__file__).parent/'ownsector238_repair_source_policy.json').read_text())
    vox = pd.DataFrame({'VOX':100.}, index=calendar)
    return source,panel['adj'],vox,policy


class RepairTests(unittest.TestCase):
    def test_fills_declared_gaps_and_preserves_scope(self):
        old,adj,vox,p = fixture(); before=deepcopy(old)
        out=repair.repair_sources(old,adj,vox,p)
        self.assertTrue(out['admitted']);self.assertEqual(out['counts']['required_cells'],4)
        self.assertEqual(out['repair_counts'],dict(unchanged=1,full_period_communication_proxy=1,dated_business_proxy=2,unresolved=0))
        self.assertEqual(old,before);self.assertEqual(out['counts']['original_eligible_cells'],8)
        self.assertEqual([r['benchmark'] for r in out['records']],['VOX','XLP','XLK','SPY'])
    def test_missing_vox_is_not_filled(self):
        old,adj,vox,p=fixture();vox=vox.iloc[:0]
        out=repair.repair_sources(old,adj,vox,p)
        self.assertFalse(out['admitted']);self.assertEqual(out['counts']['gap_cells'],1)
    def test_invalid_values_not_missing(self):
        old,adj,vox,p=fixture();vox['VOX']=True
        out=repair.repair_sources(old,adj,vox,p)
        self.assertEqual(out['counts']['invalid_endpoints'],2)
        self.assertEqual(out['counts']['missing_endpoints'],0)
    def test_all_xlc_changes_even_when_original_valid(self):
        old,adj,vox,p=fixture();adj['XLC']=100
        out=repair.repair_sources(old,adj,vox,p)
        self.assertEqual(out['records'][0]['benchmark'],'VOX')
    def test_nonmissing_override_forbidden(self):
        _,_,_,p=fixture(); r=dict(is_etf=False,benchmark='XLE',symbol='BUD',date='2020-01-31')
        self.assertEqual(repair.resolve(r,p),('XLE','unchanged',None))
    def test_publication_and_expiry_boundaries(self):
        _,_,_,p=fixture();r=dict(is_etf=False,benchmark=None,symbol='GOOGL',date='2016-02-11')
        self.assertIsNone(repair.resolve(r,p)[0]);r['date']='2016-02-12';self.assertEqual(repair.resolve(r,p)[0],'XLK')
        r['date']='2018-09-24';self.assertIsNone(repair.resolve(r,p)[0])
    def test_unknown_missing_stays_gap(self):
        _,_,_,p=fixture();r=dict(is_etf=False,benchmark=None,symbol='UNKNOWN',date='2020-06-30')
        self.assertEqual(repair.resolve(r,p),(None,'unresolved',None))
    def test_etf_always_spy(self):
        _,_,_,p=fixture();r=dict(is_etf=True,benchmark='SPY',symbol='VOX',date='2020-06-30')
        self.assertEqual(repair.resolve(r,p),('SPY','unchanged',None))
        r['benchmark']='XLC'
        with self.assertRaises(ValueError):repair.resolve(r,p)
    def test_policy_tamper_and_duplicate_rejected(self):
        _,_,_,p=fixture();p['missing_mapping_rules'][0]['benchmark']='XLE'
        with self.assertRaises(ValueError):repair.check_policy(p)
        _,_,_,p=fixture();p['missing_mapping_rules'][0]=p['missing_mapping_rules'][1]
        with self.assertRaises(ValueError):repair.check_policy(p)
    def test_physical_prefix(self):
        old,adj,vox,p=fixture();full=repair.repair_sources(old,adj,vox,p)
        for n in [0,1,2]:
            short=deepcopy(old);short['events']=short['events'][:n];short['records']=[r for r in short['records'] if r['query_index']<n]
            cutoff=short['events'][-1]['date'] if n else None
            a=adj.loc[:cutoff] if n else adj.iloc[:0];v=vox.loc[:cutoff] if n else vox.iloc[:0]
            rr=repair.repair_sources(short,a,v,p)
            self.assertEqual(rr,repair.repair_sources(old,adj,vox,p,limit=n))
            self.assertEqual(rr['events'],full['events'][:n])
    def test_missing_market_not_silently_admitted(self):
        old,adj,vox,p=fixture();adj=adj.drop(columns='SPY')
        out=repair.repair_sources(old,adj,vox,p);self.assertFalse(out['admitted']);self.assertEqual(out['counts']['market_gap_events'],1)
    def test_required_axes_and_query_corruption(self):
        old,adj,vox,p=fixture();old['records'][0]['column_index']=99
        with self.assertRaises(ValueError):repair.repair_sources(old,adj,vox,p)
        old,adj,vox,p=fixture();old['records'][0]['date']='2016-07-29'
        with self.assertRaises(ValueError):repair.repair_sources(old,adj,vox,p)
    def test_raw_vox_good(self):
        rows=[dict(date='2015-01-02T00:00:00.000Z',adjClose=70.)]
        frame=repair.parse_vox(rows,['2015-01-02']);self.assertEqual(list(frame.columns),['VOX']);self.assertEqual(float(frame.iloc[0,0]),70.)
    def test_raw_vox_bad_values(self):
        for value in [None,True,0.,-1.,float('nan'),float('inf'),'70']:
            with self.subTest(value=value),self.assertRaises(ValueError):repair.parse_vox([dict(date='2015-01-02T00:00:00Z',adjClose=value)],['2015-01-02'])
    def test_raw_vox_bad_dates(self):
        for stamp in ['2014-12-31T00:00:00Z','2015-01-03T00:00:00Z','2015-01-02','2015-01-02T01:00:00Z','2015-01-02T00:00:00-05:00']:
            with self.subTest(stamp=stamp),self.assertRaises(ValueError):repair.parse_vox([dict(date=stamp,adjClose=70.)],['2015-01-02'])
    def test_raw_vox_empty_duplicate(self):
        with self.assertRaises(ValueError):repair.parse_vox([],[])
        row=dict(date='2015-01-02T00:00:00Z',adjClose=70.)
        with self.assertRaises(ValueError):repair.parse_vox([row,row],['2015-01-02'])

if __name__=='__main__':unittest.main(verbosity=2)
