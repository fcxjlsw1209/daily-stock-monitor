"""Synthetic-only recovery race tests; no source files, orders or outcomes."""
from copy import deepcopy
import json
import unittest

import numpy as np
import pandas as pd
import recovery165_kernel as k


def repeated(pattern):
    return np.resize(np.asarray(pattern, dtype=float), 211)


def fixture():
    sessions=pd.bdate_range('2015-01-01','2017-04-05')
    names=['A','B','C','ETF','UNKNOWN']
    events=[('2016-12-30',names)]
    prices=pd.DataFrame(100.,index=sessions,columns=names+['XLK'])
    t=sessions.get_loc(pd.Timestamp(events[0][0]))
    for name,pattern in [('A',[100,80,100,100,100]),('B',[100,60,70,80,100]),
                         ('C',[100,80,100,100,100]),('XLK',[100,70,80,90,100])]:
        prices.iloc[t-252:t-41,prices.columns.get_loc(name)]=repeated(pattern)
    maps=dict(etfs=['ETF'],by_date={events[0][0]:{'A':'XLK','B':'XLK','C':'XLK','ETF':None,'UNKNOWN':None}})
    return prices,maps,events,[str(day.date()) for day in sessions]


def scalar_oracle(stock,sector):
    starts=[s for s in range(1,190) if stock[s]<stock[s-1] and sector[s]<sector[s-1]]
    if len(starts)<20:return None,None,len(starts)
    differences=[];impacts=[]
    for s in starts:
        times=[]
        for path in (stock,sector):
            times.append(next((lag for lag in range(1,22) if path[s+lag]>=path[s-1]),22))
        differences.append(times[1]-times[0])
        rs=stock[s]/stock[s-1]-1.;rb=sector[s]/sector[s-1]-1.
        impacts.append((rs-rb)/(abs(rs)+abs(rb)))
    return sum(differences)/(21*len(starts)),sum(impacts)/len(starts),len(starts)


class Recovery165(unittest.TestCase):
    def test_manual_oracle_known_recovery_impact_ranks_and_audit(self):
        p,m,e,c=fixture();d,a=k.build_features(p,m,e,c);row=a['event_history'][0]
        t=c.index(e[0][0])
        for j,name in enumerate(e[0][1][:3]):
            expected=scalar_oracle(p[name].iloc[t-252:t-41].tolist(),p['XLK'].iloc[t-252:t-41].tolist())
            self.assertEqual(row['raw_recovery'][j],expected[0])
            self.assertAlmostEqual(row['raw_control'][j],expected[1],places=14)
            self.assertEqual(row['joint_down_counts'][j],expected[2])
        self.assertEqual(row['raw_recovery'],[2/21,0.,2/21,None,None])
        np.testing.assert_allclose(row['ranks_recovery'][:3],[1/3,-1/6,1/3],atol=1e-16,rtol=0)
        self.assertEqual(row['ranks_recovery'][3:],[None,None])
        self.assertEqual(row['common_support'],[True,True,True,False,False])
        self.assertEqual(row['source_status'],['available']*3+['etf_not_applicable','missing_sector_mapping'])
        self.assertEqual(row['window_start'],c[t-252]);self.assertEqual(row['window_end'],c[t-42])
        self.assertEqual(a['start_count'],189);self.assertEqual(a['window_prices'],211)
        for suffix in ('recovery','control'):
            ranks=pd.Series(row['raw_'+suffix]).rank(method='average',pct=True)-.5
            np.testing.assert_array_equal(d['delta_'+suffix][0,:3],.04*ranks.to_numpy()[:3])
        json.dumps(a,allow_nan=False)

    def test_first_recovery_k1_k21_exact_threshold_and_censor22(self):
        for start in (1,189):
            for lag in (1,21):
                values=np.full(211,50.);values[start-1]=100.;values[start+lag]=100.
                self.assertEqual(k._first_passage(values,start),lag)
                values[start+lag]=np.nextafter(100.,0.)
                self.assertEqual(k._first_passage(values,start),22)
        for start in (0,190,True):
            with self.assertRaises(ValueError):k._first_passage(np.ones(211),start)

    def test_last_start_189_reaches_last_price_210_and_no_later_start(self):
        sector=np.arange(211.,0.,-1.)
        stock=sector.copy();stock[210]=stock[188]
        race,_,count,status=k._paired_recovery(stock,sector)
        self.assertEqual(count,189);self.assertEqual(status,'available')
        self.assertEqual(race,1/(21*189))
        stock[210]=np.nextafter(stock[188],0.)
        self.assertEqual(k._paired_recovery(stock,sector)[0],0.)

    def test_minimum_twenty_joint_strictly_down_starts(self):
        for count in (0,19,20):
            stock=np.full(211,100.);sector=stock.copy()
            stock[np.arange(1,2*count,2)]=80.;sector[np.arange(1,2*count,2)]=70.
            result=k._paired_recovery(stock,sector)
            self.assertEqual(result[2],count)
            if count<20:self.assertTrue(np.isnan(result[0]));self.assertEqual(result[3],'insufficient_joint_down')
            else:self.assertEqual(result[0],0.);self.assertEqual(result[3],'available')
        stock=np.arange(211.,0.,-1.);sector=stock.copy();stock[5]=stock[4]
        self.assertEqual(k._paired_recovery(stock,sector)[2],188)

    def test_both_censored_is_zero_and_no_variance_gate(self):
        path=np.arange(211.,0.,-1.)
        race,control,count,status=k._paired_recovery(path,path)
        self.assertEqual((race,control,count,status),(0.,0.,189,'available'))

    def test_identical_recovery_means_remain_exact_ties(self):
        stock=repeated([100,80,100,100,100]);sector=repeated([100,70,80,90,100])
        a=k._paired_recovery(stock,sector)[0]
        b=k._paired_recovery(stock*8.,sector*.125)[0]
        self.assertEqual(a,2/21);self.assertEqual(a,b)

    def test_each_paired_source_boundary_interior_missing_nonpositive_neutral(self):
        p,m,e,c=fixture();t=c.index(e[0][0])
        for name in ('A','XLK'):
            for offset in (-252,-251,-100,-42):
                for bad in (np.nan,np.inf,-np.inf,0.,-1.):
                    q=p.copy();q.iloc[t+offset,q.columns.get_loc(name)]=bad
                    d,a=k.build_features(q,m,e,c);row=a['event_history'][0]
                    with self.subTest(name=name,offset=offset,bad=bad):
                        self.assertIsNone(row['raw_recovery'][0]);self.assertIsNone(row['raw_control'][0])
                        self.assertIsNone(row['joint_down_counts'][0]);self.assertFalse(row['common_support'][0])
                        for values in d.values():self.assertEqual(values[0,0],0.);self.assertFalse(np.signbit(values[0,0]))

    def test_recent_current_future_and_preformation_poison_never_read(self):
        p,m,e,c=fixture();t=c.index(e[0][0]);expected,a=k.build_features(p,m,e,c)
        q=p.astype(object);q.iloc[:t-252]='preformation';q.iloc[t-41:]='future/recent'
        q['ETF']='unconsumed ETF';q['UNKNOWN']='unconsumed unmapped'
        actual,b=k.build_features(q,m,e,c)
        self.assertEqual(a,b)
        for key in actual:self.assertEqual(actual[key].tobytes(),expected[key].tobytes())
        actual,b=k.build_features(p.iloc[:t-41],m,e,c)
        self.assertEqual(a,b)

    def test_independent_price_units_do_not_change_measurement(self):
        p,m,e,c=fixture();expected,a=k.build_features(p,m,e,c)
        q=p.copy();q['A']*=8.;q['B']*=.25;q['C']*=32.;q['XLK']*=.125
        actual,b=k.build_features(q,m,e,c)
        self.assertEqual(a,b)
        for key in actual:np.testing.assert_array_equal(actual[key],expected[key])

    def test_shared_stock_support_excludes_etf_unknown_and_invalid_paths(self):
        p,m,e,c=fixture();p['B']=np.nan
        d,a=k.build_features(p,m,e,c);row=a['event_history'][0]
        self.assertEqual(row['ranks_recovery'],[.25,None,.25,None,None])
        self.assertEqual(row['ranks_control'],[.25,None,.25,None,None])
        for values in d.values():np.testing.assert_array_equal(values,[[.01,0.,.01,0.,0.]])
        p['C']=np.nan;d,a=k.build_features(p,m,e,c)
        for values in d.values():np.testing.assert_array_equal(values,[[.02,0.,0.,0.,0.]])
        bad=deepcopy(m);bad['by_date'][e[0][0]]['ETF']='XLK'
        with self.assertRaisesRegex(ValueError,'ETF'):k.build_features(p,bad,e,c)

    def test_initial_forced_neutral_with_source_values(self):
        p,m,e,c=fixture();day='2016-08-17';e=[(day,e[0][1])];m['by_date']={day:next(iter(m['by_date'].values()))}
        t=c.index(day)
        p.iloc[t-252:t-41,p.columns.get_loc('A')]=repeated([100,80,100,100,100])
        p.iloc[t-252:t-41,p.columns.get_loc('XLK')]=repeated([100,70,80,90,100])
        d,a=k.build_features(p,m,e,c);row=a['event_history'][0]
        self.assertTrue(row['initial_forced_neutral']);self.assertEqual(row['common_support'],[False]*5)
        self.assertIsNotNone(row['raw_recovery'][0])
        for values in d.values():self.assertEqual(values.tobytes(),np.zeros((1,5)).tobytes())

    def test_exact_prefixes_future_mapping_poison_and_no_mutation(self):
        p,m,_,c=fixture();names=['A','B','C','ETF','UNKNOWN'];mapping=next(iter(m['by_date'].values()))
        dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31','2016-11-30','2016-12-30']
        e=[(day,names) for day in dates];m['by_date']={day:dict(mapping) for day in dates}
        saved=p.copy();saved_m=deepcopy(m);full,a=k.build_features(p,m,e,c)
        for n in range(7):
            part,b=k.build_features(p,m,e,c,n)
            self.assertEqual(b['event_history'],a['event_history'][:n])
            for key in full:self.assertEqual(part[key].tobytes(),full[key][:n].tobytes())
        poison=deepcopy(m);poison['by_date'][dates[-1]]='unused future mapping'
        k.build_features(p,poison,e,c,1)
        pd.testing.assert_frame_equal(p,saved);self.assertEqual(m,saved_m)

    def test_all123_synthetic_feature_prefixes_exact(self):
        sessions=pd.bdate_range('2013-01-01','2027-01-31')
        p=pd.DataFrame({name:np.resize(pattern,len(sessions)) for name,pattern in
                       [('A',[100.,80.,100.,100.,100.]),('XLK',[100.,70.,80.,90.,100.])]},index=sessions)
        dates=['2016-08-17']+[str(day.date()) for day in pd.date_range('2016-08-31',periods=121,freq='BME')]
        e=[(day,['A']) for day in dates];m=dict(etfs=[],by_date={day:{'A':'XLK'} for day in dates});c=[str(day.date()) for day in sessions]
        full,a=k.build_features(p,m,e,c)
        for n in range(123):
            part,b=k.build_features(p,m,e,c,n)
            self.assertEqual(b['event_history'],a['event_history'][:n])
            for key in full:self.assertEqual(part[key].tobytes(),full[key][:n].tobytes())

    def test_insufficient_calendar_missing_official_session_or_benchmark(self):
        p,m,e,c=fixture();t=c.index(e[0][0])
        _,a=k.build_features(p,m,e,c[t-252:]);self.assertEqual(a['common_support_cells'],3)
        _,a=k.build_features(p,m,e,c[t-251:]);self.assertEqual(a['common_support_cells'],0)
        self.assertEqual(a['event_history'][0]['source_status'][:3],['insufficient_history']*3)
        for q in (p.drop(index=pd.Timestamp(c[t-100])),p.drop(columns='XLK')):
            _,a=k.build_features(q,m,e,c);self.assertEqual(a['common_support_cells'],0)

    def test_empty_prefix_never_reads_prices_or_mapping_values(self):
        p,m,e,c=fixture();p=p.astype(object);p[:]='poison';m['by_date'][e[0][0]]=object()
        d,a=k.build_features(p,m,e,c,0)
        self.assertEqual(a['event_history'],[]);self.assertTrue(all(v.shape==(0,5) for v in d.values()))

    def test_axis_calendar_mapping_event_and_prefix_contracts(self):
        p,m,e,c=fixture()
        for q in (p.iloc[::-1],p.rename(columns={'B':'A'}),p.set_axis(p.index.tz_localize('UTC'))):
            with self.assertRaises((ValueError,TypeError)):k.build_features(q,m,e,c)
        for changes in ({},dict(etfs=[],by_date={}),dict(etfs=['ETF','ETF'],by_date=m['by_date'])):
            with self.assertRaises(ValueError):k.build_features(p,changes,e,c)
        for target in ('SPY','QQQ',False,1):
            bad=deepcopy(m);bad['by_date'][e[0][0]]['A']=target
            with self.assertRaises(ValueError):k.build_features(p,bad,e,c)
        for limit in (-1,2,True,1.,'1'):
            with self.assertRaises(ValueError):k.build_features(p,m,e,c,limit)
        with self.assertRaisesRegex(ValueError,'month end'):k.build_features(p,m,[('2016-12-29',e[0][1])],c,0)
        with self.assertRaisesRegex(ValueError,'next session'):k.build_features(p,m,e,c[:c.index(e[0][0])+1])
        for count in (0,210,212):
            with self.assertRaises(ValueError):k._paired_recovery(np.ones(count),np.ones(211))


if __name__=='__main__':unittest.main()
