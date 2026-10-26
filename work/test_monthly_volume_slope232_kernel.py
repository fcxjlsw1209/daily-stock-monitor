"""Synthetic-only tests for fixed-window rank-volume slopes; no market I/O."""
import copy
import json
import math
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import monthly_volume_slope232_kernel as k


def fixture():
    dates = pd.bdate_range('2014-01-01', periods=900)
    pos = np.arange(len(dates), dtype=float)
    p = np.exp(.0004*pos+.0000007*pos**2)
    panel = dict(adj=pd.DataFrame({'A':p, 'B':p**.5, 'C':np.full(len(dates),4.)}, index=dates),
                 volume=pd.DataFrame({'A':1.+pos%13, 'B':np.ones(len(dates)),
                                      'C':np.full(len(dates),3.)}, index=dates))
    available = [i for i in range(300,len(dates)-1) if dates[i].month != dates[i+1].month][:3]
    meta = dict(dates=[str(dates[i].date()) for i in available],
                symbols=[['A','B','C'] for _ in available], etfs=['C'])
    calendar = dates.strftime('%Y-%m-%d').tolist()
    return panel, meta, calendar, np.ones((3,3), dtype=bool)


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(253)/252.
        self.p = np.exp(self.x**2)
        self.v = np.arange(1,254,dtype=float)

    def test_weighted_slope_matches_direct_formula(self):
        out = k.measure(self.p,self.v)
        x=list(self.x); y=[math.log(float(p))-math.log(float(self.p[0])) for p in self.p]
        w=[.5+i/252 for i in range(253)]
        total=math.fsum(w); xb=math.fsum(a*b for a,b in zip(w,x))/total
        yb=math.fsum(a*b for a,b in zip(w,y))/total
        expected=math.fsum(a*(b-xb)*(c-yb) for a,b,c in zip(w,x,y))/math.fsum(a*(b-xb)**2 for a,b in zip(w,x))
        self.assertEqual(out['candidate'],expected)
        self.assertGreater(out['candidate'],out['control'])
        self.assertEqual(out['weights'][0],.5); self.assertEqual(out['weights'][-1],1.5)
        self.assertEqual(math.fsum(out['weights']),253.)

    def test_affine_log_path_and_negative_trend(self):
        for slope in (.8,-.8):
            out=k.measure(np.exp(2+slope*self.x),self.v)
            self.assertAlmostEqual(out['candidate'],slope,places=14)
            self.assertAlmostEqual(out['control'],slope,places=14)

    def test_constant_volume_exact_copy_and_single_solve(self):
        with mock.patch.object(k,'_slope',wraps=k._slope) as solve:
            out=k.measure(self.p,np.ones(253)*7)
        self.assertEqual(solve.call_count,1)
        self.assertEqual(out['candidate'],out['control'])
        self.assertEqual(out['candidate_stats'],out['control_stats'])
        self.assertEqual(out['candidate_branch'],'constant_volume_copy')
        self.assertEqual(out['volume_ranks'],[127.]*253)
        self.assertEqual(out['weights'],[1.]*253)

    def test_constant_price_zero_no_solve_or_log(self):
        with mock.patch.object(k,'_slope',side_effect=AssertionError('solve')), mock.patch.object(k.math,'log',side_effect=AssertionError('log')):
            out=k.measure(np.full(253,3.),self.v)
        self.assertEqual(out['candidate'],0.);self.assertEqual(out['control'],0.)
        self.assertEqual(out['candidate_branch'],'constant_price_zero')
        self.assertIsNone(out['candidate_stats'])

    def test_price_scale_invariance(self):
        a=k.measure(self.p,self.v); b=k.measure(self.p*13.,self.v)
        self.assertAlmostEqual(a['candidate'],b['candidate'],places=13)
        self.assertAlmostEqual(a['control'],b['control'],places=13)

    def test_volume_scale_and_monotone_transform_invariance(self):
        a=k.measure(self.p,self.v)
        for v in (self.v*8,self.v**3):
            b=k.measure(self.p,v)
            self.assertEqual(a['weights'],b['weights'])
            self.assertEqual(a['candidate'],b['candidate'])

    def test_exact_average_ties(self):
        v=np.array([1.]*100+[2.]*100+[3.]*53)
        out=k.measure(self.p,v)
        self.assertEqual(out['volume_ranks'],[50.5]*100+[150.5]*100+[227.]*53)
        self.assertEqual(out['weights'][0],.5+49.5/252)
        self.assertEqual(math.fsum(out['weights']),253.)

    def test_neighbor_float_volume_distinction(self):
        v=np.ones(253);v[0]=np.nextafter(1.,2.)
        out=k.measure(self.p,v)
        self.assertFalse(out['constant_volume'])
        self.assertEqual(out['volume_ranks'][0],253.)
        self.assertEqual(out['volume_ranks'][1],126.5)

    def test_source_invalid_common_unavailable(self):
        for target in ('price','volume'):
            for bad,reason in [(0.,'invalid'),(-1.,'invalid'),(math.nan,'invalid'),(math.inf,'invalid'),
                               (True,'nonnumeric'),('1','nonnumeric'),(None,'nonnumeric'),(10**1000,'invalid')]:
                with self.subTest(target=target,bad=str(bad)[:15]):
                    p=self.p.astype(object);v=self.v.astype(object)
                    (p if target=='price' else v)[4]=bad
                    out=k.measure(p,v)
                    self.assertFalse(out['available']);self.assertTrue(math.isnan(out['candidate']))
                    self.assertTrue(math.isnan(out['control']));self.assertEqual(out['reason'],reason+'_'+target)

    def test_invalid_price_precedes_invalid_volume(self):
        p=self.p.copy();v=self.v.copy();p[4]=0.;v[4]=0.
        self.assertEqual(k.measure(p,v)['reason'],'invalid_price')

    def test_shape_mismatch_rejected(self):
        for p,v in [(self.p[:-1],self.v),(self.p,self.v[:-1]),(self.p[:,None],self.v)]:
            with self.assertRaises(ValueError):k.measure(p,v)

    def test_derived_nonfinite_aborts(self):
        with mock.patch.object(k.math,'log',return_value=math.inf):
            with self.assertRaises(FloatingPointError):k.measure(self.p,self.v)
        with mock.patch.object(k.math,'fsum',side_effect=OverflowError('derived')):
            with self.assertRaises(FloatingPointError):k.measure(self.p,self.v)
        with self.assertRaises(FloatingPointError):k._slope([0.,1.],[1e308,-1e308],[2.,2.])

    def test_extreme_positive_sources_remain_finite(self):
        low=math.log(np.finfo(float).tiny); high=math.log(np.finfo(float).max)
        p=np.array([math.exp(low+(high-low)*i/252) for i in range(253)])
        p[0]=np.nextafter(0.,1.)
        out=k.measure(p,self.v)
        self.assertTrue(out['available']);self.assertTrue(math.isfinite(out['candidate']))

    def test_no_input_mutation(self):
        p=self.p.copy();v=self.v.copy();k.measure(p,v)
        np.testing.assert_array_equal(p,self.p);np.testing.assert_array_equal(v,self.v)


class RankTests(unittest.TestCase):
    def test_ties_mask_and_symmetric_endpoints(self):
        got=k.centered_rank(np.array([2.,1.,2.,4.,math.nan]),np.array([1,1,1,1,0],bool))
        np.testing.assert_array_equal(got,[0.,-.5,0.,.5,0.])

    def test_constant_singleton_empty_exact_zero(self):
        for x,a in [(np.array([3.,3.]),np.array([1,1],bool)),
                    (np.array([3.,math.nan]),np.array([1,0],bool)),
                    (np.array([],float),np.array([],bool))]:
            np.testing.assert_array_equal(k.centered_rank(x,a),np.zeros(len(x)))

    def test_bad_rank_domain_rejected(self):
        for x,a in [(np.array([math.nan]),np.array([1],bool)),
                    (np.array([1.]),np.array([1])),(np.array([True]),np.array([1],bool))]:
            with self.assertRaises(ValueError):k.centered_rank(x,a)


class BuildTests(unittest.TestCase):
    def setUp(self):self.panel,self.meta,self.calendar,self.eligible=fixture()
    def run_build(self,**kwargs):
        return k.build_features(kwargs.get('panel',self.panel),kwargs.get('meta',self.meta),
            kwargs.get('calendar',self.calendar),kwargs.get('eligible',self.eligible),kwargs.get('limit'))

    def test_exact_windows_counts_and_json_audit(self):
        values,audit=self.run_build()
        self.assertEqual(set(values),{'candidate','control','available'})
        self.assertTrue(values['available'].all())
        self.assertEqual(audit['available_cells'],9)
        self.assertEqual(audit['source_price_observations'],9*253)
        self.assertEqual(audit['source_volume_observations'],9*253)
        self.assertEqual(audit['volume_rank_observations'],9*253)
        self.assertEqual(audit['descriptive_time_slope_estimates'],18)
        self.assertEqual(audit['unweighted_slope_solves'],6)
        self.assertEqual(audit['weighted_slope_solves'],3)
        self.assertEqual(audit['constant_volume_copies'],3)
        self.assertEqual(audit['constant_price_shortcuts'],3)
        self.assertEqual(audit['constant_volume_cells'],6)
        for event in audit['event_history']:
            t=event['session_position']
            self.assertEqual(event['source_dates'],self.calendar[t-294:t-41])
            self.assertEqual(len(event['source_dates']),253)
        json.dumps(audit,allow_nan=False)

    def test_ineligible_poison_unread_common_mask(self):
        self.eligible[:,0]=False
        for frame in self.panel.values():frame['A']='poison'
        values,audit=self.run_build()
        self.assertFalse(values['available'][:,0].any())
        self.assertTrue(np.isnan(values['candidate'][:,0]).all())
        self.assertTrue(values['available'][:,1:].all())
        self.assertEqual(audit['ineligible_cells'],3);self.assertEqual(audit['attempted_cells'],6)
        self.assertEqual(audit['event_history'][0]['cells'][0]['prices'],[])

    def test_initial_partial_skips_even_eligible_poison(self):
        meta=dict(dates=[k.INITIAL_PARTIAL],symbols=[['A','B','C']],etfs=['C'])
        panel={key:frame.astype(object) for key,frame in self.panel.items()}
        for frame in panel.values():frame.iloc[:,:]='poison'
        values,audit=self.run_build(meta=meta,panel=panel,eligible=np.ones((1,3),bool))
        self.assertFalse(values['available'].any());self.assertEqual(audit['initial_partial_cells'],3)
        self.assertEqual(audit['attempted_cells'],0);self.assertEqual(audit['reason_counts'],{'initial_partial':3})
        self.assertEqual(len(audit['event_history'][0]['source_dates']),253)

    def test_whole_month_not_dropped_for_one_invalid_security(self):
        t=self.calendar.index(self.meta['dates'][0]);self.panel['volume'].iloc[t-42,0]=0
        values,audit=self.run_build()
        self.assertFalse(values['available'][0,0]);self.assertTrue(values['available'][0,1:].all())
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'],'invalid_volume')

    def test_numeric_future_and_before_window_poison_ignored(self):
        a,_=self.run_build(limit=1)
        t=self.calendar.index(self.meta['dates'][0])
        panel={key:frame.astype(object) for key,frame in self.panel.items()}
        for frame in panel.values():
            frame.iloc[:t-294,:]='past';frame.iloc[t-41:,:]='future'
        b,_=self.run_build(panel=panel,limit=1)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_prefix_logical_and_physical(self):
        allvalues,_=self.run_build()
        for n in range(4):
            logical,la=self.run_build(limit=n)
            meta=dict(self.meta,dates=self.meta['dates'][:n],symbols=self.meta['symbols'][:n])
            cutoff=self.meta['dates'][n-1] if n else None
            panel={key:(frame.loc[:cutoff].copy() if n else frame.iloc[:0].copy()) for key,frame in self.panel.items()}
            physical,pa=self.run_build(panel=panel,meta=meta,eligible=self.eligible[:n])
            for key in logical:
                np.testing.assert_array_equal(logical[key],allvalues[key][:n])
                np.testing.assert_array_equal(logical[key],physical[key])
            self.assertEqual(la,pa)
        empty=dict(dates=[],symbols=[],etfs=[])
        values,_=self.run_build(meta=empty,eligible=np.zeros((0,0),bool))
        self.assertEqual(values['candidate'].shape,(0,0))

    def test_missing_session_no_calendar_compression(self):
        t=self.calendar.index(self.meta['dates'][0]);day=self.calendar[t-100]
        for key,reason in [('adj','missing_price_session'),('volume','missing_volume_session')]:
            panel=dict(self.panel);panel[key]=panel[key].drop(pd.Timestamp(day))
            values,audit=self.run_build(panel=panel,limit=1)
            self.assertFalse(values['available'].any())
            self.assertEqual(audit['reason_counts'],{reason:3})

    def test_missing_symbol_common_unavailable(self):
        for key,reason in [('adj','missing_price_symbol'),('volume','missing_volume_symbol')]:
            panel=dict(self.panel);panel[key]=panel[key].drop(columns='A')
            values,audit=self.run_build(panel=panel,limit=1)
            self.assertFalse(values['available'][0,0]);self.assertTrue(values['available'][0,1:].all())
            self.assertEqual(audit['event_history'][0]['cells'][0]['reason'],reason)

    def test_insufficient_history(self):
        meta=dict(dates=['2014-01-31'],symbols=[['A','B','C']],etfs=[])
        values,audit=self.run_build(meta=meta,eligible=self.eligible[:1])
        self.assertFalse(values['available'].any());self.assertEqual(audit['reason_counts'],{'insufficient_history':3})
        self.assertEqual(audit['event_history'][0]['source_dates'],[])

    def test_boolean_shape_and_limit_validation(self):
        for eligible in [self.eligible.astype(int),self.eligible[:,:2],self.eligible[:,0],self.eligible[:2]]:
            with self.assertRaises(ValueError):self.run_build(eligible=eligible)
        for limit in [True,-1,4,1.0]:
            with self.assertRaises(ValueError):self.run_build(limit=limit)

    def test_calendar_duplicate_order_and_format(self):
        for cal in [self.calendar[:20]+[self.calendar[19]]+self.calendar[20:],
                    list(reversed(self.calendar)),[pd.Timestamp(d) for d in self.calendar],
                    ['2014-01-04']+self.calendar,[]]:
            with self.assertRaises(ValueError):self.run_build(calendar=cal)

    def test_query_monthend_witness_and_duplicate_symbols(self):
        for meta in [dict(self.meta,dates=[self.calendar[301]]+self.meta['dates'][1:]),
                     dict(self.meta,dates=[self.meta['dates'][0]]*3),
                     dict(self.meta,symbols=[['A','A','C']]*3),dict(self.meta,etfs=['C','C'])]:
            with self.assertRaises(ValueError):self.run_build(meta=meta)
        cutoff=self.calendar.index(self.meta['dates'][-1])
        with self.assertRaises(ValueError):self.run_build(calendar=self.calendar[:cutoff+1])

    def test_frame_axes_and_panel_keys(self):
        panel=dict(self.panel);panel['adj']=pd.concat([panel['adj'].iloc[:1],panel['adj']])
        with self.assertRaises(ValueError):self.run_build(panel=panel)
        panel=dict(self.panel);panel['volume']=panel['volume'].rename(columns={'B':'A'})
        with self.assertRaises(ValueError):self.run_build(panel=panel)
        with self.assertRaises(ValueError):self.run_build(panel=dict(self.panel,close=self.panel['adj']))

    def test_no_input_mutation(self):
        panels={key:frame.copy(deep=True) for key,frame in self.panel.items()}
        meta=copy.deepcopy(self.meta);eligible=self.eligible.copy();self.run_build()
        for key in panels:pd.testing.assert_frame_equal(panels[key],self.panel[key])
        self.assertEqual(meta,self.meta);np.testing.assert_array_equal(eligible,self.eligible)


if __name__=='__main__':unittest.main()
