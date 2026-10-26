"""Synthetic source tests only; no actual prices, decisions, or returns."""
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_upside_retention181_kernel as k


def fixture():
    days=pd.bdate_range('2010-01-04',periods=365)
    x=np.arange(len(days))
    frame=pd.DataFrame({'A':100*np.exp(.002*x+.03*np.sin(.23*x)),
                        'ETF':80*np.exp(-.0002*x+.02*np.cos(.31*x))},index=days)
    meta=dict(dates=[str(days[i].date()) for i in (294,320,341)],
              symbols=[['A','ETF'] for _ in range(3)],etfs=['ETF'])
    return {'adj':frame},meta,days.strftime('%Y-%m-%d').tolist()


def one_gain(end=2.):
    p=np.full(253,end);p[0]=1.;p[1:22]=4.
    return p


class MeasureTests(unittest.TestCase):
    def test_partial_retention_differs_from_binary(self):
        x=k.measure(one_gain())
        self.assertEqual(x['anchor_indices'],[1])
        self.assertEqual(x['retention_share'],.5)
        self.assertEqual(x['binary_retention_share'],0.)
        self.assertEqual(x['candidate'],0.)
        self.assertEqual(x['control'],-.5)
        self.assertEqual(x['retention_share_estimates'],2)

    def test_full_zero_and_negative_endpoint_retention(self):
        for end,q in ((4.,1.),(1.,0.),(.5,0.)):
            x=k.measure(one_gain(end))
            self.assertEqual(x['retention_share'],q)
            self.assertEqual(x['binary_retention_share'],q)
            self.assertEqual(x['candidate'],q-.5)
            self.assertEqual(x['control'],q-.5)

    def test_extra_appreciation_capped_at_original_gain(self):
        x=k.measure(one_gain(16.))
        self.assertEqual(x['capped_retained_log_moves'][0],x['positive_log_returns'][0])
        self.assertGreater(x['retained_log_moves'][0],x['positive_log_returns'][0])
        self.assertEqual(x['candidate'],.5)

    def test_binary_comparison_exact_price_tie(self):
        tied=k.measure(one_gain(4.));below=k.measure(one_gain(np.nextafter(4.,0.)))
        above=k.measure(one_gain(np.nextafter(4.,np.inf)))
        self.assertTrue(tied['binary_recovered'][0])
        self.assertFalse(below['binary_recovered'][0])
        self.assertTrue(above['binary_recovered'][0])

    def test_last_anchor_and_last_maturity_exact(self):
        p=np.ones(253);p[231:252]=2.;p[252]=1.5
        x=k.measure(p)
        self.assertEqual(x['anchor_indices'],[231])
        self.assertAlmostEqual(x['retention_share'],math.log(1.5)/math.log(2.))
        p[232:]=3.;p[231]=1.
        x=k.measure(p)
        self.assertFalse(x['available']);self.assertEqual(x['positive_anchor_count'],0)

    def test_overlapping_anchors_all_retained_without_cooldown(self):
        p=np.exp(.01*np.arange(253));x=k.measure(p)
        self.assertEqual(x['anchor_indices'],list(range(1,232)))
        self.assertEqual(x['positive_anchor_count'],231)
        self.assertEqual(x['candidate'],.5);self.assertEqual(x['control'],.5)

    def test_gain_weighted_formula_not_equal_event_average(self):
        p=np.ones(253);p[1:22]=4.;p[22:24]=2.;p[24:45]=8.;p[45:]=2.
        x=k.measure(p);w=x['positive_log_returns']
        expected=math.fsum(x['capped_retained_log_moves'])/math.fsum(w)
        self.assertEqual(x['retention_share'],expected)
        self.assertEqual(x['binary_retention_share'],math.fsum(v for v,b in zip(w,x['binary_recovered']) if b)/math.fsum(w))
        self.assertGreaterEqual(x['retention_share'],x['binary_retention_share'])

    def test_no_positive_is_unavailable_not_zero_success(self):
        for p in (np.ones(253),np.exp(-.01*np.arange(253))):
            x=k.measure(p)
            self.assertFalse(x['available']);self.assertEqual(x['reason'],'no_positive_anchors')
            self.assertEqual(x['retention_share_estimates'],2)
            self.assertEqual(x['positive_return_sum'],0.)
            self.assertIsNone(x['retention_share']);self.assertTrue(math.isnan(x['candidate']))

    def test_one_tiny_positive_requires_no_floor_or_minimum_count(self):
        p=np.ones(253);p[1:]=np.nextafter(1.,2.)
        x=k.measure(p)
        self.assertTrue(x['available']);self.assertEqual(x['positive_anchor_count'],1)
        self.assertEqual(x['candidate'],.5)

    def test_scale_invariance_with_log_difference_roundoff(self):
        p=one_gain(2.);expected=k.measure(p)
        for scale in (1e-100,1e100,2.**-100,2.**100):
            x=k.measure(p*scale)
            self.assertEqual(x['anchor_indices'],expected['anchor_indices'])
            self.assertEqual(x['binary_recovered'],expected['binary_recovered'])
            for field in ('candidate','control','retention_share'):
                self.assertAlmostEqual(x[field],expected[field],places=12)

    def test_extreme_prices_never_form_overflowing_ratio(self):
        p=np.full(253,1e300);p[0]=1e-300
        x=k.measure(p)
        self.assertTrue(x['available']);self.assertEqual(x['candidate'],.5)
        self.assertTrue(math.isfinite(x['positive_return_sum']))

    def test_invalid_sources_and_exact_shape(self):
        for value in (0.,-1.,np.nan,np.inf,-np.inf,'bad'):
            p=one_gain().astype(object);p[150]=value;x=k.measure(p)
            self.assertFalse(x['available']);self.assertEqual(x['retention_share_estimates'],0)
        for p in (np.ones(252),np.ones(254),np.ones((253,1))):
            with self.assertRaises(ValueError):k.measure(p)

    def test_nonfinite_arithmetic_does_not_become_missing(self):
        with patch.object(k.math,'log',return_value=math.inf):
            with self.assertRaises(FloatingPointError):k.measure(one_gain())
        with patch.object(k.math,'fsum',return_value=math.inf):
            with self.assertRaises(FloatingPointError):k.measure(one_gain())


class BuildTests(unittest.TestCase):
    def test_independent_full_audit_crosscheck_on_both_fixtures(self):
        import monthly_upside_retention181_independent as independent
        from test_monthly_upside_retention181_independent import fixture as other_fixture
        for factory in (fixture,other_fixture):
            p,m,c=factory();v,a=k.build_features(p,m,c)
            self.assertTrue(independent.verify_features(p,m,c,v,a)['passed'])
            for mode in ('no_positive','missing','invalid','single_partial','extreme'):
                p,m,c=factory();m=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1])
                first=m['symbols'][0][0]
                if mode=='missing':p['adj']=p['adj'].drop(pd.Timestamp(c[10]))
                elif mode=='invalid':p['adj'].loc[pd.Timestamp(c[10]),first]=np.nan
                else:
                    prices=(np.ones(253) if mode=='no_positive' else one_gain())
                    if mode=='extreme':
                        prices=np.full(253,1e300);prices[0]=1e-300
                    p['adj'].loc[p['adj'].index[:253],first]=prices
                v,a=k.build_features(p,m,c)
                self.assertTrue(independent.verify_features(p,m,c,v,a)['passed'])

    def test_exact_window_dates_maturity_and_counts(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c)
        self.assertEqual(v['candidate'].shape,(3,2));self.assertTrue(v['available'].all())
        self.assertEqual(a['retention_share_estimates'],12)
        first=a['event_history'][0]
        self.assertEqual(first['price_start'],c[0]);self.assertEqual(first['price_end'],c[252])
        self.assertEqual(first['windows']['anchors'],dict(first_date=c[1],last_date=c[231],count=231))
        self.assertEqual(first['windows']['maturities'],dict(first_date=c[22],last_date=c[252],horizon_sessions=21))
        self.assertEqual(a['positive_anchor_count'],sum(cell['positive_anchor_count'] for row in a['event_history'] for cell in row['cells']))
        json.dumps(a,allow_nan=False)

    def test_entire_source_required_including_unselected_dates(self):
        p,m,c=fixture();p['adj']=p['adj'].drop(pd.Timestamp(c[10]))
        v,a=k.build_features(p,m,c)
        self.assertFalse(v['available'][0].any())
        self.assertEqual(a['event_history'][0]['missing_official_sessions'],[c[10]])
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_official_session')

    def test_all_42_recent_sessions_are_not_required_or_converted(self):
        p,m,c=fixture();expected=k.build_features(p,m,c,1)
        class Poison:
            def __float__(self):raise AssertionError('Recent or future data consumed')
        p['adj']=p['adj'].astype(object);p['adj'].iloc[253:]=Poison()
        got=k.build_features(p,m,c,1)
        for key in expected[0]:np.testing.assert_array_equal(got[0][key],expected[0][key])
        self.assertEqual(got[1],expected[1])
        p['adj']=p['adj'].iloc[:253]
        got=k.build_features(p,m,c,1)
        self.assertEqual(got[1],expected[1])

    def test_each_prefix_empty_and_invalid_limits(self):
        p,m,c=fixture();values,audit=k.build_features(p,m,c)
        for n in range(4):
            v,a=k.build_features(p,m,c,n)
            for key in values:np.testing.assert_array_equal(v[key],values[key][:n])
            self.assertEqual(a['event_history'],audit['event_history'][:n])
        for n in (-1,4,True,1.5,'1'):
            with self.assertRaises(ValueError):k.build_features(p,m,c,n)

    def test_etf_same_values_unknown_symbol_and_history(self):
        p,m,c=fixture();p['adj']['ETF']=p['adj']['A'];v,a=k.build_features(p,m,c)
        for key in v:np.testing.assert_array_equal(v[key][:,0],v[key][:,1])
        m['dates'][0]=c[293];m['symbols'][1][1]='UNKNOWN';v,a=k.build_features(p,m,c)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        self.assertEqual(a['event_history'][1]['cells'][1]['reason'],'missing_symbol')

    def test_source_arrays_unchanged(self):
        p,m,c=fixture();before=p['adj'].copy(deep=True);k.build_features(p,m,c)
        pd.testing.assert_frame_equal(before,p['adj'])

    def test_meta_maps_and_calendar_contract(self):
        p,m,c=fixture();bad=dict(m);bad.pop('etfs')
        with self.assertRaises(ValueError):k.build_features(p,bad,c)
        bad=dict(m);bad['symbols']=[['A','A']]+m['symbols'][1:]
        with self.assertRaises(ValueError):k.build_features(p,bad,c)
        for cc in (c[::-1],c+[c[-1]],[]):
            with self.assertRaises(ValueError):k.build_features(p,m,cc)


if __name__=='__main__':
    unittest.main(verbosity=2)
