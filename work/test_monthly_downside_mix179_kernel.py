"""Synthetic tests only: no stored research prices, forecasts, or returns."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_downside_mix179_kernel as k


def prices(returns):
    return 100. * np.cumprod(np.r_[1., 1. + np.asarray(returns)])


def fixture():
    dates = pd.bdate_range('2010-01-04', periods=365)
    x = np.arange(len(dates))
    frame = pd.DataFrame({'A': 100*np.exp(.001*x+.02*np.sin(x*.31)),
                          'ETF': 75*np.exp(-.0001*x+.01*np.cos(x*.23))}, index=dates)
    meta = dict(dates=[str(dates[i].date()) for i in (294, 320, 341)],
                symbols=[['A', 'ETF']]*3, etfs=['ETF'])
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


class MeasureTests(unittest.TestCase):
    def test_exact_windows_and_formula(self):
        p = prices(.001 + np.linspace(.002, .025, 294)*np.sin(np.arange(294)))
        out = k.measure(p)
        self.assertTrue(out['available'])
        shares = {}
        for name, lo, hi in [('old252', 0, 252), ('current63', 231, 294)]:
            r = [float(p[i]/p[i-1]-1.) for i in range(lo+1, hi+1)]
            total = math.fsum(x*x for x in r)
            down = math.fsum(x*x for x in r if x < 0.)
            self.assertEqual(out['sum_squares_'+name], total)
            self.assertEqual(out['downside_sum_squares_'+name], down)
            shares[name] = down/total
            self.assertEqual(out['share_'+name], shares[name])
        self.assertEqual(out['candidate'], shares['old252']-shares['current63'])
        self.assertEqual(out['control'], .5-shares['current63'])
        self.assertNotEqual(out['candidate'], out['control'])
        self.assertEqual(out['downside_share_estimates'], 2)

    def test_direction_and_bounds(self):
        falling = k.measure(prices(np.r_[np.full(231,.01), np.full(63,-.01)]))
        rising = k.measure(prices(np.r_[np.full(231,-.01), np.full(63,.01)]))
        self.assertLess(falling['candidate'], 0.)
        self.assertEqual(falling['control'], -.5)
        self.assertGreater(rising['candidate'], 0.)
        self.assertEqual(rising['control'], .5)
        for out in (falling, rising):
            self.assertLessEqual(abs(out['candidate']), 1.)
            self.assertLessEqual(abs(out['control']), .5)

    def test_constant_positive_and_negative_growth_valid_not_missing(self):
        for r, share, control in [(.002,0.,.5),(-.002,1.,-.5)]:
            out = k.measure(prices(np.full(294,r)))
            self.assertTrue(out['available'])
            self.assertEqual(out['share_old252'], share)
            self.assertEqual(out['share_current63'], share)
            self.assertEqual(out['candidate'], 0.)
            self.assertEqual(out['control'], control)

    def test_zero_and_zero_loss_distinguished(self):
        out = k.measure(np.ones(295))
        self.assertFalse(out['available'])
        self.assertEqual(out['reason'], 'downside_denominator_floor')
        self.assertEqual(out['downside_share_estimates'], 2)
        for name in k.WINDOWS:
            self.assertEqual(out['sum_squares_'+name], 0.)
            self.assertEqual(out['downside_sum_squares_'+name], 0.)
            self.assertIsNone(out['share_'+name])
        self.assertTrue(math.isnan(out['candidate']))

    def test_old_only_risk_and_current_only_risk_both_common_unavailable(self):
        for r in [np.r_[.01*np.sin(np.arange(231)), np.zeros(63)],
                  np.r_[np.zeros(252), .01*np.sin(np.arange(42))]]:
            out = k.measure(prices(r))
            self.assertFalse(out['available'])
            self.assertEqual(out['downside_share_estimates'], 2)
            self.assertTrue(math.isnan(out['candidate']) and math.isnan(out['control']))
            self.assertIsNone(out['share_old252'])
            self.assertIsNone(out['share_current63'])

    def test_near_floor_from_actual_synthetic_prices(self):
        low=np.ones(295);low[240:]=1.+9e-13
        high=np.ones(295);high[240:]=1.+1.1e-12
        self.assertFalse(k.measure(low)['available'])
        self.assertTrue(k.measure(high)['available'])
        out=k.measure(high);floor=out['sum_squares_current63']
        with patch.object(k,'DENOMINATOR_FLOOR',floor):
            self.assertFalse(k.measure(high)['available'])
        with patch.object(k,'DENOMINATOR_FLOOR',np.nextafter(floor,0.)):
            self.assertTrue(k.measure(high)['available'])

    def test_price_scale_invariance(self):
        p=prices(.001+.02*np.sin(np.arange(294)*.3));out=k.measure(p)
        for scale in (2.**-100,2.**100,1e-100,1e100):
            got=k.measure(p*scale)
            self.assertTrue(got['available'])
            for key in ('candidate','control','share_old252','share_current63'):
                self.assertAlmostEqual(got[key],out[key],places=12)

    def test_invalid_and_nonnumeric_sources_have_no_estimates(self):
        for value in (0.,-1.,np.nan,np.inf,-np.inf,'bad'):
            p=np.ones(295,dtype=object);p[100]=value;out=k.measure(p)
            self.assertFalse(out['available'])
            self.assertEqual(out['downside_share_estimates'],0)
            self.assertIn(out['reason'],('invalid_adjusted_price','nonnumeric_adjusted_price'))

    def test_shape_is_contract_error(self):
        for p in (np.ones(294), np.ones((295,1)), np.ones(296)):
            with self.assertRaises(ValueError):k.measure(p)

    def test_valid_source_ratio_overflow_and_underflow_are_errors(self):
        for pair in ((1e-300,1e300),(1e300,1e-300)):
            p=np.ones(295);p[:2]=pair
            with self.assertRaises(FloatingPointError):k.measure(p)

    def test_finite_ratio_square_overflow_is_error(self):
        p=np.ones(295);p[1]=1e200
        with self.assertRaises(FloatingPointError):k.measure(p)

    def test_finite_squares_sum_overflow_is_error(self):
        p=np.ones(295);p[1::2]=1e154
        with self.assertRaises(FloatingPointError):k.measure(p)


class BuildTests(unittest.TestCase):
    def test_independent_full_audit_crosscheck(self):
        import monthly_downside_mix179_independent as independent
        from test_monthly_downside_mix179_independent import fixture as other_fixture
        cases=[fixture(),other_fixture()]
        p,m,c=fixture();p['adj'].iloc[:,:]=1.;cases.append((p,m,c))
        p,m,c=fixture();p['adj']=p['adj'].drop(pd.Timestamp(c[100]));cases.append((p,m,c))
        p,m,c=fixture();p['adj'].iloc[140,0]=np.nan;cases.append((p,m,c))
        p,m,c=fixture();m['dates'][0]=c[293];m['symbols'][1]=['A','MISSING'];cases.append((p,m,c))
        p,m,c=fixture();x=np.arange(len(p['adj']))
        p['adj']['A']=1.01**x;p['adj']['ETF']=.99**x;cases.append((p,m,c))
        for p,m,c in cases:
            values,audit=k.build_features(p,m,c)
            receipt=independent.verify_features(p,m,c,values,audit)
            self.assertTrue(receipt['passed'])
            self.assertEqual(receipt['downside_share_estimates'],audit['downside_share_estimates'])

    def test_dates_counts_json_and_etf_same_treatment(self):
        p,m,c=fixture();p['adj']['ETF']=p['adj']['A'];v,a=k.build_features(p,m,c)
        self.assertEqual(v['available'].shape,(3,2));self.assertTrue(v['available'].all())
        self.assertEqual(a['downside_share_estimates'],12)
        self.assertEqual(a['checked_cells'],6)
        for key in v:np.testing.assert_array_equal(v[key][:,0],v[key][:,1])
        row=a['event_history'][0]
        self.assertEqual(row['price_start'],c[0]);self.assertEqual(row['price_end'],c[294])
        for name,lo,hi,n in [('old252',0,252,252),('current63',231,294,63)]:
            self.assertEqual(row['windows'][name],dict(price_start=c[lo],price_end=c[hi],return_count=n))
        self.assertEqual(row['cells'][0]['date'],c[294])
        self.assertFalse(a['ranking_performed']);self.assertFalse(a['portfolio_returns_calculated'])
        json.dumps(a,allow_nan=False)

    def test_all_floor_keeps_two_estimates_per_cell(self):
        p,m,c=fixture();p['adj'].iloc[:,:]=1.;v,a=k.build_features(p,m,c)
        self.assertFalse(v['available'].any());self.assertEqual(a['downside_share_estimates'],12)
        self.assertEqual(a['reason_counts'],{'downside_denominator_floor':6})
        json.dumps(a,allow_nan=False)

    def test_every_prefix_and_empty_prefix(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c)
        for n in range(4):
            nv,na=k.build_features(p,m,c,limit=n)
            for key in v:np.testing.assert_array_equal(nv[key],v[key][:n])
            self.assertEqual(na['event_history'],a['event_history'][:n])
            self.assertEqual(na['downside_share_estimates'],4*n)
        for n in (-1,4,True,1.,'1'):
            with self.assertRaises(ValueError):k.build_features(p,m,c,limit=n)

    def test_future_source_never_converted_and_extra_session_ignored(self):
        class Poison:
            def __float__(self):raise AssertionError('Future source was accessed')
        p,m,c=fixture();v,a=k.build_features(p,m,c,limit=1)
        poisoned={'adj':p['adj'].astype(object)};poisoned['adj'].iloc[295:,:]=Poison()
        extra=pd.DataFrame({'A':[Poison()],'ETF':[Poison()]},index=pd.DatetimeIndex(['2010-01-09']))
        poisoned['adj']=pd.concat([poisoned['adj'],extra]).sort_index()
        nv,na=k.build_features(poisoned,m,c,limit=1)
        for key in v:np.testing.assert_array_equal(nv[key],v[key])
        self.assertEqual(na,a)

    def test_historical_source_before_window_not_converted(self):
        p,m,c=fixture();m['dates']=[c[320]];m['symbols']=[['A','ETF']]
        v,a=k.build_features(p,m,c)
        p['adj']=p['adj'].astype(object);p['adj'].iloc[:26,:]='unused'
        nv,na=k.build_features(p,m,c)
        for key in v:np.testing.assert_array_equal(nv[key],v[key])
        self.assertEqual(na,a)

    def test_missing_session_neutral_no_fill(self):
        p,m,c=fixture();p['adj']=p['adj'].drop(pd.Timestamp(c[100]));v,a=k.build_features(p,m,c,limit=1)
        self.assertFalse(v['available'].any());self.assertEqual(a['downside_share_estimates'],0)
        self.assertEqual(a['reason_counts'],{'missing_official_session':2})
        self.assertEqual(a['event_history'][0]['missing_official_sessions'],[c[100]])

    def test_short_history_and_unknown_symbol(self):
        p,m,c=fixture();m['dates'][0]=c[293];m['symbols'][1]=['A','MISSING']
        v,a=k.build_features(p,m,c)
        self.assertFalse(v['available'][0].any());self.assertFalse(v['available'][1,1])
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        self.assertIsNone(a['event_history'][0]['windows'])
        self.assertEqual(a['event_history'][1]['cells'][1]['reason'],'missing_symbol')

    def test_missing_one_security_not_entire_cross_section(self):
        p,m,c=fixture();p['adj'].iloc[140,0]=np.nan;v,a=k.build_features(p,m,c,limit=1)
        self.assertFalse(v['available'][0,0]);self.assertTrue(v['available'][0,1])
        self.assertEqual(a['downside_share_estimates'],2)

    def test_empty_events(self):
        p,_,c=fixture();v,a=k.build_features(p,dict(dates=[],symbols=[],etfs=[]),c)
        self.assertEqual(v['available'].shape,(0,0));self.assertEqual(a['checked_cells'],0)

    def test_metadata_calendar_and_panel_errors(self):
        p,m,c=fixture();invalid=[]
        for key in ('dates','symbols','etfs'):
            bad=copy.deepcopy(m);bad.pop(key);invalid.append(bad)
        bad=copy.deepcopy(m);bad['symbols'][1]=['A','A'];invalid.append(bad)
        bad=copy.deepcopy(m);bad['symbols'][1]=['A'];invalid.append(bad)
        bad=copy.deepcopy(m);bad['dates'][1]=bad['dates'][0];invalid.append(bad)
        for bad in invalid:
            with self.assertRaises(ValueError):k.build_features(p,bad,c)
        for bad in ([],c[::-1],c+[c[-1]],['2010-01-09']):
            with self.assertRaises(ValueError):k.build_features(p,m,bad)
        with self.assertRaises(ValueError):k.build_features(dict(p,extra=1),m,c)
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'].iloc[::-1]},m,c)

    def test_inputs_unchanged(self):
        p,m,c=fixture();original=p['adj'].copy();oldm=copy.deepcopy(m);oldc=c[:]
        k.build_features(p,m,c)
        pd.testing.assert_frame_equal(p['adj'],original);self.assertEqual(m,oldm);self.assertEqual(c,oldc)


if __name__ == '__main__':
    unittest.main(verbosity=2)
