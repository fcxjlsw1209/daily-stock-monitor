"""Synthetic source arithmetic, availability, and timing tests only."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_vol_inflation178_kernel as k


def prices(returns):
    return 100.*np.cumprod(np.r_[1., 1.+np.asarray(returns)])


def fixture():
    dates = pd.bdate_range('2010-01-04', periods=365)
    x = np.arange(len(dates))
    frame = pd.DataFrame({'A': 100*np.exp(.001*x+.02*np.sin(x*.31)),
                          'ETF': 75*np.exp(.001*x+.01*np.cos(x*.23))}, index=dates)
    meta = dict(dates=[str(dates[i].date()) for i in (294, 320, 341)],
                symbols=[['A', 'ETF']]*3, etfs=['ETF'])
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


class MeasureTests(unittest.TestCase):
    def test_exact_windows_and_formula(self):
        p = prices(.001+np.linspace(.002, .025, 294)*np.sin(np.arange(294)))
        out = k.measure(p)
        self.assertTrue(out['available'])
        sigma = {}
        for name, bounds in {'old252': (0,253), 'current63': (231,295), 'current252': (42,295)}.items():
            w=p[bounds[0]:bounds[1]]; r=w[1:]/w[:-1]-1
            sigma[name]=np.std(r, ddof=1)
            self.assertEqual(out['sigma_'+name],sigma[name])
        for arm, ref in [('candidate','old252'),('control','current252')]:
            log=math.log(sigma['current63'])-math.log(sigma[ref])
            self.assertEqual(out[arm+'_log_ratio'],log)
            self.assertEqual(out[arm],-math.tanh(max(0.,log)))
        self.assertNotEqual(out['candidate'],out['control'])
        self.assertEqual(out['volatility_estimates'],3)

    def test_deterministic_current_risk_inflation_and_contraction(self):
        low=.001+.002*np.sin(np.arange(231))
        high=.001+.04*np.sin(np.arange(63))
        self.assertLess(k.measure(prices(np.r_[low,high]))['candidate'],0.)
        out=k.measure(prices(np.r_[high,np.tile(low,2)[:231]]))
        self.assertEqual(out['candidate'],0.)
        self.assertEqual(out['control'],0.)

    def test_raw_bounds(self):
        for amp in (.0001,.01,.2):
            out=k.measure(prices(.001+amp*np.sin(np.arange(294))))
            for key in ('candidate','control'):
                self.assertGreaterEqual(out[key],-1.);self.assertLessEqual(out[key],0.)

    def test_scale_invariance(self):
        p=prices(.001+.02*np.sin(np.arange(294)*.3));out=k.measure(p)
        for scale in (1e-100,1e100):
            got=k.measure(p*scale)
            for key in ('candidate','control','sigma_old252','sigma_current63','sigma_current252'):
                self.assertAlmostEqual(got[key],out[key],places=12)

    def test_all_three_computed_on_floor_failure(self):
        out=k.measure(np.ones(295))
        self.assertFalse(out['available']);self.assertEqual(out['reason'],'volatility_floor')
        self.assertEqual(out['volatility_estimates'],3)
        for key in ('sigma_old252','sigma_current63','sigma_current252'):self.assertEqual(out[key],0.)
        self.assertIsNone(out['candidate_log_ratio']);self.assertTrue(math.isnan(out['candidate']))

    def test_only_current_floor_makes_common_unavailable(self):
        p=prices(np.r_[.01*np.sin(np.arange(231)),np.zeros(63)])
        out=k.measure(p)
        self.assertGreater(out['sigma_old252'],1e-12);self.assertEqual(out['sigma_current63'],0.)
        self.assertFalse(out['available']);self.assertEqual(out['volatility_estimates'],3)

    def test_floor_strict(self):
        p=prices(.01*np.sin(np.arange(294)))
        with patch.object(k.np,'std',return_value=k.STD_FLOOR):
            self.assertFalse(k.measure(p)['available'])
        with patch.object(k.np,'std',return_value=np.nextafter(k.STD_FLOOR,math.inf)):
            self.assertTrue(k.measure(p)['available'])

    def test_invalid_source_neutral(self):
        for value in (0.,-1.,np.nan,np.inf,-np.inf):
            p=np.ones(295);p[100]=value;out=k.measure(p)
            self.assertEqual(out['reason'],'invalid_adjusted_price');self.assertEqual(out['volatility_estimates'],0)
            self.assertTrue(math.isnan(out['candidate']));self.assertTrue(math.isnan(out['control']))

    def test_nonnumeric_source_neutral(self):
        p=np.ones(295,dtype=object);p[100]='bad'
        self.assertEqual(k.measure(p)['reason'],'nonnumeric_adjusted_price')

    def test_bad_shape(self):
        for value in (np.ones(294),np.ones((295,1)),np.ones(296)):
            with self.assertRaises(ValueError):k.measure(value)

    def test_valid_source_ratio_overflow_and_underflow_are_errors(self):
        for a,b in [(1e-300,1e300),(1e300,1e-300)]:
            p=np.ones(295);p[0:2]=[a,b]
            with self.assertRaises(FloatingPointError):k.measure(p)

    def test_finite_ratio_variance_overflow_is_error(self):
        p=np.ones(295);p[1]=1e200
        with self.assertRaises(FloatingPointError):k.measure(p)


class BuildTests(unittest.TestCase):
    def test_independent_scalar_audit_crosschecks_both_fixtures_and_neutral_masks(self):
        import monthly_vol_inflation178_independent as independent
        from test_monthly_vol_inflation178_independent import fixture as other_fixture
        cases=[fixture(),other_fixture()]
        p,m,c=fixture();p['adj'].iloc[:,:]=1.;cases.append((p,m,c))
        p,m,c=fixture();p['adj']=p['adj'].drop(pd.Timestamp(c[100]));cases.append((p,m,c))
        p,m,c=fixture();p['adj'].iloc[200,0]=np.nan;cases.append((p,m,c))
        for p,m,c in cases:
            v,a=k.build_features(p,m,c)
            check=independent.verify_features(p,m,c,v,a)
            self.assertTrue(check['passed'])
            self.assertEqual(check['volatility_estimates'],a['volatility_estimates'])

    def test_audit_dates_counts_json_and_etf_same_treatment(self):
        panel,meta,calendar=fixture();v,a=k.build_features(panel,meta,calendar)
        self.assertEqual(v['available'].shape,(3,2));self.assertTrue(v['available'].all())
        self.assertEqual(a['volatility_estimates'],18);self.assertEqual(a['checked_cells'],6)
        event=a['event_history'][0]
        self.assertEqual(event['price_start'],calendar[0]);self.assertEqual(event['price_end'],calendar[294])
        for key,lo,hi,n in [('old252',0,252,252),('current63',231,294,63),('current252',42,294,252)]:
            self.assertEqual(event['windows'][key],dict(price_start=calendar[lo],price_end=calendar[hi],return_count=n))
        self.assertTrue(event['cells'][1]['available']);json.dumps(a,allow_nan=False)
        self.assertFalse(a['portfolio_returns_calculated']);self.assertFalse(a['ranking_performed'])

    def test_prefix_identical_and_limit_validation(self):
        panel,meta,calendar=fixture();allv,alla=k.build_features(panel,meta,calendar)
        for n in range(4):
            v,a=k.build_features(panel,meta,calendar,limit=n)
            for key in v:np.testing.assert_array_equal(v[key],allv[key][:n])
            self.assertEqual(a['event_history'],alla['event_history'][:n])
            self.assertEqual(a['volatility_estimates'],n*6)
        for n in (-1,4,True,1.,'1'):
            with self.assertRaises(ValueError):k.build_features(panel,meta,calendar,limit=n)

    def test_future_source_not_numeric_converted(self):
        class Poison:
            def __float__(self):raise AssertionError('Future price read')
        panel,meta,calendar=fixture();v,a=k.build_features(panel,meta,calendar,limit=1)
        p={'adj':panel['adj'].astype(object)};p['adj'].iloc[295:,:]=Poison()
        new,b=k.build_features(p,meta,calendar,limit=1)
        for key in v:np.testing.assert_array_equal(v[key],new[key])
        self.assertEqual(a,b)

    def test_missing_session_no_fill(self):
        panel,meta,calendar=fixture();panel['adj']=panel['adj'].drop(pd.Timestamp(calendar[100]))
        v,a=k.build_features(panel,meta,calendar,limit=1)
        self.assertFalse(v['available'].any());self.assertEqual(a['volatility_estimates'],0)
        self.assertEqual(a['event_history'][0]['missing_official_sessions'],[calendar[100]])
        self.assertEqual(a['reason_counts'],{'missing_official_session':2})

    def test_noncalendar_extra_row_ignored(self):
        panel,meta,calendar=fixture();v,a=k.build_features(panel,meta,calendar)
        extra=pd.DataFrame({'A':['poison'],'ETF':['poison']},index=pd.DatetimeIndex(['2010-01-09']))
        panel['adj']=pd.concat([panel['adj'],extra]).sort_index()
        vv,aa=k.build_features(panel,meta,calendar)
        for key in v:np.testing.assert_array_equal(v[key],vv[key])
        self.assertEqual(a,aa)

    def test_short_history_and_unknown_symbol(self):
        panel,meta,calendar=fixture();meta['dates'][0]=calendar[293]
        meta['symbols'][1]=['A','MISSING']
        v,a=k.build_features(panel,meta,calendar)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        self.assertEqual(a['event_history'][1]['cells'][1]['reason'],'missing_symbol')
        self.assertFalse(v['available'][0].any());self.assertFalse(v['available'][1,1])

    def test_invalid_input_does_not_change_other_symbol(self):
        panel,meta,calendar=fixture();panel['adj'].iloc[140,0]=np.nan
        v,a=k.build_features(panel,meta,calendar,limit=1)
        self.assertFalse(v['available'][0,0]);self.assertTrue(v['available'][0,1])
        self.assertEqual(a['volatility_estimates'],3)

    def test_empty_events(self):
        panel,_,calendar=fixture();v,a=k.build_features(panel,dict(dates=[],symbols=[],etfs=[]),calendar)
        self.assertEqual(v['available'].shape,(0,0));self.assertEqual(a['checked_cells'],0)

    def test_metadata_and_calendar_validation(self):
        panel,meta,calendar=fixture()
        invalid=[]
        q=copy.deepcopy(meta);q.pop('etfs');invalid.append(q)
        q=copy.deepcopy(meta);q['symbols'][1]=['A','A'];invalid.append(q)
        q=copy.deepcopy(meta);q['symbols'][1]=['A'];invalid.append(q)
        q=copy.deepcopy(meta);q['dates'][1]=q['dates'][0];invalid.append(q)
        q=copy.deepcopy(meta);q['etfs']=['ETF','ETF'];invalid.append(q)
        for q in invalid:
            with self.assertRaises(ValueError):k.build_features(panel,q,calendar)
        for c in (calendar[::-1],calendar+calendar[-1:],[],['2010-01-09']):
            with self.assertRaises(ValueError):k.build_features(panel,meta,c)

    def test_panel_validation_and_no_mutation(self):
        panel,meta,calendar=fixture();original=panel['adj'].copy();m=copy.deepcopy(meta)
        k.build_features(panel,meta,calendar);pd.testing.assert_frame_equal(panel['adj'],original);self.assertEqual(meta,m)
        with self.assertRaises(ValueError):k.build_features(dict(panel,extra=original),meta,calendar)
        panel['adj']=original.iloc[::-1]
        with self.assertRaises(ValueError):k.build_features(panel,meta,calendar)


if __name__=='__main__':
    unittest.main(verbosity=2)
