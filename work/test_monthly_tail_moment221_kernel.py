"""Synthetic source windows and arithmetic for the predeclared tail-moment contrast."""
from copy import deepcopy
import math
import unittest
import numpy as np
import pandas as pd
import monthly_tail_moment221_kernel as k


def prices(returns):
    p=[100.]
    for r in returns:p.append(p[-1]*(1.+r))
    return np.asarray(p)


def fixture():
    idx=pd.bdate_range('2014-01-01','2021-01-29')
    t=np.arange(len(idx))
    frame=pd.DataFrame({'A':100*np.exp(.0007*t+.04*np.sin(t/9)),
        'B':150*np.exp(.0004*t+.03*np.sin(t/6)),
        'E':50*np.exp(.0003*t+.02*np.sin(t/7)),
        'FLAT':np.ones(len(idx))*100},index=idx)
    dates=['2016-08-17','2016-08-31','2017-01-31','2018-06-29','2019-12-31']
    meta={'dates':dates,'symbols':[['A','B','E','FLAT'] for _ in dates],'etfs':['E']}
    return {'adj':frame},meta,list(idx.strftime('%Y-%m-%d'))


class ArithmeticTests(unittest.TestCase):
    def test_no_loss_constant_and_rising_shared_unavailable(self):
        for p in [np.ones(253)*100,prices([.001]*252)]:
            x=k.measure(p);self.assertTrue(x['source_available']);self.assertFalse(x['available'])
            self.assertEqual(x['reason'],'no_negative_returns');self.assertEqual(x['negative_day_count'],0)
            self.assertEqual(x['quadratic_risk'],0.);self.assertEqual(x['cubic_risk'],0.)
            self.assertTrue(np.isnan(x['raw_candidate']))

    def test_hand_computed_two_losses_over_all252_days(self):
        p=np.full(253,64.);p[1:]=32.;p[2:]=24.
        x=k.measure(p);self.assertEqual(x['negative_day_count'],2)
        self.assertEqual(x['momentum'],-.625)
        self.assertEqual(x['downside_square_sum'],.3125)
        self.assertEqual(x['downside_cube_sum'],.140625)
        self.assertEqual(x['quadratic_risk'],math.sqrt(.3125/252))
        self.assertEqual(x['cubic_risk'],float(np.cbrt(.140625/252)))
        self.assertEqual(x['raw_candidate'],x['momentum']/x['cubic_risk'])
        self.assertGreater(x['raw_candidate'],x['raw_control'])

    def test_positive_momentum_more_tail_penalty(self):
        x=k.measure(prices([-.1]+[.002]*251))
        self.assertGreater(x['momentum'],0.);self.assertLess(x['raw_candidate'],x['raw_control'])
        self.assertGreater(x['cubic_risk'],x['quadratic_risk'])

    def test_zero_growth_with_loss_is_available(self):
        p=np.ones(253)*100;p[1]=50
        x=k.measure(p);self.assertTrue(x['available']);self.assertEqual(x['momentum'],0.)
        self.assertEqual(x['raw_candidate'],0.);self.assertEqual(x['raw_control'],0.)

    def test_one_tiny_negative_is_not_epsilon_floored(self):
        p=np.ones(253);p[1:]=np.nextafter(1.,0.)
        x=k.measure(p);self.assertTrue(x['available']);self.assertEqual(x['negative_day_count'],1)
        self.assertGreater(x['cubic_risk'],0.);self.assertGreater(x['quadratic_risk'],0.)

    def test_binary_scale_invariance_and_no_mutation(self):
        p=prices([.002,-.001]*126);copy=p.copy()
        a,b=k.measure(p),k.measure(p*8)
        for key in ['momentum','raw_candidate','raw_control','cubic_risk','quadratic_risk']:
            self.assertEqual(a[key],b[key])
        np.testing.assert_array_equal(p,copy)

    def test_fixed_width(self):
        for n in [0,252,254]:
            with self.assertRaises(ValueError):k.measure(np.ones(n))

    def test_source_invalid_not_numeric_zero(self):
        for v in [0.,-1.,np.nan,np.inf]:
            p=np.ones(253);p[71]=v
            a=k.measure(p);self.assertFalse(a['source_available']);self.assertTrue(np.isnan(a['momentum']))
        for v in [True,'100',None]:
            p=[1.]*253;p[71]=v
            self.assertFalse(k.measure(p)['source_available'])

    def test_overflow_is_hard_error_not_missing(self):
        p=np.ones(253);p[0]=1e-300;p[-1]=1e300
        with self.assertRaises(FloatingPointError):k.measure(p)

    def test_rank_domain_ties_including_singleton(self):
        q=k.percentile_rank(np.array([5.,5.,1.,np.nan]),np.array([1,1,1,0],bool))
        np.testing.assert_array_equal(q[:3],np.array([2.5/3-.5,2.5/3-.5,1/3-.5]))
        self.assertTrue(np.isnan(q[3]));self.assertEqual(k.percentile_rank(np.array([3.]),np.array([True]))[0],.5)


class WindowTests(unittest.TestCase):
    def test_full_axes_masks_and_counts(self):
        panel,meta,cal=fixture();v,a=k.build_features(panel,meta,cal)
        self.assertEqual(set(v),set(k.FEATURE_KEYS));self.assertEqual(v['available'].shape,(5,4))
        self.assertEqual(a['source_available_cells'],20);self.assertEqual(a['available_cells'],15)
        self.assertEqual(a['downside_moment_evaluations'],40);self.assertEqual(a['normalized_momentum_scores'],30)
        self.assertTrue(v['available'][:,2].all());self.assertFalse(v['available'][:,3].any())
        for row in a['event_history']:
            t=cal.index(row['date']);self.assertEqual(row['price_start'],cal[t-294])
            self.assertEqual(row['price_end'],cal[t-42]);self.assertEqual(row['first_return_date'],cal[t-293])

    def test_latest42_prices_cannot_change_feature_or_missingness(self):
        panel,meta,cal=fixture();meta={**meta,'dates':meta['dates'][-1:],'symbols':meta['symbols'][-1:]}
        before,a=k.build_features(panel,meta,cal);t=cal.index(meta['dates'][0])
        changed={'adj':panel['adj'].copy()};changed['adj'].loc[cal[t-41]:, :]=np.nan
        after,b=k.build_features(changed,meta,cal)
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        self.assertEqual(a,b)

    def test_missing_official_session_never_shortens_window(self):
        panel,meta,cal=fixture();meta={**meta,'dates':meta['dates'][-1:],'symbols':meta['symbols'][-1:]}
        t=cal.index(meta['dates'][0]);panel['adj']=panel['adj'].drop(pd.Timestamp(cal[t-100]))
        v,a=k.build_features(panel,meta,cal);self.assertFalse(v['available'].any())
        self.assertEqual(a['reason_counts'],{'missing_official_session':4})

    def test_missing_symbol_preserves_other_candidates(self):
        panel,meta,cal=fixture();panel['adj']=panel['adj'].drop(columns='B')
        v,a=k.build_features(panel,meta,cal);self.assertFalse(v['available'][:,1].any())
        self.assertTrue(v['available'][:,0].all());self.assertEqual(a['reason_counts']['missing_symbol'],5)

    def test_initial_partial_allowed_other_partial_forbidden(self):
        panel,meta,cal=fixture();k.build_features(panel,meta,cal)
        meta['dates'][0]='2016-08-18'
        with self.assertRaises(ValueError):k.build_features(panel,meta,cal)

    def test_invalid_metadata_and_calendar_fail(self):
        panel,meta,cal=fixture()
        for bad in [cal[::-1],cal+[cal[-1]],['2014-01-04']]:
            with self.assertRaises(ValueError):k.build_features(panel,meta,bad)
        duplicate=deepcopy(meta);duplicate['symbols'][0][1]='A'
        with self.assertRaises(ValueError):k.build_features(panel,duplicate,cal)

    def test_all_physical_source_meta_prefixes(self):
        panel,meta,cal=fixture();v,a=k.build_features(panel,meta,cal)
        for n in range(6):
            small={**meta,'dates':meta['dates'][:n],'symbols':meta['symbols'][:n]}
            frame=panel['adj'].loc[:meta['dates'][n-1]] if n else panel['adj'].iloc[:0]
            vv,aa=k.build_features({'adj':frame},small,cal)
            self.assertEqual(aa['event_history'],a['event_history'][:n])
            for key in v:
                if n:np.testing.assert_array_equal(vv[key],v[key][:n])
                else:self.assertEqual(vv[key].shape,(0,0))

    def test_no_mutation_and_limit(self):
        panel,meta,cal=fixture();before=panel['adj'].copy();copy=deepcopy(meta)
        v,a=k.build_features(panel,meta,cal,limit=2);self.assertEqual(v['available'].shape,(2,4))
        pd.testing.assert_frame_equal(before,panel['adj']);self.assertEqual(copy,meta)
        for n in [-1,True,6]:
            with self.assertRaises(ValueError):k.build_features(panel,meta,cal,limit=n)


if __name__=='__main__':unittest.main(verbosity=2)
