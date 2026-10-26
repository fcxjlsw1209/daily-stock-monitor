"""Synthetic-only checks of weekly alignment, nested regressions and ranking."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_price_delay140_factor as f


def price_path(returns):
    return 100. * np.r_[1., np.cumprod(1. + np.asarray(returns))]


def vectors():
    rng = np.random.default_rng(140)
    market = rng.normal(.002, .03, 56)
    y = .001 + .2*market[4:] + .8*market[3:-1] + rng.normal(0., .005, 52)
    return price_path(y), price_path(market)


def fixture():
    sessions = pd.bdate_range('2015-01-01', periods=460)
    names = ['S'+str(i) for i in range(32)] + ['E'+str(i) for i in range(17)]
    rng = np.random.default_rng(140)
    market = rng.normal(.0003, .01, len(sessions)-1)
    data = {'SPY': price_path(market)}
    for i, name in enumerate(names):
        data[name] = price_path((.2+i*.01)*market + .3*np.roll(market, 5) +
                                rng.normal(.0001, .006, len(market)))
    meta = dict(dates=[str(sessions[420].date()), str(sessions[445].date())],
                symbols=[names[:], names[:]], etfs=names[32:])
    return {'adj': pd.DataFrame(data, index=sessions)}, meta, list(sessions)


class RegressionTests(unittest.TestCase):
    def test_known_nested_fit_and_direct_oracle(self):
        y, x = vectors(); got = f.regression_delay(y, x)
        yr = y[1:]/y[:-1]-1.; xr = x[1:]/x[:-1]-1.
        matrix = np.column_stack([np.ones(52)] + [xr[4-j:56-j] for j in range(5)])
        for label, design in (('restricted', matrix[:, :2]), ('full', matrix)):
            q, r = np.linalg.qr(design, mode='reduced')
            beta = np.linalg.solve(r, q.T @ yr)
            expected = 1.-np.square(yr-design@beta).sum()/np.square(yr-yr.mean()).sum()
            self.assertAlmostEqual(got[label+'_r2'], expected, places=12)
        self.assertTrue(got['available'])
        self.assertEqual((got['restricted_rank'], got['full_rank']), (2, 6))
        self.assertAlmostEqual(got['raw_delay'], 1.-got['restricted_r2']/got['full_r2'], places=14)
        self.assertAlmostEqual(got['raw_nonfit'], 1.-got['restricted_r2'], places=14)
        self.assertGreater(got['raw_delay'], .5)
    def test_contemporaneous_exact_fit_has_zero_delay(self):
        _, x = vectors(); market=x[1:]/x[:-1]-1.
        got = f.regression_delay(price_path(.01+1.2*market[4:]), x)
        self.assertAlmostEqual(got['full_r2'], 1., places=14)
        self.assertAlmostEqual(got['restricted_r2'], 1., places=14)
        self.assertAlmostEqual(got['raw_delay'], 0., places=14)
    def test_positive_scale_invariance(self):
        y, x = vectors(); a=f.regression_delay(y,x); b=f.regression_delay(y*3,x*7)
        for k in ('restricted_r2','full_r2','raw_delay','raw_nonfit'):
            self.assertAlmostEqual(a[k],b[k],places=12)
    def test_zero_response_or_market_variance(self):
        y,x=vectors()
        self.assertEqual(f.regression_delay(np.ones(53),x)['reason'],'zero_response_variance')
        self.assertEqual(f.regression_delay(y,np.ones(57))['reason'],'zero_market_variance')
    def test_rank_deficiency_is_joint_unavailable(self):
        market=np.tile([.01,-.01],28)
        y,_=vectors();got=f.regression_delay(y,price_path(market))
        self.assertFalse(got['available']);self.assertEqual(got['reason'],'rank_deficient_full')
        self.assertIsNone(got['raw_nonfit']);self.assertIsNone(got['raw_delay'])
    def test_full_r2_numeric_floor(self):
        _,x=vectors();market=x[1:]/x[:-1]-1.
        matrix=np.column_stack([np.ones(52)]+[market[4-j:56-j] for j in range(5)])
        q,_=np.linalg.qr(matrix,mode='complete'); y=.001+.02*q[:,6]
        got=f.regression_delay(price_path(y),x)
        self.assertEqual(got['reason'],'full_r2_below_numeric_floor')
        self.assertLess(got['full_r2'],1e-8);self.assertFalse(got['available'])
    def test_condition_gate(self):
        y,x=vectors();original=np.linalg.lstsq
        def bad_condition(*args,**kwargs):
            beta,residual,rank,singular=original(*args,**kwargs)
            singular[-1]=singular[0]/(f.MAX_CONDITION*2)
            return beta,residual,rank,singular
        with patch.object(np.linalg,'lstsq',side_effect=bad_condition):
            got=f.regression_delay(y,x)
        self.assertEqual(got['reason'],'ill_conditioned_restricted')
    def test_bad_shapes_types_and_endpoints(self):
        y,x=vectors()
        for bad in (x[:-1],np.full(57,True),np.full(57,'1.0')):
            with self.assertRaises(ValueError):f.regression_delay(y,bad)
        for value in (0.,-1.,np.nan,np.inf):
            changed=x.copy();changed[3]=value
            with self.assertRaises(ValueError):f.regression_delay(y,changed)
    def test_large_derived_return_fails_explicitly(self):
        y,x=vectors();x[0]=1e-300;x[1]=1e300
        with self.assertRaises(FloatingPointError):f.regression_delay(y,x)


class RankTests(unittest.TestCase):
    def test_endpoints_ties_and_missing(self):
        np.testing.assert_array_equal(f.centered_midrank([10.,20.,20.,30.,np.nan]),
                                      [-.5,0.,0.,.5,np.nan])
        np.testing.assert_array_equal(f.centered_midrank([3.,3.,3.]),[0.,0.,0.])
        self.assertTrue(np.isnan(f.centered_midrank([3.,np.nan])).all())
    def test_nonboolean_numeric_only(self):
        with self.assertRaises(ValueError):f.centered_midrank([True,False])


class CalendarBuilderTests(unittest.TestCase):
    def test_week_boundary_holiday_and_cutoff(self):
        p,m,s=fixture();signal=pd.Timestamp(m['dates'][0]);pos=s.index(signal);cut=s[pos-42]
        cutoff,nominal,actual=f.weekly_endpoints(s,signal)
        self.assertEqual(cutoff,cut);self.assertEqual(len(actual),57)
        self.assertTrue((nominal.weekday==4).all());self.assertLessEqual(nominal[-1],cut)
        self.assertTrue((actual<=nominal).all())
        removed=nominal[-2];self.assertIn(removed,s)
        holidays=[x for x in s if x!=removed]
        _,n,a=f.weekly_endpoints(holidays,signal)
        idx=list(n).index(removed);self.assertEqual(a[idx],removed-pd.Timedelta(days=1))
        # A mid-week cutoff may never use that week's unfinished Friday.
        self.assertTrue(all(n<=holidays[holidays.index(signal)-42]))
    def test_complete_builder_ranks_audit_and_nonmutation(self):
        p,m,s=fixture();old=p['adj'].copy();got=f.build_features(p,m,s)
        self.assertEqual(len(got['audit']['cells']),98)
        self.assertEqual(got['arrays']['common_available'].sum(),64)
        self.assertEqual(got['audit']['attempted_stock_windows'],64)
        self.assertEqual(got['audit']['descriptive_ols_calls'],128)
        self.assertEqual(got['audit']['completed_regression_pairs'],64)
        self.assertEqual(got['audit']['new_model_fits'],0)
        for i in range(2):
            for raw,rank in zip(f.EXTRA[:2],f.EXTRA[2:4]):
                np.testing.assert_array_equal(got['arrays'][rank][i],f.centered_midrank(got['arrays'][raw][i]))
                self.assertTrue(np.isnan(got['arrays'][rank][i,32:]).all())
        pd.testing.assert_frame_equal(p['adj'],old);json.dumps(got['audit'],allow_nan=False)
    def test_missing_market_endpoint_does_not_compress(self):
        p,m,s=fixture();_,_,ends=f.weekly_endpoints(s,m['dates'][0])
        p['adj'].loc[ends[20],'SPY']=np.nan;got=f.build_features(p,m,s)
        self.assertFalse(got['arrays']['common_available'][0].any())
        self.assertEqual(got['audit']['cells'][0]['reason'],'invalid_market_path')
    def test_missing_stock_endpoint_affects_only_that_stock(self):
        p,m,s=fixture();_,_,ends=f.weekly_endpoints(s,m['dates'][0])
        p['adj'].loc[ends[20],'S0']=np.nan;got=f.build_features(p,m,s)
        self.assertEqual(got['audit']['cells'][0]['reason'],'invalid_stock_path')
        for key in f.EXTRA[:4]:self.assertTrue(np.isnan(got['arrays'][key][0,0]))
        self.assertTrue(got['arrays']['common_available'][0,1:32].all())
    def test_stock_only_requires_last53_prices(self):
        p,m,s=fixture();_,_,ends=f.weekly_endpoints(s,m['dates'][0])
        expected=f.build_features(p,m,s);p['adj'].loc[ends[:4],'S0']=np.nan
        actual=f.build_features(p,m,s)
        for key in f.EXTRA:np.testing.assert_array_equal(expected['arrays'][key][0],actual['arrays'][key][0])
    def test_one_valid_stock_shared_unavailable(self):
        p,m,s=fixture();p['adj'].loc[:,['S'+str(i) for i in range(1,32)]]=np.nan
        got=f.build_features(p,m,s)
        self.assertEqual(got['audit']['cells'][0]['reason'],'insufficient_cross_section')
        self.assertTrue(got['audit']['cells'][0]['regression_available'])
        self.assertFalse(got['arrays']['common_available'].any())
        for key in f.EXTRA[:4]:self.assertTrue(np.isnan(got['arrays'][key]).all())
    def test_no_eligibility_filter_and_etfs_never_enter_ranking(self):
        p,m,s=fixture();expected=f.build_features(p,m,s);m['eligible']=[[False]*49]*2
        p['adj'].loc[:,m['etfs']]=np.nan;actual=f.build_features(p,m,s)
        for key in f.EXTRA:np.testing.assert_array_equal(expected['arrays'][key],actual['arrays'][key])
    def test_recent_future_and_unfinished_week_poison_ignored(self):
        p,m,s=fixture();m=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1])
        expected=f.build_features(p,m,s);_,_,ends=f.weekly_endpoints(s,m['dates'][0])
        p['adj'].loc[p['adj'].index>ends[-1]]=np.nan;actual=f.build_features(p,m,s)
        self.assertEqual(expected['audit'],actual['audit'])
        for key in f.EXTRA:np.testing.assert_array_equal(expected['arrays'][key],actual['arrays'][key])
    def test_prefix_same_data_no_later_month_effect(self):
        p,m,s=fixture();full=f.build_features(p,m,s)
        prefix=f.build_features(p,dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]),s)
        for key in f.EXTRA:np.testing.assert_array_equal(full['arrays'][key][:1],prefix['arrays'][key])
    def test_insufficient_history_and_bad_axes(self):
        p,m,s=fixture();m=dict(m,dates=[str(s[20].date())],symbols=m['symbols'][:1])
        got=f.build_features(p,m,s);self.assertEqual(got['audit']['cells'][0]['reason'],'insufficient_history')
        with self.assertRaises(ValueError):f.build_features(p,m,s[::-1])
        m['dates']=['2015-01-03']
        with self.assertRaises(ValueError):f.build_features(p,m,s)


if __name__=='__main__':unittest.main()
