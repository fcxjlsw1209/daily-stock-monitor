"""Invented data only: shared support, timing, risk math and rank aggregation."""
import copy
import json
import math
import unittest

import numpy as np
import pandas as pd

import monthly_risk_consensus219_kernel as kernel


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-04')
    x = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({'A': 100.+.03*x+3.*np.sin(x*.13),
        'B': 200.+.02*x+5.*np.cos(x*.23), 'FUND': 80.+.02*x+4.*np.sin(x*.37)}, index=dates)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
        symbols=[['A', 'B', 'FUND'] for _ in range(3)], etfs=['FUND'])
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


def source_dates(meta, calendar, row=0):
    t = calendar.index(meta['dates'][row])
    return pd.to_datetime(calendar[t-294:t-41])


class MeasureTests(unittest.TestCase):
    def test_mean_sample_volatility_endpoint_and_no_annualization(self):
        p = 100. + np.arange(253.) + 7.*np.sin(np.arange(253.)*.11)
        result = kernel.measure(p)
        r = p[1:]/p[:-1]-1
        self.assertTrue(result['available'])
        self.assertEqual(result['momentum'], p[-1]/p[0]-1)
        self.assertEqual(result['simple_return_sum'], math.fsum(r))
        self.assertEqual(result['mean_return'], math.fsum(r)/252)
        self.assertAlmostEqual(result['sample_vol'], r.std(ddof=1), places=16)
        self.assertGreater(abs(result['sample_vol']-r.std(ddof=0)), 1e-8)
        self.assertEqual(result['sharpe'], result['mean_return']/result['sample_vol'])
        self.assertEqual(result['start_price'], p[0]); self.assertEqual(result['end_price'], p[-1])

    def test_binary64_variance_uses_multiplication_not_power(self):
        # Regress the exact-arithmetic contract on varied synthetic252-day paths.
        # **2 may differ by one ulp from x*x even for the same binary64 input.
        rng = np.random.default_rng(219)
        any_power_difference = False
        for _ in range(64):
            p = 100.*np.exp(np.cumsum(rng.normal(.001,.03,253)))
            result = kernel.measure(p)
            returns = [float(p[i+1]/p[i]-1.) for i in range(252)]
            mean = math.fsum(returns)/252
            differences = [r-mean for r in returns]
            product_sum = math.fsum(d*d for d in differences)
            any_power_difference |= any(d*d != d**2 for d in differences)
            self.assertEqual(result['centered_square_sum'], product_sum)
            self.assertEqual(result['sample_vol'], math.sqrt(product_sum/251))
        self.assertTrue(any_power_difference)

    def test_zero_volatility_preserves_descriptive_nonzero_growth(self):
        for p, growth, mean in [(np.ones(253), 0., 0.), (np.exp2(np.arange(253.)), 2.**252-1, 1.)]:
            result = kernel.measure(p)
            self.assertTrue(result['source_available']); self.assertFalse(result['available'])
            self.assertEqual(result['reason'], 'zero_sample_volatility')
            self.assertEqual(result['momentum'], growth)
            self.assertEqual(result['mean_return'], mean)
            self.assertEqual(result['sample_vol'], 0.)
            self.assertEqual(result['centered_square_sum'], 0.)
            self.assertTrue(math.isnan(result['sharpe']))

    def test_negative_momentum_and_negative_mean_stay_available(self):
        p = 400.-np.arange(253.)+np.sin(np.arange(253.))
        result = kernel.measure(p)
        self.assertTrue(result['available'])
        self.assertLess(result['momentum'], 0.); self.assertLess(result['sharpe'], 0.)

    def test_interior_prices_affect_risk_even_with_fixed_endpoints(self):
        p = np.arange(100., 353.)
        first = kernel.measure(p); p[120] *= 1.7; changed = kernel.measure(p)
        self.assertEqual(first['momentum'], changed['momentum'])
        self.assertNotEqual(first['sample_vol'], changed['sample_vol'])
        self.assertNotEqual(first['sharpe'], changed['sharpe'])

    def test_source_validation_and_all253_observations(self):
        for bad, reason in [(True, 'nonnumeric_adjusted_price'), ('100', 'nonnumeric_adjusted_price'),
                (None, 'nonnumeric_adjusted_price'), (1+0j, 'nonnumeric_adjusted_price'),
                (0., 'invalid_adjusted_price'), (-1., 'invalid_adjusted_price'),
                (np.inf, 'invalid_adjusted_price'), (np.nan, 'invalid_adjusted_price')]:
            for j in (0, 42, 126, 251, 252):
                p = [100.]*253; p[j] = bad
                with self.subTest(bad=bad, index=j):
                    got = kernel.measure(p)
                    self.assertFalse(got['source_available']); self.assertFalse(got['available'])
                    self.assertEqual(got['reason'], reason)
        for p in ([1.]*252, np.ones((253,1)), 1.):
            with self.assertRaises(ValueError): kernel.measure(p)

    def test_valid_price_overflow_is_not_missing(self):
        p = np.ones(253); p[0] = 1e-308; p[-1] = 1e308
        with self.assertRaises(FloatingPointError): kernel.measure(p)
        p = np.ones(253); p[123] = 1e200
        with self.assertRaises(FloatingPointError): kernel.measure(p)

    def test_power_two_scale_and_input_immutability(self):
        p = np.arange(100.,353.); before = p.copy(); expected = kernel.measure(p)
        for factor in (2.**-500, 2.**500):
            got = kernel.measure(p*factor)
            for key in ('momentum','mean_return','sample_vol','sharpe','simple_return_sum','centered_square_sum'):
                self.assertEqual(got[key], expected[key])
        np.testing.assert_array_equal(p, before)


class RankTests(unittest.TestCase):
    def test_original_percentile_ties_singleton_empty(self):
        got = kernel.percentile_rank(np.array([1.,2.,2.,4.,np.nan]), np.array([True]*4+[False]))
        np.testing.assert_array_equal(got, [-.25,.125,.125,.5,np.nan])
        np.testing.assert_array_equal(kernel.percentile_rank(np.ones(3), np.ones(3,bool)), [2/3-.5]*3)
        np.testing.assert_array_equal(kernel.percentile_rank(np.array([3.,np.nan]),np.array([True,False])), [.5,np.nan])
        self.assertTrue(np.isnan(kernel.percentile_rank(np.ones(2), np.zeros(2,bool))).all())

    def test_min_mean_disagreement_penalty(self):
        q,c,equal = kernel.aggregate([-.25,.5])
        self.assertEqual(q,-.25); self.assertEqual(c,.125); self.assertFalse(equal)
        self.assertEqual(q,c-abs(-.25-.5)/2)
        a,b,_ = kernel.aggregate([-.25,.125])
        self.assertEqual(a,q); self.assertNotEqual(b,c)

    def test_exact_equal_signed_zero_copy_and_no_tolerance(self):
        for v in (.1,.5,-.25,-0.):
            q,c,equal = kernel.aggregate([v,v])
            self.assertTrue(equal); self.assertEqual(q.hex(), v.hex()); self.assertEqual(c.hex(),v.hex())
        self.assertFalse(kernel.aggregate([.1,np.nextafter(.1,1.)])[2])
        for values in ([True,0.],[1.,0.],[np.nan,0.],[.1]):
            with self.assertRaises(ValueError): kernel.aggregate(values)

    def test_rank_validation(self):
        for raw,mask in [(np.ones(2),np.ones(3,bool)), (np.ones(2),np.ones(2)),
                         (np.array([np.inf]),np.array([True]))]:
            with self.assertRaises(ValueError): kernel.percentile_rank(raw,mask)


class BuildTests(unittest.TestCase):
    def test_schema_counts_and_clocks(self):
        panel,meta,cal = fixture(); values,audit = kernel.build_features(panel,meta,cal)
        self.assertEqual(set(values), set(kernel.FEATURE_KEYS))
        self.assertEqual(audit['source_available_by_event'], [3]*3)
        self.assertEqual(audit['available_by_event'], [3]*3)
        self.assertEqual(audit['valid_price_observations'], 253*9)
        self.assertEqual(audit['simple_returns'], 252*9)
        self.assertEqual(audit['ranked_observations'], 18)
        self.assertEqual(audit['aggregated_scores'], 18)
        self.assertEqual(audit['std_ddof'],1); self.assertEqual(audit['window_returns'],252)
        self.assertEqual(audit['new_model_fits'],0); self.assertEqual(audit['predictive_model_fits'],0)
        self.assertFalse(audit['ranking_eligibility_filtered'])
        self.assertTrue(np.isfinite(values['candidate'][0]).all())
        json.dumps(audit,allow_nan=False)
        for event in audit['event_history']:
            t = event['session_position']
            self.assertEqual((event['price_start'],event['first_return_date'],event['price_end']),
                             (cal[t-294],cal[t-293],cal[t-42]))

    def test_zero_volatility_common_support_keeps_source_descriptions(self):
        panel,meta,cal = fixture(); dates = source_dates(meta,cal)
        panel['adj'].loc[dates,['B','FUND']] = 100.
        vals,audit = kernel.build_features(panel,meta,cal,limit=1)
        self.assertEqual(audit['source_available_cells'],3); self.assertEqual(audit['available_cells'],1)
        self.assertEqual(audit['simple_returns'],3*252); self.assertEqual(audit['ranked_observations'],2)
        for key in ('momentum','mean_return','sample_vol'): np.testing.assert_array_equal(vals[key][0,1:],[0.,0.])
        for key in ('sharpe','rank_momentum','rank_sharpe','candidate','control'):
            self.assertTrue(np.isnan(vals[key][0,1:]).all())
        for key in ('rank_momentum','rank_sharpe','candidate','control'): self.assertEqual(vals[key][0,0],.5)
        self.assertEqual(audit['reason_counts']['zero_sample_volatility'],2)
        json.dumps(audit,allow_nan=False)

    def test_invalid_source_all_float_missing(self):
        panel,meta,cal = fixture(); panel['adj'].loc[source_dates(meta,cal)[126],'B'] = np.nan
        vals,audit = kernel.build_features(panel,meta,cal,limit=1)
        self.assertEqual(audit['source_available_cells'],2)
        for k,v in vals.items():
            if k != 'available': self.assertTrue(np.isnan(v[0,1]))
        cell=audit['event_history'][0]['cells'][1]
        self.assertTrue(all(cell[k] is None for k in ['momentum','mean_return','sample_vol','sharpe']))

    def test_initial_and_calm_are_not_producer_filters(self):
        panel,meta,cal = fixture(); before,a = kernel.build_features(panel,meta,cal)
        meta['state_calm10'] = [True]*3; meta['eligible'] = [[False]*3]*3
        other,b = kernel.build_features(panel,meta,cal)
        for k in before: np.testing.assert_array_equal(before[k],other[k])
        self.assertEqual(a,b); self.assertTrue(np.isfinite(other['candidate'][0]).all())

    def test_direct_targets_no_second_rank(self):
        panel,meta,cal = fixture(); vals,_ = kernel.build_features(panel,meta,cal)
        for i,j in np.ndindex(vals['available'].shape):
            expected = kernel.aggregate([vals['rank_momentum'][i,j],vals['rank_sharpe'][i,j]])
            self.assertEqual(vals['candidate'][i,j],expected[0]); self.assertEqual(vals['control'][i,j],expected[1])

    def test_full_source_tie_offset_preserved(self):
        panel,meta,cal = fixture()
        panel['adj']['B']=panel['adj']['A']; panel['adj']['FUND']=panel['adj']['A']
        vals,audit = kernel.build_features(panel,meta,cal)
        for k in ('rank_momentum','rank_sharpe','candidate','control'):
            np.testing.assert_array_equal(vals[k],np.full((3,3),2/3-.5))
        self.assertTrue(all(c['exact_equal_ranks'] for e in audit['event_history'] for c in e['cells']))

    def test_structural_missing_and_no_compressed_calendar(self):
        panel,meta,cal = fixture(); missing=source_dates(meta,cal)[100]
        panel['adj']=panel['adj'].drop(index=missing,columns='B')
        _,audit=kernel.build_features(panel,meta,cal,limit=1)
        self.assertEqual([c['reason'] for c in audit['event_history'][0]['cells']],
                         ['missing_official_session','missing_symbol','missing_official_session'])
        _,audit=kernel.build_features(panel,meta,cal[cal.index(meta['dates'][0])-100:],limit=1)
        self.assertEqual(audit['reason_counts'], {'insufficient_history':3})

    def test_all_physical_prefixes_and_future_poison(self):
        panel,meta,cal=fixture(); original,a=kernel.build_features(panel,meta,cal)
        for n in range(4):
            short={'adj':panel['adj'].loc[:meta['dates'][n-1]].copy() if n else panel['adj'].iloc[:0].copy()}
            other,b=kernel.build_features(short,meta,cal,limit=n)
            for k in original: np.testing.assert_array_equal(other[k],original[k][:n])
            self.assertEqual(b['event_history'],a['event_history'][:n])
        expected,a=kernel.build_features(panel,meta,cal,limit=1)
        t=cal.index(meta['dates'][0]); panel['adj']=panel['adj'].astype(object)
        panel['adj'].loc[pd.Timestamp(cal[t-41]):]='future poison';panel['adj']['UNUSED']='poison'
        other,b=kernel.build_features(panel,meta,cal,limit=1)
        for k in expected: np.testing.assert_array_equal(other[k],expected[k])
        self.assertEqual(a,b)

    def test_reordered_identity_input_immutable_and_json_safe(self):
        panel,meta,cal=fixture(); before=panel['adj'].copy(); old=copy.deepcopy(meta)
        vals,audit=kernel.build_features(panel,meta,cal)
        reverse=dict(meta,symbols=[list(reversed(s)) for s in meta['symbols']])
        swapped,_=kernel.build_features(panel,reverse,cal)
        for k in vals: np.testing.assert_array_equal(swapped[k],vals[k][:,::-1])
        pd.testing.assert_frame_equal(before,panel['adj']);self.assertEqual(old,meta)
        json.dumps(audit,allow_nan=False)

    def test_empty_nonempty_and_bad_schema(self):
        panel,meta,cal=fixture(); empty,a=kernel.build_features(panel,meta,cal,limit=0)
        self.assertEqual(empty['available'].shape,(0,3));self.assertEqual(a['checked_cells'],0)
        for limit in (True,-1,4,1.5):
            with self.assertRaises(ValueError): kernel.build_features(panel,meta,cal,limit=limit)
        bad=copy.deepcopy(meta);bad['symbols'][0]=['A','A','FUND']
        with self.assertRaises(ValueError): kernel.build_features(panel,bad,cal)
        bad=copy.deepcopy(meta);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError): kernel.build_features(panel,bad,cal)
        with self.assertRaises(ValueError): kernel.build_features(panel,meta,list(reversed(cal)))


if __name__ == '__main__':
    unittest.main()
