"""Synthetic independent source229 checks; no archived arrays or outcomes."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import monthly_early_history229_independent as m


def fixture():
    months = pd.period_range('2011-01', '2015-12', freq='M')
    dates = np.array([str(x.end_time.date()) for x in months], dtype='datetime64[D]')
    ends = np.array([str((x+1).end_time.date()) for x in months], dtype='datetime64[D]')
    rng = np.random.default_rng(229)
    X = rng.uniform(-.45, .45, (len(dates)*3, 10))
    symbols = np.tile(['A', 'B', 'C'], len(dates))
    noise = .03*np.sin(np.arange(len(X))/3.)
    y = .2*X[:, 0]-.1*X[:, 3] + noise
    ys = {k: y*(i+1)/4. for i,k in enumerate(m.TARGETS)}
    good = np.tile([True, True, False], len(dates))
    qd = np.repeat(np.array(['2012-06-30', '2013-01-31', '2014-02-28', '2016-07-31'], dtype='datetime64[D]'),3)
    qs = np.tile(['A', 'B', 'C'], 4)
    Q = rng.uniform(-.45, .45, (len(qd), 10))
    eligible = np.tile([True, True, False], 4)
    args = (np.repeat(dates,3), np.repeat(ends,3), symbols, X, ys, good,
            qd, qs, Q, eligible)
    return args, np.ones(len(qd), bool)


def physical_prefix(args, allowed, n):
    a = copy.deepcopy(list(args))
    if n:
        year = int(a[6][n-1].astype('datetime64[Y]').astype(int)) + 1970
        use = a[1] <= np.datetime64('%04d-12-31' % (year-1))
    else:
        use = np.zeros(len(a[0]), bool)
    for k in (0, 1, 2, 3, 5):
        a[k] = a[k][use].copy()
    a[4] = {k: v[use].copy() for k,v in a[4].items()}
    for k in (6, 7, 8, 9):
        a[k] = a[k][:n].copy()
    return tuple(a), allowed[:n].copy()


def frozen_reference(args):
    path = m.ROOT / 'outputs/monthly_local_linear70/model.py'
    spec = importlib.util.spec_from_file_location('early229_synthetic_frozen_reference', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.annual_forecasts(*args)


class SourceTests(unittest.TestCase):
    def test_independent_augmented_solver_matches_frozen_model(self):
        args, allowed = fixture()
        a = list(copy.deepcopy(args));a[9] = a[9] | allowed
        expected = frozen_reference(a)
        got = m.replay_oracle(*a)
        for k in m.RAW_KEYS:
            m.close(got['raw'][k], expected['raw'][k], k)
        m.record_check(got['audit'], expected['audit'])
        np.testing.assert_array_equal(got['available'], expected['available'])
        np.testing.assert_array_equal(got['local_available'], expected['local_available'])

    def test_union_requested_and_available_counts(self):
        args, allowed = fixture();out = m.forecasts_oracle(*args, expansion_allowed=allowed)
        counts = out['audit']['counts']
        self.assertEqual((counts['query_rows'], counts['original_query_rows'], counts['expansion_requested_rows']), (12,8,4))
        self.assertEqual((counts['available_rows'], counts['original_available_rows'], counts['added_available_rows']), (9,6,3))
        self.assertEqual(tuple(counts), m.COUNT_FIELDS)
        self.assertEqual(counts['actual_regression_solves'], counts['source_target_fits']+counts['pooled_log_refits']+counts['local_solves'])

    def test_expansion_false_keeps_original_queries(self):
        args, allowed = fixture();allowed[:] = False
        out = m.forecasts_oracle(*args, expansion_allowed=allowed)
        self.assertEqual(out['audit']['counts']['added_available_rows'], 0)
        self.assertTrue(out['available'][3]);self.assertFalse(out['available'][5])

    def test_future_labels_are_not_query_qualification(self):
        args, allowed = fixture();a, allowed = physical_prefix(args, allowed, 6)
        original = m.forecasts_oracle(*a, expansion_allowed=allowed)
        b = list(copy.deepcopy(args))
        for k in (6,7,8,9):b[k]=b[k][:6]
        future = b[1] > np.datetime64('2012-12-31')
        b[3][future] = np.inf;b[5][future] = False
        for y in b[4].values():y[future] = np.nan
        got=m.forecasts_oracle(*b,expansion_allowed=allowed)
        for k in m.RAW_KEYS:m.close(got['raw'][k],original['raw'][k],k,atol=0.)
        m.record_check(got['audit'],original['audit'])
        self.assertTrue(got['available'][5])

    def test_cold_year_and_exact_own_cold_fallback(self):
        args,allowed=fixture();out=m.forecasts_oracle(*args,expansion_allowed=allowed)
        self.assertFalse(out['available'][:3].any())
        self.assertTrue(np.isnan(out['raw']['source_log'][:3]).all())
        for j in (5,8,11):
            self.assertFalse(out['local_available'][j])
            self.assertEqual(out['raw']['source_log'][j],out['raw']['local_constant'][j])
            self.assertEqual(out['raw']['source_log'][j],out['raw']['local_linear'][j])

    def test_same_security_local_correction_and_alias_not_inferred(self):
        args,allowed=fixture();out=m.forecasts_oracle(*args,expansion_allowed=allowed)
        self.assertTrue(out['local_available'][3])
        self.assertNotEqual(out['raw']['local_linear'][3],out['raw']['source_log'][3])
        a=list(copy.deepcopy(args));a[7]=a[7].astype(object);a[7][3]='AA'
        changed=m.forecasts_oracle(*a,expansion_allowed=allowed)
        self.assertFalse(changed['local_available'][3])
        self.assertEqual(changed['raw']['local_linear'][3],changed['raw']['source_log'][3])

    def test_mature_window_prior_december_and_minimum18(self):
        args,allowed=fixture();a=list(copy.deepcopy(args))
        for k in (6,7,8,9):a[k]=a[k][3:6]
        out=m.forecasts_oracle(*a,expansion_allowed=allowed[3:6])
        annual=out['audit']['replay_audit']['annual'][0]
        self.assertEqual(annual['cutoff'],'2012-12-31')
        self.assertEqual(annual['lookback_exclusive_start'],'2007-12-31')
        self.assertEqual(annual['training_months'],23)
        use=np.isin(a[0],np.unique(a[0])[:17]);a[5] &= use
        cold=m.forecasts_oracle(*a,expansion_allowed=allowed[3:6]);self.assertFalse(cold['available'].any())
        a[5]=args[5] & np.isin(args[0],np.unique(args[0])[:18])
        warm=m.forecasts_oracle(*a,expansion_allowed=allowed[3:6]);self.assertTrue(warm['available'].all())

    def test_joint_four_targets_and_singleton_months(self):
        args,allowed=fixture();a=list(copy.deepcopy(args))
        for k in (6,7,8,9):a[k]=a[k][3:6]
        a[4]['loss_utility'][0:18]=np.nan
        out=m.forecasts_oracle(*a,expansion_allowed=allowed[3:6])
        self.assertEqual(out['audit']['replay_audit']['annual'][0]['training_months'],17)
        self.assertFalse(out['available'].any())
        a=list(copy.deepcopy(args));a[5][np.arange(len(a[5]))%3==1]=False
        out=m.forecasts_oracle(*a,expansion_allowed=allowed)
        self.assertFalse(out['available'].any())

    def test_current_nonfinite_features_unavailable(self):
        for value in (np.nan,np.inf,-np.inf):
            args,allowed=fixture();args[8][5,0]=value
            out=m.forecasts_oracle(*args,expansion_allowed=allowed)
            self.assertFalse(out['available'][5]);self.assertFalse(out['audit']['union_query_mask'][5])

    def test_original_input_bytes_unchanged_and_flags(self):
        args,allowed=fixture();before=m.input_signature((args,allowed))
        out=m.forecasts_oracle(*args,expansion_allowed=allowed)
        receipt=m.check_forecasts(*args,expansion_allowed=allowed,result=out)
        self.assertEqual(m.input_signature((args,allowed)),before)
        self.assertTrue(all(receipt[k] is True for k in m.FORECAST_FLAGS))
        self.assertFalse(receipt['producer_modules_imported']);self.assertFalse(receipt['shared_numerical_primitives'])
        self.assertEqual(receipt['numerical_tolerance'],1e-12)

    def test_corrupt_prediction_mask_counts_and_schema_rejected(self):
        args,allowed=fixture();out=m.forecasts_oracle(*args,expansion_allowed=allowed)
        for mode in ('prediction','mask','count','query_filter','extra'):
            bad=copy.deepcopy(out)
            if mode=='prediction':bad['raw']['local_linear'][3]+=.01
            elif mode=='mask':bad['available'][3]=False
            elif mode=='count':bad['audit']['counts']['local_solves']+=1
            elif mode=='query_filter':bad['audit']['query_future_labels_used']=True
            else:bad['unexpected']=False
            with self.assertRaises(ValueError):m.check_forecasts(*args,expansion_allowed=allowed,result=bad)

    def test_all_physical_query_prefixes(self):
        args,allowed=fixture();full=m.forecasts_oracle(*args,expansion_allowed=allowed)
        for n in range(len(allowed)+1):
            a,ok=physical_prefix(args,allowed,n)
            out=m.forecasts_oracle(*a,expansion_allowed=ok)
            for k in m.RAW_KEYS:m.close(out['raw'][k],full['raw'][k][:n],k,atol=0.)
            np.testing.assert_array_equal(out['available'],full['available'][:n])

    def test_future_queries_do_not_change_prefix(self):
        args,allowed=fixture();a=list(copy.deepcopy(args));a[8][6:]=np.nan
        base=m.forecasts_oracle(*args,expansion_allowed=allowed)
        got=m.forecasts_oracle(*a,expansion_allowed=allowed)
        for k in m.RAW_KEYS:m.close(got['raw'][k][:6],base['raw'][k][:6],k,atol=0.)

    def test_date_bound_strict(self):
        for date in ('2016-08-17','2016-08-31','2020-01-31'):
            args,allowed=fixture();args[6][-3:]=np.datetime64(date)
            with self.assertRaises(ValueError):m.forecasts_oracle(*args,expansion_allowed=allowed)

    def test_date_encoding_and_nat_rejected(self):
        args,allowed=fixture()
        for kind in ('numeric','ns','noniso','nat'):
            a=list(copy.deepcopy(args))
            if kind=='numeric':a[6]=np.arange(len(a[6]))
            elif kind=='ns':a[6]=a[6].astype('datetime64[ns]')
            elif kind=='noniso':a[6]=a[6].astype(str).astype(object);a[6][0]='2012-6-30'
            else:a[6][0]=np.datetime64('NaT')
            with self.assertRaises((ValueError,TypeError)):m.forecasts_oracle(*a,expansion_allowed=allowed)

    def test_duplicate_and_misaligned_identity_rejected(self):
        for mode in ('train_duplicate','query_duplicate','untrimmed','query_order','two_signals','two_ends','end_month'):
            args,allowed=fixture();a=list(copy.deepcopy(args))
            if mode=='train_duplicate':a[2][1]=a[2][0]
            elif mode=='query_duplicate':a[7][1]=a[7][0]
            elif mode=='untrimmed':a[7]=a[7].astype(object);a[7][0]=' A'
            elif mode=='query_order':a[6][0]=np.datetime64('2016-07-31')
            elif mode=='two_signals':a[6][1]=np.datetime64('2012-06-29')
            elif mode=='two_ends':a[1][0]=np.datetime64('2011-02-27')
            else:a[1][0]=np.datetime64('2011-03-31')
            with self.assertRaises(ValueError):m.forecasts_oracle(*a,expansion_allowed=allowed)

    def test_masks_and_nonnumeric_arrays_rejected(self):
        for mode in ('common','query','allow','Xbool','Qobject','complex','target_shape'):
            args,allowed=fixture();a=list(copy.deepcopy(args))
            if mode=='common':a[5]=a[5].astype(int)
            elif mode=='query':a[9]=a[9].astype(int)
            elif mode=='allow':allowed=allowed.astype(int)
            elif mode=='Xbool':a[3]=a[3].astype(bool)
            elif mode=='Qobject':a[8]=a[8].astype(object)
            elif mode=='complex':a[4]['log_excess']=a[4]['log_excess'].astype(complex)
            else:a[4]['log_excess']=a[4]['log_excess'][:-1]
            with self.assertRaises(ValueError):m.forecasts_oracle(*a,expansion_allowed=allowed)

    def test_consumed_rank_bounds_only(self):
        args,allowed=fixture();args[8][3,0]=.50000001
        with self.assertRaises(ValueError):m.forecasts_oracle(*args,expansion_allowed=allowed)
        args,allowed=fixture();args[8][0,0]=5.
        out=m.forecasts_oracle(*args,expansion_allowed=allowed)
        self.assertFalse(out['available'][0])

    def test_empty_queries_and_zero_targets(self):
        args,allowed=fixture();a,ok=physical_prefix(args,allowed,0)
        out=m.forecasts_oracle(*a,expansion_allowed=ok)
        self.assertEqual(sum(out['audit']['counts'].values()),0)
        self.assertTrue(m.check_forecasts(*a,expansion_allowed=ok,result=out)['passed'])
        for y in args[4].values():y[:]=0.
        out=m.forecasts_oracle(*args,expansion_allowed=allowed)
        self.assertEqual(out['audit']['counts']['local_solves'],0)
        self.assertTrue(np.all(out['raw']['local_linear'][out['available']]==0.))

    def test_new_producer_both_fixtures_all_prefixes(self):
        import monthly_early_history229_model as producer
        import test_monthly_early_history229_model as ptest
        for args, allowed in (fixture(), ptest.fixture()):
            before=m.input_signature((args,allowed))
            for n in range(len(allowed)+1):
                a,ok=physical_prefix(args,allowed,n)
                got=producer.forecasts(*a,expansion_allowed=ok)
                self.assertTrue(m.check_forecasts(*a,expansion_allowed=ok,result=got)['passed'])
            self.assertEqual(before,m.input_signature((args,allowed)))


if __name__=='__main__':
    unittest.main()
