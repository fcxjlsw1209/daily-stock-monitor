"""Artificial-data-only tests for the shared-maturity annual target contrast."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_paired_target143_model as m


def fixture():
    ds = pd.date_range('2015-12-31', '2021-12-31', freq='ME')
    dates = np.repeat(ds.values.astype('datetime64[D]'), 3)
    ends = np.repeat((ds+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'), 3)
    available = np.repeat((ds+pd.offsets.MonthEnd(1)+pd.offsets.BDay(1)).values.astype('datetime64[D]'), 3)
    symbols = np.tile(['A','B','C'], len(ds))
    rng = np.random.default_rng(143)
    X = rng.uniform(-.5,.5,(len(dates),10))
    y = .2*X[:,0]+.1*X[:,1]**2+np.tile([-.02,.03,-.01],len(ds))
    y = (y.reshape(-1,3)-y.reshape(-1,3).mean(axis=1,keepdims=True)).ravel()
    ed = np.array(['2017-06-30','2019-01-31','2019-11-29','2021-02-26'],dtype='datetime64[D]')
    return [dates,ends,available,symbols,X,{'close':y,'open':-.7*y},np.ones(len(dates),bool),
            ed,np.array(['A','B','C','A']),rng.uniform(-.5,.5,(4,10)),np.ones(4,bool)]


def run(a,**kw):
    return m.annual_forecasts(*a,min_months=kw.get('min_months',3),min_own_months=kw.get('min_own_months',3))


def one_year(a,year=2021):
    a=copy.deepcopy(a);a[7]=np.array([str(year)+'-01-29'],dtype='datetime64[D]')
    for i in (8,9,10):a[i]=a[i][:1]
    return a


class PairedModelTests(unittest.TestCase):
    def test_zero_gap_equal_targets_exact_predictions(self):
        a=fixture();a[5]['open']=a[5]['close'].copy();z=run(a)
        np.testing.assert_array_equal(z['raw']['close'],z['raw']['open'])
        np.testing.assert_array_equal(z['pooled']['close'],z['pooled']['open'])

    def test_sign_reversal_same_cohort_linear_response(self):
        a=fixture();a[5]['open']=-a[5]['close'];z=run(a)
        np.testing.assert_allclose(z['raw']['close'],-z['raw']['open'],rtol=0,atol=1e-14)

    def test_exact_original_model_on_same_restricted_single_year(self):
        a=one_year(fixture());z=run(a);g=a[6]&(a[2]<=np.datetime64('2020-12-31'))
        for arm in m.ARMS:
            targets={k:a[5][arm] for k in m.original.TARGETS}
            r=m.original.annual_forecasts(a[0],a[1],a[3],a[4],targets,g,a[7],a[8],a[9],a[10],min_months=3,min_own_months=3)
            np.testing.assert_array_equal(z['raw'][arm],r['raw']['local_linear'])
            np.testing.assert_array_equal(z['pooled'][arm],r['raw']['source_log'])
            np.testing.assert_array_equal(z['available'],r['available'])

    def test_november_label_not_available_at_prior_december_cutoff(self):
        a=one_year(fixture());z=run(a);r=z['audit']['annual'][0]
        self.assertEqual(r['latest_calendar_end'],'2020-11-30')
        self.assertEqual(r['latest_available_session'],'2020-12-01')
        bad=a[0]>=np.datetime64('2020-11-01')
        a[4][bad]=123456.;a[5]['close'][bad]=99999.;a[5]['open'][bad]=-99999.
        p=run(a)
        for arm in m.ARMS:np.testing.assert_array_equal(z['raw'][arm],p['raw'][arm])

    def test_expired_past_rows_do_not_affect_forecast(self):
        a=one_year(fixture());z=run(a);old=a[0]<=np.datetime64('2015-12-31')
        a[4][old]=999.;a[5]['open'][old]=-999.;a[5]['close'][old]=999.
        p=run(a)
        for arm in m.ARMS:np.testing.assert_array_equal(z['raw'][arm],p['raw'][arm])

    def test_actual_scalers_rows_weights_are_shared(self):
        a=fixture();observed=[];fn=m.numeric._fit_ridge
        def capture(X,y,w,alpha):
            fit=fn(X,y,w,alpha);observed.append((X.copy(),w.copy(),fit));return fit
        with patch.object(m.numeric,'_fit_ridge',side_effect=capture):z=run(a)
        self.assertEqual(len(observed),6)
        for i in range(0,len(observed),2):
            left,right=observed[i:i+2]
            for k in (0,1):np.testing.assert_array_equal(left[k],right[k])
            for k in (0,1):np.testing.assert_array_equal(left[2][k],right[2][k])
        self.assertEqual(z['audit']['new_pooled_model_fits'],6)

    def test_independent_weighted_least_squares_pooled_prediction(self):
        a=one_year(fixture());z=run(a);g=a[6]&(a[0]>np.datetime64('2015-12-31'))&(a[2]<=np.datetime64('2020-12-31'))
        X=a[4][g];dates=a[0][g];mo,inv,count=np.unique(dates.astype('datetime64[M]'),return_inverse=True,return_counts=True)
        ages=np.datetime64('2020-12','M').astype(int)-mo.astype(int)
        w=np.exp2(-ages[inv]/12)/count[inv];w*=len(w)/sum(w)
        mean=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1)
        A=np.vstack([np.sqrt(w)[:,None]*(X-mean)/scale,np.sqrt(10)*np.eye(10)])
        for arm in m.ARMS:
            y=a[5][arm][g];intercept=np.average(y,weights=w)
            beta=np.linalg.lstsq(A,np.r_[np.sqrt(w)*(y-intercept),np.zeros(10)],rcond=None)[0]
            expected=((a[9][0]-mean)/scale).dot(beta)+intercept
            self.assertAlmostEqual(expected,z['pooled'][arm][0],places=13)

    def test_no_own_history_keeps_own_pooled_target(self):
        a=fixture();a[8]=np.array(['NEW']*4);z=run(a)
        self.assertFalse(z['local_available'].any())
        for arm in m.ARMS:np.testing.assert_array_equal(z['raw'][arm],z['pooled'][arm])
        self.assertGreater(np.max(np.abs(z['raw']['close']-z['raw']['open'])),1e-6)

    def test_one_target_missing_removes_shared_rows_and_singletons(self):
        a=one_year(fixture());base=run(a);a[5]['open'][3:5]=np.nan
        z=run(a)
        self.assertEqual(z['audit']['annual'][0]['training_rows'],base['audit']['annual'][0]['training_rows']-3)
        self.assertEqual(z['audit']['annual'][0]['training_months'],base['audit']['annual'][0]['training_months']-1)

    def test_false_common_good_ignores_poison_values(self):
        a=one_year(fixture());a[6][3:6]=False;z=run(a)
        a[4][3:6]=100.;a[5]['close'][3:6]=999.;a[5]['open'][3:6]=-999.
        p=run(a)
        for arm in m.ARMS:np.testing.assert_array_equal(z['raw'][arm],p['raw'][arm])

    def test_query_eligibility_is_separate_from_future_training(self):
        a=fixture();a[8]=np.array(['NOT_IN_TRAIN']*4);z=run(a)
        self.assertTrue(z['available'].all())
        a[10][1]=False;p=run(a)
        self.assertFalse(p['available'][1]);self.assertTrue(np.isnan(p['raw']['open'][1]))
        np.testing.assert_array_equal(p['raw']['close'][[0,2,3]],z['raw']['close'][[0,2,3]])

    def test_future_queries_and_prefixes_do_not_change_prior_predictions(self):
        a=fixture();full=run(a)
        for n in range(5):
            b=copy.deepcopy(a)
            for k in (7,8,9,10):b[k]=b[k][:n]
            z=run(b)
            for arm in m.ARMS:np.testing.assert_array_equal(z['raw'][arm],full['raw'][arm][:n])

    def test_changed_queries_do_not_change_scaler(self):
        a=fixture();z=run(a);a[9]*=-.3;p=run(a)
        for l,r in zip(z['audit']['annual'],p['audit']['annual']):
            for key in ('training_indices_sha256','weights_sha256','scaler_mean_sha256','scaler_scale_sha256'):self.assertEqual(l[key],r[key])

    def test_cold_start_has_no_output_or_fit(self):
        z=run(fixture(),min_months=200)
        self.assertEqual(z['audit']['new_pooled_model_fits'],0);self.assertFalse(z['available'].any())
        self.assertTrue(np.isnan(z['raw']['open']).all())

    def test_constant_features_zero_targets(self):
        a=fixture();a[4][:]=0.;a[9][:]=0.
        for v in a[5].values():v[:]=0.
        z=run(a)
        for arm in m.ARMS:np.testing.assert_array_equal(z['raw'][arm],np.zeros(4))

    def test_no_mutation(self):
        a=fixture();old=copy.deepcopy(a);run(a)
        for left,right in zip(a,old):
            if isinstance(left,dict):
                for k in left:np.testing.assert_array_equal(left[k],right[k])
            else:np.testing.assert_array_equal(left,right)

    def test_duplicate_used_security_month_rejected(self):
        a=fixture();a[3][4]=a[3][3]
        with self.assertRaisesRegex(ValueError,'Repeated'):run(a)

    def test_maturity_not_confused_with_calendar_end(self):
        a=fixture();a[2]=a[1].copy()
        with self.assertRaisesRegex(ValueError,'Separate'):run(a)

    def test_wrong_calendar_month_rejected(self):
        a=fixture();a[1]=a[2].copy()
        with self.assertRaisesRegex(ValueError,'Separate'):run(a)

    def test_target_shape_and_keys_rejected(self):
        for val in ({'close':[1.],'open':[1.]},{'log_excess':np.zeros(3)}):
            a=fixture();a[5]=val
            with self.assertRaises(ValueError):run(a)

    def test_boolean_and_string_features_rejected(self):
        for kind in (bool,str,complex):
            a=fixture();a[4]=a[4].astype(kind)
            with self.assertRaises(ValueError):run(a)

    def test_nonboolean_masks_rejected(self):
        for k in (6,10):
            a=fixture();a[k]=a[k].astype(int)
            with self.assertRaises(ValueError):run(a)

    def test_rank_bounds_checked_only_on_actual_rows(self):
        a=fixture();a[4][3,0]=.501
        with self.assertRaisesRegex(ValueError,'percentile'):run(a)

    def test_invalid_minimum_rejected(self):
        for val in (0,False,1.5):
            with self.assertRaises(ValueError):run(fixture(),min_months=val)
            with self.assertRaises(ValueError):run(fixture(),min_own_months=val)

    def test_invalid_date_type_rejected(self):
        for k in (0,1,2,7):
            a=fixture();a[k]=a[k].astype(int)
            with self.assertRaises((ValueError,TypeError)):run(a)

    def test_nonfinite_fit_cannot_be_reported_as_missing(self):
        with patch.object(m.numeric,'_predict',return_value=np.array([np.nan])):
            with self.assertRaises((ValueError,FloatingPointError,AssertionError)):run(one_year(fixture()))


if __name__=='__main__':unittest.main(verbosity=2)
