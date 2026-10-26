"""Synthetic-only validation for the two-target endpoint retest model."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_endpoint155_model as m


def month_ends(sessions):
    return pd.Series(sessions, index=sessions).groupby(sessions.to_period('M')).last().tolist()


def target_fixture():
    sessions = pd.bdate_range('2019-12-01', '2021-02-28')
    close = pd.DataFrame(100.0, index=sessions, columns=['A','B','C'])
    dates = np.array(['2020-01-31'] * 3, dtype='datetime64[D]')
    close.loc['2020-02-28'] = [110.0, 80.0, 100.0]
    return [close, sessions, dates, np.array(['A','B','C']),
            np.ones(3,bool), np.ones(3,bool), np.ones(3,bool)]


def annual_fixture():
    ds = pd.date_range('2013-12-31', '2021-11-30', freq='ME')
    dates = np.repeat(ds.to_numpy(dtype='datetime64[D]'), 3)
    ends = np.repeat((ds + pd.offsets.MonthEnd(1)).to_numpy(dtype='datetime64[D]'), 3)
    rng = np.random.default_rng(155)
    X = rng.uniform(-0.5, 0.5, (len(dates), 10))
    ret = .2 * X[:,0] - .1 * X[:,1] + .03
    labels = dict(dates=dates, ends=ends, symbols=np.tile(['A','B','C'],len(ds)),
                  return_target=ret, loss_target=(ret < 0).astype(float),
                  base_good=np.ones(len(dates),bool), good=np.ones(len(dates),bool))
    query_dates = np.array(['2016-01-29','2019-01-31','2019-11-29','2021-01-29'],dtype='datetime64[D]')
    return [labels, X, query_dates, np.array(['A','B','C','A']),
            rng.uniform(-.5,.5,(4,10)), np.ones(4,bool)]


def annual(a, **kwargs):
    return m.annual_forecasts(*a, event_monthend=kwargs.get('monthend',np.ones(len(a[2]),bool)))


def build_fixture():
    sessions = pd.bdate_range('2013-12-01','2021-02-28')
    ends = month_ends(sessions)
    train = [x for x in ends if pd.Timestamp('2014-01-01') <= x <= pd.Timestamp('2020-11-30')]
    lookup = {x.to_period('M'):x for x in ends}
    trainmeta = dict(dates=[str(x.date()) for x in train],
        ends=[str(lookup[x.to_period('M')+1].date()) for x in train], symbols=[['A','B','C'] for _ in train])
    meta = dict(dates=['2016-08-17','2016-08-31','2017-01-31','2020-01-31','2020-02-28'],
        ends=['2016-08-31','2016-09-30','2017-02-28','2020-02-28','2020-03-31'],
        symbols=[['A','B','C'] for _ in range(5)])
    idx = np.arange(len(sessions),dtype=float)
    close = pd.DataFrame({'A':100+idx*.02+np.sin(idx/9),
                          'B':200+idx*.01+np.sin(idx/7)*3,
                          'C':300-idx*.015+np.cos(idx/11)*4},index=sessions)
    rng = np.random.default_rng(155)
    return [close, sessions, trainmeta, meta,
            rng.uniform(-.5,.5,(len(train)*3,10)),rng.uniform(-.5,.5,(15,10)),
            np.ones(len(train)*3,bool),np.ones(len(train)*3,bool),np.ones((5,3),bool)]


class LabelsTests(unittest.TestCase):
    def test_positive_negative_and_exact_zero(self):
        z=m.build_endpoint_targets(*target_fixture())
        np.testing.assert_allclose(z['return_target'],[.1,-.2,0],rtol=0,atol=1e-15)
        np.testing.assert_array_equal(z['loss_target'],[0,1,0])
        self.assertTrue(z['good'].all())

    def test_large_gain_not_clipped_or_centered(self):
        a=target_fixture();a[0].loc['2020-02-28']=[500,600,700]
        z=m.build_endpoint_targets(*a)
        np.testing.assert_array_equal(z['return_target'],[4,5,6])
        np.testing.assert_array_equal(z['loss_target'],[0,0,0])

    def test_benchmark_and_intramonth_path_not_new_targets(self):
        a=target_fixture();z=m.build_endpoint_targets(*a)
        a[0]['SPY']=1.;a[0].loc['2020-02-14',['A','B','C']]=[.001,9000,np.nan]
        b=m.build_endpoint_targets(*a)
        for key in m.LABELS:np.testing.assert_array_equal(b[key],z[key])

    def test_inherited_false_rows_are_not_inspected(self):
        a=target_fixture();a[4][0]=False;a[0].loc['2020-02-28','A']=np.inf
        z=m.build_endpoint_targets(*a)
        self.assertTrue(np.isnan(z['return_target'][0]));self.assertFalse(z['good'][0])
        self.assertTrue(z['good'][1:].all())

    def test_each_inherited_mask_restricts(self):
        for k in (4,5,6):
            a=target_fixture();a[k][1]=False;z=m.build_endpoint_targets(*a)
            self.assertFalse(z['base_good'][1]);self.assertFalse(z['good'][1])

    def test_singleton_keeps_raw_but_not_training(self):
        a=target_fixture();a[4][:2]=False;z=m.build_endpoint_targets(*a)
        self.assertFalse(z['good'].any());self.assertEqual(z['return_target'][2],0)

    def test_absent_end_never_shortens(self):
        a=target_fixture();a[0]=a[0].loc[:'2020-02-27'];z=m.build_endpoint_targets(*a)
        self.assertFalse(z['good'].any());self.assertTrue(np.isnan(z['return_target']).all())
        np.testing.assert_array_equal(z['ends'],np.array(['2020-02-28']*3,dtype='datetime64[D]'))

    def test_missing_asset_not_zero(self):
        a=target_fixture();a[0]=a[0].drop(columns='A');z=m.build_endpoint_targets(*a)
        self.assertTrue(np.isnan(z['loss_target'][0]));self.assertFalse(z['good'][0])

    def test_zero_negative_nonfinite_and_boolean_prices_excluded(self):
        for value in (0.,-1.,np.nan,np.inf,True):
            a=target_fixture();a[0]=a[0].astype(object);a[0].loc['2020-02-28','A']=value
            z=m.build_endpoint_targets(*a);self.assertFalse(z['good'][0]);self.assertTrue(np.isnan(z['return_target'][0]))

    def test_overflow_ratio_not_label(self):
        a=target_fixture();a[0].loc['2020-01-31','A']=1e-300;a[0].loc['2020-02-28','A']=1e300
        self.assertFalse(m.build_endpoint_targets(*a)['good'][0])

    def test_common_per_security_scale_preserves_labels(self):
        a=target_fixture();z=m.build_endpoint_targets(*a);a[0]=a[0]*[2.,4.,.5]
        b=m.build_endpoint_targets(*a)
        for key in ('return_target','loss_target','good'):np.testing.assert_array_equal(z[key],b[key])

    def test_month_end_training_required(self):
        a=target_fixture();a[2][:]=np.datetime64('2020-01-30')
        with self.assertRaisesRegex(ValueError,'month-end'):m.build_endpoint_targets(*a)

    def test_expected_end_tamper_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cached label ends'):
            m.build_endpoint_targets(*target_fixture(),expected_ends=['2020-02-27']*3)

    def test_calendar_not_complete_next_month_rejected(self):
        a=target_fixture();a[1]=a[1][a[1]<'2020-02-01']
        with self.assertRaisesRegex(ValueError,'next target month'):m.build_endpoint_targets(*a)

    def test_calendar_order_duplicates_and_timezone_rejected(self):
        for kind in ('reverse','duplicate','timezone'):
            a=target_fixture()
            a[1]=a[1][::-1] if kind=='reverse' else a[1].append(a[1][-1:]) if kind=='duplicate' else a[1].tz_localize('UTC')
            with self.assertRaises(ValueError):m.build_endpoint_targets(*a)

    def test_duplicate_security_date_rejected(self):
        a=target_fixture();a[3][1]='A'
        with self.assertRaisesRegex(ValueError,'repeats'):m.build_endpoint_targets(*a)

    def test_nonboolean_mask_rejected(self):
        a=target_fixture();a[4]=a[4].astype(int)
        with self.assertRaises(ValueError):m.build_endpoint_targets(*a)

    def test_december_january_and_leap_year(self):
        a=target_fixture();a[2][:]=np.datetime64('2019-12-31')
        z=m.build_endpoint_targets(*a)
        self.assertEqual(str(z['ends'][0]),'2020-01-31')
        a[2][:]=np.datetime64('2020-01-31');self.assertEqual(str(m.build_endpoint_targets(*a)['ends'][0]),'2020-02-28')


class AnnualTests(unittest.TestCase):
    def test_fixed_hyperparameters_and_two_fits_per_year(self):
        z=annual(annual_fixture());self.assertEqual(z['audit']['new_pooled_model_fits'],6)
        self.assertEqual(z['audit']['alpha'],10);self.assertEqual(z['audit']['half_life_months'],12)
        self.assertEqual(z['audit']['minimum_mature_months'],18);self.assertEqual(z['audit']['local_model_fits'],0)

    def test_independent_weighted_least_squares(self):
        a=annual_fixture();z=annual(a);cut=np.datetime64('2020-12-31');lo=np.datetime64('2015-12-31')
        g=(a[0]['dates']>lo)&(a[0]['ends']<=cut);X=a[1][g];ds=a[0]['dates'][g]
        mo,inv,count=np.unique(ds.astype('datetime64[M]'),return_inverse=True,return_counts=True)
        age=np.datetime64('2020-12','M').astype(int)-mo.astype(int)
        w=np.exp2(-age[inv]/12)/count[inv];w*=len(w)/sum(w)
        mean=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=w))
        scale=np.where(scale>1e-12,scale,1);A=np.vstack([np.sqrt(w)[:,None]*(X-mean)/scale,np.sqrt(10)*np.eye(10)])
        for key,out in zip(m.TARGETS,('mean_return_prediction','loss_prediction')):
            y=a[0][key][g];intercept=np.average(y,weights=w)
            beta=np.linalg.lstsq(A,np.r_[np.sqrt(w)*(y-intercept),np.zeros(10)],rcond=None)[0]
            expected=(a[4][-1]-mean)/scale@beta+intercept
            self.assertAlmostEqual(z[out][-1],expected,places=13)

    def test_actual_scalers_and_weights_shared(self):
        a=annual_fixture();observed=[];fn=m.numeric._fit_ridge
        def fit(X,y,w,alpha):
            v=fn(X,y,w,alpha);observed.append((X.copy(),w.copy(),v));return v
        with patch.object(m.numeric,'_fit_ridge',side_effect=fit):annual(a)
        for i in range(0,len(observed),2):
            x,y=observed[i:i+2]
            for j in (0,1):np.testing.assert_array_equal(x[j],y[j])
            for j in (0,1):np.testing.assert_array_equal(x[2][j],y[2][j])

    def test_same_training_rows_when_one_target_missing(self):
        a=annual_fixture();a[0]['loss_target'][30:32]=np.nan
        observed=[];fn=m.numeric._fit_ridge
        def fit(X,y,w,alpha):observed.append(len(y));return fn(X,y,w,alpha)
        with patch.object(m.numeric,'_fit_ridge',side_effect=fit):z=annual(a)
        self.assertTrue(all(observed[i]==observed[i+1] for i in range(0,len(observed),2)))
        self.assertNotIn(32,z['audit']['annual'][0]['training_indices'])

    def test_end_after_cutoff_cannot_affect_past_prediction(self):
        a=annual_fixture();a[2]=a[2][-1:];a[3]=a[3][-1:];a[4]=a[4][-1:];a[5]=a[5][-1:]
        z=annual(a);future=a[0]['ends']>np.datetime64('2020-12-31')
        a[1][future]=.49;a[0]['return_target'][future]=999;a[0]['loss_target'][future]=0
        p=annual(a)
        for key in ('mean_return_prediction','loss_prediction'):np.testing.assert_array_equal(z[key],p[key])
        self.assertEqual(z['audit']['annual'][0]['latest_label_end'],'2020-12-31')

    def test_exclusive_five_year_lower_bound(self):
        a=annual_fixture();a[2]=a[2][-1:];a[3]=a[3][-1:];a[4]=a[4][-1:];a[5]=a[5][-1:]
        z=annual(a);old=a[0]['dates']<=np.datetime64('2015-12-31')
        a[1][old]=-.49;a[0]['return_target'][old]=-999;a[0]['loss_target'][old]=1
        p=annual(a)
        for key in ('mean_return_prediction','loss_prediction'):np.testing.assert_array_equal(z[key],p[key])

    def test_seventeen_months_cold_eighteen_mature(self):
        a=annual_fixture();a[0]['good'][:]=False;a[0]['good'][3:54]=True
        self.assertFalse(annual(a)['common_available'][0])
        a[0]['good'][54:57]=True;self.assertTrue(annual(a)['common_available'][0])

    def test_early_query_does_not_fit_with_future_months(self):
        a=annual_fixture();a[2][0]=np.datetime64('2014-08-29');z=annual(a)
        self.assertFalse(z['common_available'][0]);self.assertTrue(np.isnan(z['mean_return_prediction'][0]))

    def test_constant_loss_classes_are_valid_not_rebalanced(self):
        for value in (0.,1.):
            a=annual_fixture();a[0]['loss_target'][:]=value;z=annual(a)
            np.testing.assert_array_equal(z['loss_prediction'],np.full(4,value))
            self.assertTrue(z['common_available'].all())

    def test_linear_probability_not_clipped(self):
        a=annual_fixture();calls=[]
        def prediction(fit,X):calls.append(1);return np.full(len(X),-2. if len(calls)%2 else 3.)
        with patch.object(m.numeric,'_predict',side_effect=prediction):z=annual(a)
        np.testing.assert_array_equal(z['loss_prediction'],np.full(4,3.))
        self.assertFalse(z['audit']['calibrated_probabilities'])

    def test_nonfinite_fit_is_hard_failure(self):
        with patch.object(m.numeric,'_predict',return_value=np.array([np.inf])):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):annual(annual_fixture())

    def test_query_availability_independent_of_future_prices(self):
        a=annual_fixture();a[3][:]='NEVERSEEN';z=annual(a)
        self.assertTrue(z['common_available'].all())

    def test_nonmonthend_mask_prevents_fit_and_prediction(self):
        a=annual_fixture();z=annual(a,monthend=np.zeros(4,bool))
        self.assertFalse(z['common_available'].any());self.assertEqual(z['audit']['new_pooled_model_fits'],0)
        self.assertTrue(all(row['status'] == 'no_queries' for row in z['audit']['annual']))

    def test_query_prefixes_exact(self):
        a=annual_fixture();z=annual(a)
        for n in range(5):
            p=copy.deepcopy(a)
            for i in (2,3,4,5):p[i]=p[i][:n]
            r=annual(p)
            for key in ('mean_return_prediction','loss_prediction','common_available'):np.testing.assert_array_equal(r[key],z[key][:n])

    def test_query_changes_never_fit_scaler(self):
        a=annual_fixture();z=annual(a);a[4]*=-1;p=annual(a)
        for l,r in zip(z['audit']['annual'],p['audit']['annual']):
            for key in ('training_indices_sha256','weights_sha256','scaler_mean_sha256','scaler_scale_sha256'):self.assertEqual(l[key],r[key])

    def test_target_binary_validation(self):
        a=annual_fixture();a[0]['loss_target'][0]=.5
        with self.assertRaisesRegex(ValueError,'binary'):annual(a)

    def test_wrong_width_nonboolean_complex_rejected(self):
        for k,change in ((1,lambda x:x[:,:9]),(5,lambda x:x.astype(int)),(4,lambda x:x.astype(complex))):
            a=annual_fixture();a[k]=change(a[k])
            with self.assertRaises(ValueError):annual(a)

    def test_used_plain_feature_bounds(self):
        a=annual_fixture();a[4][0,0]=.6
        with self.assertRaisesRegex(ValueError,'bounds'):annual(a)

    def test_nonfinite_feature_masks_shared(self):
        a=annual_fixture();a[4][0,0]=np.nan;z=annual(a)
        self.assertFalse(z['common_available'][0]);self.assertTrue(np.isnan(z['loss_prediction'][0]))

    def test_fit_exception_not_silently_zeroed(self):
        with patch.object(m.numeric,'_fit_ridge',side_effect=FloatingPointError('synthetic')):
            with self.assertRaises(FloatingPointError):annual(annual_fixture())


class RankTests(unittest.TestCase):
    def test_exact_shared_rank_and_loss_direction(self):
        z=m.rank_predictions([[1,2,3]],[[.2,.1,.3]],[[True]*3],[[True]*3])
        np.testing.assert_array_equal(z['mean_return_rank'],[[-.5,0,.5]])
        np.testing.assert_array_equal(z['negative_loss_rank'],[[0,.5,-.5]])

    def test_ties_average_and_zero_center(self):
        z=m.rank_predictions([[1,1,2,2]],[[1,1,2,2]],[[True]*4],[[True]*4])
        np.testing.assert_allclose(z['mean_return_rank'],[[-1/3,-1/3,1/3,1/3]],atol=1e-16)
        self.assertAlmostEqual(z['mean_return_rank'].sum(),0,places=15)

    def test_all_ties_singleton_empty_zero(self):
        for mask in ([[True]*3],[[True,False,False]],[[False]*3]):
            z=m.rank_predictions([[1,1,1]],[[0,0,0]],mask,[[True]*3])
            for key in m.HANDS:np.testing.assert_array_equal(z[key],[[0,0,0]])

    def test_one_missing_prediction_removes_from_both(self):
        z=m.rank_predictions([[1,99,3]],[[.2,np.nan,.1]],[[True]*3],[[True]*3])
        np.testing.assert_array_equal(z['common_available'],[[True,False,True]])
        np.testing.assert_array_equal(z['mean_return_rank'],[[-.5,0,.5]])
        np.testing.assert_array_equal(z['negative_loss_rank'],[[-.5,0,.5]])

    def test_ineligible_values_do_not_change_population(self):
        z=m.rank_predictions([[1,99,3]],[[.2,-99,.1]],[[True]*3],[[True,False,True]])
        np.testing.assert_array_equal(z['mean_return_rank'],[[-.5,0,.5]])

    def test_rank_prefix_independent(self):
        a=np.arange(15).reshape(5,3);b=-a;mask=np.ones(a.shape,bool)
        z=m.rank_predictions(a,b,mask,mask)
        for n in range(6):
            p=m.rank_predictions(a[:n],b[:n],mask[:n],mask[:n])
            for key in (*m.HANDS,'common_available'):np.testing.assert_array_equal(z[key][:n],p[key])


class BuildTests(unittest.TestCase):
    def test_output_schema_and_partial_initial_fallback(self):
        z=m.build(*build_fixture());self.assertEqual(set(z['arrays']),set(m.EXTRA));self.assertEqual(set(z['labels']),set(m.LABELS))
        self.assertEqual(z['arrays']['common_available'].shape,(5,3));self.assertFalse(z['arrays']['common_available'][0].any())
        for key in m.HANDS:np.testing.assert_array_equal(z['arrays'][key][0],[0,0,0])
        self.assertTrue(z['arrays']['common_available'][1:].all())

    def test_noninitial_nonmonthend_rejected(self):
        a=build_fixture();a[3]['dates'][2]='2017-01-30'
        with self.assertRaisesRegex(ValueError,'initial'):m.build(*a)

    def test_other_initial_nonmonthend_rejected(self):
        a=build_fixture();a[3]['dates'][0]='2016-08-16'
        with self.assertRaisesRegex(ValueError,'initial'):m.build(*a)

    def test_tail_evaluation_end_not_used_as_label(self):
        a=build_fixture();z=m.build(*a);a[3]['ends'][-1]='2020-03-24';p=m.build(*a)
        for key in m.EXTRA:np.testing.assert_array_equal(z['arrays'][key],p['arrays'][key])
        for key in m.LABELS:np.testing.assert_array_equal(z['labels'][key],p['labels'][key])

    def test_full_query_prefixes_including_empty(self):
        a=build_fixture();z=m.build(*a)
        for n in range(6):
            b=copy.deepcopy(a)
            b[3]={k:v[:n] for k,v in b[3].items()};b[5]=b[5][:n*3];b[8]=b[8][:n]
            p=m.build(*b)
            for key in m.EXTRA:np.testing.assert_array_equal(z['arrays'][key][:n],p['arrays'][key])

    def test_future_price_changes_do_not_change_earlier_query_year(self):
        a=build_fixture();z=m.build(*a);a[0].loc['2017-01-01':]*=3
        p=m.build(*a)
        for key in m.EXTRA:np.testing.assert_array_equal(z['arrays'][key][:3],p['arrays'][key][:3])

    def test_query_price_missing_does_not_gate_predictor(self):
        a=build_fixture();a[0]=a[0].loc[:'2019-12-31'];z=m.build(*a)
        self.assertTrue(z['arrays']['common_available'][-2:].all())

    def test_cached_training_end_mismatch_rejected(self):
        a=build_fixture();a[2]['ends'][0]='2014-02-27'
        with self.assertRaisesRegex(ValueError,'Cached label ends'):m.build(*a)

    def test_duplicate_metadata_or_width_rejected(self):
        a=build_fixture();a[3]['symbols'][1]=['A','A','C']
        with self.assertRaisesRegex(ValueError,'repeats'):m.build(*a)
        a=build_fixture();a[8]=a[8][:,:2]
        with self.assertRaisesRegex(ValueError,'width'):m.build(*a)

    def test_no_inputs_mutated(self):
        a=build_fixture();before=copy.deepcopy(a);m.build(*a)
        for i,(x,y) in enumerate(zip(a,before)):
            if i==0:pd.testing.assert_frame_equal(x,y)
            elif i==1:pd.testing.assert_index_equal(x,y)
            elif isinstance(x,dict):self.assertEqual(x,y)
            else:np.testing.assert_array_equal(x,y)


if __name__ == '__main__':
    unittest.main(verbosity=2)
