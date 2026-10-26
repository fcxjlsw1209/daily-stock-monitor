"""Synthetic-only validation for the paired SPY-relative run/incidence duration model."""
import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_relative_duration225_model as m


def month_ends(sessions):
    return pd.Series(sessions, index=sessions).groupby(sessions.to_period('M')).last().tolist()


def target_fixture():
    sessions = pd.bdate_range('2019-12-01', '2021-02-28')
    close = pd.DataFrame(100.0, index=sessions, columns=['A','B','C','SPY'])
    dates = np.array(['2020-01-31'] * 3, dtype='datetime64[D]')
    close.loc['2020-02-28'] = [110.0, 80.0, 100.0, 100.0]
    return [close, sessions, dates, np.array(['A','B','C']),
            np.ones(3,bool), np.ones(3,bool), np.ones(3,bool)]


def annual_fixture():
    ds = pd.date_range('2013-12-31', '2021-11-30', freq='ME')
    dates = np.repeat(ds.to_numpy(dtype='datetime64[D]'), 3)
    ends = np.repeat((ds + pd.offsets.MonthEnd(1)).to_numpy(dtype='datetime64[D]'), 3)
    rng = np.random.default_rng(190)
    X = rng.uniform(-0.5, 0.5, (len(dates), 10))
    ret = .2 * X[:,0] - .1 * X[:,1] + .03
    labels = dict(dates=dates, ends=ends, symbols=np.tile(['A','B','C'],len(ds)),
                  candidate_target=np.maximum(0., -ret), control_target=np.maximum(np.maximum(0., -ret), np.maximum(0., -ret*.7+.01)),
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
                          'C':300-idx*.015+np.cos(idx/11)*4,
                          'SPY':400+idx*.03+np.cos(idx/13)},index=sessions)
    rng = np.random.default_rng(190)
    return [close, sessions, trainmeta, meta,
            rng.uniform(-.5,.5,(len(train)*3,10)),rng.uniform(-.5,.5,(15,10)),
            np.ones(len(train)*3,bool),np.ones(len(train)*3,bool),np.ones((5,3),bool)]


class LabelsTests(unittest.TestCase):
    def test_same_incidence_different_chronological_run(self):
        a=[100,90,90,90,100,100,100]
        b=[100,90,100,90,100,90,100]
        self.assertEqual(m.relative_path(a,[100]*7),(.5,.5))
        self.assertEqual(m.relative_path(b,[100]*7),(1/6,.5))
        self.assertEqual(m.duration_counts(a,[100]*7),(3,3,6))
        self.assertEqual(m.duration_counts(b,[100]*7),(1,3,6))

    def test_equality_breaks_run_and_end_session_is_included(self):
        self.assertEqual(m.duration_counts([10,9,9,10,9],[1]*5),(2,3,4))
        self.assertEqual(m.duration_counts([10,10,10,9],[1]*4),(1,1,3))
        self.assertEqual(m.relative_path([10,9],[1,1]),(1.,1.))
        self.assertEqual(m.relative_path([10,10],[1,1]),(0.,0.))

    def test_zero_and_all_lag_are_valid_and_ignore_depth(self):
        self.assertEqual(m.relative_path([10,10,11,12],[1]*4),(0.,0.))
        self.assertEqual(m.relative_path([10,9,8,7],[1]*4),(1.,1.))
        self.assertEqual(m.relative_path([10,1,1,10],[1]*4),(2/3,2/3))
        self.assertEqual(m.relative_path([10,9,9,10],[1]*4),(2/3,2/3))

    def test_cumulative_relative_not_one_day_or_absolute_loss(self):
        self.assertEqual(m.relative_path([100,110],[100,130]),(1.,1.))
        self.assertEqual(m.relative_path([100,80],[100,70]),(0.,0.))
        # On second target day stock rises faster, but remains behind signal-relative SPY.
        self.assertEqual(m.duration_counts([100,90,99],[100,100,100]),(2,2,2))
        z=m.build_endpoint_targets(*target_fixture())
        np.testing.assert_array_equal(z['candidate_target'],[0.,.05,0.])
        np.testing.assert_array_equal(z['control_target'],[0.,.05,0.])
        self.assertTrue(z['good'].all())

    def test_extreme_binary64_no_float_cross_products(self):
        tiny=float(np.nextafter(0.,1.));largest=float(np.finfo(float).max)
        cases=(([tiny,largest],[tiny,largest],0.),
               ([largest,tiny],[largest,tiny],0.),
               ([largest,tiny],[tiny,largest],1.),
               ([tiny,largest],[largest,tiny],0.))
        with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
            for stock,spy,expected in cases:
                self.assertEqual(m.relative_path(stock,spy),(expected,expected))

    def test_near_tie_integer_products_not_rounded_float_products(self):
        epsilon=2.**-52
        stock=[1.,1.+epsilon];spy=[1.-epsilon,1.]
        self.assertEqual(float(Fraction.from_float(stock[1])*Fraction.from_float(spy[0])),1.)
        self.assertEqual(m.relative_path(stock,spy),(1.,1.))
        for end in (np.nextafter(10.,0.),10.,np.nextafter(10.,np.inf)):
            wanted=float(end<10.)
            self.assertEqual(m.relative_path([10.,end],[1.,1.]),(wanted,wanted))

    def test_integer_ratio_rounds_once_and_scale_invariance(self):
        stock=[10,9,9,10,9,10,9];spy=[1]*7
        value=m.relative_path(stock,spy)
        self.assertEqual(value,(float(Fraction(2,6)),float(Fraction(4,6))))
        self.assertEqual(m.relative_path(np.array(stock)*8.,np.array(spy)*.125),value)

    def test_strict_numeric_nonboolean_paths(self):
        for bad in ([True,1.],['1','2'],[1,None],[1,1+0j],[],[1],[[1],[2]]):
            with self.assertRaises(ValueError):m.relative_path(bad,[1]*len(bad))
        for bad in (0.,-1.,np.nan,np.inf,-np.inf):
            with self.assertRaises(ValueError):m.relative_path([1,bad],[1,1])
        with self.assertRaises(ValueError):m.relative_path([1,2],[1,2,3])

    def test_inherited_false_rows_not_inspected_and_all_masks_restrict(self):
        for k in (4,5,6):
            a=target_fixture();a[k][0]=False;a[0]=a[0].astype(object)
            a[0].loc['2020-01-31':'2020-02-28','A']='not inspected'
            z=m.build_endpoint_targets(*a)
            self.assertTrue(np.isnan(z['candidate_target'][0]));self.assertFalse(z['base_good'][0])
            self.assertTrue(z['good'][1:].all())
            c=z['audit']['path_checks'][0]
            self.assertEqual(c['status'],'inherited_base_unavailable')
            self.assertEqual(c['invalid_stock_dates'],[])
            self.assertIsNone(c['lag_session_count']);self.assertIsNone(c['longest_lag_run'])

    def test_singleton_keeps_raw_and_counts_but_not_training(self):
        a=target_fixture();a[4][[0,2]]=False;z=m.build_endpoint_targets(*a)
        self.assertFalse(z['good'].any());self.assertEqual(z['candidate_target'][1],.05)
        self.assertEqual(z['audit']['source_path_rows'],1)
        self.assertEqual(z['audit']['lag_session_observations'],1)
        self.assertEqual(z['audit']['longest_run_session_observations'],1)
        self.assertEqual(z['audit']['months'][0]['status'],'fewer_than_two_shared_rows')

    def test_missing_interior_cannot_compress_or_join_runs(self):
        for day in ('2020-01-31','2020-02-14','2020-02-28'):
            a=target_fixture();a[0]=a[0].drop(index=day);z=m.build_endpoint_targets(*a)
            self.assertFalse(z['good'].any());self.assertTrue(np.isnan(z['candidate_target']).all())
            self.assertTrue(np.isnan(z['control_target']).all())
            self.assertEqual(z['audit']['path_checks'][0]['missing_official_sessions'],[day])
            self.assertEqual(z['audit']['source_path_rows'],0)
            self.assertEqual(z['audit']['lag_session_observations'],0)

    def test_every_consumed_source_cell_and_missing_benchmark(self):
        for symbol in ('A','SPY'):
            for day in ('2020-01-31','2020-02-14','2020-02-28'):
                for value in (0.,-1.,np.nan,np.inf,-np.inf,True,'bad'):
                    a=target_fixture();a[0]=a[0].astype(object);a[0].loc[day,symbol]=value
                    z=m.build_endpoint_targets(*a)
                    self.assertFalse(z['good'][0]);self.assertTrue(np.isnan(z['control_target'][0]))
                    self.assertIsNone(z['audit']['path_checks'][0]['longest_lag_run'])
            a=target_fixture();a[0]=a[0].drop(columns=symbol);z=m.build_endpoint_targets(*a)
            self.assertFalse(z['good'][0])

    def test_complete_calendar_witness_and_monthend_required(self):
        a=target_fixture();a[1]=a[1][a[1]<=pd.Timestamp('2020-02-28')]
        with self.assertRaisesRegex(ValueError,'next official session'):m.build_endpoint_targets(*a)
        a=target_fixture();a[2][:]=np.datetime64('2020-01-30')
        with self.assertRaisesRegex(ValueError,'official calendar month-end'):m.build_endpoint_targets(*a)
        a=target_fixture()
        with self.assertRaisesRegex(ValueError,'Cached label ends'):
            m.build_endpoint_targets(*a,expected_ends=np.array(['2020-02-27']*3,dtype='datetime64[D]'))

    def test_december_january_leap_and_custom_official_holiday(self):
        a=target_fixture();a[2][:]=np.datetime64('2019-12-31')
        self.assertEqual(str(m.build_endpoint_targets(*a)['ends'][0]),'2020-01-31')
        a=target_fixture();a[1]=a[1][a[1]!=pd.Timestamp('2020-02-17')]
        z=m.build_endpoint_targets(*a)
        self.assertEqual(z['audit']['months'][0]['target_session_count'],19)
        self.assertNotIn('2020-02-17',z['audit']['months'][0]['source_dates'])
        self.assertEqual(z['candidate_target'][1],1/19)

    def test_future_prices_and_unrelated_columns_not_consumed(self):
        a=target_fixture();z=m.build_endpoint_targets(*a)
        a[0]=a[0].astype(object);a[0].loc['2020-03-01':]='future poison';a[0]['OTHER']='ignored'
        p=m.build_endpoint_targets(*a)
        for key in m.TARGETS:np.testing.assert_array_equal(z[key],p[key])
        self.assertEqual(z['audit'],p['audit'])

    def test_duplicate_metadata_nonboolean_masks_and_axes_fail(self):
        a=target_fixture();a[3][1]='A'
        with self.assertRaises(ValueError):m.build_endpoint_targets(*a)
        a=target_fixture();a[4]=a[4].astype(int)
        with self.assertRaises(ValueError):m.build_endpoint_targets(*a)
        a=target_fixture();a[1]=a[1][::-1]
        with self.assertRaises(ValueError):m.build_endpoint_targets(*a)
        a=target_fixture();a[0]=pd.concat([a[0],a[0].iloc[-1:]])
        with self.assertRaises(ValueError):m.build_endpoint_targets(*a)

    def test_label_audit_exact_contract(self):
        z=m.build_endpoint_targets(*target_fixture())['audit']
        self.assertEqual(z['relative_target_definitions'],{'candidate':'max_consecutive_d I[R_d<1]/L','control':'sum_d I[R_d<1]/L'})
        for key in ('exact_binary_float_rational','rounded_targets_once','complete_positive_finite_paths',
                    'candidate_bounded_by_control','lag_equality_breaks_run','signal_session_excluded',
                    'target_end_included','inherited_common_good_retained','no_endpoint_shortening'):
            self.assertTrue(z[key])
        self.assertNotIn('target_bounds_independent',z)
        self.assertFalse(z['calendar_completeness_certified_by_this_function'])
        self.assertEqual(z['source_path_rows'],3);self.assertEqual(z['source_price_observations'],126)
        self.assertEqual(z['target_session_observations'],60);self.assertEqual(z['missing_path_rows'],0)
        self.assertEqual(z['lag_session_observations'],1);self.assertEqual(z['longest_run_session_observations'],1)
        self.assertEqual([r['lag_session_count'] for r in z['path_checks']],[0,1,0])
        self.assertEqual([r['longest_lag_run'] for r in z['path_checks']],[0,1,0])
        self.assertFalse(z['clipped']);self.assertFalse(z['centered'])


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
        for key,out in zip(m.TARGETS,('candidate_prediction','control_prediction')):
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
        a=annual_fixture();a[0]['control_target'][30:32]=np.nan
        observed=[];fn=m.numeric._fit_ridge
        def fit(X,y,w,alpha):observed.append(len(y));return fn(X,y,w,alpha)
        with patch.object(m.numeric,'_fit_ridge',side_effect=fit):z=annual(a)
        self.assertTrue(all(observed[i]==observed[i+1] for i in range(0,len(observed),2)))
        self.assertNotIn(32,z['audit']['annual'][0]['training_indices'])

    def test_end_after_cutoff_cannot_affect_past_prediction(self):
        a=annual_fixture();a[2]=a[2][-1:];a[3]=a[3][-1:];a[4]=a[4][-1:];a[5]=a[5][-1:]
        z=annual(a);future=a[0]['ends']>np.datetime64('2020-12-31')
        a[1][future]=.49;a[0]['candidate_target'][future]=1;a[0]['control_target'][future]=1
        p=annual(a)
        for key in ('candidate_prediction','control_prediction'):np.testing.assert_array_equal(z[key],p[key])
        self.assertEqual(z['audit']['annual'][0]['latest_label_end'],'2020-12-31')

    def test_exclusive_five_year_lower_bound(self):
        a=annual_fixture();a[2]=a[2][-1:];a[3]=a[3][-1:];a[4]=a[4][-1:];a[5]=a[5][-1:]
        z=annual(a);old=a[0]['dates']<=np.datetime64('2015-12-31')
        a[1][old]=-.49;a[0]['candidate_target'][old]=0;a[0]['control_target'][old]=1
        p=annual(a)
        for key in ('candidate_prediction','control_prediction'):np.testing.assert_array_equal(z[key],p[key])

    def test_seventeen_months_cold_eighteen_mature(self):
        a=annual_fixture();a[0]['good'][:]=False;a[0]['good'][3:54]=True
        self.assertFalse(annual(a)['common_available'][0])
        a[0]['good'][54:57]=True;self.assertTrue(annual(a)['common_available'][0])

    def test_early_query_does_not_fit_with_future_months(self):
        a=annual_fixture();a[2][0]=np.datetime64('2014-08-29');z=annual(a)
        self.assertFalse(z['common_available'][0]);self.assertTrue(np.isnan(z['candidate_prediction'][0]))

    def test_constant_targets_are_valid_not_rebalanced(self):
        for value in (0.,1.):
            a=annual_fixture();a[0]['candidate_target'][:]=value;a[0]['control_target'][:]=value;z=annual(a)
            for key in ('candidate_prediction','control_prediction'):
                np.testing.assert_array_equal(z[key],np.full(4,value))
            self.assertTrue(z['common_available'].all())
            r=m.rank_predictions(z['candidate_prediction'][None,:],z['control_prediction'][None,:],
                                 z['common_available'][None,:],np.ones((1,4),bool))
            for key in m.HANDS:np.testing.assert_array_equal(r[key],np.zeros((1,4)))
        # Either continuous target may be constant while the matched target varies.
        a=annual_fixture();a[0]['candidate_target'][:]=0.;z=annual(a)
        np.testing.assert_array_equal(z['candidate_prediction'],np.zeros(4))

    def test_linear_prediction_not_clipped(self):
        a=annual_fixture();calls=[]
        def prediction(fit,X):calls.append(1);return np.full(len(X),-2. if len(calls)%2 else 3.)
        with patch.object(m.numeric,'_predict',side_effect=prediction):z=annual(a)
        np.testing.assert_array_equal(z['control_prediction'],np.full(4,3.))
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
            for key in ('candidate_prediction','control_prediction','common_available'):np.testing.assert_array_equal(r[key],z[key][:n])

    def test_query_changes_never_fit_scaler(self):
        a=annual_fixture();z=annual(a);a[4]*=-1;p=annual(a)
        for l,r in zip(z['audit']['annual'],p['audit']['annual']):
            for key in ('training_indices_sha256','weights_sha256','scaler_mean_sha256','scaler_scale_sha256'):self.assertEqual(l[key],r[key])

    def test_control_continuous_and_independent_bounds(self):
        a=annual_fixture();a[0]['control_target'][0]=.5
        self.assertTrue(annual(a)['common_available'].all())
        for key in m.TARGETS:
            for bad in (-.01,1.01):
                a=annual_fixture();a[0][key][0]=bad
                with self.assertRaisesRegex(ValueError,'bounded in'):annual(a)

    def test_wrong_width_nonboolean_complex_rejected(self):
        for k,change in ((1,lambda x:x[:,:9]),(5,lambda x:x.astype(int)),(4,lambda x:x.astype(complex))):
            a=annual_fixture();a[k]=change(a[k])
            with self.assertRaises(ValueError):annual(a)

    def test_used_plain_feature_bounds(self):
        a=annual_fixture();a[4][0,0]=.6
        with self.assertRaisesRegex(ValueError,'bounds'):annual(a)

    def test_nonfinite_feature_masks_shared(self):
        a=annual_fixture();a[4][0,0]=np.nan;z=annual(a)
        self.assertFalse(z['common_available'][0]);self.assertTrue(np.isnan(z['control_prediction'][0]))

    def test_fit_exception_not_silently_zeroed(self):
        with patch.object(m.numeric,'_fit_ridge',side_effect=FloatingPointError('synthetic')):
            with self.assertRaises(FloatingPointError):annual(annual_fixture())

    def test_label_order_enforced_but_prediction_order_not_forced(self):
        a=annual_fixture();a[0]['candidate_target'][0]=1.;a[0]['control_target'][0]=0.
        with self.assertRaisesRegex(ValueError,'cannot exceed'):annual(a)
        calls=[]
        def prediction(fit,X):
            calls.append(1);return np.full(len(X),3. if len(calls)%2 else -2.)
        with patch.object(m.numeric,'_predict',side_effect=prediction):z=annual(annual_fixture())
        np.testing.assert_array_equal(z['candidate_prediction'],np.full(4,3.))
        np.testing.assert_array_equal(z['control_prediction'],np.full(4,-2.))

    def test_annual_target_statistics_and_accounting(self):
        a=annual_fixture();z=annual(a)
        self.assertEqual(z['audit']['new_pooled_model_fits'],sum(r['model_fits'] for r in z['audit']['annual']))
        for row in z['audit']['annual']:
            ids=row['training_indices']
            for key in m.TARGETS:
                y=a[0][key][ids]
                self.assertEqual(row['target_positive_counts'][key],int(np.count_nonzero(y>0)))
                self.assertEqual(row['target_sums'][key],float(y.sum()))
                self.assertEqual(row['target_means'][key],float(y.mean()))
                self.assertEqual(row['target_maxima'][key],float(y.max()))
                self.assertNotIn('target_counts',row);self.assertNotIn('target_prevalence',row)
                if row['status']=='fit':
                    w=m.numeric.month_balanced_weights(a[0]['dates'][ids],row['cutoff'],12.)
                    self.assertEqual(row['weighted_target_means'][key],float(np.average(y,weights=w)))
                    self.assertNotIn('weighted_target_prevalence',row)

    def test_duration_validation_bounds_and_zero_positive_counts(self):
        for value in (-1e-10,1.00001):
            a=annual_fixture();a[0]['candidate_target'][0]=value;a[0]['control_target'][0]=1.
            with self.assertRaisesRegex(ValueError,'bounded in'):annual(a)
        a=annual_fixture();a[0]['candidate_target'][:]=0.;a[0]['control_target'][:]=1.
        z=annual(a)
        np.testing.assert_array_equal(z['candidate_prediction'],np.zeros(4))
        np.testing.assert_array_equal(z['control_prediction'],np.ones(4))
        # Ordered target bounds do not require identical positive counts in the low-level annual API.
        for row in z['audit']['annual']:
            self.assertEqual(row['target_positive_counts']['candidate_target'],0)
            self.assertEqual(row['target_positive_counts']['control_target'],row['training_rows'])

    def test_empty_target_statistics_not_nan_or_event_counts(self):
        a=annual_fixture();a[0]['good'][:]=False;z=annual(a)
        for row in z['audit']['annual']:
            for key in m.TARGETS:
                self.assertEqual(row['target_positive_counts'][key],0)
                self.assertEqual(row['target_sums'][key],0.)
                self.assertIsNone(row['target_means'][key]);self.assertIsNone(row['target_maxima'][key])
            self.assertNotIn('weighted_target_means',row)
        self.assertEqual(z['audit']['new_pooled_model_fits'],0)

    def test_constant_fractional_duration_valid(self):
        a=annual_fixture();a[0]['candidate_target'][:]=.25;a[0]['control_target'][:]=1.
        z=annual(a)
        np.testing.assert_array_equal(z['candidate_prediction'],np.full(4,.25))
        np.testing.assert_array_equal(z['control_prediction'],np.ones(4))



class RankTests(unittest.TestCase):
    def test_exact_shared_rank_and_loss_direction(self):
        z=m.rank_predictions([[1,2,3]],[[.2,.1,.3]],[[True]*3],[[True]*3])
        np.testing.assert_array_equal(z['candidate_rank'],[[.5,0,-.5]])
        np.testing.assert_array_equal(z['control_rank'],[[0,.5,-.5]])

    def test_ties_average_and_zero_center(self):
        z=m.rank_predictions([[1,1,2,2]],[[1,1,2,2]],[[True]*4],[[True]*4])
        np.testing.assert_allclose(z['candidate_rank'],[[1/3,1/3,-1/3,-1/3]],atol=1e-16)
        self.assertAlmostEqual(z['candidate_rank'].sum(),0,places=15)

    def test_all_ties_singleton_empty_zero(self):
        for mask in ([[True]*3],[[True,False,False]],[[False]*3]):
            z=m.rank_predictions([[1,1,1]],[[0,0,0]],mask,[[True]*3])
            for key in m.HANDS:np.testing.assert_array_equal(z[key],[[0,0,0]])

    def test_one_missing_prediction_removes_from_both(self):
        z=m.rank_predictions([[1,99,3]],[[.2,np.nan,.1]],[[True]*3],[[True]*3])
        np.testing.assert_array_equal(z['common_available'],[[True,False,True]])
        np.testing.assert_array_equal(z['candidate_rank'],[[.5,0,-.5]])
        np.testing.assert_array_equal(z['control_rank'],[[-.5,0,.5]])

    def test_ineligible_values_do_not_change_population(self):
        z=m.rank_predictions([[1,99,3]],[[.2,-99,.1]],[[True]*3],[[True,False,True]])
        np.testing.assert_array_equal(z['candidate_rank'],[[.5,0,-.5]])

    def test_rank_prefix_independent(self):
        a=np.arange(15).reshape(5,3);b=-a;mask=np.ones(a.shape,bool)
        z=m.rank_predictions(a,b,mask,mask)
        for n in range(6):
            p=m.rank_predictions(a[:n],b[:n],mask[:n],mask[:n])
            for key in (*m.HANDS,'common_available'):np.testing.assert_array_equal(z[key][:n],p[key])

    def test_infinite_forecast_rejected(self):
        with self.assertRaisesRegex(ValueError,'Infinite'):
            m.rank_predictions([[np.inf,0]],[[0,0]],[[True,True]],[[True,True]])


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

    def test_exact_source_cohort_parity_with209_without_reusing_targets(self):
        import monthly_relative_path209_model as previous
        args=build_fixture();current=m.build(*args)
        old=previous.build_endpoint_targets(args[0],args[1],
            np.repeat(np.array(args[2]['dates'],dtype='datetime64[D]'),3),
            np.array(args[2]['symbols']).reshape(-1),args[6],args[7],
            np.isfinite(args[4]).all(axis=1),expected_ends=np.repeat(np.array(args[2]['ends'],dtype='datetime64[D]'),3))
        for key in ('dates','ends','symbols','base_good','good'):
            np.testing.assert_array_equal(current['labels'][key],old[key])
        self.assertTrue(np.any(current['labels']['candidate_target']!=old['candidate_target']))
        self.assertTrue(np.any(current['labels']['control_target']!=old['control_target']))
        self.assertTrue(np.all(current['labels']['candidate_target']<=current['labels']['control_target']))

    def test_physical_mature_training_query_source_prefixes(self):
        args=build_fixture();full=m.build(*args)
        for n in (0,1,2,3,5):
            short=copy.deepcopy(args)
            short[3]={k:v[:n] for k,v in short[3].items()}
            short[5]=short[5][:3*n];short[8]=short[8][:n]
            if n:
                last=pd.Timestamp(short[3]['dates'][-1])
                cutoff=np.datetime64('%04d-12-31'%(last.year-1))
                keep=np.asarray(short[2]['ends'],dtype='datetime64[D]')<=cutoff
                short[0]=short[0].loc[:last].copy()
            else:
                keep=np.zeros(len(short[2]['dates']),bool);short[0]=short[0].iloc[:0].copy()
            short[2]={k:[v for v,g in zip(seq,keep) if g] for k,seq in short[2].items()}
            keep=np.repeat(keep,3)
            for i in (4,6,7):short[i]=short[i][keep]
            result=m.build(*short)
            for key in m.EXTRA:np.testing.assert_array_equal(result['arrays'][key],full['arrays'][key][:n])


if __name__ == '__main__':
    unittest.main(verbosity=2)
