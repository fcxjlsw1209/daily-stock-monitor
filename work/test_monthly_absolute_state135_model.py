"""Synthetic mapping tests only; never loads market data or fits a model."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import monthly_absolute_state135_model as m


def model(sd=.004, variances=(1., 4.)):
    return dict(available=True, status='available', reason=None, fit_attempted=True,
                scaler=dict(mean=.001, sd=sd),
                parameters=dict(means=[-.1, .2], variances=list(variances),
                                initial=[.4, .6], transition=[[.97, .03], [.03, .97]]),
                filtered_posterior=[.7, .3],
                certificate=dict(converged=True, all_arithmetic_finite=True,
                                 low_variance_state_first=True))


def fixture(sd=.004):
    year = dict(year=2020, fit_attempted=True, status='available', reason=None,
                model=model(sd), cutoff='2019-12-31', first_invalid_query_date=None)
    signals = [dict(signal_index=i, signal_date=day, year=2020, available=True,
                    reason=None, first_invalid_query_date=None,
                    filtered_probability=p, memoryless_probability=q)
               for i, (day, p, q) in enumerate([
                   ('2020-01-31', .7, .4), ('2020-02-28', 0., 1.), ('2020-03-31', 1., 0.)])]
    return dict(years=[year], signals=signals, new_model_fits=1), np.array([.7, 0., 1.]), np.array([.4, 1., 0.])


class ClassificationTests(unittest.TestCase):
    def test_three_exact_mapping_cases_and_both_probability_streams(self):
        for sd, expected, mask in ((.001, [1., 1., 1.], [True, True]),
                                   (.02, [0., 0., 0.], [False, False]),
                                   (.004, None, [True, False])):
            with self.subTest(sd=sd):
                a, f, q = fixture(sd); r = m.transform(a, f, q)
                np.testing.assert_array_equal(r['filtered'], f if expected is None else expected)
                np.testing.assert_array_equal(r['memoryless'], q if expected is None else expected)
                self.assertEqual(r['audit']['years'][0]['quiet_state_mask'], mask)

    def test_threshold_equality_is_not_quiet_no_epsilon(self):
        sd = .1/math.sqrt(252)
        d = model(sd, (1., 4.)); r = m.classify_model(d)
        self.assertEqual(r['annualized_state_volatility'][0], .1)
        self.assertEqual(r['quiet_state_mask'], [False, False])
        d['scaler']['sd'] = math.nextafter(sd, 0.)
        self.assertEqual(m.classify_model(d)['quiet_state_mask'], [True, False])
        d['scaler']['sd'] = math.nextafter(sd, math.inf)
        self.assertEqual(m.classify_model(d)['quiet_state_mask'], [False, False])

    def test_high_state_at_threshold_is_nonquiet(self):
        r = m.classify_model(model(.1/math.sqrt(252), (.25, 1.)))
        self.assertEqual(r['quiet_state_mask'], [True, False])
        self.assertEqual(r['annualized_state_volatility'][1], .1)

    def test_annualization_order_and_mean_irrelevance(self):
        d = model(); r = m.classify_model(d)
        self.assertEqual(r['annualized_state_volatility'],
                         [.004*math.sqrt(v)*math.sqrt(252) for v in (1., 4.)])
        d['scaler']['mean'] = 100.; d['parameters']['means'] = [80., -90.]
        self.assertEqual(m.classify_model(d), r)

    def test_relative_labels_are_not_absolute_classification(self):
        self.assertEqual(m.classify_model(model(.02))['quiet_state_mask'], [False, False])
        self.assertEqual(m.classify_model(model(.001))['quiet_state_mask'], [True, True])

    def test_unordered_equal_negative_nonfinite_variances_rejected(self):
        for variances in ((4.,1.),(1.,1.),(0.,1.),(-1.,2.),(1.,math.inf),(1.,math.nan)):
            with self.subTest(variances=variances), self.assertRaises(ValueError):
                m.classify_model(model(variances=variances))

    def test_bad_scales_and_certification_rejected(self):
        for sd in (0., -1., math.inf, math.nan, True, '.004', None):
            with self.subTest(sd=sd), self.assertRaises(ValueError):m.classify_model(model(sd))
        for key in ('converged', 'all_arithmetic_finite', 'low_variance_state_first'):
            d=model(); d['certificate'][key]=False
            with self.subTest(key=key), self.assertRaises(ValueError):m.classify_model(d)

    def test_malformed_other_model_parameters_rejected(self):
        for key, value in [('means',[0.]),('means',[0.,math.nan]),('initial',[.7,.7]),
                           ('transition',[[.9,.2],[.1,.9]])]:
            d=model(); d['parameters'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):m.classify_model(d)
        d=model(); d['filtered_posterior']=[False,True]
        with self.assertRaises(ValueError):m.classify_model(d)

    def test_derived_overflow_is_hard_failure(self):
        with self.assertRaises(FloatingPointError):m.classify_model(model(1e308))


class TransformTests(unittest.TestCase):
    def test_deep_copy_preserves_all_source_evidence_and_no_new_fit(self):
        a,f,q=fixture(); original=copy.deepcopy(a); fc=f.copy(); qc=q.copy()
        r=m.transform(a,f,q)
        self.assertEqual(a,original);np.testing.assert_array_equal(f,fc);np.testing.assert_array_equal(q,qc)
        self.assertEqual(r['audit']['new_model_fits'],0);self.assertEqual(r['audit']['reused_model_fits'],1)
        self.assertEqual(r['audit']['years'][0]['model'],a['years'][0]['model'])
        self.assertIsNot(r['audit']['years'][0]['model'],a['years'][0]['model'])
        for old,new in zip(a['signals'],r['audit']['signals']):
            self.assertEqual(new['original_filtered_probability'],old['filtered_probability'])
            self.assertEqual(new['original_memoryless_probability'],old['memoryless_probability'])
            for key in ('available','reason','year','signal_index','signal_date','first_invalid_query_date'):
                self.assertEqual(old[key],new[key])
        json.dumps(r['audit'],allow_nan=False)

    def test_missing_query_stays_nan_even_when_both_states_quiet(self):
        a,f,q=fixture(.001); a['years'][0]['first_invalid_query_date']='2020-02-03'
        for i in (1,2):
            a['signals'][i].update(available=False,reason='invalid_query_history',
                filtered_probability=None,memoryless_probability=None,first_invalid_query_date='2020-02-03')
            f[i]=q[i]=math.nan
        r=m.transform(a,f,q)
        self.assertEqual(r['filtered'][0],1.);self.assertTrue(np.isnan(r['filtered'][1:]).all())
        self.assertTrue(np.isnan(r['memoryless'][1:]).all());json.dumps(r['audit'],allow_nan=False)
        self.assertIsNone(r['audit']['signals'][1]['filtered_probability'])

    def test_unavailable_year_before_fit(self):
        a,f,q=fixture();a['new_model_fits']=0
        a['years'][0].update(fit_attempted=False,status='unavailable',reason='incomplete_training_prices',model=None)
        for row in a['signals']:row.update(available=False,reason='incomplete_training_prices',filtered_probability=None,memoryless_probability=None)
        f[:]=q[:]=math.nan;r=m.transform(a,f,q)
        self.assertTrue(np.isnan(r['filtered']).all());self.assertEqual(r['audit']['reused_model_fits'],0)
        self.assertIsNone(r['audit']['years'][0]['annualized_state_volatility'])

    def test_unavailable_postfit_model_is_not_reclassified(self):
        a,f,q=fixture();a['years'][0].update(status='unavailable',reason='unseparated_state_variances',
            model=dict(available=False,status='unavailable',fit_attempted=True,parameters=None))
        for row in a['signals']:row.update(available=False,reason='unseparated_state_variances',filtered_probability=None,memoryless_probability=None)
        f[:]=q[:]=math.nan;r=m.transform(a,f,q)
        self.assertTrue(np.isnan(r['filtered']).all());self.assertEqual(r['audit']['reused_model_fits'],1)

    def test_available_bad_probability_rejected_even_for_constant_mapping(self):
        for sd in (.001,.02):
            for value in (-.01,1.01,math.nan,math.inf,True,'.7',None):
                a,f,q=fixture(sd);a['signals'][0]['filtered_probability']=value
                with self.subTest(sd=sd,value=value),self.assertRaises(ValueError):m.transform(a,f,q)

    def test_arrays_must_match_source_values_exactly(self):
        a,f,q=fixture();f[0]=math.nextafter(f[0],1.)
        with self.assertRaises(ValueError):m.transform(a,f,q)
        a,f,q=fixture();q[0]=.3
        with self.assertRaises(ValueError):m.transform(a,f,q)

    def test_array_shape_type_and_infinity_rejected(self):
        a,f,q=fixture()
        for bad in ([True,False,True],['.7','0','1'],f[:,None],f[:2],[.7,math.inf,1.]):
            with self.subTest(bad=str(bad)),self.assertRaises(ValueError):m.transform(a,bad,q)

    def test_unavailable_arrays_require_nan_and_audit_null(self):
        a,f,q=fixture();a['signals'][0].update(available=False,reason='invalid_query_history',filtered_probability=None,memoryless_probability=None)
        with self.assertRaises(ValueError):m.transform(a,f,q)
        f[0]=q[0]=math.nan;a['signals'][0]['filtered_probability']=math.nan
        with self.assertRaises(ValueError):m.transform(a,f,q)

    def test_dates_indices_years_and_source_count_are_bound(self):
        mutations=[lambda a:a['signals'][0].update(signal_index=1),
                   lambda a:a['signals'][0].update(year=2019),
                   lambda a:a['signals'][1].update(signal_date='2020-01-31'),
                   lambda a:a.update(new_model_fits=0),
                   lambda a:a.update(new_model_fits=True),
                   lambda a:a['years'].append(copy.deepcopy(a['years'][0]))]
        for mutate in mutations:
            a,f,q=fixture();mutate(a)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):m.transform(a,f,q)

    def test_source_signal_year_availability_must_agree(self):
        a,f,q=fixture();a['years'][0].update(status='unavailable',reason='bad',model=None,fit_attempted=False);a['new_model_fits']=0
        with self.assertRaises(ValueError):m.transform(a,f,q)
        a,f,q=fixture();a['signals'][0]['reason']='unexpected'
        with self.assertRaises(ValueError):m.transform(a,f,q)

    def test_empty_mapping(self):
        r=m.transform(dict(years=[],signals=[],new_model_fits=0),[],[])
        self.assertEqual(r['filtered'].shape,(0,));self.assertEqual(r['audit']['reused_model_fits'],0)

    def test_mapping_is_not_idempotently_reapplied_to_absolute_audit(self):
        a,f,q=fixture();r=m.transform(a,f,q)
        with self.assertRaises(ValueError):m.transform(r['audit'],r['filtered'],r['memoryless'])

    def test_future_extension_does_not_change_existing_signal_mapping(self):
        a,f,q=fixture();short=copy.deepcopy(a);short['signals']=short['signals'][:1]
        r0=m.transform(short,f[:1],q[:1]);r1=m.transform(a,f,q)
        self.assertEqual(r0['audit']['signals'][0],r1['audit']['signals'][0])
        np.testing.assert_array_equal(r0['filtered'],r1['filtered'][:1])
        extra=copy.deepcopy(a['years'][0]);extra['year']=2021;extra['model']=model(.02)
        a['years'].append(extra);a['new_model_fits']=2
        a['signals'].append(dict(signal_index=3,signal_date='2021-01-29',year=2021,
            available=True,reason=None,filtered_probability=.9,memoryless_probability=.2))
        r2=m.transform(a,np.r_[f,.9],np.r_[q,.2])
        self.assertEqual(r1['audit']['signals'],r2['audit']['signals'][:3])
        self.assertEqual(r2['filtered'][3],0.)

    def test_no_file_io_or_fitting_dependency(self):
        a,f,q=fixture()
        with patch('builtins.open',side_effect=AssertionError('I/O forbidden')):
            r=m.transform(a,f,q)
        self.assertEqual(r['audit']['new_model_fits'],0)
        self.assertFalse(hasattr(m,'fit'));self.assertFalse(hasattr(m,'filter_path'))


if __name__=='__main__':unittest.main()
