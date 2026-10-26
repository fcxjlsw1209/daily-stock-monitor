"""Synthetic-only tests for the two-month prediction confirmation transform."""
import copy
import json
import math
import unittest

import numpy as np
import monthly_persistence217_model as model


def fixture():
    """Two securities, changed column order and a 53-week-year boundary."""
    meta = dict(dates=['2020-11-30','2020-12-31','2021-01-29','2021-02-26'],
                ends=['2020-12-31','2021-01-29','2021-02-26','2021-03-31'],
                symbols=[['A','B'],['B','A'],['A','B'],['B','A']])
    # A corrections +2,+4,-2,-4; B corrections -2,-4,-6,zero.
    p = np.array([12.,8.,6.,14.,8.,4.,10.,6.])
    source = dict(raw_local_linear=p, raw_source_log=np.full(8,10.), available=np.ones(8,bool))
    return meta, source, np.ones((4,2),bool)


def physical_prefix(args, n):
    meta, source, use = args
    width = use.shape[1]
    return ({k: copy.deepcopy(v[:n]) for k,v in meta.items()},
            {k: np.asarray(v)[:n*width].copy() for k,v in source.items()}, use[:n].copy())


def record(out, row):
    return next(r for r in out['audit']['records'] if r['row'] == row)


class PersistenceTests(unittest.TestCase):
    def test_analytic_branches_exact_identity_and_column_change(self):
        args=fixture(); out=model.build_forecasts(*args)
        np.testing.assert_array_equal(out['raw']['confirmed'],[12,8,6,14,10,4,10,6])
        np.testing.assert_array_equal(out['raw']['half'],[12,8,8,12,9,7,10,8])
        self.assertEqual(record(out,2)['prior_row'],1)
        self.assertEqual(record(out,3)['prior_row'],0)
        self.assertEqual(record(out,4)['prior_row'],3)
        self.assertEqual(record(out,4)['status'],'unconfirmed')
        self.assertEqual(record(out,6)['status'],'current_zero')
        self.assertIsNone(record(out,6)['previous_difference'])

    def test_53_week_year_crosses_by_calendar_month(self):
        out=model.build_forecasts(*fixture())
        for row in (4,5): self.assertTrue(record(out,row)['cross_year'])
        for row in (0,1,2,3,6,7): self.assertFalse(record(out,row)['cross_year'])
        self.assertEqual(out['audit']['counts']['cross_year_supported_rows'],2)

    def test_exact_count_identities_and_json(self):
        out=model.build_forecasts(*fixture()); a=out['audit']; c=a['counts']
        self.assertEqual(c,dict(query_rows=8,cached_forecast_rows=8,supported_rows=6,
                               unsupported_rows=2,current_zero_rows=1,confirmed_rows=4,
                               unconfirmed_rows=1,cross_year_supported_rows=2))
        self.assertEqual(set(c),set(model.COUNT_FIELDS))
        self.assertEqual(c['cached_forecast_rows'],c['supported_rows']+c['unsupported_rows'])
        self.assertEqual(c['supported_rows'],c['current_zero_rows']+c['confirmed_rows']+c['unconfirmed_rows'])
        self.assertEqual([r['row'] for r in a['records']],list(range(8)))
        for key in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits','new_parameters'):
            self.assertEqual(a[key],0)
        json.dumps(a,allow_nan=False)

    def test_missing_month_never_bridges_to_older_month(self):
        meta,source,use=fixture(); keep=[0,2,3]; ids=np.array([0,1,4,5,6,7])
        meta={k:[v[i] for i in keep] for k,v in meta.items()}
        source={k:v[ids].copy() for k,v in source.items()}
        out=model.build_forecasts(meta,source,use[keep])
        for j in (2,3):
            self.assertEqual(record(out,j)['reason'],'missing_previous_formal_month')
            self.assertIsNone(record(out,j)['prior_row'])
            self.assertEqual(out['raw']['half'][j],source['raw_local_linear'][j])

    def test_partial_is_excluded_but_formal_same_month_is_used(self):
        meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],
                  ends=['2016-08-31','2016-09-30','2016-10-31'],symbols=[['A']]*3)
        source=dict(raw_local_linear=np.array([1.,2.,3.]),raw_source_log=np.zeros(3),available=np.ones(3,bool))
        out=model.build_forecasts(meta,source,np.ones((3,1),bool))
        self.assertEqual(record(out,1)['reason'],'missing_previous_formal_month')
        self.assertEqual(record(out,2)['prior_row'],1)
        meta2={k:[v[0],v[2]] for k,v in meta.items()}
        source2={k:v[[0,2]] for k,v in source.items()}
        out2=model.build_forecasts(meta2,source2,np.ones((2,1),bool))
        self.assertEqual(record(out2,1)['reason'],'missing_previous_formal_month')

    def test_exact_ids_no_alias_and_new_security_unsupported(self):
        args=fixture(); args[0]['symbols'][1][1]='A-B'
        out=model.build_forecasts(*args)
        self.assertEqual(record(out,3)['reason'],'missing_previous_identity')
        self.assertEqual(record(out,4)['reason'],'missing_previous_identity')
        self.assertIsNone(record(out,4)['prior_row'])
        self.assertEqual(out['raw']['confirmed'][4],8.)

    def test_reason_precedence_current_outside_before_prior_lookup(self):
        args=fixture(); args[2][0,0]=False; args[2][1,1]=False
        out=model.build_forecasts(*args)
        for j in (0,3):
            r=record(out,j)
            self.assertEqual(r['reason'],'current_outside_universe')
            self.assertIsNone(r['prior_row']); self.assertIsNone(r['current_difference'])
        self.assertEqual(record(out,4)['reason'],'previous_outside_universe')
        self.assertEqual(record(out,4)['prior_row'],3)

    def test_prior_unavailable_precedes_prior_outside_original_U(self):
        args=fixture(); args[1]['available'][3]=False
        for key in ('raw_local_linear','raw_source_log'): args[1][key][3]=np.nan
        args[2][1,1]=False
        out=model.build_forecasts(*args)
        r=record(out,4)
        self.assertEqual(r['reason'],'previous_unavailable')
        self.assertEqual(r['prior_row'],3)
        self.assertNotIn(3,[r['row'] for r in out['audit']['records']])
        self.assertTrue(np.isnan(out['raw']['confirmed'][3]))
        self.assertFalse(out['available'][3])

    def test_current_zero_preserves_negative_zero_without_prior_arithmetic(self):
        args=fixture(); args[1]['raw_local_linear'][0]=1e308; args[1]['raw_source_log'][0]=-1e308
        args[1]['raw_local_linear'][3]=-0.; args[1]['raw_source_log'][3]=0.
        out=model.build_forecasts(*args); r=record(out,3)
        self.assertEqual(r['status'],'current_zero'); self.assertIsNone(r['previous_difference'])
        for arm in out['raw']:
            self.assertEqual(out['raw'][arm][3].item().hex(),(-0.).hex())

    def test_previous_zero_does_not_confirm_either_direction(self):
        for zero in (0.,-0.):
            for value in (2.,-2.):
                args=fixture(); args[1]['raw_local_linear'][0]=zero; args[1]['raw_source_log'][0]=zero
                args[1]['raw_local_linear'][3]=value; args[1]['raw_source_log'][3]=0.
                out=model.build_forecasts(*args)
                self.assertEqual(record(out,3)['status'],'unconfirmed')
                self.assertEqual(out['raw']['confirmed'][3],0.)
                self.assertEqual(out['raw']['half'][3],0.5*value)

    def test_unconfirmed_copies_pooled_negative_zero_exactly(self):
        args=fixture(); args[1]['raw_local_linear'][3]=-1.; args[1]['raw_source_log'][3]=-0.
        out=model.build_forecasts(*args)
        self.assertEqual(record(out,3)['status'],'unconfirmed')
        self.assertEqual(out['raw']['confirmed'][3].item().hex(),(-0.).hex())

    def test_unsupported_differences_are_never_computed(self):
        args=fixture(); args[1]['raw_local_linear'][0]=1e308; args[1]['raw_source_log'][0]=-1e308
        args[2][0,0]=False
        out=model.build_forecasts(*args)
        self.assertEqual(out['raw']['half'][0],1e308)
        for j in (0,3):
            self.assertEqual(record(out,j)['status'],'unsupported')
            self.assertIsNone(record(out,j)['current_difference'])
            self.assertIsNone(record(out,j)['previous_difference'])

    def test_valid_input_difference_overflow_hardfails_current_and_previous(self):
        for j in (0,3):
            args=fixture(); args[1]['raw_local_linear'][j]=1e308; args[1]['raw_source_log'][j]=-1e308
            with self.assertRaises(FloatingPointError): model.build_forecasts(*args)

    def test_no_extra_source_fields_or_labels_consumed(self):
        args=fixture(); expected=model.build_forecasts(*args)
        args[1]['target_log_excess']=object(); args[1]['raw_candidate']=object()
        args[1]['local_available']=object(); args[0]['training_labels']=object()
        actual=model.build_forecasts(*args)
        self.assertEqual(actual['audit'],expected['audit'])
        for key in actual['raw']: np.testing.assert_array_equal(actual['raw'][key],expected['raw'][key])

    def test_all_physical_prefixes_and_future_poison(self):
        args=fixture(); out=model.build_forecasts(*args)
        for n in range(5):
            prefix=model.build_forecasts(*physical_prefix(args,n))
            for key in out['raw']: np.testing.assert_array_equal(prefix['raw'][key],out['raw'][key][:2*n])
            self.assertEqual(prefix['audit']['records'],[r for r in out['audit']['records'] if r['row']<2*n])
            poisoned=copy.deepcopy(args)
            for key in poisoned[1]:
                poisoned[1][key]=poisoned[1][key].astype(object); poisoned[1][key][2*n:]=object()
            poisoned=(poisoned[0],poisoned[1],poisoned[2].astype(object))
            poisoned[2][n:]=object()
            for key in ('dates','ends','symbols'): poisoned[0][key][n:]=[object()]*(4-n)
            clean=model.build_forecasts(*physical_prefix(poisoned,n))
            self.assertEqual(clean['audit'],prefix['audit'])
            for key in clean['raw']: np.testing.assert_array_equal(clean['raw'][key],prefix['raw'][key])

    def test_input_nonmutation_and_detached_output(self):
        args=fixture(); before=copy.deepcopy(args); out=model.build_forecasts(*args)
        self.assertEqual(args[0],before[0])
        for key in args[1]: np.testing.assert_array_equal(args[1][key],before[1][key])
        np.testing.assert_array_equal(args[2],before[2])
        out['raw']['confirmed'][0]=999.; out['available'][0]=False
        self.assertEqual(args[1]['raw_local_linear'][0],12.); self.assertTrue(args[1]['available'][0])

    def test_metadata_guards_dates_ends_ids_and_duplicate_formal_month(self):
        mutations=[lambda a:a[0]['dates'].__setitem__(1,a[0]['dates'][0]),
                   lambda a:a[0]['dates'].__setitem__(0,'2020-1-30'),
                   lambda a:a[0]['ends'].__setitem__(0,'2020-11-30'),
                   lambda a:a[0]['ends'].__setitem__(0,'2021-01-29'),
                   lambda a:a[0]['symbols'][0].__setitem__(1,'A'),
                   lambda a:a[0]['symbols'][0].__setitem__(0,' A'),
                   lambda a:a[0]['symbols'][0].append('Z'),
                   lambda a:a[0]['dates'].__setitem__(2,'2020-12-31')]
        for mutate in mutations:
            args=fixture(); mutate(args)
            with self.assertRaises(ValueError): model.build_forecasts(*args)
        args=fixture(); args[0]['dates'][1]='2020-11-29'
        with self.assertRaises(ValueError): model.build_forecasts(*args)
        args=fixture(); args[0]['dates'][0]='2016-08-17'
        with self.assertRaises(ValueError): model.build_forecasts(*args)

    def test_mask_shape_type_and_numeric_guards(self):
        mutations=[lambda a:a[1].__setitem__('available',np.ones(8,int)),
                   lambda a:a[1].__setitem__('available',np.ones((4,2),bool)),
                   lambda a:a[1].__setitem__('raw_local_linear',np.zeros((4,2))),
                   lambda a:a[1].__setitem__('raw_source_log',np.zeros(7)),
                   lambda a:a[1]['raw_local_linear'].__setitem__(0,np.nan),
                   lambda a:a[1]['raw_source_log'].__setitem__(0,np.inf),
                   lambda a:a[1].__setitem__('raw_source_log',np.full(8,'1')),
                   lambda a:a[1].__setitem__('raw_local_linear',np.ones(8,bool))]
        for mutate in mutations:
            args=fixture(); mutate(args)
            with self.assertRaises((ValueError,FloatingPointError)): model.build_forecasts(*args)
        for use in (np.ones((4,2),int),np.ones(8,bool),np.ones((4,0),bool)):
            args=fixture()
            with self.assertRaises(ValueError): model.build_forecasts(args[0],args[1],use)

    def test_unavailable_source_requires_nan_in_both_predictors(self):
        args=fixture(); args[1]['available'][0]=False
        with self.assertRaises(ValueError): model.build_forecasts(*args)
        args[1]['raw_local_linear'][0]=np.nan
        with self.assertRaises(ValueError): model.build_forecasts(*args)
        args[1]['raw_source_log'][0]=np.nan
        out=model.build_forecasts(*args)
        self.assertFalse(out['available'][0]); self.assertTrue(np.isnan(out['raw']['half'][0]))

    def test_calendar_gap_of_year_is_not_previous_month(self):
        args=fixture(); args[0]['dates'][2:] = ['2022-01-31','2022-02-28']
        args[0]['ends'][2:] = ['2022-02-28','2022-03-31']
        out=model.build_forecasts(*args)
        self.assertEqual(record(out,4)['reason'],'missing_previous_formal_month')
        self.assertFalse(record(out,4)['cross_year'])
        self.assertEqual(record(out,7)['prior_row'],4)

    def test_empty_prefix_zero_counts_and_unchanged_shapes(self):
        out=model.build_forecasts(*physical_prefix(fixture(),0))
        self.assertEqual(out['audit']['records'],[])
        self.assertEqual(out['audit']['counts'],{k:0 for k in model.COUNT_FIELDS})
        self.assertEqual(out['available'].shape,(0,))
        for v in out['raw'].values(): self.assertEqual(v.shape,(0,)); self.assertEqual(v.dtype,np.dtype('float64'))


if __name__=='__main__':
    unittest.main(verbosity=2)
