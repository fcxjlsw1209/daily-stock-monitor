"""Independent invented-input checks; no actual217 sources, labels or returns."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_persistence217_independent as oracle


def fixture():
    dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31',
           '2016-12-30','2017-01-31','2017-02-28']
    ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30',
          '2017-01-31','2017-02-28','2017-03-31']
    names=['A','ETF','B']
    meta=dict(dates=dates,ends=ends,symbols=[names.copy() for _ in dates],etfs=['ETF'])
    p=np.array([[1.,-1.,1.],[1.,-2.,0.],[2.,-1.,3.],[-1.,-2.,0.],
                [3.,2.,1.],[2.,-3.,-1.],[0.,-2.,2.]]).ravel()
    source=dict(raw_local_linear=p,raw_source_log=np.zeros(len(p)),available=np.ones(len(p),bool))
    return [meta,source,np.ones((len(dates),3),bool)]


def prefix(args,n):
    meta,source,use=copy.deepcopy(args);width=use.shape[1]
    for key in ('dates','ends','symbols'):meta[key]=meta[key][:n]
    for key in source:source[key]=source[key][:n*width]
    return [meta,source,use[:n]]


def record(built,row):
    return next(r for r in built['audit']['records'] if r['row']==row)


def rank_fixture():
    source=np.array([[0.,.1,.2,.3],[.2,.1,0.,-.1],[0.,.2,.1,.3],[0.,.1,.2,.3]])
    shape=source.shape
    arrays=dict(baseline=np.zeros(shape),eligible=np.ones(shape,bool),
        forecast_available=np.ones(shape,bool),model_available=np.ones(shape,bool),
        local_available=np.zeros(shape,bool),raw_source_log=source.copy(),
        raw_local_constant=source+.02,raw_local_linear=source[:,::-1].copy(),
        rank_log_excess=np.vstack([oracle.centered_rank(x) for x in source]),
        state_calm10=np.array([True,True,False,True]),
        state_weak_narrow=np.array([False,False,True,False]),extension_available=np.ones(shape,bool))
    forecast=dict(raw=dict(confirmed=np.array([[.1,.1,.2,.2],[.3,.1,.1,.2],
        [.1,.2,.1,.2],[.2,.3,.1,0.]]).ravel(),half=np.array([[0.,.3,.2,.1],
        [.2,.3,.1,0.],[.3,.2,.1,0.],[0.,.2,.1,.3]]).ravel()),available=np.ones(source.size,bool))
    return arrays,forecast


class Tests(unittest.TestCase):
    def test_analytic_branches_counts_and_same_negative_sign(self):
        args=fixture();b=oracle.reconstruct(*args)
        expected=np.array([[1.,-1.,1.],[1.,-2.,0.],[2.,-1.,0.],[0.,-2.,0.],
                           [3.,2.,1.],[2.,0.,0.],[0.,-2.,0.]]).ravel()
        half=np.array([[1.,-1.,1.],[1.,-2.,0.],[1.,-.5,1.5],[-.5,-1.,0.],
                       [3.,2.,1.],[1.,-1.5,-.5],[0.,-1.,1.]]).ravel()
        np.testing.assert_array_equal(b['raw']['confirmed'],expected)
        np.testing.assert_array_equal(b['raw']['half'],half)
        self.assertEqual(b['audit']['counts'],dict(query_rows=21,cached_forecast_rows=21,
            supported_rows=12,unsupported_rows=9,current_zero_rows=2,confirmed_rows=5,
            unconfirmed_rows=5,cross_year_supported_rows=3))
        proof=oracle.check_forecasts(*args,b)
        self.assertTrue(all(proof[k] is True for k in oracle.MODEL_FLAGS))
        for key in oracle.ZERO_FIELDS:self.assertEqual(proof[key],0)
        self.assertEqual(record(b,7)['status'],'confirmed')

    def test_initial_partial_never_supplies_formal_history(self):
        args=fixture();b=oracle.reconstruct(*args)
        for row in range(6):
            self.assertEqual(record(b,row)['reason'],'missing_previous_formal_month')
            self.assertIsNone(record(b,row)['prior_row'])
        self.assertEqual(record(b,6)['prior_row'],3)
        # If August formal is absent, September cannot use August partial.
        for key in ('dates','ends','symbols'):del args[0][key][1]
        for key in args[1]:args[1][key]=np.delete(args[1][key],range(3,6))
        args[2]=np.delete(args[2],1,axis=0)
        self.assertEqual(record(oracle.reconstruct(*args),3)['reason'],'missing_previous_formal_month')

    def test_missing_month_not_nearest_occurrence(self):
        b=oracle.reconstruct(*fixture())
        for row in (12,13,14):
            r=record(b,row);self.assertEqual(r['status'],'unsupported')
            self.assertEqual(r['reason'],'missing_previous_formal_month');self.assertIsNone(r['prior_row'])
        self.assertEqual(record(b,15)['prior_row'],12)

    def test_exact_identity_not_column_position_or_alias(self):
        args=fixture();args[0]['symbols'][1]=['ETF','A','NEW']
        b=oracle.reconstruct(*args)
        self.assertEqual(record(b,6)['prior_row'],4)
        self.assertEqual(record(b,7)['prior_row'],3)
        self.assertEqual(record(b,8)['reason'],'missing_previous_identity')
        self.assertIsNone(record(b,8)['prior_row'])
        args[0]['symbols'][1][1]='a'
        self.assertEqual(record(oracle.reconstruct(*args),6)['reason'],'missing_previous_identity')

    def test_current_mask_precedence_and_no_unused_arithmetic(self):
        args=fixture();args[2][2,0]=False
        args[1]['raw_local_linear'][6]=1e308;args[1]['raw_source_log'][6]=-1e308
        # Next month cannot use the masked row; no overflow is evaluated anywhere.
        b=oracle.reconstruct(*args);r=record(b,6)
        self.assertEqual(r['reason'],'current_outside_universe');self.assertIsNone(r['prior_row'])
        self.assertIsNone(r['current_difference']);self.assertIsNone(r['previous_difference'])
        self.assertEqual(record(b,9)['reason'],'previous_outside_universe')
        self.assertEqual(b['raw']['confirmed'][6],1e308);self.assertTrue(b['available'][6])
        args=fixture();args[2][0,0]=False
        self.assertEqual(record(oracle.reconstruct(*args),0)['reason'],'current_outside_universe')

    def test_previous_unavailable_before_previous_universe(self):
        args=fixture();args[1]['available'][3]=False;args[2][1,0]=False
        for key in ('raw_local_linear','raw_source_log'):args[1][key][3]=np.nan
        b=oracle.reconstruct(*args);r=record(b,6)
        self.assertEqual(r['reason'],'previous_unavailable');self.assertEqual(r['prior_row'],3)
        self.assertIsNone(r['current_difference']);self.assertFalse(r['cross_year'])
        self.assertEqual(len(b['audit']['records']),20)
        for key in ('confirmed','half'):self.assertTrue(np.isnan(b['raw'][key][3]))

    def test_previous_outside_universe_preserves_current_domain(self):
        args=fixture();args[2][1,1]=False;b=oracle.reconstruct(*args)
        self.assertEqual(record(b,7)['reason'],'previous_outside_universe')
        self.assertEqual(record(b,7)['prior_row'],4)
        self.assertTrue(np.array_equal(b['available'],args[1]['available']))
        self.assertEqual(b['raw']['confirmed'][7],args[1]['raw_local_linear'][7])

    def test_current_zero_skips_previous_difference_and_preserves_signed_zero(self):
        args=prefix(fixture(),3);args[1]['raw_local_linear'][3]=1e308;args[1]['raw_source_log'][3]=-1e308
        args[1]['raw_local_linear'][6]=-0.;args[1]['raw_source_log'][6]=0.
        b=oracle.reconstruct(*args);r=record(b,6)
        self.assertEqual(r['status'],'current_zero');self.assertIsNone(r['previous_difference'])
        for key in ('confirmed','half'):
            self.assertEqual(np.float64(b['raw'][key][6]).tobytes(),np.float64(-0.).tobytes())
        bad=copy.deepcopy(b);bad['raw']['confirmed'][6]=0.
        with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)

    def test_previous_zero_unconfirmed_and_copy_pooled_signed_zero(self):
        args=fixture();args[1]['raw_source_log'][8]=-0.;b=oracle.reconstruct(*args)
        r=record(b,8);self.assertEqual(r['status'],'unconfirmed');self.assertEqual(r['previous_difference'],0.)
        self.assertEqual(np.float64(b['raw']['confirmed'][8]).tobytes(),np.float64(-0.).tobytes())
        self.assertEqual(b['raw']['half'][8],1.5)

    def test_cross_year_uses_stored_previous_annual_prediction(self):
        args=fixture();b=oracle.reconstruct(*args)
        for row in (15,16,17):self.assertTrue(record(b,row)['cross_year'])
        args[2][4,0]=False;b=oracle.reconstruct(*args)
        self.assertFalse(record(b,15)['cross_year']);self.assertEqual(b['audit']['counts']['cross_year_supported_rows'],2)
        args=fixture();args[1]['raw_local_linear'][12]=-3.;b=oracle.reconstruct(*args)
        self.assertEqual(record(b,15)['status'],'unconfirmed')
        self.assertEqual(record(b,15)['previous_difference'],-3.)

    def test_half_is_pooled_plus_half_difference_not_half_prediction(self):
        args=prefix(fixture(),3)
        args[1]['raw_source_log'][3]=10.;args[1]['raw_local_linear'][3]=12.
        args[1]['raw_source_log'][6]=20.;args[1]['raw_local_linear'][6]=24.
        b=oracle.reconstruct(*args)
        self.assertEqual(b['raw']['confirmed'][6],24.);self.assertEqual(b['raw']['half'][6],22.)
        self.assertNotEqual(b['raw']['half'][6],12.)

    def test_supported_current_and_previous_overflow_hardfail(self):
        for which in ('current','previous'):
            args=prefix(fixture(),3);row=6 if which=='current' else 3
            args[1]['raw_local_linear'][row]=1e308;args[1]['raw_source_log'][row]=-1e308
            with self.assertRaises(FloatingPointError):oracle.reconstruct(*args)
        # Same signs are compared rather than multiplied (product would overflow).
        args=prefix(fixture(),3);args[1]['raw_local_linear'][3]=1e200;args[1]['raw_local_linear'][6]=1e200
        self.assertEqual(record(oracle.reconstruct(*args),6)['status'],'confirmed')

    def test_full_physical_prefixes_and_future_poison_removed(self):
        args=fixture();whole=oracle.reconstruct(*args)
        for n in range(len(args[0]['dates'])+1):
            tiny=prefix(args,n);b=oracle.reconstruct(*tiny);oracle.check_forecasts(*tiny,b)
            for key in ('confirmed','half'):np.testing.assert_array_equal(b['raw'][key],whole['raw'][key][:n*3])
            poison=copy.deepcopy(args)
            for key in ('raw_local_linear','raw_source_log'):
                poison[1][key]=poison[1][key].astype(object);poison[1][key][n*3:]='FUTURE'
            for key in ('dates','ends','symbols'):poison[0][key][n:]=['FUTURE']*(len(args[0][key])-n)
            small=prefix(poison,n);got=oracle.reconstruct(*small)
            oracle.record_check(got['audit'],b['audit'])
        empty=oracle.reconstruct(*prefix(args,0));self.assertEqual(empty['audit']['records'],[])
        self.assertTrue(all(v==0 for v in empty['audit']['counts'].values()))

    def test_all123_physical_month_prefixes(self):
        dates=pd.date_range('2016-08-31',periods=121,freq='ME')
        ds=['2016-08-17']+[str(x.date()) for x in dates]
        es=['2016-08-31']+[str((x+pd.offsets.MonthEnd(1)).date()) for x in dates]
        qm=dict(dates=ds,ends=es,symbols=[['A','ETF'] for _ in ds])
        p=np.array([float(i%7-3) for i in range(244)])
        args=[qm,dict(raw_local_linear=p,raw_source_log=np.full(244,.25),available=np.ones(244,bool)),np.ones((122,2),bool)]
        whole=oracle.reconstruct(*args)
        for n in range(123):
            small=prefix(args,n);b=oracle.reconstruct(*small)
            for key in ('confirmed','half'):np.testing.assert_array_equal(b['raw'][key],whole['raw'][key][:2*n])
            self.assertEqual(b['audit']['counts']['query_rows'],2*n)

    def test_no_labels_or_unused_source_fields_consumed(self):
        args=fixture();expected=oracle.reconstruct(*args)
        args[0]['labels']='POISON'
        for key in ('local_available','raw_local_constant','raw_candidate','raw_control'):
            args[1][key]=np.full(21,'DO NOT CONSUME',object)
        got=oracle.reconstruct(*args);oracle.record_check(got['audit'],expected['audit'])

    def test_malformed_dates_identity_and_months_fail(self):
        for change in ('duplicate_month','duplicate_id','bad_end','bad_partial','order','iso','width'):
            args=fixture();m=args[0]
            if change=='duplicate_month':m['dates'][3]='2016-09-29'
            elif change=='duplicate_id':m['symbols'][2][1]='A'
            elif change=='bad_end':m['ends'][2]='2016-09-30'
            elif change=='bad_partial':m['ends'][0]='2016-09-30'
            elif change=='order':m['dates'][2]='2016-07-31'
            elif change=='iso':m['dates'][2]='2016-9-30'
            else:m['symbols'][2].pop()
            with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_boolean_masks_numeric_sources_and_shapes_strict(self):
        for change in ('U','available','raw_bool','raw_text','nan','inf','width','missing'):
            args=fixture()
            if change=='U':args[2]=args[2].astype(int)
            elif change=='available':args[1]['available']=args[1]['available'].astype(int)
            elif change in ('raw_bool','raw_text'):
                args[1]['raw_source_log']=args[1]['raw_source_log'].astype(object)
                args[1]['raw_source_log'][0]=True if change=='raw_bool' else '0.2'
            elif change=='nan':args[1]['raw_local_linear'][0]=np.nan
            elif change=='inf':args[1]['raw_source_log'][0]=np.inf
            elif change=='width':args[2]=np.zeros((7,0),bool)
            else:del args[1]['raw_source_log']
            with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_input_immutability(self):
        args=fixture();before=copy.deepcopy(args);b=oracle.reconstruct(*args);oracle.check_forecasts(*args,b)
        b['raw']['confirmed'][:]=999.;self.assertEqual(args[0],before[0])
        for key in args[1]:np.testing.assert_array_equal(args[1][key],before[1][key])
        np.testing.assert_array_equal(args[2],before[2])

    def test_every_count_and_audit_record_corruption_rejected(self):
        args=fixture();b=oracle.reconstruct(*args)
        for key in oracle.MODEL_COUNT_FIELDS:
            bad=copy.deepcopy(b);bad['audit']['counts'][key]+=1
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)
        for key,value in [('prior_row',0),('status','confirmed'),('reason','confirmed'),
                          ('current_difference',1.),('previous_difference',1.),('cross_year',True)]:
            bad=copy.deepcopy(b);bad['audit']['records'][0][key]=value
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)
        for key in oracle.ZERO_FIELDS:
            bad=copy.deepcopy(b);bad['audit'][key]=1
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)
        for key in ('confirmed','half'):
            bad=copy.deepcopy(b);bad['raw'][key][6]=np.nextafter(bad['raw'][key][6],np.inf)
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)

    def test_rank_slot_full_domain_readiness_and_initial_zero(self):
        a,b=rank_fixture();rr,dd,ready=oracle.rank_oracle(a,b)
        proof=oracle.verify_rank_slot(a,b,rr,dd);self.assertTrue(all(proof[k] for k in oracle.RANK_FLAGS))
        for arm in oracle.NEW_RAW:
            np.testing.assert_array_equal(dd[arm][0],0.);self.assertLessEqual(abs(dd[arm]).max(),.0125)
        b['raw']['confirmed'][4]=np.nan;b['available'][4]=False
        rr,dd,ready=oracle.rank_oracle(a,b)
        for arm in oracle.NEW_RAW:
            self.assertFalse(ready[arm][1]);np.testing.assert_array_equal(dd[arm][1],0.)
        bad=copy.deepcopy(dd);bad[oracle.ARMS[1]][0,0]=1e-16
        with self.assertRaises(ValueError):oracle.verify_rank_slot(a,b,rr,bad)

    def test_accounting_pinned_and_producer_independent(self):
        self.assertTrue(callable(oracle.accounting().metrics))
        with patch.object(oracle,'ACCOUNTING_SHA','0'*64):
            with self.assertRaises(ValueError):oracle.accounting()

    def test_blackbox_both_fixtures_all_prefixes_and_host_interfaces(self):
        # The independent implementation is already complete; import only to cross-check outputs.
        import monthly_persistence217_model as producer
        import test_monthly_persistence217_model as producer_test
        import monthly_persistence217_host as host
        for args in (fixture(),producer_test.fixture()):
            whole=oracle.reconstruct(*args);nrows,width=args[2].shape
            for n in range(nrows+1):
                for small in (prefix(args,n),host.prefix_inputs(*args,n)):
                    built=producer.build_forecasts(*small)
                    proof=oracle.check_forecasts(*small,built)
                    host.require_model_audit(proof,built,expected_rows=n*width)
                    for key in ('confirmed','half'):
                        np.testing.assert_array_equal(built['raw'][key],whole['raw'][key][:n*width])
        import test_monthly_persistence217_host as host_test
        arrays,meta,forecast=host_test.fixture()
        rr,dd=host.rank_and_delta(host_test.SyntheticStudy(),arrays,forecast,len(meta['dates']))
        oracle.verify_rank_slot(arrays,forecast,rr,dd)


if __name__=='__main__':unittest.main(verbosity=2)
