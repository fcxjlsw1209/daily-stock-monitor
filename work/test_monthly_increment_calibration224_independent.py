"""Independent invented-cohort, arithmetic and causal-prefix checks only."""
import copy
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import monthly_increment_calibration224_independent as oracle


def fixture(end='2019-02-28'):
    dates=pd.date_range('2016-08-31',end,freq='ME');names=['ONE','FUND','TWO'];width=3
    tm=dict(dates=[str(d.date()) for d in dates],
        ends=[str((d+pd.offsets.MonthEnd(1)).date()) for d in dates],
        symbols=[names.copy() for _ in dates])
    qm=dict(dates=['2016-08-17']+tm['dates'],ends=['2016-08-31']+tm['ends'],
        symbols=[names.copy()]+copy.deepcopy(tm['symbols']),etfs=['FUND'])
    size=len(dates)*width
    train=dict(offsets=np.arange(len(dates)+1,dtype=int)*width,
        dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),width),
        ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),width),
        train_plain=np.zeros((size,10)),common_good=np.ones(size,bool))
    for key in oracle.TARGETS:train['target_'+key]=np.zeros(size)
    # Exact binary increments with three distinct analytic own slopes.
    d=np.array([1.,2.,-.5]);k=np.array([.25,.75,-.5]);b=np.array([.125,-.25,.5])
    train['target_log_excess']=np.tile(b+k*d,len(dates))
    p=np.tile(b+d,len(qm['dates']));base=np.tile(b,len(qm['dates']))
    source=dict(raw_local_linear=p,raw_source_log=base,available=np.ones(len(p),bool),
        local_available=np.zeros(len(p),bool),raw_candidate=np.full(len(p),999.))
    return [train,tm,qm,source,np.ones((len(qm['dates']),width),bool)]


def prefix(args,n):
    train,tm,qm,source,use=copy.deepcopy(args);width=use.shape[1]
    for key in ('dates','ends','symbols'):qm[key]=qm[key][:n]
    for key in source:source[key]=source[key][:n*width]
    use=use[:n]
    cutoff='%04d-12-31'%(int(qm['dates'][-1][:4])-1) if n else None
    months=[i for i,e in enumerate(tm['ends']) if cutoff is not None and e<=cutoff]
    ids=np.array([j for i in months for j in range(int(train['offsets'][i]),int(train['offsets'][i+1]))],dtype=int)
    for key in ('dates','ends','symbols'):tm[key]=[tm[key][i] for i in months]
    train={key:value[ids] for key,value in train.items() if key!='offsets'}
    train['offsets']=np.r_[0,np.cumsum(np.array([len(ss) for ss in tm['symbols']],dtype=int))]
    return train,tm,qm,source,use


def latest(out):return out['audit']['annual'][-1]
def group(out,symbol='ONE'):return next(g for g in latest(out)['groups'] if g['symbol']==symbol)
def mature(args):return [i for i,e in enumerate(args[1]['ends']) if e<='2018-12-31']


def rank_fixture():
    s=np.array([[0.,.1,.2,.3],[.2,.1,0.,-.1],[0.,.2,.1,.3],[0.,.1,.2,.3]])
    shape=s.shape
    a=dict(eligible=np.ones(shape,bool),forecast_available=np.ones(shape,bool),
        model_available=np.ones(shape,bool),raw_source_log=s.copy(),
        raw_local_constant=s+.02,raw_local_linear=s[:,::-1].copy(),
        rank_log_excess=np.vstack([oracle.centered_rank(x) for x in s]),
        state_calm10=np.array([True,True,False,True]),
        state_weak_narrow=np.array([False,False,True,False]),extension_available=np.ones(shape,bool))
    f=dict(raw=dict(candidate=np.tile([.2,.1,.1,.3],4),control=np.tile([.3,.2,.1,0.],4)),
        available=np.ones(s.size,bool))
    return a,f


class ArithmeticTests(unittest.TestCase):
    def test_closed_form_and_strict_clip_boundaries(self):
        for n,a,c in [(-2.,0.,True),(0.,0.,False),(1.,.25,False),(4.,1.,False),(8.,1.,True)]:
            self.assertEqual(oracle.coefficient(n,4.),(a,c))
        for n,d in [(1.,0.),(1.,-1.),(True,1.),(math.inf,1.)]:
            with self.assertRaises((ValueError,FloatingPointError)):oracle.coefficient(n,d)

    def test_blend_exact_endpoints_equal_predictions_signed_zero(self):
        self.assertEqual(oracle.blend(5.,1.,.25),2.)
        self.assertEqual(oracle.blend(1e308,-1e308,0.),-1e308)
        self.assertEqual(oracle.blend(1e308,-1e308,1.),1e308)
        for a in (0.,.5,1.):
            self.assertEqual(oracle.blend(-0.,0.,a).hex(),(-0.).hex())
            self.assertEqual(oracle.blend(.7,.7,a).hex(),(.7).hex())
        for a in (True,-.1,1.1,math.inf):
            with self.assertRaises((ValueError,FloatingPointError)):oracle.blend(1.,0.,a)

    def test_overflow_underflow_and_signed_cancellation(self):
        tiny=float(np.nextafter(0.,1.))
        for a,b in ((1e308,2.),(1e-200,1e-200),(-1e-200,1e-200)):
            with self.assertRaises(FloatingPointError):oracle.product(a,b,'test')
        for a in (tiny,-tiny):
            with self.assertRaises(FloatingPointError):oracle.mean(a,2,'test')
        with self.assertRaises(FloatingPointError):oracle.coefficient(tiny,1e308)
        with self.assertRaises(FloatingPointError):oracle.blend(1e308,-1e308,.5)
        self.assertEqual(oracle.mean(oracle.fsum([1.,-1.],'cancel'),2,'mean'),0.)
        self.assertEqual(oracle.product(0.,1e308,'zero'),0.)


class CohortTests(unittest.TestCase):
    def test_analytic_own_slopes_pooled_equal_month_and_honest_fits(self):
        args=fixture();out=oracle.reconstruct(*args);row=latest(out)
        self.assertEqual(row['calibration_months'],28);self.assertEqual(row['calibration_rows'],84)
        self.assertEqual(row['numerator'],math.fsum([3.125/3.]*28))
        self.assertEqual(row['denominator'],math.fsum([5.25/3.]*28))
        for name,a,clipped in [('ONE',.25,False),('FUND',.75,False),('TWO',0.,True)]:
            g=group(out,name);self.assertEqual(g['alpha'],a);self.assertEqual(g['clipped'],clipped)
            self.assertEqual(g['own_months'],28);self.assertEqual(g['status'],'calibrated')
        self.assertEqual(row['parameter_estimates'],4)
        self.assertEqual(out['audit']['counts']['pooled_fits'],1)
        self.assertEqual(out['audit']['counts']['own_fits'],3)
        proof=oracle.check_forecasts(*args,out)
        self.assertTrue(all(proof[k] for k in oracle.MODEL_FLAGS));self.assertEqual(proof['new_model_fits'],4)
        self.assertEqual(proof['predictive_model_fits'],4);self.assertEqual(proof['new_ridge_fits'],0)

    def test_first_three_years_pool_cold_exact_original_copy(self):
        args=fixture();out=oracle.reconstruct(*args)
        for row,months in zip(out['audit']['annual'][:3],(0,4,16)):
            self.assertEqual(row['calibration_months'],months);self.assertEqual(row['status'],'cold_start')
            self.assertEqual(row['own_fits'],0)
            for g in row['groups']:
                self.assertEqual(g['status'],'pooled_not_ready');self.assertEqual(g['fallback'],'original')
                for key in ('candidate','control'):
                    ids=g['query_row_ids'];np.testing.assert_array_equal(out['raw'][key][ids],args[3]['raw_local_linear'][ids])

    def test_own_threshold_18_common_retained_months_and_pool_fallback(self):
        for n in (17,18):
            args=fixture()
            for i in mature(args)[n:]:args[0]['common_good'][3*i]=False
            out=oracle.reconstruct(*args);g=group(out)
            self.assertEqual(g['own_months'],n)
            self.assertEqual(g['status'],'calibrated' if n==18 else 'cold_start')
            if n==17:
                self.assertIsNone(g['alpha']);self.assertEqual(g['applied_alpha'],latest(out)['alpha'])
                ids=g['query_row_ids'];np.testing.assert_array_equal(out['raw']['candidate'][ids],out['raw']['control'][ids])

    def test_own_zero_denominator_uses_pool_no_own_fit(self):
        args=fixture()
        for i in mature(args):args[3]['raw_local_linear'][(i+1)*3]=args[3]['raw_source_log'][(i+1)*3]
        out=oracle.reconstruct(*args);g=group(out)
        self.assertEqual(g['own_months'],28);self.assertEqual(g['denominator'],0.)
        self.assertEqual(g['status'],'zero_denominator');self.assertEqual(g['fallback'],'pooled')
        self.assertEqual(latest(out)['zero_denominator_groups'],1);self.assertEqual(latest(out)['own_fits'],2)

    def test_pool_zero_denominator_blocks_own_fits(self):
        args=fixture()
        for i in mature(args):args[3]['raw_local_linear'][(i+1)*3:(i+2)*3]=args[3]['raw_source_log'][(i+1)*3:(i+2)*3]
        out=oracle.reconstruct(*args);row=latest(out)
        self.assertEqual(row['status'],'zero_denominator');self.assertIsNone(row['alpha'])
        self.assertEqual(row['parameter_estimates'],0);self.assertEqual(row['zero_denominator_groups'],0)
        self.assertTrue(all(g['status']=='pooled_not_ready' for g in row['groups']))
        self.assertEqual(out['raw']['candidate'].tobytes(),args[3]['raw_local_linear'].tobytes())

    def test_small_and_all_invalid_months_kept_no_terms_computed(self):
        args=fixture();args[4][1,1:]=False
        args[3]['raw_local_linear'][3]=1e308;args[3]['raw_source_log'][3]=-1e308
        args[0]['common_good'][3:6]=False
        out=oracle.reconstruct(*args);row=latest(out);small,empty=row['months'][:2]
        self.assertFalse(small['retained']);self.assertEqual(small['valid_rows'],1)
        self.assertEqual(empty['valid_rows'],0);self.assertEqual(empty['rows'],[])
        for key in ('numerator_sum','denominator_sum','numerator_mean','denominator_mean'):self.assertIsNone(small[key])
        for key in ('disagreement','residual_target','product','square'):self.assertIsNone(small['rows'][0][key])
        self.assertEqual(row['small_month_rows'],1)
        self.assertEqual(row['joined_label_rows'],row['invalid_label_rows']+row['small_month_rows']+row['calibration_rows'])

    def test_pool_not_restricted_to_current_ids_or_current_U(self):
        args=fixture();before=oracle.reconstruct(*args)
        for i,d in enumerate(args[2]['dates']):
            if d.startswith('2019'):
                args[2]['symbols'][i][1]='NEW';args[4][i]=False
        for i,d in enumerate(args[1]['dates']):
            if d.startswith('2019'):args[1]['symbols'][i][1]='NEW'
        out=oracle.reconstruct(*args);row=latest(out)
        self.assertEqual(row['alpha'],latest(before)['alpha'])
        self.assertTrue(any(r['symbol']=='FUND' for m in row['months'] for r in m['rows']))
        self.assertEqual(group(out,'NEW')['own_months'],0);self.assertEqual(group(out,'NEW')['fallback'],'pooled')
        self.assertTrue(out['available'].all())

    def test_equal_month_weight_differs_from_row_weight(self):
        args=fixture();args[3]['raw_source_log'][:]=0.;args[3]['raw_local_linear'][:]=1.
        args[0]['target_log_excess'][:]=np.tile([0.,0.,3.],len(args[1]['dates']))
        for i in mature(args)[14:]:args[4][i+1,2]=False
        row=latest(oracle.reconstruct(*args))
        self.assertEqual(row['numerator'],14.);self.assertEqual(row['denominator'],28.)
        self.assertEqual(row['alpha'],.5);self.assertNotEqual(row['alpha'],42./70.)

    def test_year_with_no_current_available_still_pool_estimates(self):
        args=fixture()
        for i,d in enumerate(args[2]['dates']):
            if d.startswith('2019'):
                for key in ('raw_local_linear','raw_source_log'):args[3][key][3*i:3*i+3]=np.nan
                args[3]['available'][3*i:3*i+3]=False
        out=oracle.reconstruct(*args);row=latest(out)
        self.assertEqual(row['prediction_rows'],0);self.assertEqual(row['groups'],[])
        self.assertEqual(row['status'],'calibrated');self.assertEqual(row['parameter_estimates'],1)

    def test_initial_partial_excluded_before_exact_join(self):
        args=fixture();args[3]['raw_local_linear'][:3]=1e308;args[3]['raw_source_log'][:3]=-1e308
        out=oracle.reconstruct(*args);row=latest(out)
        self.assertEqual(row['skipped_initial_rows'],3)
        self.assertEqual(row['months'][0]['date'],'2016-08-31')
        self.assertEqual(row['months'][0]['rows'][0]['source_row_id'],3)
        self.assertEqual(row['months'][0]['rows'][0]['training_row_id'],0)

    def test_common10_all_four_targets_and_original_U_required(self):
        for key in oracle.TARGETS:
            args=fixture();args[0]['target_'+key][0]=np.nan
            row=latest(oracle.reconstruct(*args));self.assertEqual(row['invalid_label_rows'],1)
        args=fixture();args[0]['train_plain'][0,9]=np.nan;args[4][2,0]=False
        row=latest(oracle.reconstruct(*args));self.assertEqual(row['invalid_label_rows'],1)
        self.assertEqual(row['joined_label_rows'],83);self.assertEqual(row['calibration_rows'],82)

    def test_mature_missing_identity_and_consumed_types_hard_fail(self):
        args=fixture();args[1]['symbols'][0][0]='WRONG'
        with self.assertRaises(ValueError):oracle.reconstruct(*args)
        args=fixture();args[0]['common_good']=args[0]['common_good'].astype(object);args[0]['common_good'][0]=1
        with self.assertRaises(ValueError):oracle.reconstruct(*args)
        args=fixture();args[0]['target_log_excess']=args[0]['target_log_excess'].astype(object);args[0]['target_log_excess'][0]='bad'
        with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_future_label_poison_is_not_consumed(self):
        args=fixture();before=oracle.reconstruct(*args)
        for key in ('train_plain',)+tuple('target_'+k for k in oracle.TARGETS):
            args[0][key]=args[0][key].astype(object)
            for i,end in enumerate(args[1]['ends']):
                if end>'2018-12-31':args[0][key][i*3:i*3+3]='future poison'
        after=oracle.reconstruct(*args)
        oracle.record_check(before['audit'],after['audit'])
        for key in before['raw']:oracle.close(before['raw'][key],after['raw'][key],key)

    def test_five_year_exclusive_signal_start(self):
        args=fixture('2023-02-28');row=latest(oracle.reconstruct(*args))
        self.assertEqual(row['lookback_exclusive_start'],'2017-12-31')
        self.assertEqual(row['months'][0]['date'],'2018-01-31')
        self.assertTrue(all(m['end']<='2022-12-31' for m in row['months']))

    def test_invalid_source_and_metadata_rejected(self):
        for mutate in (lambda a:a[3]['available'].__setitem__(0,False),
                       lambda a:a[3]['raw_source_log'].__setitem__(0,math.inf),
                       lambda a:a[2]['symbols'][0].__setitem__(0,'FUND'),
                       lambda a:a[2]['ends'].__setitem__(0,'2016-9-1')):
            args=fixture();mutate(args)
            with self.assertRaises(ValueError):oracle.reconstruct(*args)

    def test_input_immutable_and_irrelevant_source_fields_ignored(self):
        args=fixture();before=copy.deepcopy(args);out=oracle.reconstruct(*args)
        for k in args[0]:np.testing.assert_array_equal(args[0][k],before[0][k])
        for k in args[3]:np.testing.assert_array_equal(args[3][k],before[3][k])
        args[3]['local_available'][:]=True;args[3]['raw_candidate'][:]=math.nan
        other=oracle.reconstruct(*args);oracle.record_check(other['audit'],out['audit'])

    def test_check_catches_output_audit_fit_and_signedzero_tampering(self):
        args=fixture();out=oracle.reconstruct(*args)
        for mutate in (lambda b:b['audit']['counts'].__setitem__('parameter_estimates',0),
                       lambda b:b['raw']['candidate'].__setitem__(-1,999.),
                       lambda b:b['available'].__setitem__(-1,False)):
            bad=copy.deepcopy(out);mutate(bad)
            with self.assertRaises(ValueError):oracle.check_forecasts(*args,bad)
        args=fixture();args[3]['raw_local_linear'][0]=-0.;args[3]['raw_source_log'][0]=0.
        out=oracle.reconstruct(*args);out['raw']['candidate'][0]=0.
        with self.assertRaises(ValueError):oracle.check_forecasts(*args,out)

    def test_empty_physical_prefix_has_no_estimates(self):
        args=prefix(fixture(),0);out=oracle.reconstruct(*args)
        self.assertEqual(out['audit']['annual'],[]);self.assertTrue(all(v==0 for v in out['audit']['counts'].values()))
        self.assertEqual(out['raw']['candidate'].shape,(0,));oracle.check_forecasts(*args,out)

    def test_both_fixtures_all_physical_prefixes_full_audit_exact_cross(self):
        import monthly_increment_calibration224_model as producer
        from test_monthly_increment_calibration224_model import fixture as other_fixture,physical_prefix
        for args,cut in ((fixture(),prefix),(other_fixture(initial=True),physical_prefix)):
            full=oracle.reconstruct(*args)
            for n in range(len(args[2]['dates'])+1):
                small=cut(args,n);actual=producer.build_forecasts(*small)
                proof=oracle.check_forecasts(*small,actual)
                self.assertTrue(proof['passed']);self.assertFalse(proof['producer_modules_imported'])
                for key in ('candidate','control'):
                    oracle.close(actual['raw'][key],full['raw'][key][:n*args[4].shape[1]],'prefix '+key)


class RankTests(unittest.TestCase):
    def test_original_quarter_slot_and_initial_zero(self):
        a,f=rank_fixture();r,d,ready=oracle.rank_oracle(a,f)
        proof=oracle.verify_rank_slot(a,f,r,d)
        self.assertTrue(all(proof[k] for k in oracle.RANK_FLAGS))
        for arm in oracle.NEW_RAW:
            self.assertTrue(ready[arm].all());self.assertTrue((d[arm][0]==0.).all())
            old=oracle.centered_rank(a['raw_local_linear'][2])
            np.testing.assert_array_equal(d[arm][2],.0125*(r[arm][2]-old))

    def test_paired_whole_row_readiness_no_support_filter(self):
        a,f=rank_fixture();f['available'][5]=False
        for key in f['raw']:f['raw'][key][5]=np.nan
        r,d,ready=oracle.rank_oracle(a,f)
        for arm in oracle.NEW_RAW:
            self.assertFalse(ready[arm][1]);self.assertTrue((d[arm][1]==0.).all())
        a,f=rank_fixture();a['eligible'][2,1]=False
        r,d,_=oracle.rank_oracle(a,f)
        for arm in oracle.NEW_RAW:self.assertEqual(d[arm][2,1],0.)

    def test_exact_source_copy_uses_original_ranks_and_mutation_detected(self):
        a,f=rank_fixture();f['raw']['candidate']=a['raw_source_log'].ravel().copy()
        r,d,_=oracle.rank_oracle(a,f);arm='own_increment_calibration'
        np.testing.assert_array_equal(r[arm],a['rank_log_excess'])
        d[arm][1,0]+=.001
        with self.assertRaises(ValueError):oracle.verify_rank_slot(a,f,r,d)


if __name__=='__main__':unittest.main()
