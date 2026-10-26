"""Round211 synthetic-only joins, convex curves and causal cache prefixes."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import monthly_forecast_blend211_model as m


def fixture(end='2019-04-30'):
    days=pd.date_range('2016-08-31',end,freq='ME')
    names=['A','B','FUND'];width=len(names);n=len(days)
    tm=dict(dates=[str(d.date()) for d in days],ends=[str((d+pd.offsets.MonthEnd(1)).date()) for d in days],
            symbols=[names.copy() for _ in days])
    qm=copy.deepcopy(tm);qm['dates'][0]='2016-08-17';qm['ends'][0]='2016-08-31';qm['etfs']=['FUND']
    own=np.tile(np.array([-.125,.25,.125]),n);peer=np.zeros(n*width)
    y=own*.25
    train=dict(offsets=np.arange(n+1,dtype=int)*width,dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),width),
        ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),width),train_plain=np.zeros((n*width,10)),
        common_good=np.ones(n*width,bool))
    for key in m.TARGETS:train['target_'+key]=y.copy()
    source=dict(raw_candidate=peer,raw_control=own,raw_source_log=np.zeros(n*width),
                available=np.ones(n*width,bool),local_available=np.zeros(n*width,bool),
                raw_local_linear=own.copy(),raw_local_constant=own.copy())
    return train,tm,qm,source,np.ones((n,width),bool)


def prefix(inputs,n,physical=True):
    train,tm,qm,source,eligible=copy.deepcopy(inputs);width=eligible.shape[1]
    for key in ('dates','ends','symbols'):qm[key]=qm[key][:n]
    source={k:v[:n*width] for k,v in source.items()};eligible=eligible[:n]
    if physical:
        cutoff=f"{int(qm['dates'][-1][:4])-1:04d}-12-31" if n else None
        ids=[i for i,e in enumerate(tm['ends']) if cutoff is not None and e<=cutoff]
        rows=[j for i in ids for j in range(int(train['offsets'][i]),int(train['offsets'][i+1]))]
        for key in ('dates','ends','symbols'):tm[key]=[tm[key][i] for i in ids]
        for key in list(train):
            if key!='offsets':train[key]=train[key][rows]
        train['offsets']=np.r_[0,np.cumsum([len(x) for x in tm['symbols']],dtype=int)]
    return train,tm,qm,source,eligible


class ConvexTests(unittest.TestCase):
    def test_known_quadratic_solution_and_clip(self):
        for N,D,w in ((.25,1.,.25),(-1.,2.,0.),(3.,2.,1.)):
            alpha,ratio,clipped,status=m.coefficient(N,D,18)
            self.assertEqual(alpha,w);self.assertEqual(status,'learned');self.assertEqual(ratio,N/D)
            self.assertEqual(clipped,ratio!=alpha)
            def curve(a):return D*a*a-2*N*a
            for other in np.linspace(0,1,101):self.assertLessEqual(curve(alpha),curve(other)+1e-14)

    def test_support_zero_denominator_and_tiny_positive_no_floor(self):
        self.assertEqual(m.coefficient(1.,2.,17),(.5,None,False,'cold_start'))
        self.assertEqual(m.coefficient(0.,0.,18),(.5,None,False,'zero_denominator'))
        self.assertEqual(m.coefficient(2.5e-201,1e-200,18)[0],.25)
        with self.assertRaises(FloatingPointError):m.coefficient(1e308,1e-308,18)

    def test_blend_endpoint_identity_signed_zero_and_half_order(self):
        self.assertEqual(np.float64(m.blend(-0.,1.,0.)).tobytes(),np.float64(-0.).tobytes())
        self.assertEqual(np.float64(m.blend(1.,-0.,1.)).tobytes(),np.float64(-0.).tobytes())
        self.assertEqual(m.blend(1e308,1e308,.5),1e308)
        self.assertEqual(m.blend(.2,.3,.5),.2+.5*(.3-.2))
        self.assertEqual(m.blend(-1e308,1e308,1.),1e308)
        with self.assertRaises(FloatingPointError):m.blend(-1e308,1e308,.5)

    def test_bad_numeric_and_convex_inputs(self):
        for alpha in (-.1,1.1,True,'0.5'):
            with self.assertRaises(ValueError):m.blend(0.,1.,alpha)
        for x in (np.inf,np.nan):
            with self.assertRaises(FloatingPointError):m.coefficient(x,1.,18)
        with self.assertRaises(ValueError):m.coefficient(0.,-1.,18)
        with self.assertRaises(ValueError):m.coefficient(0.,1.,True)


class BuildTests(unittest.TestCase):
    def test_complete_build_schema_clock_and_actual_fit_counts(self):
        inputs=fixture();z=m.build_forecasts(*inputs)
        self.assertEqual(set(z),{'raw','available','audit'});self.assertEqual(set(z['raw']),{'candidate','control'})
        rows=z['audit']['annual'];self.assertEqual([r['year'] for r in rows],[2016,2017,2018,2019])
        self.assertEqual([r['status'] for r in rows],['cold_start']*3+['learned'])
        self.assertEqual(rows[-1]['alpha'],.25)
        self.assertEqual(rows[-1]['cutoff'],'2018-12-31');self.assertEqual(rows[-1]['lookback_exclusive_start'],'2013-12-31')
        self.assertEqual(rows[-1]['latest_label_end'],'2018-12-31')
        self.assertEqual(rows[-1]['calibration_months'],27)
        counts=z['audit']['counts'];self.assertEqual(set(counts),set(m.COUNT_FIELDS))
        self.assertEqual(counts['annual_coefficient_fits'],1);self.assertEqual(counts['cold_start_years'],3)
        self.assertEqual(z['audit']['new_model_fits'],1);self.assertEqual(z['audit']['new_ridge_fits'],0)
        self.assertEqual(counts['cached_forecast_rows'],len(z['available']))
        for r in rows:
            for month in r['months']:
                self.assertLessEqual(month['end'],r['cutoff']);self.assertGreater(month['date'],r['lookback_exclusive_start'])
                self.assertNotEqual(month['date'],m.INITIAL)
        for row in rows[:3]:
            use=[i for i,d in enumerate(inputs[2]['dates']) if d[:4]==str(row['year'])]
            for i in use:np.testing.assert_array_equal(z['raw']['candidate'][3*i:3*i+3],z['raw']['control'][3*i:3*i+3])

    def test_all_physical_prefix_forecasts_and_no_mutation(self):
        a=fixture();before=copy.deepcopy(a);full=m.build_forecasts(*a)
        for n in range(len(a[2]['dates'])+1):
            got=m.build_forecasts(*prefix(a,n))
            for key in ('candidate','control'):np.testing.assert_array_equal(got['raw'][key],full['raw'][key][:n*3])
            np.testing.assert_array_equal(got['available'],full['available'][:n*3])
        for j in (0,3):
            for k in a[j]:np.testing.assert_array_equal(a[j][k],before[j][k])
        self.assertEqual(a[1:3],before[1:3]);np.testing.assert_array_equal(a[4],before[4])

    def test_future_train_numeric_poison_and_prefix_zero(self):
        a=prefix(fixture(),8,physical=False);expected=m.build_forecasts(*a)
        future=a[0]['ends']>np.datetime64('2016-12-31')
        for key in ('train_plain','common_good')+tuple('target_'+k for k in m.TARGETS):
            a[0][key]=a[0][key].astype(object);a[0][key][future]='future poison'
        got=m.build_forecasts(*a)
        self.assertEqual(got['audit'],expected['audit'])
        np.testing.assert_array_equal(got['raw']['candidate'],expected['raw']['candidate'])
        zero=m.build_forecasts(*prefix(a,0,physical=False))
        self.assertEqual(zero['audit']['annual'],[]);self.assertTrue(all(v==0 for v in zero['audit']['counts'].values()))

    def test_current_labels_do_not_change_current_year_weight(self):
        a=fixture();expected=m.build_forecasts(*a)
        future=a[0]['ends']>np.datetime64('2018-12-31')
        for key in m.TARGETS:a[0]['target_'+key][future]=999.
        a[0]['common_good'][future]=False
        got=m.build_forecasts(*a)
        self.assertEqual(got['audit'],expected['audit'])
        np.testing.assert_array_equal(got['raw']['candidate'],expected['raw']['candidate'])

    def test_exact_join_symbol_end_and_offsets_fail(self):
        for mutation in ('symbol','end','offset','date','duplicate'):
            a=list(fixture())
            if mutation=='symbol':a[1]['symbols'][1][0]='OTHER'
            elif mutation=='end':a[2]['ends'][1]='2016-10-30'
            elif mutation=='offset':a[0]['offsets'][2]+=1
            elif mutation=='date':a[0]['dates'][3]=np.datetime64('2016-09-29')
            else:a[1]['symbols'][1][0]='B'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.build_forecasts(*a)

    def test_initial_partial_skips_wrong_month_label(self):
        a=fixture();a[0]['target_log_excess'][:3]=1e308
        z=m.build_forecasts(*a)
        self.assertEqual(z['audit']['annual'][-1]['skipped_initial_rows'],3)
        self.assertEqual(z['audit']['annual'][-1]['alpha'],.25)

    def test_actual_partial_then_same_month_end_query_is_allowed(self):
        a=list(fixture())
        for key,value in (('dates','2016-08-31'),('ends','2016-09-30'),('symbols',['A','B','FUND'])):
            a[2][key].insert(1,value)
        for key,values in a[3].items():a[3][key]=np.concatenate((values[:3],values[:3],values[3:]))
        a[4]=np.concatenate((a[4][:1],a[4][:1],a[4][1:]))
        result=m.build_forecasts(*a);r=result['audit']['annual'][-1]
        self.assertEqual(r['skipped_initial_rows'],3);self.assertEqual(r['calibration_months'],28)
        self.assertEqual(r['months'][0]['date'],'2016-08-31');self.assertEqual(r['months'][0]['source_row_ids'],[3,4,5])
        self.assertEqual(r['months'][0]['training_row_ids'],[0,1,2]);self.assertEqual(r['alpha'],.25)
        for n in (0,1,2,3,len(a[2]['dates'])):
            small=m.build_forecasts(*prefix(a,n))
            np.testing.assert_array_equal(small['raw']['candidate'],result['raw']['candidate'][:3*n])
        bad=copy.deepcopy(a);bad[2]['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):m.build_forecasts(*bad)
        bad=copy.deepcopy(a);bad[1]['dates'][0]=m.INITIAL;bad[1]['ends'][0]='2016-08-31'
        with self.assertRaises(ValueError):m.build_forecasts(*bad)

    def test_five_year_exclusive_start_and_month_maturity(self):
        a=fixture('2024-04-30');row=m.build_forecasts(*a)['audit']['annual'][-1]
        self.assertEqual(row['lookback_exclusive_start'],'2018-12-31')
        self.assertEqual(row['months'][0]['date'],'2019-01-31')
        self.assertEqual(row['months'][-1]['date'],'2023-11-30')
        self.assertEqual(row['latest_label_end'],'2023-12-31')
        self.assertEqual(row['calibration_months'],59)

    def test_seventeen_and_eighteen_common_month_boundary(self):
        a=fixture();a[4][1:10]=False
        row=m.build_forecasts(*a)['audit']['annual'][-1]
        self.assertEqual((row['calibration_months'],row['status'],row['alpha']),(18,'learned',.25))
        a[4][10]=False
        row=m.build_forecasts(*a)['audit']['annual'][-1]
        self.assertEqual((row['calibration_months'],row['status'],row['alpha']),(17,'cold_start',.5))

    def test_no_local_filter_or_current_query_eligibility_filter(self):
        a=fixture();base=m.build_forecasts(*a)
        a[3]['local_available']=np.full(len(a[3]['available']),'ignored',dtype=object)
        use=[i for i,d in enumerate(a[2]['dates']) if d.startswith('2019')];a[4][use]=False
        z=m.build_forecasts(*a)
        self.assertEqual(z['audit'],base['audit'])
        np.testing.assert_array_equal(z['raw']['candidate'],base['raw']['candidate'])

    def test_common_invalid_counts_skip_numeric_and_small_month(self):
        a=fixture();a[0]['common_good'][3:5]=False
        a[0]['train_plain']=a[0]['train_plain'].astype(object);a[0]['train_plain'][3:5]='invalid base ignored'
        z=m.build_forecasts(*a);r=z['audit']['annual'][-1];month=r['months'][0]
        self.assertEqual(month['invalid_label_rows'],2);self.assertEqual(month['usable_rows'],1)
        self.assertFalse(month['accepted']);self.assertEqual(month['numerator_terms'],[])
        self.assertEqual(r['small_month_rows'],1);self.assertEqual(r['calibration_months'],26)

    def test_finite_all_four_targets_and_feature_support(self):
        for key in ('target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility','train_plain'):
            a=fixture();a[0][key][3]=np.nan
            r=m.build_forecasts(*a)['audit']['annual'][-1]
            self.assertEqual(r['invalid_label_rows'],1);self.assertEqual(r['months'][0]['usable_rows'],2)
        for bad in (True,'0'):
            a=fixture();a[0]['target_log_excess']=a[0]['target_log_excess'].astype(object);a[0]['target_log_excess'][3]=bad
            with self.assertRaises(ValueError):m.build_forecasts(*a)

    def test_month_equal_not_row_equal(self):
        a=fixture();a[4][1,2]=False
        # First admitted month has two rows and zero target; all remaining months retain three.
        a[0]['target_log_excess'][3:6]=0.
        r=m.build_forecasts(*a)['audit']['annual'][-1]
        N=math.fsum(x['numerator_mean'] for x in r['months'] if x['accepted'])
        D=math.fsum(x['denominator_mean'] for x in r['months'] if x['accepted'])
        self.assertEqual(r['alpha'],N/D)
        row_weighted=math.fsum(v for x in r['months'] for v in x['numerator_terms'])/math.fsum(v for x in r['months'] for v in x['denominator_terms'])
        self.assertNotEqual(r['alpha'],row_weighted)

    def test_zero_difference_and_clipped_fit_counts(self):
        for target,expected in ((0.,0.),(1.,1.)):
            a=fixture();a[3]['raw_candidate'][:]=0.;a[3]['raw_control'][:]=.125
            a[0]['target_log_excess'][:]=-.1 if target==0. else 1.
            z=m.build_forecasts(*a);r=z['audit']['annual'][-1]
            self.assertEqual(r['alpha'],expected);self.assertTrue(r['clipped'])
            self.assertEqual(z['audit']['counts']['clipped_coefficient_years'],1)
        a=fixture();a[3]['raw_candidate']=a[3]['raw_control'].copy()
        z=m.build_forecasts(*a);self.assertEqual(z['audit']['annual'][-1]['status'],'zero_denominator')
        self.assertEqual(z['audit']['new_model_fits'],0)
        np.testing.assert_array_equal(z['raw']['candidate'],a[3]['raw_control'])

    def test_unavailable_preserved_and_invalid_source_protocol(self):
        a=fixture();a[3]['available'][3]=False
        for key in ('raw_candidate','raw_control','raw_source_log'):a[3][key][3]=np.nan
        z=m.build_forecasts(*a);self.assertTrue(np.isnan(z['raw']['candidate'][3]));self.assertFalse(z['available'][3])
        for bad in (0.,np.inf):
            b=copy.deepcopy(a);b[3]['raw_candidate'][3]=bad
            with self.assertRaises(ValueError):m.build_forecasts(*b)
        a=fixture();a[3]['available']=a[3]['available'].astype(int)
        with self.assertRaises(ValueError):m.build_forecasts(*a)

    def test_finite_loss_arithmetic_overflow_is_not_missing(self):
        a=fixture();a[3]['raw_control'][3]=1e200
        with self.assertRaises(FloatingPointError):m.build_forecasts(*a)
        a=fixture();a[0]['target_log_excess'][3]=1e308;a[3]['raw_candidate'][3]=-1e308
        with self.assertRaises(FloatingPointError):m.build_forecasts(*a)

    def test_metadata_shapes_and_immutable_dependency(self):
        a=fixture();a[2]['symbols'][1].pop()
        with self.assertRaises(ValueError):m.build_forecasts(*a)
        a=fixture();a[0]['train_plain']=a[0]['train_plain'][:,:9]
        with self.assertRaises(ValueError):m.build_forecasts(*a)
        self.assertEqual(len(m.dependency_files()),1)
        self.assertEqual(m.dependency_files()[0].name,'monthly_forecast_blend211_model.py')


if __name__=='__main__':
    unittest.main(verbosity=2)
