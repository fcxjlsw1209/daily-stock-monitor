"""Synthetic-only annual cohort, arithmetic and immutable cached-input tests."""
import calendar
import copy
import math
import unittest
import numpy as np
import monthly_increment_calibration224_model as model


def month_pair(year, month):
    total=year*12+month
    ny,nm=total//12,total%12+1
    return ('%04d-%02d-%02d'%(year,month,calendar.monthrange(year,month)[1]),
            '%04d-%02d-%02d'%(ny,nm,calendar.monthrange(ny,nm)[1]))


def fixture(n_history=18, symbols=('A','B','QQQ'), start=(2017,6),
            current=((2019,1),(2019,6)), initial=False):
    pairs=[month_pair((start[0]*12+start[1]-1+i)//12,
                     (start[0]*12+start[1]-1+i)%12+1) for i in range(n_history)]
    pairs += [month_pair(y,m) for y,m in current]
    if initial:pairs=[('2016-08-17','2016-08-31')]+pairs
    width=len(symbols)
    qm=dict(dates=[p[0] for p in pairs],ends=[p[1] for p in pairs],symbols=[list(symbols) for p in pairs])
    formal=[p for p in pairs if p[0]!=model.INITIAL]
    tm=dict(dates=[p[0] for p in formal],ends=[p[1] for p in formal],symbols=[list(symbols) for p in formal])
    total=len(formal)*width
    train=dict(offsets=np.arange(len(formal)+1,dtype=int)*width,
        dates=np.repeat(np.array(tm['dates']),width),ends=np.repeat(np.array(tm['ends']),width),
        train_plain=np.zeros((total,10)),common_good=np.ones(total,bool))
    train.update({'target_'+k:np.zeros(total) for k in model.TARGETS})
    ds=np.array([1.,2.,-1.,.5][:width]); slopes=np.array([.25,.75,1.,0.][:width])
    p=np.tile(ds,(len(pairs),1)); b=np.zeros_like(p)
    train['target_log_excess'][:n_history*width]=np.tile(ds*slopes,n_history)
    source=dict(raw_local_linear=p.reshape(-1).copy(),raw_source_log=b.reshape(-1).copy(),
        available=np.ones(p.size,bool),raw_candidate=np.full(p.size,999.),raw_control=np.full(p.size,-999.),
        raw_local_constant=np.full(p.size,888.),local_available=np.zeros(p.size,bool))
    return train,tm,qm,source,np.ones(p.shape,bool)


def physical_prefix(args,n):
    train,tm,qm,source,use=args; width=use.shape[1]
    smallqm={k:copy.deepcopy(v[:n]) for k,v in qm.items()}
    cutoff=str(int(qm['dates'][n-1][:4])-1)+'-12-31' if n else None
    keep=[i for i,e in enumerate(tm['ends']) if cutoff is not None and e<=cutoff]
    ids=np.array([j for i in keep for j in range(train['offsets'][i],train['offsets'][i+1])],dtype=int)
    smalltm={k:[copy.deepcopy(v[i]) for i in keep] for k,v in tm.items()}
    smalltrain={k:np.asarray(v)[ids].copy() for k,v in train.items() if k!='offsets'}
    smalltrain['offsets']=np.r_[0,np.cumsum(np.array([len(tm['symbols'][i]) for i in keep],dtype=int))]
    return smalltrain,smalltm,smallqm,{k:np.asarray(v)[:n*width].copy() for k,v in source.items()},use[:n].copy()


def annual(out,year=2019):
    return next(r for r in out['audit']['annual'] if r['year']==year)


def group(out,symbol='A',year=2019):
    return next(g for g in annual(out,year)['groups'] if g['symbol']==symbol)


class ScalarTests(unittest.TestCase):
    def test_coefficient_endpoints_and_strict_clipping(self):
        for n,alpha,clipped in [(-1.,0.,True),(0.,0.,False),(.5,.25,False),(2.,1.,False),(3.,1.,True)]:
            self.assertEqual(model.coefficient(n,2.,18),(alpha,clipped,'calibrated'))
        self.assertEqual(model.coefficient(1.,2.,17),(None,False,'cold_start'))
        self.assertEqual(model.coefficient(0.,0.,18),(None,False,'zero_denominator'))

    def test_convex_blend_exact_endpoints_and_signed_zero_precedence(self):
        self.assertEqual(model.blend(4.,2.,0.),2.)
        self.assertEqual(model.blend(4.,2.,1.),4.)
        self.assertEqual(model.blend(4.,2.,.25),2.5)
        for alpha in (0.,.5,1.):
            self.assertEqual(model.blend(-0.,0.,alpha).hex(),(-0.).hex())
        self.assertEqual(model.blend(1e308,-1e308,0.),-1e308)

    def test_nonfinite_arithmetic_and_nonzero_underflow(self):
        for a,b in ((1e308,2.),(1e-200,1e-200),(-1e-200,1e-200)):
            with self.assertRaises(FloatingPointError):model.product(a,b)
        tiny=float(np.nextafter(0.,1.))
        for sign in (1.,-1.):
            with self.assertRaises(FloatingPointError):model.mean(sign*tiny,2)
        with self.assertRaises(FloatingPointError):model.coefficient(tiny,1e308,18)
        with self.assertRaises(FloatingPointError):model.blend(1e308,-1e308,.5)
        self.assertEqual(model.product(0.,1e308),0.)
        self.assertEqual(model.mean(math.fsum([1.,-1.]),2),0.)

    def test_scalar_numeric_type_and_support_guards(self):
        for args in ((True,1.,18),(1.,-1.,18),(1.,1.,True),(math.nan,1.,18)):
            with self.assertRaises((ValueError,FloatingPointError)):model.coefficient(*args)
        for alpha in (-.1,1.1,True,math.inf):
            with self.assertRaises((ValueError,FloatingPointError)):model.blend(1.,0.,alpha)


class ModelTests(unittest.TestCase):
    def test_analytic_18_month_own_and_pooled_calibration(self):
        args=fixture(); out=model.build_forecasts(*args); row=annual(out)
        self.assertEqual(row['calibration_months'],18);self.assertEqual(row['calibration_rows'],54)
        self.assertEqual(row['alpha'],math.fsum([4.25/3]*18)/math.fsum([6./3]*18))
        for symbol,alpha in zip(('A','B','QQQ'),(.25,.75,1.)):
            g=group(out,symbol);self.assertEqual(g['alpha'],alpha)
            self.assertEqual(g['own_months'],18);self.assertEqual(g['status'],'calibrated')
            self.assertEqual(g['query_row_ids'],[54+('A','B','QQQ').index(symbol),57+('A','B','QQQ').index(symbol)])
        counts=out['audit']['counts'];self.assertEqual(counts['parameter_estimates'],4)
        self.assertEqual(counts['pooled_fits'],1);self.assertEqual(counts['own_fits'],3)
        self.assertEqual(out['audit']['new_model_fits'],4);self.assertEqual(out['audit']['predictive_model_fits'],4)

    def test_17_months_copy_original_and_no_own_fits(self):
        args=fixture(n_history=17);out=model.build_forecasts(*args)
        self.assertEqual(annual(out)['status'],'cold_start')
        self.assertTrue(all(g['status']=='pooled_not_ready' for g in annual(out)['groups']))
        for raw in out['raw'].values():np.testing.assert_array_equal(raw,args[3]['raw_local_linear'])
        self.assertEqual(out['audit']['counts']['parameter_estimates'],0)

    def test_pooled_zero_denominator_copy_original(self):
        args=fixture();args[3]['raw_source_log'][:54]=args[3]['raw_local_linear'][:54]
        out=model.build_forecasts(*args);self.assertEqual(annual(out)['status'],'zero_denominator')
        self.assertEqual(out['audit']['counts']['zero_denominator_years'],1)
        self.assertEqual(group(out)['status'],'pooled_not_ready')
        self.assertEqual(out['audit']['counts']['zero_denominator_groups'],0)
        for raw in out['raw'].values():np.testing.assert_array_equal(raw,args[3]['raw_local_linear'])

    def test_own_cold_and_zero_denominator_use_identical_pool_alpha(self):
        args=fixture();args[4][0,0]=False
        ids=np.arange(1,54,3);args[3]['raw_source_log'][ids]=args[3]['raw_local_linear'][ids]
        out=model.build_forecasts(*args)
        self.assertEqual(group(out,'A')['status'],'cold_start')
        self.assertEqual(group(out,'B')['status'],'zero_denominator')
        for symbol in ('A','B'):
            g=group(out,symbol);self.assertEqual(g['applied_alpha'],annual(out)['alpha'])
            np.testing.assert_array_equal(out['raw']['candidate'][g['query_row_ids']],out['raw']['control'][g['query_row_ids']])

    def test_pool_keeps_historical_ids_absent_from_current_query(self):
        args=fixture();before=model.build_forecasts(*args)
        for i in (-1,-2):args[2]['symbols'][i][1]='NEW';args[1]['symbols'][i][1]='NEW'
        out=model.build_forecasts(*args)
        self.assertEqual(annual(out)['alpha'],annual(before)['alpha'])
        self.assertTrue(any(h['symbol']=='B' for m in annual(out)['months'] for h in m['rows']))
        self.assertEqual(group(out,'NEW')['own_rows'],0)
        self.assertEqual(group(out,'NEW')['fallback'],'pooled')

    def test_month_equal_pool_differs_from_row_weighting(self):
        args=fixture();args[3]['raw_local_linear'][:54]=1.
        args[0]['target_log_excess'][:54]=np.tile([0.,0.,3.],18)
        args[4][9:18,2]=False
        row=annual(model.build_forecasts(*args))
        self.assertEqual(row['numerator'],9.);self.assertEqual(row['denominator'],18.)
        self.assertEqual(row['alpha'],.5);self.assertNotEqual(row['alpha'],27/45)

    def test_small_month_excluded_from_both_and_terms_not_evaluated(self):
        args=fixture();args[4][0,1:]=False
        args[3]['raw_local_linear'][0]=1e308;args[3]['raw_source_log'][0]=-1e308
        out=model.build_forecasts(*args);row=annual(out);m=row['months'][0]
        self.assertEqual(row['small_month_rows'],1);self.assertEqual(row['calibration_months'],17)
        self.assertFalse(m['retained']);self.assertIsNone(m['numerator_sum'])
        self.assertIsNone(m['rows'][0]['disagreement']);self.assertIsNone(m['rows'][0]['square'])
        self.assertEqual(group(out)['own_rows'],17)

    def test_all_original_targets_and_ten_features_gate_common_cohort(self):
        for key in ('common_good','train_plain')+tuple('target_'+k for k in model.TARGETS):
            args=fixture()
            if key=='common_good':args[0][key][0]=False
            elif key=='train_plain':args[0][key][0,7]=np.nan
            else:args[0][key][0]=np.nan
            row=annual(model.build_forecasts(*args))
            self.assertEqual(row['invalid_label_rows'],1);self.assertEqual(row['calibration_rows'],53)
            self.assertEqual(row['months'][0]['valid_rows'],2)

    def test_missing_mature_label_fails_even_historical_not_current_id(self):
        args=fixture();args[1]['symbols'][0][1]='UNKNOWN'
        for i in (-1,-2):args[2]['symbols'][i][1]='NEW';args[1]['symbols'][i][1]='NEW'
        with self.assertRaisesRegex(ValueError,'Missing mature exact'):model.build_forecasts(*args)

    def test_original_U_and_source_mask_intersection(self):
        args=fixture();args[4][0,0]=False
        args[3]['available'][1]=False;args[3]['raw_local_linear'][1]=np.nan;args[3]['raw_source_log'][1]=np.nan
        out=model.build_forecasts(*args);row=annual(out)
        self.assertEqual(row['months'][0]['joined_rows'],1);self.assertFalse(row['months'][0]['retained'])
        np.testing.assert_array_equal(out['available'],args[3]['available'])
        for raw in out['raw'].values():self.assertTrue(np.isnan(raw[1]))

    def test_current_U_and_local_available_do_not_filter_predictions(self):
        args=fixture();before=model.build_forecasts(*args);args[4][-2:]=False
        args[3]['local_available'][:]=False;out=model.build_forecasts(*args)
        self.assertEqual(out['audit'],before['audit'])
        for key in out['raw']:np.testing.assert_array_equal(out['raw'][key],before['raw'][key])

    def test_no_current_available_still_calibrates_annual_pool(self):
        args=fixture();args[3]['available'][-6:]=False
        args[3]['raw_local_linear'][-6:]=np.nan;args[3]['raw_source_log'][-6:]=np.nan
        out=model.build_forecasts(*args);row=annual(out)
        self.assertEqual(row['status'],'calibrated');self.assertEqual(row['prediction_rows'],0)
        self.assertEqual(row['groups'],[]);self.assertEqual(row['parameter_estimates'],1)

    def test_future_label_objects_never_consumed(self):
        args=fixture();before=model.build_forecasts(*args)
        for key in ['common_good','train_plain']+['target_'+k for k in model.TARGETS]:
            args[0][key]=args[0][key].astype(object);args[0][key][-6:]=object()
        out=model.build_forecasts(*args);self.assertEqual(out['audit'],before['audit'])
        for key in out['raw']:np.testing.assert_array_equal(out['raw'][key],before['raw'][key])

    def test_exclusive_five_year_signal_boundary(self):
        args=fixture(start=(2013,12));row=annual(model.build_forecasts(*args))
        self.assertEqual(row['calibration_months'],17);self.assertEqual(row['months'][0]['date'],'2014-01-31')
        self.assertEqual(row['lookback_exclusive_start'],'2013-12-31')

    def test_initial_partial_excluded_and_duplicate_formal_month_rejected(self):
        args=fixture(initial=True);row=annual(model.build_forecasts(*args))
        self.assertEqual(row['skipped_initial_rows'],3);self.assertEqual(row['calibration_months'],18)
        self.assertTrue(all(m['date']!='2016-08-17' for m in row['months']))
        args[2]['dates'][2]=args[2]['dates'][1]
        with self.assertRaises(ValueError):model.build_forecasts(*args)

    def test_unrelated_source_columns_are_not_consumed(self):
        args=fixture();before=model.build_forecasts(*args)
        for key in ('raw_candidate','raw_control','raw_local_constant','local_available'):
            args[3][key]=np.array([object()]*len(args[3][key]))
        out=model.build_forecasts(*args);self.assertEqual(out['audit'],before['audit'])

    def test_metadata_offsets_identity_masks_and_finite_source_are_strict(self):
        for mode in ('offset','identity','bool','mask','source'):
            args=fixture()
            if mode=='offset':args[0]['offsets'][1]+=1
            elif mode=='identity':args[0]['dates'][0]='2017-07-31'
            elif mode=='bool':args=list(args);args[4]=args[4].astype(int)
            elif mode=='mask':args[3]['available'][0]=False
            else:args[3]['raw_source_log'][0]=np.inf
            with self.assertRaises((ValueError,FloatingPointError)):model.build_forecasts(*args)

    def test_all_physical_prefixes_no_mutation_and_truthful_fit_counts(self):
        args=fixture();before=copy.deepcopy(args);full=model.build_forecasts(*args)
        for n in range(len(args[2]['dates'])+1):
            out=model.build_forecasts(*physical_prefix(args,n))
            for key in out['raw']:np.testing.assert_array_equal(out['raw'][key],full['raw'][key][:n*3])
            np.testing.assert_array_equal(out['available'],full['available'][:n*3])
            counts=out['audit']['counts']
            self.assertEqual(set(counts),set(model.COUNT_FIELDS))
            self.assertEqual(counts['parameter_estimates'],counts['pooled_fits']+counts['own_fits'])
            self.assertEqual(out['audit']['new_ridge_fits'],0);self.assertEqual(out['audit']['new_tree_fits'],0)
        for first,last in zip(args,before):
            if isinstance(first,dict):
                for key in first:
                    if isinstance(first[key],np.ndarray):np.testing.assert_array_equal(first[key],last[key])
                    else:self.assertEqual(first[key],last[key])
            else:np.testing.assert_array_equal(first,last)


if __name__=='__main__':unittest.main(verbosity=2)
