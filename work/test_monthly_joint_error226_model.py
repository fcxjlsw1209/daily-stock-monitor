"""Pure synthetic tests for common-clock error risk and fixed boundary domains."""
import calendar
import copy
import math
import unittest
import numpy as np
import monthly_joint_error226_model as model


def month_pair(year, month):
    total=year*12+month
    ny,nm=total//12,total%12+1
    return (f'{year:04d}-{month:02d}-{calendar.monthrange(year,month)[1]:02d}',
            f'{ny:04d}-{nm:02d}-{calendar.monthrange(ny,nm)[1]:02d}')


def fixture(n_history=18, symbols=('A','B','C','D','E','F'), start=(2017,6),
            current=((2019,1),(2019,6)), initial=False):
    pairs=[month_pair((start[0]*12+start[1]-1+i)//12,
                     (start[0]*12+start[1]-1+i)%12+1) for i in range(n_history)]
    pairs += [month_pair(y,m) for y,m in current]
    if initial:pairs=[(model.INITIAL,'2016-08-31')]+pairs
    width=len(symbols)
    qm=dict(dates=[p[0] for p in pairs],ends=[p[1] for p in pairs],symbols=[list(symbols) for p in pairs])
    formal=[p for p in pairs if p[0]!=model.INITIAL]
    tm=dict(dates=[p[0] for p in formal],ends=[p[1] for p in formal],symbols=[list(symbols) for p in formal])
    total=len(formal)*width
    train=dict(offsets=np.arange(len(formal)+1,dtype=int)*width,
        dates=np.repeat(np.array(tm['dates']),width),ends=np.repeat(np.array(tm['ends']),width),
        train_plain=np.zeros((total,10)),common_good=np.ones(total,bool))
    train.update({'target_'+k:np.zeros(total) for k in model.TARGETS})
    pattern=np.array([1.,1.,1.,-1.,0.,2.][:width])
    p=np.array([pattern*(1. if i%2==0 else -1.) for i in range(len(pairs))])
    source=dict(raw_local_linear=p.reshape(-1).copy(), available=np.ones(p.size,bool),
        raw_source_log=np.full(p.size,99.),raw_candidate=np.full(p.size,999.),
        raw_control=np.full(p.size,-999.),raw_local_constant=np.full(p.size,888.),
        local_available=np.zeros(p.size,bool))
    pre=np.tile(np.array([.4,.3,.2,.195,.185,.1][:width]),(len(pairs),1))
    return train,tm,qm,source,np.ones(p.shape,bool),pre,np.ones(p.shape,bool)


def physical_prefix(args,n):
    train,tm,qm,source,use,pre,eligible=args; width=use.shape[1]
    smallqm={k:copy.deepcopy(v[:n]) for k,v in qm.items()}
    cutoff=str(int(qm['dates'][n-1][:4])-1)+'-12-31' if n else None
    keep=[i for i,e in enumerate(tm['ends']) if cutoff is not None and e<=cutoff]
    ids=np.array([j for i in keep for j in range(train['offsets'][i],train['offsets'][i+1])],dtype=int)
    smalltm={k:[copy.deepcopy(v[i]) for i in keep] for k,v in tm.items()}
    smalltrain={k:np.asarray(v)[ids].copy() for k,v in train.items() if k!='offsets'}
    smalltrain['offsets']=np.r_[0,np.cumsum(np.array([len(tm['symbols'][i]) for i in keep],dtype=int))]
    return (smalltrain,smalltm,smallqm,{k:np.asarray(v)[:n*width].copy() for k,v in source.items()},
            use[:n].copy(),pre[:n].copy(),eligible[:n].copy())


def event(out,index=-1):
    return out['audit']['event_history'][index]


class ScalarTests(unittest.TestCase):
    def test_common_and_opposite_error_analytic_contrast(self):
        errors=np.array([[1.,1.,1.,-1.,0.],[-1.,-1.,-1.,1.,0.]])
        got=model.measure(errors)
        self.assertEqual(got['means'],[0.]*5)
        self.assertEqual(got['sum_squares'],[2.,2.,2.,2.,0.])
        self.assertEqual(got['candidate'],[2.,2./9.,8./9.])
        self.assertEqual(got['control'],[2./3.,2./3.,4./9.])
        self.assertEqual(model.effect_ranks(got['candidate']),[-.5,.5,0.])
        self.assertEqual(model.effect_ranks(got['control']),[-.25,-.25,.5])

    def test_orthogonal_columns_joint_matches_diagonal(self):
        got=model.measure([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])
        self.assertAlmostEqual(got['candidate'][0],got['control'][0],places=15)
        self.assertEqual(got['control'],[4./9.])

    def test_column_location_invariance(self):
        matrix=np.array([[1.,1.,-1.,2.],[-1.,-1.,1.,-2.]])
        baseline=model.measure(matrix)
        shifted=model.measure(matrix+np.array([10.,-4.,7.,8.]))
        for key in ('sum_squares','variances','candidate','control'):
            self.assertEqual(baseline[key],shifted[key])

    def test_zero_risk_and_constant_effects(self):
        got=model.measure([[4.,2.,-3.,0.]]*18)
        self.assertEqual(got['candidate'],[0.,0.]);self.assertEqual(got['control'],[0.,0.])
        self.assertEqual(model.effect_ranks([0.,0.,0.]),[0.,0.,0.])
        self.assertEqual(model.effect_ranks([2.,2.,1.]),[-.25,-.25,.5])

    def test_arithmetic_failures_not_missing_data(self):
        for errors in ([[1e308,0.,0.],[-1e308,0.,0.]],
                       [[1e-200,0.,0.],[-1e-200,0.,0.]],
                       [[math.inf,0.,0.],[1.,0.,0.]]):
            with self.assertRaises(FloatingPointError):model.measure(errors)
        tiny=float(np.nextafter(0.,1.))
        with self.assertRaises(FloatingPointError):model._divide(tiny,2,'mean')
        with self.assertRaises(FloatingPointError):model._divide(-tiny,2,'mean')
        self.assertEqual(model._divide(math.fsum([1.,-1.]),2,'mean'),0.)

    def test_shapes_types_and_negative_risk(self):
        for errors in ([1.,2.,3.],[[1.,2.,3.]],[[True,1.,2.],[False,2.,1.]]):
            with self.assertRaises(ValueError):model.measure(np.array(errors,dtype=object))
        for risks in ([],[-1.,0.],[True,1.]):
            with self.assertRaises(ValueError):model.effect_ranks(risks)


class BuildTests(unittest.TestCase):
    def test_exact18_month_same_H_and_parameter_accounting(self):
        args=fixture();out=model.build(*args);row=event(out)
        self.assertEqual(row['status'],'ready');self.assertEqual(len(row['H']),18)
        self.assertEqual(row['ordered_top2'],[0,1]);self.assertEqual(row['J'],[2,3,4])
        self.assertEqual(row['B'],[0,1,2,3,4]);self.assertEqual(row['counts']['error_observations'],90)
        self.assertEqual(out['arrays']['candidate_effect'][-1].tolist(),[0.,0.,-.5,.5,0.,0.])
        self.assertEqual(out['arrays']['control_effect'][-1].tolist(),[0.,0.,-.25,-.25,.5,0.])
        self.assertTrue(np.isnan(out['arrays']['candidate_risk'][-1,[0,1,5]]).all())
        c=out['audit']['counts'];self.assertEqual(c['supported_queries'],2)
        self.assertEqual(c['parameter_estimates'],32)
        for key in model.COUNT_FIELDS:
            self.assertEqual(c[key],sum(e['counts'][key] for e in out['audit']['event_history']))
        self.assertEqual(out['audit']['new_statistical_estimation_queries'],2)
        self.assertEqual(out['audit']['source_model_refits'],0)
        self.assertNotIn('new_model_fits',out['audit'])

    def test_17_month_history_joint_fallback(self):
        out=model.build(*fixture(n_history=17))
        self.assertEqual(event(out)['status'],'insufficient_history')
        self.assertFalse(out['arrays']['ready'][-1])
        for key in ('candidate_risk','control_risk'):self.assertTrue(np.isnan(out['arrays'][key][-1]).all())
        for key in ('candidate_effect','control_effect'):self.assertTrue((out['arrays'][key][-1]==0).all())
        self.assertEqual(out['audit']['parameter_estimates'],0)

    def test_domain_top2_lexical_ties_no_fifth_expansion(self):
        args=fixture();args[5][-1]=[.4,.4,.2,.2,.2,.2]
        out=model.build(*args);row=event(out)
        self.assertEqual(row['ordered_top2'],[0,1]);self.assertEqual(row['J'],[2,3,4])
        self.assertNotIn(5,row['B'])
        args[2]['symbols'][-1][0:2]=['Z','A']
        out=model.build(*args);self.assertEqual(event(out)['ordered_top2'],[1,0])

    def test_gap_exact_inclusive_no_epsilon(self):
        args=fixture();args[5][-1]=[.4,.3,0.,-.02,-float(np.nextafter(.02,math.inf)),-.3]
        row=event(model.build(*args));self.assertEqual(row['J'],[2,3])
        args[5][-1,3]=-float(np.nextafter(.02,math.inf))
        row=event(model.build(*args));self.assertEqual(row['status'],'insufficient_candidates')
        self.assertEqual(row['J'],[2])

    def test_cash_two_and_small_boundary_do_not_read_history(self):
        for number in (0,1,2,3):
            args=fixture();args[6][-1]=False;args[6][-1,:number]=True
            row=event(model.build(*args))
            self.assertEqual(row['status'],'insufficient_candidates')
            self.assertEqual(row['history'],[])

    def test_missing_security_drops_whole_month_not_remaining_history(self):
        args=fixture(n_history=19,start=(2017,5));args[2]['symbols'][0][4]='OTHER'
        out=model.build(*args);row=event(out)
        self.assertTrue(out['arrays']['ready'][-1]);self.assertEqual(len(row['H']),18)
        self.assertEqual(row['counts']['missing_security_rows'],1)
        self.assertEqual(row['counts']['incomplete_history_months'],1)
        self.assertEqual(len(row['history'][0]['rows']),5)
        self.assertNotIn(args[2]['dates'][0],row['H'])

    def test_missing_current_B_history_never_narrows_domain(self):
        args=fixture();args[2]['symbols'][-1][4]='NEW'
        row=event(model.build(*args));self.assertEqual(row['J'],[2,3,4])
        self.assertEqual(row['status'],'insufficient_history');self.assertEqual(row['H'],[])

    def test_historical_U_and_source_availability_common_month_gate(self):
        for mode in ('U','source'):
            args=fixture(n_history=19,start=(2017,5))
            if mode=='U':args[4][0,3]=False
            else:args[3]['available'][3]=False;args[3]['raw_local_linear'][3]=np.nan
            row=event(model.build(*args));self.assertEqual(len(row['H']),18)
            self.assertEqual(row['counts']['unsupported_source_rows'],1)
            self.assertEqual(row['history'][0]['rows'][3]['reason'],
                             'historical_use_false' if mode=='U' else 'source_unavailable')

    def test_common_good_all_four_targets_and_ten_features(self):
        for key in ('common_good','train_plain')+tuple('target_'+k for k in model.TARGETS):
            args=fixture(n_history=19,start=(2017,5))
            if key=='common_good':args[0][key][3]=False
            elif key=='train_plain':args[0][key][3,9]=np.nan
            else:args[0][key][3]=np.nan
            row=event(model.build(*args));self.assertEqual(len(row['H']),18)
            self.assertEqual(row['counts']['invalid_label_rows'],1)

    def test_exact_training_join_missing_is_hard_not_fallback(self):
        args=fixture();args[1]['symbols'][0][3]='OTHER'
        with self.assertRaisesRegex(ValueError,'Missing mature exact'):model.build(*args)
        args[4][0,3]=False
        self.assertEqual(event(model.build(*args))['status'],'insufficient_history')

    def test_future_values_not_consumed_and_current_y_does_not_affect(self):
        args=fixture();before=model.build(*args)
        args[0]['train_plain']=args[0]['train_plain'].astype(object)
        args[0]['train_plain'][-12:,:]='poison_future'
        for target in model.TARGETS:
            args[0]['target_'+target]=args[0]['target_'+target].astype(object)
            args[0]['target_'+target][-12:]='future_outcome'
        after=model.build(*args)
        self.assertEqual(after['audit'],before['audit'])
        for key in model.ARRAY_KEYS:np.testing.assert_array_equal(after['arrays'][key],before['arrays'][key])

    def test_no_local_history_filter_and_other_source_fields_ignored(self):
        args=fixture();before=model.build(*args)
        for key in set(args[3])-{'available','raw_local_linear'}:args[3][key]=np.array(['unused'],object)
        self.assertEqual(model.build(*args)['audit'],before['audit'])

    def test_current_source_mask_does_not_shrink_predeclared_domain(self):
        args=fixture();args[3]['available'][-6:]=False;args[3]['raw_local_linear'][-6:]=np.nan
        out=model.build(*args);self.assertTrue(out['arrays']['ready'][-1])
        self.assertEqual(event(out)['J'],[2,3,4])

    def test_annual_cutoff_and_lower_exclusive_five_year_clock(self):
        args=fixture(n_history=62,start=(2013,12),current=((2020,1),))
        row=event(model.build(*args));self.assertEqual(row['cutoff'],'2019-12-31')
        self.assertEqual(row['lookback_exclusive_start'],'2014-12-31')
        self.assertTrue(all(d>'2014-12-31' for d in row['H']))
        self.assertNotIn('2014-12-31',row['H'])
        args=fixture(n_history=18,start=(2017,7))
        row=event(model.build(*args));self.assertEqual(len(row['H']),17)
        self.assertNotIn('2018-12-31',row['H'])

    def test_initial_partial_fallback_and_excluded_from_history(self):
        args=fixture(initial=True);out=model.build(*args)
        self.assertEqual(event(out,0)['status'],'initial_partial')
        self.assertFalse(out['arrays']['ready'][0]);self.assertEqual(event(out,0)['history'],[])
        self.assertEqual(event(out)['counts']['skipped_initial_months'],1)
        self.assertNotIn(model.INITIAL,event(out)['H'])

    def test_constant_full_support_is_ready_zero_effect_not_fallback(self):
        args=fixture();args[3]['raw_local_linear'][:108]=np.tile([1.,2.,3.,4.,5.,6.],18)
        out=model.build(*args);row=event(out)
        self.assertEqual(row['status'],'ready');self.assertTrue(out['arrays']['ready'][-1])
        self.assertTrue((out['arrays']['candidate_effect'][-1]==0).all())
        self.assertTrue((out['arrays']['candidate_risk'][-1,[2,3,4]]==0).all())
        self.assertEqual(row['counts']['parameter_estimates'],16)

    def test_all_physical_prefixes_match_full_events_and_arrays(self):
        args=fixture();full=model.build(*args)
        for n in range(len(args[2]['dates'])+1):
            built=model.build(*physical_prefix(args,n))
            self.assertEqual(built['audit']['event_history'],full['audit']['event_history'][:n])
            for key in model.ARRAY_KEYS:np.testing.assert_array_equal(built['arrays'][key],full['arrays'][key][:n])
        empty=model.build(*physical_prefix(args,0))
        self.assertTrue(all(v==0 for v in empty['audit']['counts'].values()))

    def test_future_query_poison_is_removed_by_physical_prefix(self):
        args=list(fixture());small=physical_prefix(args,19);expected=model.build(*small)
        args[3]['raw_local_linear']=args[3]['raw_local_linear'].astype(object)
        args[3]['raw_local_linear'][-6:]='future_poison'
        args[5]=args[5].astype(object);args[5][-1]='future_poison'
        self.assertEqual(model.build(*physical_prefix(args,19))['audit'],expected['audit'])

    def test_metadata_shapes_and_exact_booleans_fail_closed(self):
        for change in ('offsets','duplicate','end','use','eligible','pre','source'):
            args=list(fixture())
            if change=='offsets':args[0]['offsets']=args[0]['offsets'].astype(float)
            elif change=='duplicate':args[2]['symbols'][-1][1]='A'
            elif change=='end':args[2]['ends'][-1]='2019-06-30'
            elif change=='use':args[4]=args[4].astype(int)
            elif change=='eligible':args[6]=args[6].astype(int)
            elif change=='pre':args[5][-1,0]=math.nan
            else:args[3]['available']=args[3]['available'].astype(int)
            with self.assertRaises((ValueError,FloatingPointError)):model.build(*args)

    def test_current_p_source_mask_and_error_overflow_hard_fail(self):
        args=fixture();args[3]['raw_local_linear'][0]=math.nan
        with self.assertRaises(ValueError):model.build(*args)
        args=fixture();args[3]['raw_local_linear'][0]=1e308;args[0]['target_log_excess'][0]=-1e308
        with self.assertRaises(FloatingPointError):model.build(*args)


if __name__=='__main__':
    unittest.main()
