"""Synthetic-only independent WLS, source clocks and original-episode tests."""
import copy
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_volume_slope232_independent as oracle


def fixture():
    days=pd.bdate_range('2014-01-01',periods=1000);t=np.arange(len(days),dtype=float)
    prices=pd.DataFrame(dict(A=np.exp(.001*t+.08*np.sin(t/33)),
        B=np.exp(.0004*t+.11*np.cos(t/27)),C=np.exp(-.0001*t+.02*np.cos(t/17)),
        D=np.full(len(t),10.),E=np.exp(.002*t)),index=days)
    volume=pd.DataFrame(dict(A=100.+(t%41),B=200.+(t%37)*3.,C=np.full(len(t),500.),
        D=50.+(t%29),E=np.full(len(t),1.)),index=days)
    ds=['2016-08-17','2016-08-31','2016-09-30','2016-10-31']
    meta=dict(dates=ds,ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'],
        symbols=[['A','B','C','D','E'] for _ in ds],etfs=['E'])
    return {'adj':prices,'volume':volume},meta,days.strftime('%Y-%m-%d').tolist(),np.tile([True]*4+[False],(4,1))


def physical_prefix(args,n):
    p,m,c,e=copy.deepcopy(args)
    for key in ('dates','ends','symbols'):
        if key in m:m[key]=m[key][:n]
    for key in p:p[key]=p[key].loc[:m['dates'][-1]].copy() if n else p[key].iloc[:0].copy()
    return p,m,c,e[:n].copy()


def one():
    p,m,c,e=fixture()
    for k in ('dates','ends','symbols'):m[k]=m[k][-1:]
    return p,m,c,e[-1:].copy()


def original(args):
    from test_monthly_joint_error226_independent import original as inherited_fixture
    _,m,_,e=args;n,w=e.shape
    pre=np.tile(np.linspace(.025,.005,w),(n,1))
    a=inherited_fixture(pre,e)
    a['state_calm10']=np.arange(n)%2==0
    a['state_weak_narrow']=np.arange(n)%3==0
    a['core_weight']=np.where(a['state_calm10'],.5,.52)[:,None]
    return a


def record(a,i=-1,j=0):return a['event_history'][i]['cells'][j]


class ArithmeticTests(unittest.TestCase):
    def test_independent_hand_formula_with_nonuniform_weights(self):
        p=[math.exp((i/252.)**2) for i in range(253)];v=[float(i%7+1) for i in range(253)]
        z=oracle.measure(p,v);x=[i/252. for i in range(253)]
        for key,w in (('control_stats',[1.]*253),('candidate_stats',z['weights'])):
            y=[math.log(q)-math.log(p[0]) for q in p];W=math.fsum(w)
            xb=math.fsum(wi*xi for wi,xi in zip(w,x))/W;yb=math.fsum(wi*yi for wi,yi in zip(w,y))/W
            numerator=math.fsum(wi*(xi-xb)*(yi-yb) for wi,xi,yi in zip(w,x,y))
            denominator=math.fsum(wi*(xi-xb)**2 for wi,xi in zip(w,x))
            self.assertEqual(z[key],dict(weight_sum=W,x_mean=xb,y_mean=yb,
                numerator=numerator,denominator=denominator,slope=numerator/denominator))
        self.assertNotEqual(z['candidate'],z['control'])

    def test_constant_volume_exact_copy_and_true_compute_counts(self):
        p=[math.exp(.5*i/252.+.1*math.sin(i/33)) for i in range(253)]
        z=oracle.measure(p,[7.]*253)
        self.assertEqual(z['weights'],[1.]*253);self.assertEqual(z['volume_ranks'],[127.]*253)
        self.assertEqual(z['candidate_branch'],'constant_volume_copy')
        self.assertEqual(np.float64(z['candidate']).tobytes(),np.float64(z['control']).tobytes())
        self.assertEqual(z['candidate_stats'],z['control_stats'])

    def test_constant_price_zero_and_no_solves(self):
        z=oracle.measure([10.]*253,[float(i+1) for i in range(253)])
        self.assertEqual(z['candidate'],0.);self.assertEqual(z['control'],0.)
        self.assertIsNone(z['candidate_stats']);self.assertIsNone(z['control_stats'])
        self.assertEqual(z['candidate_branch'],'constant_price_zero')
        self.assertEqual(z['control_branch'],'constant_price_zero')

    def test_linear_log_price_weight_invariance(self):
        for direction in (-.3,0.,.7):
            p=[math.exp(2.+direction*i/252.) for i in range(253)]
            z=oracle.measure(p,[float((i*19)%253+1) for i in range(253)])
            self.assertAlmostEqual(z['candidate'],direction,places=13)
            self.assertAlmostEqual(z['control'],direction,places=13)

    def test_constant_rescaling_and_rank_ties_mean_one(self):
        p=[math.exp(.3*i/252.+.1*math.cos(i/19)) for i in range(253)]
        v=[float(i%11+1) for i in range(253)];a=oracle.measure(p,v)
        b=oracle.measure([q*8 for q in p],[q*16 for q in v])
        self.assertEqual(a['weights'],b['weights']);self.assertEqual(a['volume_ranks'],b['volume_ranks'])
        self.assertAlmostEqual(a['candidate'],b['candidate'],places=13)
        self.assertAlmostEqual(a['control'],b['control'],places=13)
        self.assertAlmostEqual(math.fsum(a['weights']),253.,places=12)
        self.assertTrue(all(.5<=w<=1.5 for w in a['weights']))

    def test_same_endpoints_and_direction_not_forced(self):
        p=[math.exp((i/252.)**2) for i in range(253)]
        forward=oracle.measure(p,[float(i+1) for i in range(253)])
        reverse=oracle.measure(p,[float(253-i) for i in range(253)])
        self.assertEqual(forward['control'],reverse['control'])
        self.assertGreater(forward['candidate'],forward['control'])
        self.assertLess(reverse['candidate'],reverse['control'])
        q=[math.exp(i/252.) for i in range(253)]
        self.assertEqual((p[0],p[-1]),(q[0],q[-1]))
        self.assertNotEqual(forward['candidate'],oracle.measure(q,[float(i+1) for i in range(253)])['candidate'])

    def test_extreme_positive_sources_log_difference_stable(self):
        p=[1.]*253;p[0]=float(np.nextafter(0.,1.));p[126]=1e308
        z=oracle.measure(p,[1e308 if i%2 else 1e-300 for i in range(253)])
        self.assertTrue(math.isfinite(z['candidate']));self.assertTrue(math.isfinite(z['control']))
        with self.assertRaises(FloatingPointError):oracle.slope_stats([0.,1.],[1e308,1e308],[1.,1.])

    def test_volume_rank_is_not_a_split_unit_certificate(self):
        v=[float(i+1) for i in range(253)];mixed=v[:126]+[q/10 for q in v[126:]]
        self.assertNotEqual(oracle.average_ranks(v),oracle.average_ranks(mixed))
        self.assertEqual(oracle.average_ranks(v),oracle.average_ranks([q*10 for q in v]))

    def test_centered_rank_exact_ties_and_constant_zero(self):
        np.testing.assert_array_equal(oracle.centered_rank([1.,1.,3.]),[-.25,-.25,.5])
        np.testing.assert_array_equal(oracle.centered_rank([4.,2.]),[.5,-.5])
        for v in ([],[1.],[1.,1.],[-0.,0.,0.]):self.assertFalse(np.any(oracle.centered_rank(v)))
        np.testing.assert_array_equal(oracle.centered_rank([1.,np.nextafter(1.,2.)]),[-.5,.5])


class SourceTests(unittest.TestCase):
    def test_source_clock_counts_and_exact_zero_source_uses(self):
        args=fixture();v,a=oracle.independent_features(*args)
        self.assertEqual(a['available_cells'],12);self.assertEqual(a['attempted_cells'],12)
        self.assertEqual(a['initial_partial_cells'],5);self.assertEqual(a['ineligible_cells'],3)
        self.assertEqual(a['unweighted_slope_solves'],9);self.assertEqual(a['weighted_slope_solves'],6)
        self.assertEqual(a['constant_volume_copies'],3);self.assertEqual(a['constant_price_shortcuts'],3)
        self.assertEqual(a['source_price_observations'],253*12);self.assertEqual(a['source_volume_observations'],253*12)
        self.assertEqual(a['descriptive_time_slope_estimates'],24);self.assertEqual(a['cross_sectional_rank_observations'],0)
        for ev in a['event_history']:
            t=args[2].index(ev['date']);self.assertEqual(ev['source_start'],args[2][t-294])
            self.assertEqual(ev['source_end'],args[2][t-42]);self.assertEqual(len(ev['source_dates']),253)
        self.assertTrue(np.isnan(v['candidate'][0]).all())

    def test_future_and_latest42_prices_volume_poison_ignored(self):
        args=one();before=oracle.independent_features(*args);t=args[2].index(args[1]['dates'][0]);cut=args[2][t-42]
        for f in args[0].values(): f.loc[f.index>cut,:]=np.nan
        after=oracle.independent_features(*args)
        oracle.record_check(after[1],before[1])
        for k in oracle.VALUE_KEYS:oracle.exact_array(after[0][k],before[0][k],k)

    def test_noneligible_and_initial_never_consume_numeric_cells(self):
        args=one();before=oracle.independent_features(*args)
        for k in args[0]:args[0][k]=args[0][k].drop(columns='E')
        oracle.record_check(oracle.independent_features(*args)[1],before[1])
        args=fixture()
        for f in args[0].values():f[:]=np.nan
        v,a=oracle.independent_features(*args,limit=1)
        self.assertFalse(v['available'].any());self.assertEqual(a['attempted_cells'],0)
        self.assertTrue(all(c['reason']=='initial_partial' for c in a['event_history'][0]['cells']))

    def test_individual_bad_cell_does_not_remove_other_candidates(self):
        args=one();t=args[2].index(args[1]['dates'][0]);day=args[2][t-80]
        args[0]['volume'].loc[day,'A']=0.
        v,a=oracle.independent_features(*args)
        self.assertEqual(record(a)['reason'],'invalid_volume');self.assertEqual(v['available'].sum(),3)
        self.assertTrue(np.isnan(v['candidate'][0,0]) and np.isnan(v['control'][0,0]))
        self.assertEqual(a['available_cells'],3);self.assertTrue(args[3][0,0])

    def test_source_reason_precedence_and_separate_axes(self):
        for which,reason in (('ps','missing_price_symbol'),('vs','missing_volume_symbol'),
                ('pd','missing_price_session'),('vd','missing_volume_session')):
            args=one();t=args[2].index(args[1]['dates'][0]);day=args[2][t-80]
            role='adj' if which[0]=='p' else 'volume'
            args[0][role]=args[0][role].drop(columns='A') if which[1]=='s' else args[0][role].drop(index=day)
            self.assertEqual(record(oracle.independent_features(*args)[1])['reason'],reason)
        args=one();args[0]['volume']=args[0]['volume'].drop(columns='A');args[0]['adj']=args[0]['adj'].iloc[:0]
        self.assertEqual(record(oracle.independent_features(*args)[1])['reason'],'missing_volume_symbol')

    def test_numeric_bool_string_zero_negative_nan_inf_rejected(self):
        for role,label in (('adj','price'),('volume','volume')):
            for bad,reason in ((True,'nonnumeric_'),('3','nonnumeric_'),(None,'nonnumeric_'),
                    (0.,'invalid_'),(-1.,'invalid_'),(math.nan,'invalid_'),(math.inf,'invalid_'),(10**400,'invalid_')):
                args=one();t=args[2].index(args[1]['dates'][0]);day=args[2][t-80]
                args[0][role]['A']=args[0][role]['A'].astype(object);args[0][role].at[pd.Timestamp(day),'A']=bad
                self.assertEqual(record(oracle.independent_features(*args)[1])['reason'],reason+label)

    def test_insufficient_calendar_is_common_missing_not_compressed(self):
        args=list(one());t=args[2].index(args[1]['dates'][0]);args[2]=args[2][t-293:]
        v,a=oracle.independent_features(*args)
        self.assertFalse(v['available'].any());self.assertIsNone(a['event_history'][0]['source_start'])
        self.assertEqual(record(a)['reason'],'insufficient_history')

    def test_metadata_calendar_shape_and_boolean_gates(self):
        for kind in ('integerelig','missingetfs','duplicateetfs','duplicateid','notmonthend','isotime','unsorted','duplicateframe'):
            args=list(one())
            if kind=='integerelig':args[3]=args[3].astype(int)
            elif kind=='missingetfs':args[1].pop('etfs')
            elif kind=='duplicateetfs':args[1]['etfs']=['E','E']
            elif kind=='duplicateid':args[1]['symbols'][0][1]='A'
            elif kind=='notmonthend':args[1]['dates'][0]='2016-10-28'
            elif kind=='isotime':args[1]['dates'][0]+='T00:00:00'
            elif kind=='unsorted':args[2][1],args[2][2]=args[2][2],args[2][1]
            else:args[0]['adj'].columns=['A','A','C','D','E']
            with self.assertRaises(ValueError):oracle.independent_features(*args)
        args=one();args[1]['ends']='unused-future-label-field'
        self.assertEqual(oracle.independent_features(*args)[1]['available_cells'],4)

    def test_all_physical_logical_prefixes_and_future_metadata(self):
        args=fixture();full=oracle.independent_features(*args)
        for n in range(5):
            left=oracle.independent_features(*args,limit=n);right=oracle.independent_features(*physical_prefix(args,n))
            oracle.record_check(left[1],right[1]);oracle.record_check(left[1]['event_history'],full[1]['event_history'][:n])
            for k in oracle.VALUE_KEYS:
                oracle.exact_array(left[0][k],full[0][k][:n],k);oracle.exact_array(right[0][k],left[0][k],k)
        args[1]['dates'][-1]='bad';args[1]['symbols'][-1]=['future-bad']
        oracle.record_check(oracle.independent_features(*args,limit=3)[1]['event_history'],full[1]['event_history'][:3])
        p,m,c,e=physical_prefix(fixture(),0);v,a=oracle.independent_features(p,m,c,np.zeros((0,0),bool))
        self.assertEqual(v['candidate'].shape,(0,0));self.assertEqual(a['symbols_per_event'],0)

    def test_full_audit_and_value_mutations_rejected(self):
        args=one();v,a=oracle.independent_features(*args);check=oracle.verify_features(*args,v,a)
        self.assertTrue(all(check[k] is True for k in oracle.FEATURE_FLAGS))
        for kind in ('raw','mask','count','weight','clock','copy'):
            vv,aa=copy.deepcopy((v,a))
            if kind=='raw':vv['candidate'][0,0]+=1e-10
            elif kind=='mask':vv['available'][0,0]=False
            elif kind=='count':aa['weighted_slope_solves']+=1
            elif kind=='weight':record(aa)['weights'][0]+=.01
            elif kind=='clock':aa['event_history'][0]['source_end']=aa['event_history'][0]['date']
            else:record(aa,j=2)['candidate_branch']='weighted_slope'
            with self.assertRaises(ValueError):oracle.verify_features(*args,vv,aa)


class SelectionTests(unittest.TestCase):
    def test_original_episode_after_tiny_prefactor_exact_and_inputs_unchanged(self):
        args=fixture();v,a=oracle.independent_features(*args);o=original(args)
        # The old helper used scalar successive subtraction for both-active
        # states. Test its exact parity on nonoverlap; the dedicated test below
        # checks the production NumPy grouping for both-active states.
        o['state_weak_narrow'][o['state_calm10']]=False
        saved=copy.deepcopy(o)
        scores,choices=oracle.selection_oracle(o,args[1],v)
        receipt=oracle.verify_selection(o,args[1],v,scores,choices)
        self.assertTrue(all(receipt[k] is True for k in oracle.SELECTION_FLAGS));self.assertLessEqual(receipt['max_absolute_delta'],.005)
        for key in o:oracle.exact_array(o[key],saved[key],key)
        for arm in oracle.ARMS[1:]:
            oracle.exact_array(scores[arm][0],scores[oracle.ARMS[0]][0],'initial unchanged')
            self.assertEqual(choices[arm][0],choices[oracle.ARMS[0]][0])
        helper=oracle.frozen_source_oracle().old.original_helper();x=helper.original_prefix(o,args[1],None)
        ranks=oracle.ranked_features(v,x['eligible'],args[1]['dates'])
        for arm,key in zip(oracle.ARMS[1:],('candidate','control')):
            pre=x['local'].copy();use=ranks[key]!=0.;pre[use]+=.01*ranks[key][use]
            selected,score,_=helper.episode(pre,x)
            oracle.exact_array(scores[arm],score,arm);self.assertEqual(choices[arm],selected)

    def test_budget_keeps_original_numpy_operation_grouping(self):
        args=one();o=original(args);o['state_calm10'][:]=True;o['state_weak_narrow'][:]=True
        o['core_weight'][:]=.5
        mask=args[3].copy();v={'available':mask,'candidate':np.where(mask,1.,np.nan),'control':np.where(mask,1.,np.nan)}
        scores,choices=oracle.selection_oracle(o,args[1],v)
        budget=float(np.minimum(.05,np.maximum(0.,.10-(.025*np.array([True])+.05*np.array([True]))))[0])
        self.assertNotEqual(budget,min(.05,max(0.,.10-.025-.05)))
        expected=o['pre_score'].copy();expected[0,2]+=budget*(-.5);expected[0,3]+=budget*.5
        for arm in oracle.ARMS:oracle.exact_array(scores[arm],expected,arm+' canonical grouped budget')

    def test_old_episode_effect_not_replaced_or_applied_twice(self):
        args=one();v,a=oracle.independent_features(*args);o=original(args)
        scores,choices=oracle.selection_oracle(o,args[1],v)
        wrong={k:x.copy() for k,x in scores.items()};wrong[oracle.ARMS[1]][0,3]+=.01
        with self.assertRaises(ValueError):oracle.verify_selection(o,args[1],v,wrong,choices)
        before=scores[oracle.ARMS[1]].copy();o['effects'][:]=0.
        after,_=oracle.selection_oracle(o,args[1],v)
        self.assertFalse(np.array_equal(before,after[oracle.ARMS[1]]))

    def test_missing_and_constant_ranks_exact_incumbent(self):
        args=one();o=original(args)
        for supported in (False,True):
            mask=args[3].copy() if supported else np.zeros_like(args[3])
            v={'available':mask,'candidate':np.where(mask,1.,np.nan),'control':np.where(mask,2.,np.nan)}
            s,c=oracle.selection_oracle(o,args[1],v)
            for arm in oracle.ARMS[1:]:oracle.exact_array(s[arm],s[oracle.ARMS[0]],'neutral');self.assertEqual(c[arm],c[oracle.ARMS[0]])

    def test_common_domain_requires_original_eligibility_and_first_partial(self):
        args=fixture();v,a=oracle.independent_features(*args);o=original(args)
        for i,j in ((0,0),(1,4)):
            z=copy.deepcopy(v);z['available'][i,j]=True;z['candidate'][i,j]=z['control'][i,j]=1.
            with self.assertRaises(ValueError):oracle.selection_oracle(o,args[1],z)

    def test_decision_prefixes_exact(self):
        args=fixture();v,a=oracle.independent_features(*args);o=original(args);s,c=oracle.selection_oracle(o,args[1],v)
        for n in range(5):
            m=copy.deepcopy(args[1])
            for key in ('dates','ends','symbols'):m[key]=m[key][:n]
            sp,cp=oracle.selection_oracle({k:z[:n].copy() for k,z in o.items()},m,{k:z[:n].copy() for k,z in v.items()})
            for arm in oracle.ARMS:oracle.exact_array(sp[arm],s[arm][:n],arm);self.assertEqual(cp[arm],c[arm][:n])

    def test_one_two_three_eligible_retention(self):
        for count in (1,2,3):
            args=one();args[3][:]=False;args[3][0,:count]=True;v,a=oracle.independent_features(*args)
            s,c=oracle.selection_oracle(original(args),args[1],v)
            for arm in oracle.ARMS:self.assertEqual(len(c[arm][0]),0 if count==1 else count)

    def test_oracle_does_not_import_new_producer(self):
        text=inspect.getsource(oracle)
        self.assertNotIn('import monthly_volume_slope232_kernel',text)
        self.assertNotIn('import monthly_volume_slope232_model',text)
        self.assertNotIn('from monthly_volume_slope232',text)


class BilateralTests(unittest.TestCase):
    def test_both_fixtures_full_values_audit_all_prefixes(self):
        import monthly_volume_slope232_kernel as producer
        from test_monthly_volume_slope232_kernel import fixture as producer_fixture
        for args in (fixture(),producer_fixture()):
            full=producer.build_features(*args)
            oracle.verify_features(*args,*full)
            for n in range(len(args[1]['dates'])+1):
                logical=producer.build_features(*args,limit=n)
                oracle.verify_features(*args,*logical,limit=n)
                physical=physical_prefix(args,n);out=producer.build_features(*physical)
                oracle.verify_features(*physical,*out)
                oracle.record_check(logical[1],out[1])
                for k in oracle.VALUE_KEYS:oracle.exact_array(out[0][k],full[0][k][:n],k)


if __name__=='__main__':unittest.main()
