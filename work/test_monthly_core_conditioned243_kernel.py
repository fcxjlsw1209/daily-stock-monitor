"""Synthetic exact-rational projection and frozen Local67 integration checks."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import monthly_core_conditioned243_kernel as kernel


def study():
    path=kernel.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('core_conditioned243_test_local67',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    s=study();names=list('ABCDEF');shape=(4,6);z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,True,False,False]);weak=np.array([False,False,False,True]);cw=np.where(calm,.5,.52)[:,None]
    x=np.tile([.5,.25,.125,0.,-.25,-.5],(4,1));core=cw*x
    support=np.tile([.125,-.125,.25,0.,-.25,.0625],(4,1))
    a=dict(baseline=core+(1.-cw)*support,core=core,core_weight=cw,eligible=np.ones(shape,bool),
        effects=np.tile([0.,0.,0.,.5,.3,.5],(4,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.4,.3,.2,.195,.19,.189],(4,1)),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    m=dict(dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],
        ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'],symbols=[names.copy() for _ in range(4)],etfs=['F'])
    return s,a,m


def physical(s,a,m,n):
    aa={k:np.asarray(v)[:n].copy() for k,v in a.items()};mm=deepcopy(m)
    for key in ('dates','ends','symbols'):mm[key]=mm[key][:n]
    return s,aa,mm


def run(args,n=None):
    s,a,m=args;return kernel.decisions(s,a,m,len(m['dates']) if n is None else n)


def same_arrays(case,left,right):
    case.assertEqual(tuple(left),tuple(right))
    for key in left:
        case.assertEqual(left[key].shape,right[key].shape);case.assertEqual(left[key].dtype,right[key].dtype)
        case.assertEqual(left[key].tobytes(),right[key].tobytes(),key)


def unpack(pair):return Fraction(int(pair[0]),int(pair[1]))


class ProjectionTests(unittest.TestCase):
    def test_exact_fraction_orthogonality_and_ties(self):
        x=np.array([-.5,-.25,0.,.25,.5]);s=np.array([.25,-.125,0.,-.125,.25])
        p=kernel.project(x,s);self.assertEqual(p['branch'],'ranked');self.assertEqual(p['beta'],['0','1'])
        e=[unpack(v) for v in p['residual']];mx=unpack(p['mean_core'])
        self.assertEqual(sum(e),0);self.assertEqual(sum((Fraction(float(v))-mx)*z for v,z in zip(x,e)),0)
        np.testing.assert_array_equal(p['candidate'],[.375,-.375,0.,-.375,.375])
        np.testing.assert_array_equal(p['candidate'],p['control'])

    def test_nonzero_slope_removes_core_ordering(self):
        x=np.array([-.5,-.25,0.,.25,.5]);s=x/2+np.array([.125,-.0625,-.125,-.0625,.125])
        p=kernel.project(x,s);self.assertEqual(p['beta'],['1','2'])
        self.assertFalse(np.array_equal(p['candidate'],p['control']))
        e=[unpack(v) for v in p['residual']]
        self.assertEqual(sum(e),0);self.assertEqual(sum(Fraction(float(v))*z for v,z in zip(x,e)),0)

    def test_constant_core_keeps_targets_zero_not_support_replacement(self):
        p=kernel.project([.125]*4,[.25,.125,-.125,0.])
        self.assertEqual(p['branch'],'constant_core');self.assertIsNone(p['beta'])
        self.assertEqual(p['residual'],[None]*4);self.assertEqual(p['residual_display'],[None]*4)
        np.testing.assert_array_equal(p['candidate'],np.zeros(4));np.testing.assert_array_equal(p['control'],np.zeros(4))

    def test_exact_affine_and_constant_support_are_constant_residual(self):
        x=np.array([-.5,-.25,0.,.25,.5])
        for s in (x*.5+.125,np.full(5,.125)):
            p=kernel.project(x,s);self.assertEqual(p['branch'],'constant_residual');self.assertEqual(p['residual'],[['0','1']]*5)
            np.testing.assert_array_equal(p['candidate'],np.zeros(5));np.testing.assert_array_equal(p['control'],np.zeros(5))

    def test_singleton_and_two_points_are_degenerate(self):
        self.assertEqual(kernel.project([.25],[.125])['branch'],'constant_core')
        self.assertEqual(kernel.project([-.25,.25],[.25,-.125])['branch'],'constant_residual')

    def test_nextafter_near_collinear_not_rounded_away(self):
        x=np.array([-.25,0.,.125,.25]);s=x.copy();s[2]=np.nextafter(s[2],np.inf)
        p=kernel.project(x,s);self.assertEqual(p['branch'],'ranked')
        self.assertGreater(len(set(tuple(v) for v in p['residual'])),1)
        self.assertFalse(np.array_equal(p['candidate'],np.zeros(4)))
        self.assertEqual(sum(unpack(v) for v in p['residual']),0)

    def test_display_underflow_does_not_create_false_ties(self):
        tiny=np.nextafter(0.,1.);p=kernel.project([-.5,0.,.5],[0.,0.,tiny])
        self.assertEqual(p['branch'],'ranked');self.assertTrue(all(v==0. for v in p['residual_display']))
        np.testing.assert_array_equal(p['candidate'],[.25,-.5,.25])
        self.assertNotEqual(p['residual'][0],p['residual'][1])

    def test_affine_core_and_positive_support_invariances(self):
        x=np.array([-.25,-.125,0.,.125,.25]);s=np.array([.125,-.0625,0.,-.125,.0625]);p=kernel.project(x,s)
        for xx,ss in ((x*2+.125,s),(-x,s),(x,s*2+.125)):
            q=kernel.project(xx,ss);np.testing.assert_array_equal(q['candidate'],p['candidate']);np.testing.assert_array_equal(q['control'],p['control'])

    def test_permutation_equivariance_and_tie_ranks(self):
        x=np.array([-.5,-.25,0.,.25,.5]);s=np.array([.25,0.,-.25,0.,.25]);p=kernel.project(x,s);perm=[3,0,4,2,1]
        q=kernel.project(x[perm],s[perm]);np.testing.assert_array_equal(q['candidate'],p['candidate'][perm]);np.testing.assert_array_equal(q['control'],p['control'][perm])

    def test_ranked_residual_is_not_claimed_orthogonal(self):
        p=kernel.project([-.5,-.25,0.,.125,.5],[.125,.25,-.125,0.,-.25])
        x=[Fraction(v) for v in [-.5,-.25,0.,.125,.5]];mx=sum(x)/5
        cov=sum((v-mx)*Fraction(float(q)) for v,q in zip(x,p['candidate']))
        self.assertNotEqual(cov,0)

    def test_invalid_projection_inputs_fail(self):
        for x,s in (([],[]),([0.],[0.,1.]),([True],[.1]),([0.],['.1']),([0.],[np.nan]),([0.],[np.inf]),([0.],[10**1000])):
            with self.subTest(x=x),self.assertRaises((ValueError,TypeError)):kernel.project(x,s)


class DecisionTests(unittest.TestCase):
    def test_schema_counts_and_one_original_episode_per_arm(self):
        s,a,m=fixture()
        with patch.object(s,'modified_pre_score',wraps=s.modified_pre_score) as pre,patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:
            o,sc,p,au=run((s,a,m));self.assertEqual(pre.call_count,1);self.assertEqual(ep.call_count,3)
        self.assertEqual(tuple(o),kernel.ARMS);self.assertEqual(tuple(au),('schema','counts','events'))
        self.assertEqual(tuple(au['counts']),kernel.COUNT_FIELDS);self.assertEqual(au['counts']['descriptive_ols_projections'],4)
        self.assertEqual(au['counts']['ranked_cells'],24);self.assertEqual(au['counts']['eligible_cells'],24)
        self.assertEqual(au['counts']['initial_partial_events'],1);self.assertEqual(au['counts']['original_episode_calls'],3)
        self.assertEqual(au['counts']['new_model_fits'],0);self.assertEqual(au['counts']['predictive_model_fits'],0)
        for event in au['events']:
            self.assertEqual(tuple(event),kernel.EVENT_KEYS)
            for cell in event['cells']:self.assertEqual(tuple(cell),kernel.CELL_KEYS)
        json.dumps(au,allow_nan=False)

    def test_incumbent_exact_original_and_partial_all_arms(self):
        s,a,m=fixture();o,sc,p,au=run((s,a,m));old,full=s.choices(s.configs()[4],a,m,len(m['dates']),policy=dict(s.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],s.rp.make_orders(old,full,m));np.testing.assert_array_equal(sc[kernel.ARMS[0]],full)
        for arm in kernel.ARMS:
            self.assertEqual(o[arm][0],o[kernel.ARMS[0]][0]);np.testing.assert_array_equal(sc[arm][0],sc[kernel.ARMS[0]][0]);np.testing.assert_array_equal(p[arm][0],p[kernel.ARMS[0]][0])
        self.assertTrue(all(c['candidate_delta']==0. and c['control_delta']==0. for c in au['events'][0]['cells']))
        self.assertEqual(au['events'][0]['branch'],'ranked')

    def test_exact_funding_original_active_mask(self):
        s,a,m=fixture();a['extension_available'][:]=True;a['forecast_available'][:]=True
        a['eligible'][1,1]=False;a['forecast_available'][1,2]=False
        o,sc,p,au=run((s,a,m))
        for i,event in enumerate(au['events']):
            for j,c in enumerate(event['cells']):
                active=bool(a['state_calm10'][i] and not a['state_weak_narrow'][i] and a['forecast_available'][i,j] and a['eligible'][i,j] and np.isfinite(a['rank_log_excess'][i,j]))
                self.assertEqual(c['active'],active)
                w=(1.-float(a['core_weight'][i,0]))-(.025*a['state_calm10'][i]+.05*a['state_weak_narrow'][i]+.01*active)
                self.assertEqual(c['weight'],w)
                if i:
                    self.assertEqual(c['candidate_delta'],w*(c['candidate_target']-c['support']))
                    self.assertEqual(c['control_delta'],w*(c['control_target']-c['support']))
                    for arm,key in ((kernel.ARMS[1],'candidate_delta'),(kernel.ARMS[2],'control_delta')):
                        expected=p[kernel.ARMS[0]][i,j]+c[key] if c[key]!=0. else p[kernel.ARMS[0]][i,j]
                        self.assertEqual(p[arm][i,j],expected)

    def test_full_domain_before_eligibility_and_etf_labels(self):
        s,a,m=fixture();base=run((s,a,m))[3];a['eligible'][:,3:]=False;m['etfs']=list('ABCDEF')
        changed=run((s,a,m))[3]
        for original,event in zip(base['events'],changed['events']):
            for left,right in zip(original['cells'],event['cells']):
                for key in ('core_rank','support','residual','candidate_target','control_target'):self.assertEqual(left[key],right[key])
        self.assertEqual(changed['counts']['ranked_cells'],24)

    def test_constant_core_noop_preserves_nonzero_support(self):
        s,a,m=fixture();a['state_calm10'][:]=True;a['core_weight'][:]=.5;a['core'][:]=.0625
        a['baseline']=a['core']+.5*np.tile([.25,.125,0.,-.125,.0625,-.25],(4,1))
        o,sc,p,au=run((s,a,m));self.assertEqual(au['counts']['constant_core_events'],4);self.assertEqual(au['counts']['descriptive_ols_projections'],0)
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]]);np.testing.assert_array_equal(p[arm],p[kernel.ARMS[0]]);np.testing.assert_array_equal(sc[arm],sc[kernel.ARMS[0]])

    def test_exact_affine_source_noop_both_arms(self):
        s,a,m=fixture();a['state_calm10'][:]=True;a['core_weight'][:]=.5
        x=np.tile([-.5,-.25,0.,.25,.375,.5],(4,1));a['core']=.5*x;a['baseline']=a['core']+.5*(.125+.25*x)
        o,sc,p,au=run((s,a,m));self.assertEqual(au['counts']['constant_residual_events'],4);self.assertEqual(au['counts']['descriptive_ols_projections'],4)
        self.assertEqual(au['counts']['candidate_nonzero_delta_cells'],0);self.assertEqual(au['counts']['control_nonzero_delta_cells'],0)
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]]);np.testing.assert_array_equal(p[arm],p[kernel.ARMS[0]])

    def test_single_security_shape_and_cash(self):
        s,a,m=fixture()
        for key,value in list(a.items()):
            if value.ndim==2 and value.shape[1]==6:a[key]=value[:,:1]
        m['symbols']=[['A'] for _ in m['dates']];o,sc,p,au=run((s,a,m))
        self.assertEqual(au['counts']['constant_core_events'],4)
        self.assertTrue(all(order['symbols']==[] for arm in o.values() for order in arm))

    def test_zero_one_two_eligible_preserve_cash_or_halves(self):
        for count in (0,1,2):
            s,a,m=fixture();a['eligible'][:,count:]=False;o,_,_,au=run((s,a,m))
            for arm in o.values():
                for order in arm:self.assertEqual(set(order['symbols']),{'A','B'} if count==2 else set())
            self.assertEqual(au['counts']['descriptive_ols_projections'],4)

    def test_original_support_and_effects_pass_to_every_episode(self):
        s,a,m=fixture();original=s.ep.select_boundary;calls=[]
        def observed(score,good,names,effect,source,policy):
            np.testing.assert_array_equal(source['baseline'],a['baseline']);np.testing.assert_array_equal(source['core'],a['core'])
            np.testing.assert_array_equal(effect,a['effects']);np.testing.assert_array_equal(good,a['eligible']);calls.append(score.copy())
            return original(score,good,names,effect,source,policy)
        with patch.object(s.ep,'select_boundary',side_effect=observed):o,sc,p,au=run((s,a,m))
        self.assertEqual(len(calls),3)
        for arm,score in zip(kernel.ARMS,calls):np.testing.assert_array_equal(score,p[arm])

    def test_logical_and_physical_prefixes_exact(self):
        s,a,m=fixture();o,sc,p,au=run((s,a,m))
        for n in range(5):
            oo,ss,pp,aa=run((s,a,m),n);po,ps,pr,pa=run(physical(s,a,m,n))
            self.assertEqual(oo,po);same_arrays(self,ss,ps);same_arrays(self,pp,pr);self.assertEqual(aa,pa)
            self.assertEqual(aa['events'],au['events'][:n]);self.assertEqual(aa['counts']['original_episode_calls'],3)
            for arm in kernel.ARMS:self.assertEqual(oo[arm],o[arm][:n]);np.testing.assert_array_equal(ss[arm],sc[arm][:n]);np.testing.assert_array_equal(pp[arm],p[arm][:n])

    def test_future_poison_ignored(self):
        s,a,m=fixture();expected=run((s,a,m),1)
        for key in ('baseline','core','effects','pre_score','raw_local_constant','raw_local_linear','raw_source_log','rank_log_excess','old_ml4'):a[key][1:]=np.nan
        a['core_weight']=a['core_weight'].astype(object);a['core_weight'][1:]=None
        m['dates'][1:]=['bad']*3;m['ends'][1:]=['bad']*3;m['symbols'][1:]=[['bad']]*3
        actual=run((s,a,m),1);self.assertEqual(actual[0],expected[0]);same_arrays(self,actual[1],expected[1]);same_arrays(self,actual[2],expected[2]);self.assertEqual(actual[3],expected[3])

    def test_core_weight_shape_variants(self):
        s,a,m=fixture();expected=run((s,a,m))
        for cw in (a['core_weight'][:,0],np.broadcast_to(a['core_weight'],a['core'].shape).copy()):
            aa=deepcopy(a);aa['core_weight']=cw;actual=run((s,aa,m));self.assertEqual(actual[0],expected[0]);self.assertEqual(actual[3],expected[3]);same_arrays(self,actual[1],expected[1])

    def test_no_source_or_metadata_mutation(self):
        s,a,m=fixture();before=deepcopy(a);mm=deepcopy(m);run((s,a,m));same_arrays(self,a,before);self.assertEqual(m,mm)

    def test_mutating_original_callable_rejected(self):
        s,a,m=fixture();original=s.modified_pre_score
        def mutated(spec,arrays,n):
            out=original(spec,arrays,n);arrays['effects'][0,0]+=.01;return out
        with patch.object(s,'modified_pre_score',side_effect=mutated),self.assertRaises(ValueError):run((s,a,m))

    def test_nonfinite_and_nonboolean_required_sources_fail(self):
        for key in ('baseline','core','effects','pre_score','core_weight'):
            s,a,m=fixture();a[key].flat[0]=np.inf
            with self.subTest(key=key),self.assertRaises((ValueError,FloatingPointError)):run((s,a,m))
        for key in ('baseline','core','effects','core_weight'):
            s,a,m=fixture();a[key]=a[key].astype(object);a[key].flat[0]=True
            with self.subTest(key=key),self.assertRaises(ValueError):run((s,a,m))

    def test_invalid_axes_identity_state_and_bounded_source_fail(self):
        for kind in ('duplicate','date','end','state','eligible','weight','support','core'):
            s,a,m=fixture()
            if kind=='duplicate':m['symbols'][0][1]='A'
            elif kind=='date':m['dates'][0]='2016-8-17'
            elif kind=='end':m['ends'][0]=m['dates'][0]
            elif kind=='state':a['state_calm10']=a['state_calm10'][:,None]
            elif kind=='eligible':a['eligible']=a['eligible'].astype(int)
            elif kind=='weight':a['core_weight'][0,0]=.51
            elif kind=='support':a['baseline'][0,0]=100.
            else:a['core'][0,0]=1.;a['baseline'][0,0]=1.
            with self.subTest(kind=kind),self.assertRaises(ValueError):run((s,a,m))

    def test_postscore_nonfinite_rejected(self):
        s,a,m=fixture();original=s.ep.select_boundary
        def invalid(*args):
            choices,score=original(*args);score[0,0]=np.inf;return choices,score
        with patch.object(s.ep,'select_boundary',side_effect=invalid),self.assertRaises(ValueError):run((s,a,m))

    def test_regular_first_row_not_implicitly_partial(self):
        s,a,m=fixture();a={k:v[1:].copy() for k,v in a.items()}
        for key in ('dates','ends','symbols'):m[key]=m[key][1:]
        o,sc,p,au=run((s,a,m));self.assertEqual(au['counts']['initial_partial_events'],0)
        self.assertTrue(any(c['candidate_delta']!=0. for c in au['events'][0]['cells']))

    def test_final_choices_keep_original_boundary_not_global_post_rank(self):
        s,a,m=fixture();a['pre_score'][:]=[.3,.21,.2,.195,.19,.189];a['effects'][:]=[0.,0.,.5,.5,0.,0.]
        o,sc,p,au=run((s,a,m));i=1;names=m['symbols'][i]
        global_post=sorted(range(6),key=lambda j:(-sc[kernel.ARMS[0]][i,j],names[j]))[:3]
        self.assertNotEqual(au['events'][i]['incumbent_choices'],global_post)
        for arm,key in zip(kernel.ARMS,('incumbent_choices','candidate_choices','control_choices')):
            order=sorted(range(6),key=lambda j:(-p[arm][i,j],names[j]));third=order[2]
            boundary=[j for j in order[2:5] if p[arm][i,third]-p[arm][i,j]<=.02]
            winner=min(boundary,key=lambda j:(-sc[arm][i,j],names[j]))
            expected=sorted(order[:2]+[winner],key=lambda j:(-sc[arm][i,j],names[j]))
            self.assertEqual(au['events'][i][key],expected)

    def test_dependency_source_pin_and_frozen_files_list(self):
        self.assertEqual(kernel.SOURCE_SHA,kernel.sha256(kernel.SOURCE.read_bytes()).hexdigest())
        self.assertEqual(len(kernel.dependency_files()),6);self.assertIn(kernel.SOURCE,kernel.dependency_files())


if __name__=='__main__':unittest.main(verbosity=2)
