"""Synthetic frozen-selector integration; no original arrays or price loads."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import monthly_core_veto241_kernel as kernel


def study():
    path=kernel.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('core_veto241_test_local67',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    s=study();names=list('ABCDEF');shape=(3,6);z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,False,False]);weak=np.array([False,False,True]);cw=np.where(calm,.5,.52)[:,None]
    core=np.tile([.24,.22,.10,.09,.11,.12],(3,1))
    a=dict(baseline=core.copy(),core=core.copy(),core_weight=cw,eligible=np.ones(shape,bool),
        effects=np.tile([0.,0.,0.,.5,.3,.5],(3,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.4,.3,.2,.195,.19,.189],(3,1)),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    m=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],ends=['2016-08-31','2016-09-30','2016-10-31'],symbols=[names.copy() for _ in range(3)],etfs=[])
    return s,a,m


def physical(s,a,m,n):
    aa={k:v[:n].copy() for k,v in a.items()};mm=deepcopy(m)
    for key in ('dates','ends','symbols'):mm[key]=mm[key][:n]
    return s,aa,mm


def run(args,n=None):
    s,a,m=args;return kernel.decisions(s,a,m,len(m['dates']) if n is None else n)


def same_arrays(case,left,right):
    case.assertEqual(tuple(left),tuple(right))
    for key in left:
        case.assertEqual(left[key].shape,right[key].shape);case.assertEqual(left[key].dtype,right[key].dtype)
        case.assertEqual(left[key].tobytes(),right[key].tobytes(),key)


class KernelTests(unittest.TestCase):
    def test_exact_schema_and_frozen_original_calls_once(self):
        s,a,m=fixture()
        with patch.object(s,'modified_pre_score',wraps=s.modified_pre_score) as pre,patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as episode:
            orders,scores,pres,audit=run((s,a,m));self.assertEqual(pre.call_count,1);self.assertEqual(episode.call_count,1)
        self.assertEqual(tuple(orders),kernel.ARMS);self.assertEqual(tuple(scores),kernel.ARMS);self.assertEqual(tuple(pres),kernel.ARMS)
        self.assertEqual(audit['schema'],'monthly_core_veto241_decisions_v1');self.assertEqual(tuple(audit['counts']),kernel.COUNT_FIELDS)
        self.assertEqual(audit['counts'],dict(events=3,symbols_per_event=6,eligible_cells=18,initial_partial_events=1,three_seat_events=3,boundary_cells=9,core_compared_cells=6,vetoed_cells=2,vetoed_winner_events=2,candidate_changed_holdings=2,control_changed_holdings=2,original_episode_calls=1,new_model_fits=0))
        self.assertTrue(all(tuple(e)==kernel.EVENT_KEYS for e in audit['events']));json.dumps(audit,allow_nan=False)

    def test_veto_selects_admissible_next_winner_and_keeps_full_F(self):
        s,a,m=fixture();o,sc,p,au=run((s,a,m))
        self.assertEqual(o[kernel.ARMS[0]][1]['symbols'],['A','B','D'])
        self.assertEqual(o[kernel.ARMS[1]][1]['symbols'],['A','B','E'])
        self.assertEqual(o[kernel.ARMS[2]][1]['symbols'],['A','B','C'])
        self.assertEqual(sc[kernel.ARMS[0]].tobytes(),sc[kernel.ARMS[1]].tobytes())
        self.assertGreater(sc[kernel.ARMS[1]][1,3],sc[kernel.ARMS[1]][1,4])
        self.assertEqual(au['events'][1]['vetoed_indices'],[3]);self.assertEqual(au['events'][1]['admitted_indices'],[2,4])
        np.testing.assert_array_equal(sc[kernel.ARMS[2]][1:],p[kernel.ARMS[2]][1:])

    def test_named_partial_all_arms_exact_incumbent_even_when_F_differs_P(self):
        o,sc,p,audit=run(fixture());self.assertFalse(np.array_equal(sc[kernel.ARMS[0]][0],p[kernel.ARMS[0]][0]))
        for arm in kernel.ARMS:
            self.assertEqual(o[arm][0],o[kernel.ARMS[0]][0]);np.testing.assert_array_equal(sc[arm][0],sc[kernel.ARMS[0]][0])
        e=audit['events'][0];self.assertTrue(e['initial_partial']);self.assertEqual(e['admitted_indices'],e['boundary_indices']);self.assertEqual(e['vetoed_indices'],[])

    def test_equal_core_admits_original_winner_exactly(self):
        s,a,m=fixture();a['core'][:,3]=a['core'][:,2];a['baseline']=a['core'].copy()
        o,sc,p,au=run((s,a,m));self.assertEqual(o[kernel.ARMS[1]],o[kernel.ARMS[0]])
        self.assertEqual(au['counts']['vetoed_winner_events'],0)

    def test_nextafter_core_boundary_is_strict_no_epsilon(self):
        for direction,expected in ((-np.inf,'E'),(np.inf,'D')):
            s,a,m=fixture();a['core'][:,3]=np.nextafter(a['core'][:,2],direction);a['baseline']=a['core'].copy()
            o,_,_,_=run((s,a,m));self.assertIn(expected,o[kernel.ARMS[1]][1]['symbols'])

    def test_prethird_invariant_prevents_loss_of_third_seat(self):
        s,a,m=fixture();a['core'][:,3:]=.01;a['baseline']=a['core'].copy()
        o,_,_,au=run((s,a,m));self.assertEqual(o[kernel.ARMS[1]][1]['symbols'],['A','B','C'])
        self.assertEqual(au['events'][1]['admitted_indices'],[2]);self.assertEqual(len(o[kernel.ARMS[1]][1]['symbols']),3)

    def test_top_two_locked_even_when_third_F_is_higher(self):
        s,a,m=fixture();a['pre_score'][:,0]=.215;a['pre_score'][:,1]=.210;a['core'][:,3]=.11;a['baseline']=a['core'].copy()
        o,_,_,_=run((s,a,m));self.assertEqual(o[kernel.ARMS[1]][1]['symbols'],['D','A','B'])
        self.assertEqual(set(o[kernel.ARMS[1]][1]['symbols']),{'A','B','D'})

    def test_rank_six_excluded_despite_high_core_and_effect(self):
        s,a,m=fixture();a['core'][:,5]=.25;a['baseline']=a['core'].copy();a['effects'][:,5]=100.
        o,sc,p,au=run((s,a,m));self.assertNotIn(5,au['events'][1]['boundary_indices'])
        self.assertEqual(sc[kernel.ARMS[1]][1,5],p[kernel.ARMS[1]][1,5]);self.assertNotIn('F',o[kernel.ARMS[1]][1]['symbols'])

    def test_gap_excludes_rank_five(self):
        s,a,m=fixture();a['pre_score'][:,4]=.17;a['pre_score'][:,5]=.16
        o,_,_,au=run((s,a,m));self.assertEqual(au['events'][1]['boundary_indices'],[2,3])
        self.assertEqual(o[kernel.ARMS[1]][1]['symbols'],['A','B','C'])

    def test_alphabetical_ties_in_pre_and_final(self):
        s,a,m=fixture();a['pre_score'][:,2:]=.2;a['effects'][:]=0.;a['core'][:]=.1;a['baseline']=a['core'].copy()
        for names in m['symbols']:names[2:]=['Z','Y','X','W']
        o,_,_,au=run((s,a,m));self.assertEqual(au['events'][1]['pre_order'],[0,1,5,4,3,2])
        for arm in kernel.ARMS:self.assertEqual(o[arm][1]['symbols'],['A','B','W'])

    def test_zero_one_two_eligible_preserve_cash_or_halves(self):
        for count in (0,1,2):
            s,a,m=fixture();a['eligible'][:,count:]=False;o,_,_,au=run((s,a,m))
            for arm in kernel.ARMS:
                for order in o[arm]:self.assertEqual(order['symbols'],['A','B'] if count==2 else [])
            self.assertEqual(au['counts']['boundary_cells'],0)
            self.assertTrue(all(e['pre_third'] is None and e['core_anchor'] is None and e['original_winner'] is None for e in au['events']))

    def test_all_ordinary_rows_keep_exact_original_postscore(self):
        o,sc,p,au=run(fixture());same_arrays(self,{a:sc[a] for a in kernel.ARMS[:2]},{a:sc[kernel.ARMS[0]].copy() for a in kernel.ARMS[:2]})
        for arm in kernel.ARMS:np.testing.assert_array_equal(p[arm],p[kernel.ARMS[0]])

    def test_logical_and_physical_prefixes_exact(self):
        s,a,m=fixture();o,sc,p,au=run((s,a,m))
        for n in range(4):
            oo,ss,pp,aa=run((s,a,m),n);po,ps,pr,pa=run(physical(s,a,m,n))
            self.assertEqual(oo,po);same_arrays(self,ss,ps);same_arrays(self,pp,pr);self.assertEqual(aa,pa)
            self.assertEqual(aa['events'],au['events'][:n]);self.assertEqual(aa['counts']['original_episode_calls'],1)
            for arm in kernel.ARMS:self.assertEqual(oo[arm],o[arm][:n]);np.testing.assert_array_equal(ss[arm],sc[arm][:n]);np.testing.assert_array_equal(pp[arm],p[arm][:n])

    def test_future_numeric_and_metadata_poison_ignored(self):
        s,a,m=fixture();expected=run((s,a,m),1)
        for key in ('baseline','core','effects','pre_score','raw_local_constant','raw_local_linear','raw_source_log','rank_log_excess','old_ml4'):a[key][1:]=np.nan
        m['dates'][1:]=['bad','bad'];m['ends'][1:]=['bad','bad'];m['symbols'][1:]=[['bad'],['bad']]
        actual=run((s,a,m),1);self.assertEqual(actual[0],expected[0]);same_arrays(self,actual[1],expected[1]);same_arrays(self,actual[2],expected[2]);self.assertEqual(actual[3],expected[3])

    def test_original_arrays_and_metadata_not_mutated(self):
        s,a,m=fixture();aa=deepcopy(a);mm=deepcopy(m);run((s,a,m))
        same_arrays(self,a,aa);self.assertEqual(m,mm)

    def test_mutating_pre_callable_rejected(self):
        s,a,m=fixture();original=s.modified_pre_score
        def mutated(spec,arrays,n):
            out=original(spec,arrays,n);arrays['effects'][0,0]+=.01;return out
        with patch.object(s,'modified_pre_score',side_effect=mutated),self.assertRaises(ValueError):run((s,a,m))

    def test_nonfinite_required_source_or_pre_rejected(self):
        for key in ('baseline','core','effects','pre_score','core_weight'):
            s,a,m=fixture();a[key].flat[0]=np.nan
            with self.subTest(key=key),self.assertRaises(ValueError):run((s,a,m))

    def test_nonfinite_original_post_rejected(self):
        s,a,m=fixture();original=s.ep.select_boundary
        def invalid(*args):
            ch,sc=original(*args);sc[0,0]=np.inf;return ch,sc
        with patch.object(s.ep,'select_boundary',side_effect=invalid),self.assertRaises(ValueError):run((s,a,m))

    def test_common_positive_state_core_coefficient_required(self):
        for bad in (.51,0.,-.5):
            s,a,m=fixture();a['core_weight'][1,0]=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):run((s,a,m))
        s,a,m=fixture();a['core_weight']=np.broadcast_to(a['core_weight'],a['core'].shape).copy();a['core_weight'][1,3]=.5
        with self.assertRaises(ValueError):run((s,a,m))

    def test_core_weight_vector_and_matrix_representations_agree(self):
        s,a,m=fixture();base=run((s,a,m));a['core_weight']=a['core_weight'][:,0].copy();vector=run((s,a,m));self.assertEqual(base[0],vector[0]);self.assertEqual(base[3],vector[3])
        a['core_weight']=np.broadcast_to(a['core_weight'][:,None],a['core'].shape).copy();matrix=run((s,a,m));self.assertEqual(base[3],matrix[3]);same_arrays(self,base[1],matrix[1])

    def test_boolean_numeric_and_nonboolean_E_rejected(self):
        s,a,m=fixture();a['core']=a['core'].astype(object);a['core'][0,0]=True
        with self.assertRaises(ValueError):run((s,a,m))
        s,a,m=fixture();a['eligible']=a['eligible'].astype(float)
        with self.assertRaises(ValueError):run((s,a,m))

    def test_bad_axes_duplicate_symbols_or_prefix_rejected(self):
        s,a,m=fixture();m['symbols'][1][1]=m['symbols'][1][0]
        with self.assertRaises(ValueError):run((s,a,m))
        for n in (-1,4,True):
            with self.subTest(n=n),self.assertRaises(ValueError):run(fixture(),n)

    def test_dependencies_are_original_selector_sources(self):
        files=kernel.dependency_files();self.assertEqual(len(files),5)
        self.assertTrue(all(p.is_file() for p in files));self.assertIn(kernel.ROOT/'outputs/monthly_daily_episodes/episodes.py',files)
        self.assertEqual(kernel.POLICY,dict(budget=.05,ceiling=5,gap=.02))


if __name__=='__main__':unittest.main(verbosity=2)
