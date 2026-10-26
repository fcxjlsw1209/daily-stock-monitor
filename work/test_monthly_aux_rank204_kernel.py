"""Fabricated cache-only tests; no stored strategy arrays or market data read."""
import copy
import unittest

import numpy as np

import monthly_aux_rank204_kernel as k


def make_arrays(support, calm=None, weak=None):
    support = np.asarray(support, dtype=float)
    rows, width = support.shape
    calm = np.ones(rows, bool) if calm is None else np.asarray(calm, dtype=bool)
    weak = np.zeros(rows, bool) if weak is None else np.asarray(weak, dtype=bool)
    cw = np.where(calm, .5, .52)[:, None]
    core = np.zeros((rows, width))
    return dict(baseline=core + (1. - cw)*support, core=core, core_weight=cw,
        state_calm10=calm, state_weak_narrow=weak, extension_available=np.ones((rows, width), bool),
        forecast_available=np.ones((rows, width), bool), eligible=np.ones((rows, width), bool),
        rank_log_excess=np.zeros((rows, width)))


def fixture():
    a = make_arrays([[-.25, -.125, 0., .03125, .25, .375],
        [-.25, -.25, .125, .125, .125, .125], [.125]*6,
        [-.25, -.125, 0., .03125, .25, .375]], calm=[True, False, True, True], weak=[False, True, False, False])
    a['extension_available'][3, 0] = False
    a['forecast_available'][3, 1] = False
    a['eligible'][3, 2] = False
    a['rank_log_excess'][3, 3] = np.nan
    a['rank_log_excess'][3, 4] = np.inf
    return a


class AuxRankTests(unittest.TestCase):
    def test_formula_funding_and_full_size_delta(self):
        a = fixture();v,z = k.build_features(a)
        expected = (a['baseline']-a['core'])/(1.-a['core_weight'])
        np.testing.assert_array_equal(v['support'], expected)
        np.testing.assert_array_equal(v['weight'][0], np.full(6, .5-(.025+.01)))
        np.testing.assert_array_equal(v['weight'][1], np.full(6, .48-.05))
        np.testing.assert_array_equal(v['weight'][3], [.5-.025]*5+[.5-(.025+.01)])
        np.testing.assert_array_equal(v['candidate_delta'][3], v['weight'][3]*(v['candidate'][3]-v['support'][3]))
        self.assertGreater(np.abs(v['candidate_delta'][3]).max(), .01)
        self.assertEqual(z['active_forecast_cells'], 13)

    def test_average_ties_rank_endpoints_and_affine_control(self):
        a=make_arrays([[-.25,-.25,0.,.125,.5,.5]]*2);v,z=k.build_features(a)
        np.testing.assert_array_equal(v['candidate'][1], np.array([.1,.1,.4,.6,.9,.9])-.5)
        s=v['support'][1];q=v['candidate'][1]
        expected=q.min()+(s-s.min())*(q.max()-q.min())/(s.max()-s.min())
        expected[s==s.min()]=q.min();expected[s==s.max()]=q.max()
        np.testing.assert_array_equal(v['control'][1],expected)
        self.assertEqual(v['candidate'][1,0],v['control'][1,0])
        self.assertEqual(v['candidate'][1,-1],v['control'][1,-1])
        self.assertFalse(np.array_equal(v['candidate'][1],v['control'][1]))

    def test_two_levels_copy_exact_with_unequal_ties(self):
        v,z=k.build_features(make_arrays([[-.25,-.25,-.25,.125,.125]]*2))
        np.testing.assert_array_equal(v['candidate'],v['control'])
        np.testing.assert_array_equal(v['candidate_delta'],v['control_delta'])
        self.assertEqual(z['two_level_rows'],2)
        self.assertTrue(all(x['two_levels'] for x in z['event_history']))

    def test_constant_nonzero_rows_are_noop_not_subtraction(self):
        v,z=k.build_features(make_arrays([[.125]*4,[-.25]*4]))
        for key in ('candidate','control','candidate_delta','control_delta'):
            np.testing.assert_array_equal(v[key],np.zeros((2,4)))
        self.assertEqual(z['constant_rows'],2)
        self.assertTrue(np.any(v['support']!=0.))

    def test_singleton_is_constant_noop(self):
        v,z=k.build_features(make_arrays([[.3],[-.2]]))
        self.assertEqual(z['constant_rows'],2)
        np.testing.assert_array_equal(v['candidate_delta'],[[0.],[0.]])
        np.testing.assert_array_equal(v['candidate'],[[0.],[0.]])

    def test_initial_only_delta_neutrality(self):
        v,z=k.build_features(fixture())
        self.assertGreater(np.abs(v['candidate'][0]).max(),0.)
        self.assertGreater(np.abs(v['control'][0]).max(),0.)
        self.assertEqual(z['rank_rows'],4)
        np.testing.assert_array_equal(v['candidate_delta'][0],np.zeros(6))
        np.testing.assert_array_equal(v['control_delta'][0],np.zeros(6))

    def test_no_rank_eligibility_or_forecast_filter(self):
        a=fixture();v,z=k.build_features(a)
        for key in ('eligible','extension_available','forecast_available'):
            b=copy.deepcopy(a);b[key][:]=False;w,zz=k.build_features(b)
            for field in ('support','candidate','control'):
                np.testing.assert_array_equal(w[field],v[field])
            self.assertEqual(zz['checked_cells'],24)
            self.assertFalse(zz['ranking_eligibility_filtered'])

    def test_finite_logrank_only_changes_inherited_funding(self):
        a=make_arrays([[-.25,0.,.25]]*3,calm=[True,False,True],weak=[False,False,True])
        a['rank_log_excess'][:]=[[np.nan,np.inf,-np.inf]]*3
        v,z=k.build_features(a)
        self.assertEqual(z['active_forecast_cells'],0)
        np.testing.assert_array_equal(v['weight'],[[.475]*3,[.48]*3,[.5-(.025+.05)]*3])
        self.assertTrue(np.isfinite(v['candidate']).all())

    def test_cw_three_shapes_and_exact_calm_relationship(self):
        a=fixture();v,z=k.build_features(a)
        for weight in (a['core_weight'][:,0],np.broadcast_to(a['core_weight'],a['baseline'].shape).copy()):
            b=copy.deepcopy(a);b['core_weight']=weight;w,zz=k.build_features(b)
            for key in v:np.testing.assert_array_equal(w[key],v[key])
            self.assertEqual(z,zz)
        b=copy.deepcopy(a);b['core_weight'][1,0]=.5
        with self.assertRaisesRegex(ValueError,'calm'):k.build_features(b)
        b=copy.deepcopy(a);b['core_weight'][0,0]=np.nextafter(.5,1.)
        with self.assertRaisesRegex(ValueError,'calm'):k.build_features(b)

    def test_boolean_inputs_reject_truthy_nonboolean_values(self):
        for key in ('state_calm10','state_weak_narrow','extension_available','forecast_available','eligible'):
            for invalid in (1,0.,'True',None):
                a=fixture();a[key]=a[key].astype(object);a[key].flat[0]=invalid
                with self.subTest(key=key,invalid=invalid):
                    with self.assertRaisesRegex(ValueError,'boolean'):k.build_features(a)

    def test_numeric_inputs_reject_nonreal_and_nonfinite_required(self):
        for key in ('baseline','core','core_weight','rank_log_excess'):
            for invalid in (True,'0.1',1+0j,None):
                a=fixture();a[key]=a[key].astype(object);a[key].flat[0]=invalid
                with self.subTest(key=key,invalid=invalid):
                    with self.assertRaises(ValueError):k.build_features(a)
        for key in ('baseline','core','core_weight'):
            for invalid in (np.nan,np.inf,-np.inf):
                a=fixture();a[key].flat[0]=invalid
                with self.assertRaises(ValueError):k.build_features(a)

    def test_bounds_validate_without_clipping_and_overflow_fail(self):
        a=make_arrays([[-.5-5e-13,0.,.5+5e-13]]*2);v,z=k.build_features(a)
        self.assertLess(v['support'][1,0],-.5)
        self.assertGreater(v['support'][1,2],.5)
        a=make_arrays([[-.5-2e-12,0.,.5]]*2)
        with self.assertRaisesRegex(ValueError,'bounded'):k.build_features(a)
        a=fixture();a['baseline'][0,0]=np.finfo(float).max;a['core'][0,0]=-np.finfo(float).max
        with self.assertRaises(FloatingPointError):k.build_features(a)

    def test_every_prefix_and_physically_truncated_cache_exact(self):
        a=fixture();v,z=k.build_features(a)
        for n in range(5):
            w,zz=k.build_features(a,n)
            small={key:value[:n].copy() for key,value in a.items()}
            x,zx=k.build_features(small)
            for key in v:
                np.testing.assert_array_equal(w[key],v[key][:n]);np.testing.assert_array_equal(w[key],x[key])
                self.assertEqual(w[key].dtype,np.float64)
            self.assertEqual(zz,zx);self.assertEqual(zz['event_history'],z['event_history'][:n])
            self.assertEqual(zz['rank_rows'],n);self.assertEqual(zz['checked_cells'],n*6)

    def test_future_object_poison_all_consumed_inputs(self):
        a=fixture();v,z=k.build_features(a,2)
        b={key:value.astype(object) for key,value in a.items()}
        for value in b.values():value[2:]='FUTURE'
        w,zz=k.build_features(b,2)
        for key in v:np.testing.assert_array_equal(w[key],v[key])
        self.assertEqual(zz,z)
        with self.assertRaises(ValueError):k.build_features(b)
        for value in b.values():value[:]='FUTURE'
        w,zz=k.build_features(b,0)
        self.assertEqual(zz['events'],0)
        self.assertTrue(all(x.shape==(0,6) for x in w.values()))

    def test_immutability_unused_arrays_and_detached_outputs(self):
        a=fixture();before=copy.deepcopy(a);a['unused_future']=object();v,z=k.build_features(a)
        for key,value in before.items():np.testing.assert_array_equal(a[key],value)
        for value in v.values():value[:]=123.
        for key,value in before.items():np.testing.assert_array_equal(a[key],value)

    def test_permutation_equivariance_and_symmetric_audit(self):
        a=fixture();v,z=k.build_features(a);perm=[4,0,5,2,1,3]
        b={key:(value[:,perm] if value.ndim==2 and value.shape[1]==6 else value.copy()) for key,value in a.items()}
        w,zz=k.build_features(b)
        for key in v:np.testing.assert_array_equal(w[key],v[key][:,perm])
        self.assertEqual(z,zz)

    def test_positive_affine_support_equivariance(self):
        s=np.array([[-.25,-.125,0.,.03125,.25,.375]]*2)
        v,z=k.build_features(make_arrays(s));w,zz=k.build_features(make_arrays(s*.5+.0625))
        np.testing.assert_array_equal(v['candidate'],w['candidate'])
        np.testing.assert_array_equal(v['control'],w['control'])
        self.assertFalse(np.array_equal(v['candidate_delta'],w['candidate_delta']))

    def test_no_epsilon_tie_or_nearconstant_snapping(self):
        s=np.array([.125,np.nextafter(.125,1.),.25,.375])
        v,z=k.build_features(make_arrays([s,s]))
        self.assertEqual(z['event_history'][1]['distinct_levels'],4)
        self.assertLess(v['candidate'][1,0],v['candidate'][1,1])
        s=np.array([.125+i*np.spacing(.125) for i in range(4)])
        v,z=k.build_features(make_arrays([s,s]))
        self.assertFalse(z['event_history'][1]['constant'])
        np.testing.assert_array_equal(v['candidate'][1],np.arange(4)/3.-.5)

    def test_distribution_spacing_changes_control_not_ordinal_candidate(self):
        v,z=k.build_features(make_arrays([[-.25,-.125,0.,.25]]*2))
        w,zz=k.build_features(make_arrays([[-.25,.0625,.125,.25]]*2))
        np.testing.assert_array_equal(v['candidate'],w['candidate'])
        self.assertFalse(np.array_equal(v['control'][1,1:-1],w['control'][1,1:-1]))
        np.testing.assert_array_equal(v['control'][1,[0,-1]],w['control'][1,[0,-1]])

    def test_synthetic_full49_domain_not_eligible_subset(self):
        s=np.arange(49,dtype=float)/64.-.375
        a=make_arrays([s,s]);a['eligible'][:,3:]=False;v,z=k.build_features(a)
        np.testing.assert_array_equal(v['candidate'][1],np.arange(49)/48.-.5)
        self.assertEqual(z['checked_cells'],98)
        self.assertEqual(z['active_forecast_cells'],6)

    def test_missing_keys_shape_and_limit_fail(self):
        with self.assertRaises(ValueError):k.build_features([])
        for key in k.REQUIRED:
            a=fixture();del a[key]
            with self.assertRaises(ValueError):k.build_features(a)
        for key in k.REQUIRED:
            a=fixture();a[key]=np.asarray(a[key]).reshape(-1)[:1]
            with self.subTest(key=key):
                with self.assertRaises(ValueError):k.build_features(a)
        for limit in (-1,5,True,1.,'1'):
            with self.assertRaises(ValueError):k.build_features(fixture(),limit)
        a=make_arrays(np.empty((2,0)))
        with self.assertRaises(ValueError):k.build_features(a)


if __name__=='__main__':
    unittest.main(verbosity=2)
