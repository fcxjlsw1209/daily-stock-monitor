import unittest
import numpy as np
from monthly_formation142_features import formation_scenarios,centered_rank


def fixture(n=420):
    t=np.arange(n,dtype=float)[:,None]
    rates=np.array([.001,.002,.003])[None,:]+.001*np.sin(t/5+np.arange(3)[None,:])
    return 100*np.exp(np.cumsum(rates,axis=0))


class FormationTests(unittest.TestCase):
    def test_average_ties_and_missing(self):
        np.testing.assert_allclose(centered_rank([3,1,1,np.nan]),[.5,0,0,0])
    def test_all_missing_neutral(self):
        np.testing.assert_array_equal(centered_rank([np.nan,np.nan]),[0,0])
    def test_calm_blocks_cover210_once_and_stop42(self):
        x=formation_scenarios(fixture(),350,calm=True)
        self.assertTrue(x['ready']);self.assertEqual(x['scenario_core_ranks'].shape,(10,3))
        deleted=[i for b in x['blocks'] for i in range(b['start_price_index']+1,b['end_price_index']+1)]
        self.assertEqual(deleted,list(range(99,309)))
        self.assertEqual(x['core_weight'],.5)
    def test_other_blocks_cover252(self):
        x=formation_scenarios(fixture(),350,calm=False)
        self.assertEqual(x['scenario_core_ranks'].shape,(12,3))
        self.assertEqual(x['blocks'][0]['start_price_index'],56)
        self.assertEqual(x['blocks'][-1]['end_price_index'],308)
        self.assertEqual(x['core_weight'],.52)
    def test_delete_ratio_matches_independent_daily_product(self):
        p=fixture();x=formation_scenarios(p,350,calm=True)
        daily=p[99:309]/p[98:308]
        for k in range(10):
            keep=np.ones(210,dtype=bool);keep[k*21:(k+1)*21]=False
            expected=np.prod(daily[keep],axis=0)-1
            np.testing.assert_array_equal(centered_rank(expected),x['scenario_core_ranks'][k])
    def test_other_divisor_is_original63_sample_std(self):
        p=fixture();x=formation_scenarios(p,350,calm=False)
        v=np.std(p[288:351]/p[287:350]-1,axis=0,ddof=1)
        expected=(p[308]/p[56]-1)/v
        np.testing.assert_array_equal(centered_rank(expected),x['core_ranks'])
    def test_no_mutation(self):
        p=fixture();old=p.copy();formation_scenarios(p,350,calm=False)
        np.testing.assert_array_equal(p,old)
    def test_future_object_suffix_not_converted(self):
        p=fixture().astype(object);x=formation_scenarios(p,350,calm=False)
        p[351:]='future is unreadable';y=formation_scenarios(p,350,calm=False)
        np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])
    def test_calm_latest42_not_read(self):
        p=fixture().astype(object);x=formation_scenarios(p,350,calm=True)
        p[309:351]='ignored';y=formation_scenarios(p,350,calm=True)
        np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])
    def test_prior_to_formation_not_read(self):
        p=fixture().astype(object);x=formation_scenarios(p,350,calm=True)
        p[:98]='old';y=formation_scenarios(p,350,calm=True)
        np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])
    def test_internal_gap_requires_fallback(self):
        p=fixture();p[180,0]=np.nan;x=formation_scenarios(p,350,calm=True)
        self.assertFalse(x['ready']);self.assertEqual(x['reason'],'incomplete_originally_measurable_formation')
    def test_missing_endpoint_remains_neutral(self):
        p=fixture();p[98,0]=np.nan;x=formation_scenarios(p,350,calm=True)
        self.assertTrue(x['ready']);self.assertEqual(x['core_ranks'][0],0)
        np.testing.assert_array_equal(x['scenario_core_ranks'][:,0],0)
    def test_zero_volatility_is_missing(self):
        p=fixture();p[:,0]=100;x=formation_scenarios(p,350,calm=False)
        self.assertTrue(x['ready']);self.assertFalse(x['original_finite_mask'][0])
    def test_short_history_fallback(self):
        self.assertEqual(formation_scenarios(fixture(),200,calm=True)['reason'],'insufficient_formation_history')
    def test_infinity_error(self):
        p=fixture();p[200,0]=np.inf
        with self.assertRaises(ValueError):formation_scenarios(p,350,calm=True)
    def test_finite_price_underflow_is_error_not_new_missing_rank(self):
        p=np.full((420,3),100.)
        p[98,0]=1e300;p[99:309,0]=1e-300
        with self.assertRaises(FloatingPointError):formation_scenarios(p,350,calm=True)
    def test_finite_price_overflow_is_error(self):
        p=np.full((420,3),100.)
        p[98,0]=1e-300;p[308,0]=1e300
        with self.assertRaises(FloatingPointError):formation_scenarios(p,350,calm=True)
    def test_internal_block_overflow_is_error(self):
        p=np.full((420,3),100.)
        p[119,0]=1e-300;p[140,0]=1e300
        with self.assertRaises(FloatingPointError):formation_scenarios(p,350,calm=True)
    def test_internal_block_underflow_is_error(self):
        p=np.full((420,3),100.)
        p[119,0]=1e300;p[140,0]=1e-300
        with self.assertRaises(FloatingPointError):formation_scenarios(p,350,calm=True)
    def test_fixed_unit_rescaling_preserves_scenarios(self):
        p=fixture()
        for calm in [True,False]:
            x=formation_scenarios(p,350,calm=calm)
            y=formation_scenarios(p*np.array([.01,100,3.]),350,calm=calm)
            np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])
    def test_bad_shape_index_state(self):
        for index in [True,350.0,-1,999]:
            with self.assertRaises(ValueError):formation_scenarios(fixture(),index,calm=True)
        with self.assertRaises(ValueError):formation_scenarios(fixture(),350,calm='yes')
        with self.assertRaises(ValueError):centered_rank([1,np.inf])
    def test_signal_prefix_equivalence(self):
        p=fixture()
        for end in [294,300,350,400]:
            x=formation_scenarios(p,end,calm=False);y=formation_scenarios(p[:end+1],end,calm=False)
            np.testing.assert_array_equal(x['scenario_core_ranks'],y['scenario_core_ranks'])

if __name__=='__main__':unittest.main()
