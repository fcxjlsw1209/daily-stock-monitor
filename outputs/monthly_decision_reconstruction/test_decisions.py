import unittest
import numpy as np
from decisions import maximum_margin, CORE, SIDE, STRICT


class DecisionTests(unittest.TestCase):
    def test_feasible_selection_has_positive_margin_and_verified_dual(self):
        D=np.zeros((3,len(CORE+SIDE)))
        D[:,:len(CORE)]=.4
        r=maximum_margin(D)
        self.assertGreater(r['margin'],STRICT)
        self.assertLess(r['primal_dual_gap'],1e-8)
        self.assertGreaterEqual(r['core_mass'],.5-1e-8)

    def test_contradictory_months_cannot_be_glued_together(self):
        D=np.zeros((1,len(CORE+SIDE)));D[:,:len(CORE)]=.4
        # The exact opposite ranking in the same feature state conflicts.
        r=maximum_margin(np.vstack([D,-D]))
        self.assertLess(r['margin'],-STRICT)
        self.assertEqual(r['status'],'not_separable_in_family')

    def test_identical_features_are_not_counted_as_exact_reconstruction(self):
        r=maximum_margin(np.zeros((4,len(CORE+SIDE))))
        self.assertAlmostEqual(r['margin'],0)
        self.assertEqual(r['status'],'tie_boundary')


class SmallCorrectionTests(unittest.TestCase):
    def test_threshold_model_respects_absolute_correction_bound(self):
        import threshold_probe as t
        X=np.zeros((4,len(CORE+SIDE)))
        X[0,:len(CORE)]=.4
        X[1,:len(CORE)]=.3
        X[2,:len(CORE)]=.2
        X[3,:len(CORE)]=-.4
        row={'date':'test','symbols':['A','B','C','D'],
             'y':np.array([True,True,True,False]),'X':{'test':X}}
        fit=t.solve([row],'test','half')
        picks,score,correction=t.choices(row,fit,'test')
        self.assertEqual(set(picks),{'A','B','C'})
        self.assertGreater(fit['margin'],STRICT)
        self.assertLessEqual(np.abs(correction).max(),.05+1e-8)
        self.assertLess(fit['primal_dual_gap'],1e-8)

    def test_a_reversed_identical_state_is_not_a_common_rule(self):
        import threshold_probe as t
        X=np.zeros((6,len(CORE+SIDE)))
        X[:3,:len(CORE)]=.4;X[3:,:len(CORE)]=-.4
        first={'date':'first','symbols':list('ABCDEF'),
               'y':np.array([True,True,True,False,False,False]),'X':{'test':X}}
        later=dict(first,date='later',y=~first['y'])
        fit=t.solve([first,later],'test','half')
        self.assertLessEqual(fit['margin'],STRICT)


if __name__=='__main__':unittest.main()
