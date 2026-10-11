import unittest
import numpy as np
from infer import simplex, l1ball, constrained_ridge, score, CORE, SIDE


class InferenceTests(unittest.TestCase):
    def test_projection_budgets(self):
        x = np.array([-5., 2., 4., -1.])
        p = simplex(x, .6)
        self.assertTrue((p >= 0).all())
        self.assertAlmostEqual(p.sum(), .6)
        np.testing.assert_allclose(simplex(p, .6), p)
        b = l1ball(x, .4)
        self.assertAlmostEqual(np.abs(b).sum(), .4)
        self.assertTrue((np.sign(b[b != 0]) == np.sign(x[b != 0])).all())

    def test_constrained_fit_respects_momentum_even_if_labels_oppose_it(self):
        rng = np.random.default_rng(1701)
        d = rng.normal(size=(150, len(CORE+SIDE)))
        d[:, :len(CORE)] = -np.abs(d[:, :len(CORE)])
        w, _ = constrained_ridge(d, np.full(150, 1/150), .6)
        self.assertAlmostEqual(w[:len(CORE)].sum(), .6)
        self.assertTrue((w[:len(CORE)] >= 0).all())
        self.assertLessEqual(np.abs(w[len(CORE):]).sum(), .4 + 1e-10)

    def test_small_correction_cannot_overrule_bound(self):
        model = {'w':np.zeros(len(CORE+SIDE)), 'z':np.full(6, 1e6)}
        row = {'X':np.zeros((2, len(CORE+SIDE))), 'Z':np.array([np.ones(6), -np.ones(6)])}
        np.testing.assert_allclose(score(model, row), [.05, -.05])


if __name__ == '__main__':
    unittest.main()
