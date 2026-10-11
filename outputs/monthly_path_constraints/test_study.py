import unittest
import numpy as np
from study import indices_at, projected, landmark_penalty
from explain import fit_pairwise, project, explanations_for
from joint_rule import penalties


class Tests(unittest.TestCase):
    def test_sparse_projection_matches_interp(self):
        x = np.array([0., 1., 4., 5.]); q = np.array([-.2, .7, 1., 3., 6.])
        y = np.array([[0., 2., 1., 4.], [1., 3., 6., 2.]])
        np.testing.assert_allclose(projected(y, indices_at(x, q)), np.array([np.interp(q, x, z) for z in y]))

    def test_month_moments_reproduce_sharpe_and_covariance(self):
        a = np.array([.01, -.02, .03]); b = np.array([-.01, .06]); r = np.r_[a, b]
        q = np.array([-.01, .01, .02, .03, .01]); n = len(r)
        total = a.sum() + b.sum(); square = (a*a).sum() + (b*b).sum()
        sd = np.sqrt((square-total*total/n)/(n-1))
        self.assertAlmostEqual(sd, r.std(ddof=1))
        cov = (a*q[:3]).sum() + (b*q[3:]).sum() - total*q.sum()/n
        cor = cov/np.sqrt((square-total*total/n)*((q*q).sum()-q.sum()**2/n))
        self.assertAlmostEqual(cor, np.corrcoef(r, q)[0, 1])

    def test_wrong_date_of_identical_maximum_drawdown_is_penalized(self):
        a=np.array([[0.,-.3446,-.20],[0.,-.20,-.3446]])
        result=landmark_penalty(a,np.array([2]))
        self.assertGreater(result[0],100.)
        self.assertEqual(result[1],0.)

    def test_bounded_pairwise_fit_stays_finite(self):
        rng=np.random.default_rng(1);X=rng.uniform(-1,1,(600,5));b=rng.uniform(-.5,.5,600)
        coef,loss,_=fit_pairwise(X,b)
        self.assertTrue(np.isfinite(coef).all() and np.isfinite(loss))
        self.assertLessEqual(np.abs(coef).sum(),.5+1e-12)
        self.assertLessEqual(loss,np.mean(np.minimum(b,0)**2))
        self.assertAlmostEqual(np.abs(project(np.array([4.,-2.]))).sum(),.5)

    def test_joint_constraint_penalizes_nonleaders_not_top_three(self):
        symbols=['A','B','C','D','E']
        d={'E':np.zeros((1,2)),'meta':[{'symbols':symbols}],
           'combos':[np.array([[0,1,2],[2,3,4]])]}
        rows=[{'symbols':symbols,'X':np.zeros((5,5)),
               'cores':{'raw':np.array([1.,.9,.8,.1,0.])}}]
        pen,deficit=penalties(d,rows,np.zeros(5))
        self.assertEqual(pen[0,0],0.)
        self.assertEqual(deficit[0,0],0.)
        self.assertAlmostEqual(deficit[0,1],.25)
        self.assertAlmostEqual(pen[0,1],16.)

    def test_explanation_selection_does_not_include_child_path(self):
        rows={f'parent__{family}':{} for family in ['raw','risk','mixed']}
        rows['parent__joint_rule__raw']={'pairwise_loss':0.}
        self.assertEqual(set(explanations_for(rows,'parent')),{'parent__raw','parent__risk','parent__mixed'})


if __name__ == '__main__': unittest.main()
