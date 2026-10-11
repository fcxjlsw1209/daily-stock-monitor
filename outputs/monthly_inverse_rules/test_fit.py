from itertools import combinations
import unittest
import numpy as np
from fit import triple_partition,objective,project_l1


class InverseRuleTests(unittest.TestCase):
    def test_triple_partition_matches_enumeration(self):
        logit=np.array([[.2,-.5,.7,1.2,4.],[100.,99.,97.,-2.,0.]])
        mask=np.array([[1,1,1,1,0],[1,1,1,1,1]],bool)
        actual,inc=triple_partition(logit,mask)
        for m in range(2):
            triples=list(combinations(np.flatnonzero(mask[m]),3));a=np.array([logit[m,list(h)].sum() for h in triples]);q=np.exp(a-a.max());q/=q.sum()
            expected=a.max()+np.log(np.exp(a-a.max()).sum());counts=np.zeros(5)
            for triple,w in zip(triples,q):counts[list(triple)]+=w
            self.assertAlmostEqual(actual[m],expected,places=11);np.testing.assert_allclose(inc[m],counts,atol=1e-12)
        np.testing.assert_allclose(inc.sum(axis=1),3.,atol=1e-12)

    def test_gradient_matches_finite_difference(self):
        rng=np.random.default_rng(1);X=rng.normal(0,.2,(2,5,4));C=rng.uniform(0,1,(2,5,3));bags=np.array([[0,1,2],[1,3,4]])
        d=dict(X=X,C=C,mask=np.ones((2,5),bool),BX=X[:,bags].mean(axis=2),BC=C[:,bags].mean(axis=2),logprior=np.log(np.array([[.6,.4],[.3,.7]])),month_weight=np.array([.2,.8]))
        w=rng.normal(0,.01,4);loss,grad=objective(w,d,0)
        for i in range(4):
            step=np.zeros(4);step[i]=1e-6
            numerical=(objective(w+step,d,0)[0]-objective(w-step,d,0)[0])/2e-6
            self.assertAlmostEqual(grad[i],numerical,places=6)

    def test_l1_projection_bound(self):
        x=np.array([.8,-.3,.2]);y=project_l1(x)
        self.assertAlmostEqual(abs(y).sum(),.5)
        np.testing.assert_allclose(project_l1(np.array([.1,-.2])),[.1,-.2])


if __name__=='__main__':unittest.main()
