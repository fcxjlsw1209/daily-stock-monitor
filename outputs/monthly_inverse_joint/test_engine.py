import unittest
import numpy as np
from engine import soft_objective,hard_objective,path_loss,project


class JointTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(420);M,K,T,D=3,5,7,4
        path=np.exp(np.concatenate([np.zeros((M,K,1)),np.cumsum(rng.normal(.003,.02,(M,K,T-1)),axis=2)],axis=2))
        return dict(F=rng.normal(0,.1,(M,K,D)),C=rng.uniform(.1,.9,(M,K)),mask=np.ones((M,K),bool),paths=path,
            R=path[:,:,[1,2,3,4,5]],Y=rng.normal(.03,.02,(M,5)),W=np.full((M,5),.2),month_weight=np.array([.3,.6,.8]),targetdd=np.array([-.1,-.08,-.06]))

    def test_analytic_gradient_includes_compounding_and_drawdown(self):
        d=self.fixture();coef=np.array([.01,-.03,.07,.02]);loss,grad=soft_objective(coef,d,.05)
        self.assertTrue(np.isfinite(loss))
        for i in range(len(coef)):
            h=np.zeros_like(coef);h[i]=1e-6
            numeric=(soft_objective(coef+h,d,.05)[0]-soft_objective(coef-h,d,.05)[0])/2e-6
            self.assertAlmostEqual(numeric,grad[i],places=7)

    def test_hard_selection_is_one_portfolio_per_month(self):
        d=self.fixture();w=np.array([.1,.2,-.2,.1]);loss,ids=hard_objective(w[None,:],d)
        expected=np.argmax(.5*d['C']+np.einsum('mkd,d->mk',d['F'],w),axis=1)
        np.testing.assert_array_equal(ids[0],expected)
        actual=path_loss(d['paths'][np.arange(3),expected],d['R'][np.arange(3),expected],d)[0]
        self.assertAlmostEqual(actual,loss[0],places=12)

    def test_masked_candidates_have_no_effect(self):
        d=self.fixture();d['mask'][:,-1]=False;w=np.ones(4)*.01;before=soft_objective(w,d,.04)
        d['C'][:,-1]=100;d['F'][:,-1]=200;after=soft_objective(w,d,.04)
        self.assertAlmostEqual(before[0],after[0],places=12);np.testing.assert_allclose(before[1],after[1],atol=1e-12)

    def test_group_weight_constraints(self):
        w=np.arange(17)*.1
        a=project(w,'stock_only');b=project(w,'portfolio')
        self.assertLessEqual(abs(a[:15]).sum(),.5+1e-12);self.assertEqual(abs(a[15:]).sum(),0)
        self.assertLessEqual(abs(b[:15]).sum(),.4+1e-12);self.assertLessEqual(abs(b[15:]).sum(),.1+1e-12)


if __name__=='__main__':unittest.main()
