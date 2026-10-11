import unittest
import numpy as np
import experiment as e

class ListwiseTests(unittest.TestCase):
    def test_gradient(self):
        rng=np.random.default_rng(12);E=rng.normal(size=(6,7,4));allow=np.ones((6,7),bool);allow[0,6]=False
        labels=np.array([0,1,2,3,4,5]);pw=np.full(6,1/6);w=np.array([.6,.2,.1,.1]);prior=np.full(4,.25)
        f,g=e.likelihood(w,E,allow,labels,pw,.05,.1,prior);eps=1e-6
        num=[(e.likelihood(w+np.eye(4)[j]*eps,E,allow,labels,pw,.05,.1,prior)[0]-e.likelihood(w-np.eye(4)[j]*eps,E,allow,labels,pw,.05,.1,prior)[0])/(2*eps) for j in range(4)]
        np.testing.assert_allclose(g,num,rtol=1e-6,atol=1e-7)

    def test_sequential_choice_removes_previous_pick(self):
        d=dict(order=np.array([[2,0,1]]),eligible=np.ones((1,4),bool));X=np.arange(8).reshape(1,4,2)
        E,allow,lab,pw=e.events(X,d,np.array([True]))
        self.assertEqual(lab.tolist(),[2,0,1]);self.assertFalse(allow[1,2]);self.assertFalse(allow[2,0]);self.assertFalse(allow[2,2])
        np.testing.assert_allclose(pw,[1/3]*3)

    def test_direction_of_ordered_likelihood(self):
        E=np.eye(3)[None,:,:];allowed=np.ones((1,3),bool);labels=np.array([0]);pw=np.array([1.]);prior=np.ones(3)/3
        good=e.likelihood(np.array([.7,.2,.1]),E,allowed,labels,pw,.1,0,prior)[0]
        bad=e.likelihood(np.array([.1,.2,.7]),E,allowed,labels,pw,.1,0,prior)[0]
        self.assertLess(good,bad)

    def test_training_cutoffs_and_group_budgets(self):
        m=e.read(e.P/'INPUTS.json')
        for rows in m['ml_sources'].values():
            for r in rows:self.assertLessEqual(r['latest_training_label'],r['cutoff']);self.assertLessEqual(r['cutoff'],r['date'])
        for r in e.read(e.P/'TRIALS.json'):
            A,b,prior=e.limits(r['spec']);w=np.array(r['weights'])
            self.assertAlmostEqual(w.sum(),1,places=7);self.assertTrue((w>=-1e-7).all())
            self.assertTrue((np.einsum('ij,j->i',A,w,optimize=False)<=b+1e-7).all())

    def test_future_orders_excluded_from_events(self):
        m=e.read(e.P/'INPUTS.json');d=e.pack('percentile',m);X,spec=e.design(d,m['families'][0],True);dev=np.array(m['development'])
        original=e.events(X,d,dev)
        changed=dict(d,order=d['order'].copy());changed['order'][~dev]=-1
        new=e.events(X,changed,dev)
        for a,b in zip(original,new):np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
