import unittest
import numpy as np
from search import screen

class ScreenTests(unittest.TestCase):
    def make(self):
        rng=np.random.default_rng(861)
        return dict(X=rng.normal(size=(4,7,5)),eligible=np.array([[1,1,1,1,1,0,0],[1,1,0,0,0,0,0],[1,0,0,0,0,0,0],[1,1,1,1,1,1,1]],bool),
                    ratios=rng.uniform(.8,1.2,(4,7)),valid=np.ones((4,7),bool),labels=np.array([[1,1,1,0,0,0,0]]*4,bool),label_mask=np.ones(4,bool))
    def test_vectorized_scores_match_slow_reference_with_two_and_cash(self):
        data=self.make();rec=[(0,-1,-1,1,0,0),(0,1,2,.6,.3,.1),(4,3,-1,.7,.3,0)]
        z=screen(data,rec,['2020-12-31','2021-01-29','2021-02-26','2021-03-31'],['2021-01-29','2021-02-26','2021-03-31','2021-04-30'])
        for k,conf in enumerate(rec):
            rr=[];exact=0
            for t in range(4):
                sc=sum(w*data['X'][t,:,j] for j,w in zip(conf[:3],conf[3:]) if j>=0)
                good=np.flatnonzero(data['eligible'][t]);chosen=sorted(good,key=lambda j:(-sc[j],j))[:3]
                if len(chosen)<2:chosen=[]
                ret=float(data['ratios'][t,chosen].mean()) if chosen else 1.
                rr.append(ret);exact+=int(sum(data['labels'][t,j] for j in chosen)==3)
            self.assertTrue(np.allclose(z['period'][k],rr));self.assertEqual(z['exact'][k],exact)
    def test_unpriceable_selection_is_rejected_not_removed_from_pool(self):
        d=self.make();d['X'][:]=0;d['X'][:,:,0]=np.arange(7)[None,:]
        d['valid'][0,4]=False
        z=screen(d,[(0,-1,-1,1,0,0)],['2020-12-31']*4,['2021-01-29']*4)
        self.assertFalse(z['valid'][0]);self.assertIn(4,z['picks'][0,0])
    def test_returns_cannot_change_selection(self):
        d=self.make();args=([(0,1,-1,.8,.2,0)],['2020-12-31']*4,['2021-01-29']*4)
        a=screen(d,*args);d['ratios']*=10;b=screen(d,*args)
        self.assertTrue(np.array_equal(a['picks'],b['picks']))
    def test_ties_have_stable_symbol_order(self):
        d=self.make();d['X'][:]=0
        z=screen(d,[(0,-1,-1,1,0,0)],['2020-12-31']*4,['2021-01-29']*4)
        self.assertEqual(z['picks'][0,0].tolist(),[0,1,2])
if __name__=='__main__':unittest.main()
