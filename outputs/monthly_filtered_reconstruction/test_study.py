import unittest,copy
import numpy as np
from filtered_study import mask,pick,evaluate,feasibility,fit_soft,fit_max_months
from compact import metrics,compact_scores

class StudyTests(unittest.TestCase):
    def toy(self,contradiction=False):
        vals=np.array([.4,.3,.2,.1,0.,-.1]);x=np.zeros((2,6,3));x[:,:,0]=vals
        labels=np.zeros((2,6),bool);labels[:,:3]=True
        if contradiction:labels[1]=~labels[1]
        names=['A','B','C','D','E','F'];targets=[[s for s,v in zip(names,row) if v] for row in labels]
        return dict(key='fixture',dates=['2021-01-29','2021-02-26'],symbols=[names,names],records=[dict(date='2021-01-29',end='2021-02-26',target=targets[0],included_A=True,included_B=True),dict(date='2021-02-26',end='2021-03-31',target=targets[1],included_A=True,included_B=False)],
                    design=x,X=x,eligible=np.ones((2,6),bool),labels=labels,names=['momentum','support','small'],core=1,side=1,small=1)

    def test_month_masks_are_frozen_and_not_selection_dependent(self):
        d=self.toy();self.assertEqual(mask(d,'A').sum(),2);self.assertEqual(mask(d,'B').sum(),1)
        d['design']*=-1
        self.assertEqual(mask(d,'B').tolist(),[True,False])

    def test_top3_tie_order_and_gate(self):
        scores=np.zeros((1,5));eligible=np.array([[False,True,True,True,True]])
        self.assertEqual(pick(scores,eligible).tolist(),[[1,2,3]])

    def test_known_uniform_rule_recovers_all_months(self):
        r=feasibility(self.toy(),'A')
        self.assertTrue(r['strict_full_fit']);self.assertEqual(r['evaluations']['A']['exact_periods'],2)

    def test_incompatible_labels_reject_full_fit(self):
        r=feasibility(self.toy(True),'A');self.assertLess(r['margin'],0)
        self.assertFalse(r['strict_full_fit'])

    def test_mip_optimizes_whole_months_and_reports_bound(self):
        r=fit_max_months(self.toy(True),'A',seconds=5)
        self.assertEqual(r['status'],0);self.assertEqual(r['evaluations']['A']['exact_periods'],1)
        self.assertAlmostEqual(r['objective_lower_bound'],1)

    def test_soft_fit_preserves_core_and_small_correction_budget(self):
        r=fit_soft(self.toy(),'A',.001);w=np.array(r['weights'])
        self.assertGreaterEqual(w[0],.5-1e-7);self.assertLessEqual(w[-1],.1+1e-7);self.assertAlmostEqual(w.sum(),1)

    def test_batch_and_individual_matching_agree(self):
        d=self.toy();scores=compact_scores(d,[(0,-1,-1,1,0,0),(0,1,-1,.5,.5,0)])
        z=metrics(d,scores)
        for sl in ['A','B']:
            for i in range(2):
                e=evaluate(d,scores[i],sl)
                self.assertEqual(z[sl][0][i],e['exact_periods']);self.assertEqual(z[sl][1][i],e['shared_positions'])

class RetentionTests(unittest.TestCase):
    toy=StudyTests.toy
    def test_recursive_decisions_ignore_author_labels(self):
        from retention import recursive_scores
        d=self.toy();scores=d['design'][:,:,0]
        _,a=recursive_scores(d,scores,buffer=5)
        d['labels']=~d['labels']
        for row in d['records']:row['target']=['D','E','F']
        _,b=recursive_scores(d,scores,buffer=5)
        self.assertEqual(a,b)

    def test_future_row_does_not_change_prior_decision(self):
        from retention import recursive_scores
        d=self.toy();scores=d['design'][:,:,0].copy()
        _,a=recursive_scores(d,scores,buffer=5,bonus=.1)
        scores[-1]*=-20
        _,b=recursive_scores(d,scores,buffer=5,bonus=.1)
        self.assertEqual(a[0],b[0])

if __name__=='__main__':unittest.main()
