import unittest
import numpy as np
from fit_policy import fit,design,evaluate,FEATURES,CORE,pairs
from reconstruct import max_margin

class RankingTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(582)
        out=[]
        for k in range(6):
            x=rng.uniform(-.5,.5,(12,len(FEATURES)))
            target=np.argsort(-x[:,0])[:3]
            out.append(dict(date=f'2021-{k+1:02d}-01',end=f'2021-{k+2:02d}-01',symbols=[str(i) for i in range(12)],
                            target=[str(i) for i in target],y=np.isin(np.arange(12),target),valid=True,regime=k%2,
                            X={'universe_percentile':x}))
        return out
    def test_hard_margin_explains_known_uniform_rule(self):
        rows=self.rows();D,_=pairs(rows,'universe_percentile','linear70')
        w,res=max_margin(D,active=[0])
        self.assertGreater(res['margin'],0)
        self.assertAlmostEqual(w[0],1)
        self.assertTrue(np.all(w[1:]==0))
        self.assertEqual(evaluate(rows,w,'universe_percentile','linear70')['exact_months'],6)
    def test_soft_fit_respects_budgets_and_positive_core(self):
        for m,b in [('linear70',(.7,.3,0)),('regime70',(.7,.2,.1)),('shape70',(.7,.2,.1))]:
            w,r=fit(self.rows(),'universe_percentile',m)
            self.assertGreaterEqual(w[:len(CORE)].min(),-1e-8)
            self.assertAlmostEqual(w[:len(CORE)].sum(),b[0],places=7)
            self.assertLessEqual(abs(w[len(CORE):len(FEATURES)]).sum(),b[1]+1e-7)
            self.assertLessEqual(abs(w[len(FEATURES):]).sum(),b[2]+1e-7)
    def test_interactions_bounded_and_do_not_mutate_input(self):
        row=self.rows()[0];before=row['X']['universe_percentile'].copy()
        for model in ['regime70','shape70']:
            x,names=design(row,'universe_percentile',model)
            self.assertEqual(x.shape[1],len(names))
            self.assertLessEqual(abs(x).max(),.5)
            self.assertTrue(np.array_equal(row['X']['universe_percentile'],before))
    def test_invalid_universe_not_used_as_partial_label(self):
        rows=self.rows();D,w=pairs(rows,'universe_percentile','linear70')
        bad=dict(rows[0],valid=False,X={'universe_percentile':np.full_like(rows[0]['X']['universe_percentile'],np.nan)})
        DD,ww=pairs(rows+[bad],'universe_percentile','linear70')
        self.assertTrue(np.array_equal(D,DD));self.assertTrue(np.array_equal(w,ww))

class MonthlyModelTimingTests(unittest.TestCase):
    def test_future_or_unmatured_return_labels_cannot_change_fit(self):
        from monthly_ml import fit_ridge_asof
        rng=np.random.default_rng(59)
        old=dict(start='2020-01-31',end='2020-02-28',X=rng.normal(size=(30,5)),y=rng.normal(size=30),excluded=0)
        future=dict(start='2020-02-28',end='2020-03-31',X=np.full((3,5),np.nan),y=np.full(3,np.nan),excluded=0)
        a,ma,aa=fit_ridge_asof([old],'2020-02-28',1)
        b,mb,bb=fit_ridge_asof([old,future],'2020-02-28',1)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(ma,mb);self.assertEqual(aa,bb)
        self.assertEqual(aa['train_end'],'2020-02-28')
    def test_five_year_window_excludes_older_labels(self):
        from monthly_ml import fit_ridge_asof
        rng=np.random.default_rng(64)
        valid=dict(start='2020-01-31',end='2020-02-28',X=rng.normal(size=(30,5)),y=rng.normal(size=30),excluded=0)
        stale=dict(start='2014-01-31',end='2014-02-28',X=np.full((3,5),np.nan),y=np.full(3,np.nan),excluded=0)
        a,ma,aa=fit_ridge_asof([valid],'2020-02-28',1)
        b,mb,bb=fit_ridge_asof([stale,valid],'2020-02-28',1)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(aa,bb)

if __name__=='__main__':unittest.main()
