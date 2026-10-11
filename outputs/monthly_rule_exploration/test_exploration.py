import unittest
from itertools import combinations,product
import numpy as np
from explore import CORE,SIDE,FEATURES,design,fit_rule,perturb_weights,choose
from joint_explore import mixed_fit

class ExplorationTests(unittest.TestCase):
    def test_mixed_solver_matches_exhaustive_small_problem(self):
        rng=np.random.default_rng(7654);symbols=list('ABCDE')
        rows=[dict(symbols=symbols,X={'universe_percentile':rng.uniform(-.5,.5,(5,len(FEATURES)))}) for _ in range(2)]
        cs=[[list(c) for c in combinations(symbols,3)][:4],[list(c) for c in combinations(symbols,3)][3:7]]
        for kind in ['linear','quadratic','threshold']:
            with self.subTest(kind=kind):
                expected=max(fit_rule(rows,list(path),'universe_percentile',kind)['margin'] for path in product(*cs))
                result=mixed_fit(rows,cs,'universe_percentile',kind,seconds=20)
                self.assertTrue(result['proven_optimal'])
                self.assertAlmostEqual(result['fit']['margin'],expected,places=7)
                # Forbid its chosen path and check that the remaining path optimum is also exact.
                remaining=[p for p in product(range(4),repeat=2) if list(p)!=result['path']]
                expected2=max(fit_rule(rows,[cs[i][j] for i,j in enumerate(p)],'universe_percentile',kind)['margin'] for p in remaining)
                alternate=mixed_fit(rows,cs,'universe_percentile',kind,[result['path']],seconds=20)
                self.assertTrue(alternate['proven_optimal'])
                self.assertAlmostEqual(alternate['fit']['margin'],expected2,places=7)

    def test_nonlinear_side_never_changes_core_response(self):
        rng=np.random.default_rng(55);X=rng.uniform(-.5,.5,(30,len(FEATURES)))
        for kind in ['quadratic','threshold']:
            a=design(X,kind)
            for j in range(len(CORE)):
                Y=X.copy();Y[:,j]=np.minimum(.5,Y[:,j]+.1);b=design(Y,kind)
                self.assertTrue(np.array_equal(a[:,len(FEATURES):],b[:,len(FEATURES):]))
            self.assertLessEqual(abs(a[:,len(FEATURES):]).max(),.5+1e-12)
            model=rng.normal(size=a.shape[1]-len(FEATURES));model*=.1/abs(model).sum()
            self.assertLessEqual(abs(np.einsum('ij,j->i',a[:,len(FEATURES):],model,optimize=False)).max(),.05+1e-12)

    def test_rounding_and_perturbations_preserve_group_budgets(self):
        rng=np.random.default_rng(765);w=rng.normal(size=len(FEATURES)+len(SIDE));w[:len(CORE)]=abs(w[:len(CORE)])
        groups=[slice(0,len(CORE)),slice(len(CORE),len(FEATURES)),slice(len(FEATURES),None)]
        for ids,mass in zip(groups,[.5,.4,.1]):w[ids]*=mass/abs(w[ids]).sum()
        for v in [np.round(w,3),w*rng.uniform(.99,1.01,len(w))]:
            result=perturb_weights(w,v)
            for ids in groups:self.assertAlmostEqual(abs(result[ids]).sum(),abs(w[ids]).sum(),places=12)
            self.assertGreaterEqual(result[:len(CORE)].min(),0)

class CoreSummaryTests(unittest.TestCase):
    def test_core_summary_is_monotone_and_independent_of_risk_slots(self):
        from monotone_core import transformed
        rng=np.random.default_rng(92);x=rng.uniform(-.5,.5,(20,len(FEATURES)))
        before=x.copy();a=transformed({'X':{'universe_percentile':x}})['X']['universe_percentile']
        self.assertTrue(np.array_equal(x,before))
        changed=x.copy();changed[:,3:6]*=-1
        b=transformed({'X':{'universe_percentile':changed}})['X']['universe_percentile']
        self.assertTrue(np.array_equal(a,b))
        for j in range(3):
            changed=x.copy();changed[:,j]=np.minimum(.5,changed[:,j]+.1)
            b=transformed({'X':{'universe_percentile':changed}})['X']['universe_percentile']
            self.assertTrue(np.all(b[:,:6]>=a[:,:6]-1e-12))
            self.assertTrue(np.array_equal(a[:,6:],b[:,6:]))

    def test_feature_names_do_not_mutate_shared_model_definition(self):
        from explore import feature_names
        names=feature_names('linear');old=list(FEATURES);names[0]='temporary_alias'
        self.assertEqual(FEATURES,old)

class RollingTimingTests(unittest.TestCase):
    def test_future_labels_do_not_change_ridge_prediction(self):
        import pandas as pd
        from rolling_model import prediction
        rng=np.random.default_rng(987);dates=pd.date_range('2017-12-31',periods=31,freq='ME')
        data=[dict(start=str(dates[i].date()),end=str(dates[i+1].date()),X=rng.uniform(-.5,.5,(5,len(FEATURES))),y=rng.uniform(-.5,.5,5)) for i in range(30)]
        row=dict(date=str(dates[-1].date()),X={'universe_percentile':rng.uniform(-.5,.5,(6,len(FEATURES)))})
        a,aa=prediction(row,data,'universe_percentile')
        future=dict(start=row['date'],end='2020-07-31',X=np.full((5,len(FEATURES)),np.nan),y=np.full(5,np.nan))
        b,bb=prediction(row,data+[future],'universe_percentile')
        self.assertTrue(np.array_equal(a,b));self.assertEqual(aa,bb)
        self.assertLessEqual(max(abs(a)),.05+1e-12)
        self.assertLessEqual(aa['last_training_label_end'],row['date'])

if __name__=='__main__':unittest.main()
