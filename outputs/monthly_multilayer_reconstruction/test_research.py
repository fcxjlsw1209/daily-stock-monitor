import sys, unittest, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import research as r

class MonthlyArchitectureTests(unittest.TestCase):
    def synthetic(self,n=360):
        rng=np.random.default_rng(73);idx=pd.bdate_range('2018-01-01',periods=n)
        px=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0003,.02,(n,2)),axis=0)),index=idx,columns=['SPY','A'])
        raw=dict(close=px.copy(),high=px*1.03,low=px*.97,opening=px.shift(1).fillna(px.iloc[0]),volume=pd.DataFrame(rng.uniform(1e5,2e5,(n,2)),index=idx,columns=px.columns),sector_return=px.pct_change(fill_method=None))
        return px,raw

    def test_features_prefix_invariant_to_future(self):
        px,raw=self.synthetic();cut=310
        a=r.shape_features(px,raw);b=r.shape_features(px.iloc[:cut],{k:v.iloc[:cut] for k,v in raw.items()})
        for k in a:np.testing.assert_allclose(a[k].iloc[:cut],b[k],equal_nan=True,err_msg=k)
        px2=px.copy();px2.iloc[cut:]*=17;raw2={k:v.copy() for k,v in raw.items()}
        for k in raw2:raw2[k].iloc[cut:]*=9
        c=r.shape_features(px2,raw2)
        for k in a:np.testing.assert_allclose(a[k].iloc[:cut],c[k].iloc[:cut],equal_nan=True,err_msg=k)

    def test_split_adjusted_ohlc_shapes(self):
        px,raw=self.synthetic();a=r.shape_features(px,raw);braw={k:v.copy() for k,v in raw.items()}
        for k in ['close','high','low','opening']:braw[k].iloc[250:]/=4
        b=r.shape_features(px,braw)
        for k in a:np.testing.assert_allclose(a[k],b[k],equal_nan=True,atol=1e-12,err_msg=k)

    def test_training_matured_labels_only(self):
        meta=dict(dates=['2014-12-31','2015-12-31','2020-11-30','2020-12-31','2021-01-29'],ends=['2015-01-30','2016-01-29','2020-12-31','2021-01-29',None])
        a=dict(eligible=np.ones((5,2),bool),X=np.arange(20).reshape(5,2,2),targets=np.ones((5,2,3)))
        X,y,age,audit=r.training_rows(meta,a,'2020-12-31')
        self.assertEqual(audit['rows'],[(1,0),(1,1),(2,0),(2,1)])
        a['targets'][3:]=1e9
        X2,y2,age2,_=r.training_rows(meta,a,'2020-12-31')
        np.testing.assert_array_equal(y,y2);np.testing.assert_array_equal(X,X2)

    def test_half_life_weighting(self):
        w=r.recency_weights(np.array([0.,12.,24.]),12)
        self.assertAlmostEqual(w[0]/w[1],2);self.assertAlmostEqual(w[0]/w[2],4);self.assertAlmostEqual(w.mean(),1)
        np.testing.assert_array_equal(r.recency_weights(np.array([0.,12.]),0),[1,1])

    def test_weak_gate_restores_support(self):
        s=dict(core=np.full((2,3),.2),support=np.full((2,3),.1),core_weight=.6,small_weight=0)
        zero=np.zeros((2,3));tree=np.full((2,3),.5)
        sc=r.compose(s,zero,zero,tree,zero,np.array([False,True]),[0,0,.05,0])
        np.testing.assert_allclose(sc[0],.3);np.testing.assert_allclose(sc[1],.2+.1*(1-.05/.4)+.05*.5)

    def test_budget_and_pure_momentum(self):
        s=dict(core=np.ones((2,3))*.3,support=np.zeros((2,3)),core_weight=1,small_weight=0)
        self.assertTrue(r.allowed(s,[.1,.025,.025,.05]));self.assertFalse(r.allowed(s,[.1,.05,.05,0]))
        z=np.ones((2,3))*.5
        np.testing.assert_allclose(r.compose(s,z,z,z,z,np.ones(2,bool),[.1,.025,.025,.05]),.3*.8+.5*.2)
        s['core_weight']=.6;s['small_weight']=.075
        self.assertFalse(r.allowed(s,[.1,.025,.025,.05]))

    def test_ranking_ignores_ineligible_predictions(self):
        a=r.centered_rank(np.array([[1.,9999.,2.]]),np.array([[True,False,True]]))
        np.testing.assert_allclose(a,[[0,0,.5]])

    def test_ridge_weights_and_reproducibility(self):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng=np.random.default_rng(5);X=rng.normal(size=(120,5));y=X[:,0]+rng.normal(size=120)
        weights=r.recency_weights(np.linspace(0,60,120),12)
        m1=Ridge(alpha=1,solver='cholesky').fit(X,y,sample_weight=weights);m2=Ridge(alpha=1,solver='cholesky').fit(X,y,sample_weight=weights)
        np.testing.assert_array_equal(m1.predict(X),m2.predict(X))
        params=dict(max_depth=2,max_iter=10,early_stopping=False,random_state=1729)
        a=HistGradientBoostingRegressor(**params).fit(X,y);b=HistGradientBoostingRegressor(**params).fit(X,y)
        np.testing.assert_array_equal(a.predict(X),b.predict(X))

    def test_frozen_masks_and_saved_predictions(self):
        m=r.read(r.P/'FEATURES.json');old=r.read(r.O/'monthly_filtered_reconstruction/PLAN.json')
        self.assertEqual(m['records'],old['records']);self.assertEqual(sum(z['included_A'] for z in m['records']),54);self.assertEqual(sum(z['included_B'] for z in m['records']),39)
        if (r.P/'PREDICTIONS.npz').exists():
            z=np.load(r.P/'PREDICTIONS.npz');self.assertTrue(np.isfinite(z['raw']).all());self.assertLessEqual(abs(z['ranks']).max(),.5)
            for q in r.read(r.P/'TRAINING_AUDIT.json'):self.assertLessEqual(q['train_end'],q['cutoff']);self.assertEqual(r.sha(q['model_file']),q['model_sha256'])

    def test_sources_and_observers_unchanged(self):
        for name in ['PROTECTED_BEFORE.json','SOURCE_HASHES.json']:
            for f,h in r.read(r.P/name).items():self.assertEqual(r.sha(f),h,f)

    def test_safe_weighted_ridge_equivalence(self):
        from sklearn.linear_model import Ridge
        rng=np.random.default_rng(17);X=rng.normal(size=(120,9));y=rng.normal(size=120)
        w=r.recency_weights(np.linspace(0,60,120),24)
        safe=r.fit_weighted_ridge(X,y,w,10)
        ref=Ridge(alpha=10,fit_intercept=True,solver='cholesky').fit(X,y,sample_weight=w)
        np.testing.assert_allclose(safe['coef'],ref.coef_,atol=1e-10,rtol=1e-10)
        self.assertAlmostEqual(safe['intercept'],ref.intercept_,places=10)

    def test_metrics_gate_and_stable_ties(self):
        d=dict(eligible=np.array([[True,True,True,True],[True,False,False,False],[True,True,False,False]]),labels=np.array([[True,True,True,False],[True,False,False,False],[True,True,False,False]]),masks={'all':np.ones(3,bool)})
        out=r.metrics(d,np.zeros((3,4)))['all']
        np.testing.assert_array_equal(out,[1,5])

    def test_saved_winner_metrics_replay(self):
        if not (r.P/'LEADERS.json').exists():return
        data=r.load_eval();seeds=r.seeds(data);pred=np.load(r.P/'PREDICTIONS.npz')['ranks']
        for label,l in r.read(r.P/'LEADERS.json').items():
            sc=r.compose(seeds[l['trial']['seed']],*pred[l['bundle']['model_ids']],data['shapes'][:,:,l['trial']['shape']],data['weak'],l['weights'])
            self.assertEqual(r.metrics(data,sc,True),{k:l[k] for k in ['metrics','details']})
            self.assertTrue(r.allowed(seeds[l['trial']['seed']],l['weights']))
        self.assertEqual(r.read(r.P/'ABLATIONS.json')[0]['metrics']['A']['exact'],17)

    def test_saved_numeric_validation(self):
        if not (r.P/'NUMERICAL_VALIDATION.json').exists():return
        x=r.read(r.P/'NUMERICAL_VALIDATION.json')
        self.assertTrue(x['all_prediction_ranks_identical']);self.assertLess(x['max_prediction_difference'],1e-8)

    def test_performance_fixed_units_not_daily_rebalancing(self):
        import performance as pf
        idx=pd.to_datetime(['2024-01-31','2024-02-01','2024-02-29'])
        px=pd.DataFrame({'A':[100,200,100],'B':[100,100,200]},index=idx)
        orders=[dict(date='2024-01-31',end='2024-02-29',signal_date='2024-01-31',symbols=['A','B'],scores=[])]
        independent=pf.independently_compound(px,orders,idx[0],idx[-1])
        st,eq,trades,holds=pf.engine.replay(px,px,orders,0,'close',idx[0],idx[-1])
        np.testing.assert_allclose(eq,[100000,150000,150000]);np.testing.assert_allclose(eq,independent)

    def test_performance_daily_statistics(self):
        if not (r.P/'PERFORMANCE.json').exists():return
        info=r.read(r.P/'PERFORMANCE.json');data=r.read(r.P/'PERFORMANCE_DAILY.json')
        idx=pd.to_datetime([x['date'] for x in data]);years=(idx[-1]-idx[0]).days/365.2425
        for name,modes in info['runs'].items():
            for mode,stats in modes.items():
                values=pd.Series([x[name+'__'+mode] for x in data],index=idx);rr=values.pct_change(fill_method=None).dropna()
                self.assertAlmostEqual(stats['cagr'],(values.iloc[-1]/100000)**(1/years)-1,places=12)
                self.assertAlmostEqual(stats['sharpe'],rr.mean()/rr.std(ddof=1)*np.sqrt(252),places=12)
                self.assertAlmostEqual(stats['max_drawdown'],(values/values.cummax()-1).min(),places=12)
        self.assertEqual(info['checks']['author_reported_months_rounding_matches'],69)
        self.assertTrue(info['checks']['model_selection_unchanged'])

if __name__=='__main__':unittest.main()
