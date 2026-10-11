import unittest
import state_rank as s

class StateRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.d,cls.L,cls.x,cls.px,cls.op,cls.f,cls.a=s.load()
        cls.data=dict(s.np.load(s.P/'TRAIN_DATA.npz'));cls.meta=s.read(s.P/'TRAIN_DATA.json')

    def test_pair_laplacian_matches_explicit_weighted_pairs(self):
        rng=s.np.random.default_rng(719);X=rng.normal(size=(13,7));y=rng.normal(size=13);past=rng.normal(size=13)
        g,r=s.pair_moments(X,y,past);rank=s.pd.Series(y).rank(pct=True).to_numpy()-.5
        anchor=set(s.np.argsort(-past)[:8]);top=set(s.np.argsort(-y)[:3]);gg=s.np.zeros((7,7));rr=s.np.zeros(7);total=0
        for i in range(13):
            for j in range(i+1,13):
                if i not in anchor and j not in anchor:continue
                w=3 if (i in top)!=(j in top) else 1;dx=X[i]-X[j];dy=rank[i]-rank[j]
                gg+=w*s.np.outer(dx,dx);rr+=w*dx*dy;total+=w
        s.np.testing.assert_allclose(g,gg/total,atol=1e-12);s.np.testing.assert_allclose(r,rr/total,atol=1e-12)

    def test_maturity_window_rejects_not_yet_known_labels(self):
        meta=dict(dates=['2010-12-01','2011-01-01','2015-12-01','2015-12-30','2016-01-02'],ends=['2011-01-01','2011-02-01','2015-12-31','2016-01-30','2016-02-01'])
        self.assertEqual(s.training_indices(meta,'2015-12-31'),[1,2])

    def test_baseline_and_residual_exact_decision_parity(self):
        for row,name in [(s.grid()[0],'baseline'),(s.grid()[1],'primary')]:
            chosen,sc=s.choices(row['spec'],self.a,self.m,self.d,122);ref=s.read(s.prior.P/(name+'_ORDERS.json'))
            for i,ids in enumerate(chosen):
                self.assertEqual([self.m['symbols'][i][j] for j in ids],ref[i]['symbols'])
                s.np.testing.assert_allclose(sc[i,ids],ref[i]['scores'],atol=1e-12)

    def test_later_arrays_cannot_change_first88_decisions(self):
        changed={k:v.copy() for k,v in self.a.items()}
        for a in changed.values():a[88:]=0
        for row in s.grid()[::9]:self.assertEqual(s.choices(row['spec'],self.a,self.m,self.d,88)[0],s.choices(row['spec'],changed,self.m,self.d,88)[0])

    def test_future_label_mutation_cannot_change_annual_fit(self):
        cutoff='2015-12-31';changed={k:v.copy() for k,v in self.data.items()}
        for i,end in enumerate(self.meta['ends']):
            if end>cutoff:
                sl=slice(changed['offsets'][i],changed['offsets'][i+1]);changed['Y'][sl]=50;changed['adverse'][sl]=100;changed['X'][sl]=2;changed['past_score'][sl]=-3
        for cfg in s.MODEL_CONFIGS[::2]:
            c1,b1,_=s.fit_year(self.data,self.meta,cutoff,cfg);c2,b2,_=s.fit_year(changed,self.meta,cutoff,cfg)
            s.np.testing.assert_array_equal(c1,c2);self.assertEqual(b1,b2)

    def test_forecasts_only_use_pre_signal_mature_labels(self):
        audit=s.read(s.P/'FORECAST_AUDIT.json');self.assertEqual(len(audit),66)
        for row in audit:
            self.assertLessEqual(row['latest_label_end'],row['cutoff'])
            self.assertGreater(row['security_examples'],1000)
            for date,end in zip(self.m['dates'],self.m['ends']):
                if s.pd.Timestamp(end).year==row['year']:self.assertLessEqual(row['cutoff'],date)
        for k,a in self.a.items():
            if k.startswith('ml_'):
                self.assertTrue(s.np.isfinite(a).all());self.assertTrue((abs(a)<=.5+1e-12).all())

    def test_weekly_pool_origin_not_future(self):
        for date,origin in zip(self.meta['dates'],self.meta['pool_origins']):
            if origin!='inherited_pre2016_historical_proxy':self.assertLessEqual(origin,date)
            else:self.assertLess(date,'2016-08-17')
        self.assertTrue(s.np.isfinite(self.data['X']).all())

    def test_forward_target_and_missing_path(self):
        px=s.pd.DataFrame(dict(A=[100,90,110,120],B=[100,s.np.nan,110,120],SPY=[100,101,102,103]),index=s.pd.date_range('2020-01-01',periods=4))
        end,y,a,g=s.forward_labels(px,px.index[0],['A','B'],3)
        self.assertEqual(end,'2020-01-04');s.np.testing.assert_allclose(y,[.17,.17]);self.assertAlmostEqual(a[0],.1);self.assertEqual(g.tolist(),[True,False])

    def test_state_prefix_invariance(self):
        i=20;date=self.m['dates'][i];ss=self.m['symbols'][i];cut=self.px.loc[:date]
        a=s.state_row(self.px,date,ss,set(self.m['etfs']));b=s.state_row(cut,date,ss,set(self.m['etfs']));self.assertEqual(a,b)

    def test_long_momentum_excludes_latest42(self):
        px=self.px[['SPY','QQQ','AAPL']].loc[:'2019-12-31'].copy();raw=dict(close=px,open=px,high=px*1.01,low=px*.99,volume=px*0+1000)
        f,_=s.raw_features(px,raw);changed=px.copy();changed.iloc[-42:]*=s.np.linspace(.9,1.3,42)[:,None];g,_=s.raw_features(changed,raw)
        for k in ['mom252','mom294','accel','path']:s.np.testing.assert_allclose(f[k].iloc[-1],g[k].iloc[-1],atol=1e-12)

    def test_combined_budget_does_not_shrink_core(self):
        spec=[dict(key='profile_sector',state='global_',amount=.1),dict(key='ml_2',state='global_',amount=.1)]
        result=s.score(spec,self.a,122);core=self.a['core'];w=self.a['core_weight'];support=(self.a['baseline']-core)/(1-w)
        expected=core+(1-w-.1)*support+.05*self.a['profile_sector']+.05*self.a['ml_2']
        s.np.testing.assert_allclose(result,expected,atol=1e-12);self.assertTrue((w>=.5).all())

    def test_predeclared_grid_size_and_no_date_specific_rules(self):
        self.assertEqual(len(s.grid()),104)
        for row in s.grid():
            for z in row['spec']:
                self.assertLessEqual(z['amount'],.1);self.assertNotIn('ticker',z);self.assertNotIn('date',z)

if __name__=='__main__':unittest.main()
