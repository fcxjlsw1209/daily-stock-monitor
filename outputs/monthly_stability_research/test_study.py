import unittest
import stability as s
import local_refine as local


class StabilityStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.d,cls.L,cls.x,cls.px,cls.op,cls.f=s.load()

    def test_baseline_decisions_reproduce_frozen_full_history(self):
        ch,_=s.decisions({'baseline':True},self.m,self.d,self.L,self.x,self.f,122)
        got=[[self.m['symbols'][i][j] for j in ids] for i,ids in enumerate(ch)]
        expected=[o['symbols'] for o in s.read(s.ex.P/'return_primary_ORDERS.json')]
        self.assertEqual(got,expected)

    def test_future_prices_do_not_change_past_features(self):
        day='2018-06-29';i=self.m['dates'].index(day)
        one=dict(dates=[day],symbols=[self.m['symbols'][i]])
        cols=list(set(one['symbols'][0]))
        px=self.px.loc[:'2018-07-31',cols].copy()
        a=s.features(px,one)
        px.loc[px.index>day]*=3.5
        b=s.features(px,one)
        for key in a:s.np.testing.assert_allclose(a[key],b[key],rtol=0,atol=0)

    def test_skipped_momentum_and_aligned_risk_ignore_latest42_sessions(self):
        day='2018-06-29';i=self.m['dates'].index(day)
        one=dict(dates=[day],symbols=[self.m['symbols'][i]])
        px=self.px.loc[:day,one['symbols'][0]].copy();a=s.features(px,one)
        px.iloc[-42:]*=s.np.linspace(.5,2.,42)[:,None]
        b=s.features(px,one)
        for key in ['252_raw','294_raw','252_aligned','294_aligned','294_path','path','accel']:
            s.np.testing.assert_allclose(a[key],b[key],rtol=0,atol=0)

    def test_grid_obeys_long_core_and_monthly_rules(self):
        grid=s.specs();self.assertEqual(len(grid),1441)
        for row in grid[1:]:
            z=row['spec'];self.assertGreaterEqual(z['core_weight'],.5)
            self.assertIn(z['retain'],[3,5])
            self.assertAlmostEqual(sum(s.SUPPORTS[z['support']]),1.)
            self.assertGreaterEqual(1-z['core_weight']-.10,0.)

    def test_later_events_cannot_change_training_decisions(self):
        spec=s.specs()[317]['spec'];a=s.decisions(spec,self.m,self.d,self.L,self.x,self.f,88)[0]
        altered={k:v.copy() for k,v in self.f.items()}
        for v in altered.values():v[88:]=1000.
        b=s.decisions(spec,self.m,self.d,self.L,self.x,altered,88)[0]
        self.assertEqual(a,b)

    def test_weak_early_or_recent_performance_not_admitted(self):
        base=dict(training=dict(sharpe=1.5,max_drawdown=-.3),early=dict(cagr=.35),recent=dict(cagr=.6))
        row=dict(training=dict(sharpe=2.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.9))
        self.assertFalse(s.acceptable(row,base))
        row['early']['cagr']=.4;row['recent']['cagr']=.4
        self.assertFalse(s.acceptable(row,base))

    def test_zero_mixture_exactly_retains_baseline(self):
        spec=dict(alternative=s.specs()[12]['spec'],blend=0.,state='calm10',retain=3,
                  addon='none',addon_budget=0.,addon_state='global_')
        st=local.states(self.px,self.m,self.d)
        a=local.choose(spec,self.m,self.d,self.L,self.x,self.f,st,122)[0]
        b=s.decisions({'baseline':True},self.m,self.d,self.L,self.x,self.f,122)[0]
        self.assertEqual(a,b)

    def test_market_state_uses_only_past_prices(self):
        day='2018-06-29';i=self.m['dates'].index(day)
        one=dict(dates=[day]);d={k:v[i:i+1] for k,v in self.d.items()}
        px=self.px[['SPY','QQQ']].copy();a=local.states(px,one,d)
        px.loc[px.index>day]*=100.;b=local.states(px,one,d)
        for k in a:s.np.testing.assert_array_equal(a[k],b[k])

    def test_local_frozen_is_training_selected_with_constraints(self):
        frozen=s.read(s.P/'LOCAL_FROZEN.json');base=s.read(s.P/'TRAIN_TRIALS.json')[0]
        self.assertEqual(s.sha(s.P/'LOCAL_FROZEN.json'),(s.P/'LOCAL_FROZEN.sha256').read_text().strip())
        for z in frozen['selected'].values():
            self.assertTrue(s.acceptable(z,base))
            spec=z['spec'];cw=spec['alternative'].get('core_weight',.52)
            self.assertGreaterEqual((1-spec['blend'])*.52+spec['blend']*cw,.5)
            self.assertLessEqual(spec['addon_budget'],.1)


if __name__=='__main__':unittest.main()
