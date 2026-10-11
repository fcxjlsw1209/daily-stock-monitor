import unittest
import blend_research as s

class BlendResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.d,cls.L,cls.x,cls.px,cls.op,cls.f,cls.a,cls.recs=s.load()
        cls.meta=s.read(s.P/'MONTHLY_TRAIN.json');cls.data=dict(s.np.load(s.P/'MONTHLY_TRAIN.npz'));cls.pred=dict(s.np.load(s.P/'PREDICTIONS.npz'))

    def test_previous_controls_exact122_decisions(self):
        for row,name in [(s.grid(self.recs)[0],'baseline'),(s.grid(self.recs)[2],'primary')]:
            ch,sc=s.old.choices(row['spec'],self.a,self.m,self.d,122);ref=s.read(s.old.P/(name+'_ORDERS.json'))
            for i,ids in enumerate(ch):
                self.assertEqual([self.m['symbols'][i][j] for j in ids],ref[i]['symbols']);s.np.testing.assert_allclose(sc[i,ids],ref[i]['scores'],atol=1e-12)

    def test_monthly_labels_end_at_next_actual_monthend(self):
        for date,end in zip(self.meta['dates'],self.meta['ends']):
            a,b=s.pd.Timestamp(date),s.pd.Timestamp(end)
            self.assertEqual(b.to_period('M'),a.to_period('M')+1)
            self.assertEqual(a,self.px.index[self.px.index.to_period('M')==a.to_period('M')][-1])
            self.assertEqual(b,self.px.index[self.px.index.to_period('M')==b.to_period('M')][-1])
        self.assertEqual(len(self.meta['dates']),180)

    def test_future_month_does_not_enter_year_training(self):
        cutoff='2020-12-31';ids=s.old.training_indices(self.meta,cutoff)
        self.assertIn('2020-11-30',[self.meta['dates'][i] for i in ids]);self.assertNotIn('2020-12-31',[self.meta['dates'][i] for i in ids])
        altered={k:v.copy() for k,v in self.data.items()}
        for i,end in enumerate(self.meta['ends']):
            if end>cutoff:
                sl=slice(altered['offsets'][i],altered['offsets'][i+1]);altered['Y'][sl]=1;altered['adverse'][sl]=1;altered['X'][sl]=2
        for target in ['return','utility','adverse']:
            f1,_=s.fit_point(self.data,self.meta,cutoff,target,10);f2,_=s.fit_point(altered,self.meta,cutoff,target,10)
            s.np.testing.assert_array_equal(f1['coef'],f2['coef']);self.assertEqual(f1['intercept'],f2['intercept'])

    def test_label_path_and_missing_day_not_imputed(self):
        px=s.pd.DataFrame(dict(A=[100,80,120],B=[100,s.np.nan,120],SPY=[100,100,110]),index=s.pd.to_datetime(['2020-01-31','2020-02-03','2020-02-28']))
        y,a,g=s.label_to_end(px,px.index[0],px.index[-1],['A','B']);s.np.testing.assert_allclose(y,[.1,.1]);self.assertAlmostEqual(a[0],.2);self.assertEqual(g.tolist(),[True,False])

    def test_rank_mixture_endpoints_and_interpolation(self):
        a=self.pred['W_return_10_rank'];b=self.pred['M_return_10_rank']
        for w in [0.,.25,1.]:
            rec=dict(family='frequency_mix',terms=[('W_return_10',w),('M_return_10',1-w)])
            got=s.recipe_scores(rec,self.pred,self.a,self.d['eligible']);s.np.testing.assert_allclose(got,w*a+(1-w)*b,atol=1e-12)

    def test_utility_subtracts_adverse_not_adds(self):
        pred={'M_return_10_raw':s.np.array([[.10,.11,.12]]),'M_adverse_10_raw':s.np.array([[0.,.20,.01]])}
        recipe=dict(family='economic_utility',mode='M',return_alpha=10,risk_alpha=10,penalty=.5)
        z=s.recipe_scores(recipe,pred,{},s.np.ones((1,3),bool));self.assertEqual(s.np.argsort(-z[0]).tolist(),[2,0,1])

    def test_regime_switch_uses_known_weak_gate(self):
        rec=dict(family='regime_mix',mode='W',return_alpha=10,utility_alpha=100)
        z=s.recipe_scores(rec,self.pred,self.a,self.d['eligible']);w=s.np.where(self.a['state_weak_narrow'][:,None],.25,.75)
        s.np.testing.assert_allclose(z,w*self.pred['W_return_10_rank']+(1-w)*self.pred['W_utility_100_rank'])

    def test_original_weekly_predictions_exact(self):
        for key,oldkey in [('W_return_10','ml_0'),('W_return_100','ml_1'),('W_utility_10','ml_4'),('W_utility_100','ml_5')]:s.np.testing.assert_allclose(self.pred[key+'_rank'],self.a[oldkey],atol=1e-12)

    def test_forecast_audit_all_dates_prior_to_signal(self):
        audit=s.read(s.P/'FORECAST_AUDIT.json');self.assertEqual(len(audit),132)
        for r in audit:
            self.assertLessEqual(r['latest_label_end'],r['cutoff'])
            for date,end in zip(self.m['dates'],self.m['ends']):
                if s.pd.Timestamp(end).year==r['year']:self.assertLessEqual(r['cutoff'],date)
        for k,v in self.pred.items():
            self.assertTrue(s.np.isfinite(v).all())
            if 'adverse' in k and k.endswith('_raw'):self.assertTrue(((v>=0)&(v<=1)).all())

    def test_recipes_finite_and_duplicate_free(self):
        self.assertEqual(len(self.recs),80);self.assertEqual(len(s.grid(self.recs)),963)
        keys=[s.json.dumps({k:v for k,v in r.items() if k!='id'},sort_keys=True) for r in self.recs];self.assertEqual(len(keys),len(set(keys)))

    def test_later_features_do_not_change_training_choices(self):
        changed={k:v.copy() for k,v in self.a.items()}
        for a in changed.values():a[88:]=0
        for row in s.grid(self.recs)[::79]:self.assertEqual(s.old.choices(row['spec'],self.a,self.m,self.d,88)[0],s.old.choices(row['spec'],changed,self.m,self.d,88)[0])

    def test_pool_origin_known_and_features_finite(self):
        for date,origin in zip(self.meta['dates'],self.meta['pool_origins']):
            if origin=='inherited_pre2016_historical_proxy':self.assertLess(date,'2016-08-17')
            else:self.assertLessEqual(origin,date)
        self.assertTrue(s.np.isfinite(self.data['X']).all())

if __name__=='__main__':unittest.main()
