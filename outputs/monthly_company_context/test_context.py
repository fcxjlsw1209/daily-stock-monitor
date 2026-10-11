import unittest
import context_research as s

class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.d,cls.px,cls.op,cls.a,cls.oldpred=s.load();cls.pred=dict(s.np.load(s.P/'FORECASTS.npz'))
        cls.data=dict(s.np.load(s.blend.P/'MONTHLY_TRAIN.npz'));cls.meta=s.read(s.blend.P/'MONTHLY_TRAIN.json');cls.details=s.read(s.P/'COMPANY_CONTEXT_DETAILS.json')

    def test_previous_orders122_exact(self):
        for row,name in [(s.grid()[0],'baseline'),(s.grid()[1],'primary')]:
            ch,sc=s.boundary_choices(row,self.a,self.pred,self.m,self.d,122);ref=s.read(s.old.P/(name+'_ORDERS.json'))
            for i,ids in enumerate(ch):
                self.assertEqual([self.m['symbols'][i][j] for j in ids],ref[i]['symbols']);s.np.testing.assert_allclose(sc[i,ids],ref[i]['scores'],atol=1e-12)

    def test_fixed_grid_count_and_uniqueness(self):
        rows=s.grid();self.assertEqual(len(rows),98);self.assertEqual(len(s.model_configs()),12)
        keys=[s.json.dumps({k:v for k,v in r.items() if k!='id'},sort_keys=True) for r in rows];self.assertEqual(len(keys),len(set(keys)))

    def test_all_forecasts_finite_and_global_identical(self):
        for v in self.pred.values():self.assertTrue(s.np.isfinite(v).all())
        s.np.testing.assert_allclose(self.pred['0_raw'],self.oldpred['M_return_100_raw'],atol=1e-12)

    def test_annual_mature_labels_only_and_future_data_independence(self):
        cutoff='2020-12-31';rows,ss,dates,ends=s.annual_rows(self.data,self.meta,cutoff)
        self.assertLessEqual(max(ends),cutoff);self.assertNotIn('2020-12-31',dates)
        altered={k:v.copy() for k,v in self.data.items()}
        for i,end in enumerate(self.meta['ends']):
            if end>cutoff:
                sl=slice(altered['offsets'][i],altered['offsets'][i+1]);altered['Y'][sl]=999;altered['good'][sl]=True;altered['X'][sl]=999
        got=s.annual_rows(altered,self.meta,cutoff)
        s.np.testing.assert_array_equal(got[0],rows);s.np.testing.assert_array_equal(altered['Y'][got[0]],self.data['Y'][rows]);s.np.testing.assert_array_equal(altered['X'][got[0]],self.data['X'][rows])

    def test_no_history_falls_back_to_global(self):
        p,info=s.analogue(s.np.array([]),[],s.np.empty((0,8)),s.np.empty((0,3)),s.np.zeros(8),s.np.zeros(3),'2020-01-31',.12,dict(kind='company_context',bandwidth=.5,prior_mass=6))
        self.assertEqual(p,.12);self.assertEqual(info['own_weight'],0.)

    def test_small_sample_is_shrunk_not_treated_as_certain(self):
        args=(s.np.array([1.]),['2020-01-31'],s.np.zeros((1,8)),s.np.zeros((1,3)),s.np.zeros(8),s.np.zeros(3),'2020-01-31',0.)
        p,info=s.analogue(*args,dict(kind='company_mean',prior_mass=6))
        self.assertAlmostEqual(p,1/7);self.assertAlmostEqual(info['own_weight'],1/7)
        p2,_=s.analogue(*args,dict(kind='company_mean',prior_mass=12));self.assertLess(p2,p)

    def test_realized_magnitude_not_win_rate(self):
        def estimate(y):return s.analogue(s.np.array(y),['2020-01-31']*2,s.np.zeros((2,8)),s.np.zeros((2,3)),s.np.zeros(8),s.np.zeros(3),'2020-01-31',0.,dict(kind='company_mean',prior_mass=6))[0]
        self.assertGreater(estimate([.20,-.01]),estimate([.01,-.20]))

    def test_matching_market_state_upweights_relevant_observation(self):
        args=(s.np.array([.20,-.20]),['2020-01-31']*2,s.np.zeros((2,8)),s.np.array([[1,1,1],[-1,-1,-1]]),s.np.zeros(8),s.np.ones(3),'2020-01-31',0.)
        mean,_=s.analogue(*args,dict(kind='company_shape',bandwidth=.5,prior_mass=6));context,_=s.analogue(*args,dict(kind='company_context',bandwidth=.5,prior_mass=6))
        self.assertAlmostEqual(mean,0.);self.assertGreater(context,0.)

    def test_zero_feature_rows_recover_only_same_date_context(self):
        names=self.meta['feature_names'];n=len(names);raw=s.np.zeros((4,n));raw[0]=.2;raw[2]=.3;st=s.np.array([1,1,-1,-1])
        X=s.np.column_stack([raw,raw*st[:,None],raw*st[:,None],raw*st[:,None]])
        _,state=s.context_features(X,names,s.np.array([0,2,4]));s.np.testing.assert_array_equal(state,s.np.tile(st[:,None],(1,3)))

    def test_top_two_locked_and_only_one_name_can_change(self):
        basech,base=s.boundary_choices(s.grid()[1],self.a,self.pred,self.m,self.d,122)
        for row in s.grid()[2:]:
            ch,_=s.boundary_choices(row,self.a,self.pred,self.m,self.d,122)
            for i,picks in enumerate(ch):
                rank=sorted(s.np.flatnonzero(self.d['eligible'][i]),key=lambda j:(-base[i,j],self.m['symbols'][i][j]))
                if len(rank)>=3:
                    self.assertTrue(set(rank[:2]).issubset(picks));self.assertLessEqual(len(set(picks)-set(basech[i])),1)
                    for j in set(picks)-set(basech[i]):
                        self.assertIn(j,rank[:row['ceiling']]);self.assertLessEqual(base[i,rank[2]]-base[i,j],row['gap']+1e-12)
                self.assertIn(len(picks),[0,2,3]);self.assertTrue(self.d['eligible'][i,picks].all())

    def test_no_signal_returns_global_not_company_bump(self):
        n=0
        for z in self.details:
            if z['model_id']>=8 and z['signal_type']<=0:
                self.assertEqual(z['prediction'],z['global_prediction']);self.assertEqual(z['own_weight'],0);n+=1
        self.assertGreater(n,0)

    def test_own_history_samples_have_mature_labels_and_shrink(self):
        for z in self.details:
            if z['latest_label_end'] is not None:self.assertLessEqual(z['latest_label_end'],z['cutoff'])
            self.assertLessEqual(z['cutoff'],z['date']);self.assertGreaterEqual(z['own_weight'],0);self.assertLess(z['own_weight'],1)
            self.assertLessEqual(z['observations'],60)

    def test_signal_state_exact_matching_and_symbol_identity(self):
        sig=dict(s.np.load(s.P/'SIGNALS.npz'));audit={z['year']:z['cutoff'] for z in s.read(s.P/'FORECAST_AUDIT.json')};cache={}
        for year,cutoff in audit.items():cache[cutoff]=s.annual_rows(self.data,self.meta,cutoff)
        for z in self.details[::127]:
            rows,ss,dates,ends=cache[z['cutoff']];rr=rows[ss==z['symbol']]
            if z['model_id']>=8:rr=rr[sig['train'][rr]==z['signal_type']] if z['signal_type']>0 else s.np.array([],int)
            self.assertEqual(len(rr),z['observations'])

    def test_prediction_future_tail_cannot_change_training_choices(self):
        pred={k:v.copy() for k,v in self.pred.items()}
        for v in pred.values():v[88:]=999
        for row in s.grid()[::7]:self.assertEqual(s.boundary_choices(row,self.a,self.pred,self.m,self.d,88)[0],s.boundary_choices(row,self.a,pred,self.m,self.d,88)[0])

if __name__=='__main__':unittest.main()
