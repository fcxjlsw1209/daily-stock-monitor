from pathlib import Path
p=Path(__file__).resolve().parents[1]
s=(p/'work/test_monthly_price_delay140_runner.py').read_text().replace('price_delay140','formation142').replace('round140','round142').replace("'round':140","'round':142").replace('round=140','round=142')
start=s.index('class PlacementTests');end=s.index('\n\nclass SelectionTests',start)
s=s[:start]+'''class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.m,self.a,t,px,_=synthetic();self.b=FakeBackend(self.m,self.a,px)
        self.a.update(formation_rank=np.tile(np.linspace(-.45,.5,49),(122,12,1)),
            formation_count=np.full(122,12,dtype=np.int64),formation_ready=np.ones(122,bool),
            formation_valid=np.ones((122,49),bool))
    def test_baseline_only_calls_episode_once_for_three_arms(self):
        c,s,details=r.choices_all(self.a,self.m,1,self.b,POLICY)
        self.assertEqual(self.b.episodes,1)
        self.assertEqual(set(c),set(r.NAMES))
        for name in r.NAMES[1:]:
            np.testing.assert_array_equal(s[name],self.a['baseline'][:1])
            self.assertFalse(details[name][0]['fallback'])
    def test_missing_exact_order_and_final_score_fallback(self):
        self.a['formation_ready'][:]=False
        c,s,d=r.choices_all(self.a,self.m,2,self.b,POLICY)
        for name in r.NAMES[1:]:
            self.assertEqual(c[name],c[r.NAMES[0]])
            self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
    def test_unchanged_core_exact_order_and_final_score_fallback(self):
        self.a['formation_rank'][:]=self.a['core'][:,None,:]/self.a['core_weight'][:,None,None]
        c,s,d=r.choices_all(self.a,self.m,2,self.b,POLICY)
        for name in r.NAMES[1:]:
            self.assertTrue(all(x['fallback'] for x in d[name]))
            self.assertEqual(c[name],c[r.NAMES[0]])
            self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
    def test_future_suffix_not_read(self):
        c,s,_=r.choices_all(self.a,self.m,1,self.b,POLICY)
        for key in ('baseline','core','core_weight','effects'):
            self.a[key][1:]=np.nan
        self.a['formation_rank']=self.a['formation_rank'].astype(object)
        self.a['formation_rank'][1:]='future'
        d,t,_=r.choices_all(self.a,self.m,1,self.b,POLICY)
        self.assertEqual(c,d)
        for name in r.NAMES:self.assertEqual(s[name].tobytes(),t[name].tobytes())
    def test_input_nonmutation_and_prefix(self):
        before={k:v.tobytes() for k,v in self.a.items()}
        c,s,_=r.choices_all(self.a,self.m,3,self.b,POLICY)
        for n in range(4):
            d,t,_=r.choices_all(self.a,self.m,n,self.b,POLICY)
            for name in r.NAMES:
                self.assertEqual(d[name],c[name][:n]);self.assertEqual(t[name].tobytes(),s[name][:n].tobytes())
        self.assertEqual(before,{k:v.tobytes() for k,v in self.a.items()})
    def test_invalid_prefix(self):
        for n in (-1,123,True,.5):
            with self.assertRaises(ValueError):r.choices_all(self.a,self.m,n,self.b,POLICY)


class FeatureIntegrationTests(unittest.TestCase):
    def setUp(self):
        axis=pd.bdate_range('2015-01-01',periods=400)
        rng=np.random.default_rng(142)
        self.px=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0005,.01,(400,50)),axis=0)),index=axis,columns=['SPY']+['S'+str(i) for i in range(49)])
        self.m=dict(dates=[str(axis[-1].date())],symbols=[list(self.px.columns[1:])])
        self.sessions=[str(x.date()) for x in axis]
        stock=self.px.iloc[:,1:];sigma=stock.pct_change(fill_method=None).rolling(63).std().iloc[-1]
        calm=bool(self.px.SPY.pct_change(fill_method=None).rolling(63).std().iloc[-1]*np.sqrt(252)<.1)
        cw=.5 if calm else .52
        core=(stock.iloc[-43]/stock.iloc[-253 if calm else -295]-1)
        if not calm:core=core/sigma
        self.a=dict(state_calm10=np.array([calm]),core_weight=np.array([cw]),core=(cw*(core.rank(pct=True).fillna(.5)-.5)).to_numpy()[None,:])
    def test_exact_core_mask_and_historical_window(self):
        result=r.build_features({'adj':self.px},self.m,self.a,self.sessions)
        self.assertTrue(result['arrays']['formation_ready'][0]);self.assertTrue(result['arrays']['formation_valid'].all())
        self.assertEqual(result['arrays']['formation_count'][0],12)
        self.assertEqual(result['audit']['months'][0]['blocks'][-1]['end_price_index'],357)
    def test_rank_parity_failure_stops(self):
        self.a['core'][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'weighted core parity'):r.build_features({'adj':self.px},self.m,self.a,self.sessions)
    def test_calm_parity_failure_stops(self):
        self.a['state_calm10'][0]=not self.a['state_calm10'][0]
        with self.assertRaisesRegex(ValueError,'calm state parity'):r.build_features({'adj':self.px},self.m,self.a,self.sessions)
    def test_calendar_gap_stops(self):
        with self.assertRaisesRegex(ValueError,'official formation'):r.build_features({'adj':self.px.drop(self.px.index[200])},self.m,self.a,self.sessions)
    def test_future_price_suffix_does_not_change_features(self):
        expected=r.build_features({'adj':self.px},self.m,self.a,self.sessions)
        extra=pd.DataFrame(np.nan,index=pd.bdate_range(self.px.index[-1]+pd.Timedelta(days=1),periods=10),columns=self.px.columns)
        got=r.build_features({'adj':pd.concat([self.px,extra])},self.m,self.a,self.sessions)
        for key in r.EXTRA:np.testing.assert_array_equal(got['arrays'][key],expected['arrays'][key])
''' + s[end:]
start=s.index('        def features(');end=s.index('        self.runner=self.make_runner()',start)
s=s[:start]+'''        def features(panel,meta,old,sessions):
            arrays=dict(formation_rank=np.full((122,12,49),np.nan),formation_count=np.zeros(122,dtype=np.int64),
                formation_ready=np.zeros(122,bool),formation_valid=np.zeros((122,49),bool))
            months=[dict(signal_index=i,date=day,calm=True,horizon=252,ready=False,reason='synthetic_missing',blocks=[],original_finite_mask=[False]*49,selectors={}) for i,day in enumerate(meta['dates'])]
            return dict(arrays=arrays,audit=dict(months=months,new_model_fits=0))
        p=patch.object(r,'build_features',side_effect=features);p.start();self.addCleanup(p.stop)
''' + s[end:]
s=s.replace("len(r.read(self.out/'FEATURE_AUDIT.json')['cells']),5978","len(r.read(self.out/'FEATURE_AUDIT.json')['months']),122")
s=s.replace("{'delay_vs_incumbent','nonfit_vs_incumbent','delay_vs_nonfit'}","{'worst_vs_incumbent','mean_vs_incumbent','worst_vs_mean'}")
s=s.replace("['price_delay']","['formation_scenario']")
with (p/'work/test_monthly_formation142_runner.py').open('x') as f:f.write(s)
