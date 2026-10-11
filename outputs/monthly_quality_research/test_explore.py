import unittest
import explore as e


class QualityStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.d,cls.L,cls.x,cls.px,cls.op,cls.f,cls.a=e.load()
        cls.sectors=e.pd.read_csv(e.O/'index_union/alias_resolved_inputs/sectors.csv.gz',index_col=0,parse_dates=True,low_memory=False)

    def test_zero_change_exact122_observed_candidate(self):
        ch,sc=e.choices([],self.a,self.m,self.d,122)
        rows=e.read(e.old.P/'robust_control_ORDERS.json')
        for i,ids in enumerate(ch):
            self.assertEqual([self.m['symbols'][i][j] for j in ids],rows[i]['symbols'])
            e.np.testing.assert_allclose(sc[i,ids],rows[i]['scores'],atol=1e-12,rtol=0)

    def test_training_is_unchanged_by_later_arrays(self):
        a={k:v.copy() for k,v in self.a.items()}
        for v in a.values():v[88:]=0
        for row in e.grid(a)[::13]:
            self.assertEqual(e.choices(row['spec'],self.a,self.m,self.d,88)[0],e.choices(row['spec'],a,self.m,self.d,88)[0])

    def test_joint_residual_removes_known_market_sector_beta(self):
        rng=e.np.random.default_rng(21);r=e.pd.DataFrame(rng.normal(0,.01,(400,2)),columns=['SPY','XLK'])
        r['A']=2*r.SPY+.5*r.XLK
        px=(1+r).cumprod()*100;sectors=e.pd.DataFrame({'A':'Technology'},index=r.index)
        residual=e.residual_returns(px,sectors,126)
        e.np.testing.assert_allclose(residual.A.iloc[130:],0,atol=1e-12)

    def test_residual_prefix_and_latest42_exclusion(self):
        cols=['SPY','XLK','AAPL'];px=self.px[cols].loc[:'2019-12-31'].copy();sectors=self.sectors.reindex(px.index)
        original=e.residual_returns(px,sectors,126)
        cutoff='2018-11-30';changed=px.copy();changed.loc[changed.index>cutoff]*=3
        perturbed=e.residual_returns(changed,sectors,126)
        e.pd.testing.assert_frame_equal(original.loc[:cutoff],perturbed.loc[:cutoff])
        a=original.rolling(210).sum().shift(42).iloc[-1]
        changed=px.copy();changed.iloc[-42:]*=e.np.linspace(.8,1.2,42)[:,None]
        b=e.residual_returns(changed,sectors,126).rolling(210).sum().shift(42).iloc[-1]
        e.np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)

    def test_etfs_are_neutral_in_stock_residual_factor(self):
        for key,arr in self.a.items():
            if not key.startswith('resid_'):continue
            for i,ss in enumerate(self.m['symbols']):
                for j,s in enumerate(ss):
                    if s in self.m['etfs']:self.assertEqual(arr[i,j],0)

    def test_feature_prefix_matches_full(self):
        day='2017-06-30';i=self.m['dates'].index(day)
        m=dict(dates=[day],symbols=[self.m['symbols'][i]],etfs=self.m['etfs'])
        d={k:v[i:i+1] for k,v in self.d.items()};x={k:v[i:i+1] for k,v in self.x.items()};f={k:v[i:i+1] for k,v in self.f.items()}
        L={k:dict(score=v['score'][i:i+1],active=v['active'][i:i+1]) for k,v in self.L.items()}
        a,st=e.make_features(self.px.loc[:day],m,self.sectors.loc[:day],d,L,x,f)
        for key,v in a.items():e.np.testing.assert_allclose(v,self.a[key][i:i+1],atol=1e-12,rtol=0,err_msg=key)

    def test_finite_grid_and_support_budgets(self):
        rows=e.grid(self.a);self.assertEqual(len(rows),163)
        for row in rows[1:]:
            z=row['spec'][0]
            if z['family']=='residual':self.assertLessEqual(z['amount'],.1)
            else:self.assertLessEqual(z['amount'],1)
            self.assertIn(z['state'],['global_','calm10','highvol'])

    def test_segment_constraint_prevents_recent_only_win(self):
        base=dict(training=dict(sharpe=1.6,max_drawdown=-.31),early=dict(cagr=.45),recent=dict(cagr=.7))
        row=dict(training=dict(sharpe=2,max_drawdown=-.2),early=dict(cagr=.4),recent=dict(cagr=1.5))
        self.assertFalse(e.allowed(row,base))


if __name__=='__main__':unittest.main()
