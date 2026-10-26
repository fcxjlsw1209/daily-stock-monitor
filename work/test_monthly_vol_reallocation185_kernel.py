"""Synthetic-only source, algebra and prefix tests for the fixed three doses."""
import copy
import unittest

import numpy as np
import pandas as pd

import monthly_vol_reallocation185_kernel as k


def fixture(width=6, rows=3):
    dates=pd.bdate_range('2010-01-04',periods=520)
    t=np.arange(len(dates),dtype=float);names=['S'+str(j) for j in range(width)]
    data={s:(50.+j)*np.cumprod(1.+.0003*(j+1)+(.003+.0008*j)*np.sin(t*(.27+j*.03)+j))
          for j,s in enumerate(names)}
    data['SPY']=100.*np.cumprod(1.+.0002+.004*np.sin(t*.31))
    px=pd.DataFrame(data,index=dates)
    chosen=[dates[294+21*i] for i in range(rows)]
    meta=dict(dates=[str(d.date()) for d in chosen],symbols=[names[:] for _ in chosen],etfs=names[-1:])
    a,s=sources(px,meta)
    return dict(adj=px),meta,dates.strftime('%Y-%m-%d').tolist(),a,s


def sources(px,meta):
    r=px.pct_change(fill_method=None);s63=r.rolling(63).std();s126=r.rolling(126).std()
    calm=(s63.SPY*np.sqrt(252.)<.10).loc[meta['dates']].to_numpy()
    high=(s63.SPY>s63.SPY.rolling(252).median()).loc[meta['dates']].to_numpy()
    ranks=lambda x:x.rank(method='average',pct=True).fillna(.5).to_numpy()-.5
    v=np.stack([ranks(s126.loc[day,ss]) for day,ss in zip(meta['dates'],meta['symbols'])])
    cw=np.where(calm,.5,.52)[:,None];vw=np.where(calm,.2,.219375*(1.-np.where(high,.10,0.)/.48))[:,None]
    core=cw*np.broadcast_to(np.linspace(-.4,.4,v.shape[1]),v.shape)
    other=np.broadcast_to(np.linspace(.3,-.3,v.shape[1]),v.shape)
    baseline=core+vw*v+(1.-cw-vw)*other
    a=dict(baseline=baseline,core=core,core_weight=cw,state_calm10=calm,
           eligible=np.ones(v.shape,bool),state_weak_narrow=np.zeros(len(calm),bool))
    s=dict(vol=v,vol_weight=vw,state_highvol=high,core=core.copy(),core_weight=cw.copy())
    return a,s


class FeatureTests(unittest.TestCase):
    def test_independent_complete_audit_on_both_synthetic_fixtures(self):
        import monthly_vol_reallocation185_independent as independent
        from test_monthly_vol_reallocation185_independent import fixture as other_fixture
        for make in (fixture,other_fixture):
            p,m,c,a,s=make()
            values,audit=k.build_features(p,m,c,a,s)
            self.assertTrue(independent.verify_features(p,m,c,a,s,values,audit)['passed'])
            deltas,detail=k.build_deltas(values,a,s)
            self.assertTrue(independent.verify_deltas(values,a,s,deltas,detail)['passed'])

    def test_exact_original_rank_and_unit_formula(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s)
        np.testing.assert_array_equal(v['vol_rank'],s['vol'])
        np.testing.assert_array_equal(v['unit_delta'],(1.-a['core_weight'])*s['vol']-(a['baseline']-a['core']))
        self.assertEqual(z['checked_cells'],18);self.assertEqual(z['source_vol_finite_cells'],18)
        self.assertEqual(z['volatility_rank_reconstructions'],3);self.assertEqual(z['predictive_model_fits'],0)
        self.assertTrue(z['no_new_availability_filter']);self.assertFalse(z['ranking_eligibility_filtered'])
        for row in z['event_history']:
            self.assertEqual(sum(row['vol_domain']),6)
            np.testing.assert_allclose(row['nonvol_donor_budget'],.3,rtol=0,atol=1e-15)

    def test_missing_neutral_rank_is_not_an_extra_delta_mask(self):
        p,m,c,a,s=fixture();p['adj'].loc[p['adj'].index[200:280],'S0']=np.nan
        a,s=sources(p['adj'],m);v,z=k.build_features(p,m,c,a,s)
        self.assertEqual(v['vol_rank'][1,0],0.)
        self.assertFalse(z['event_history'][1]['vol_domain'][0]);self.assertGreater(z['source_vol_missing_neutral_cells'],0)
        d,az=k.build_deltas(v,a,s)
        self.assertNotEqual(d['vol_reallocate_0050'][1,0],0.)
        self.assertEqual(d['vol_reallocate_0050'][1,0],-.05*(a['baseline'][1,0]-a['core'][1,0]))

    def test_ties_zero_vol_and_original_full_domain(self):
        p,m,c,a,s=fixture();p['adj']['S0']=100.;p['adj']['S1']=100.
        a,s=sources(p['adj'],m);v,z=k.build_features(p,m,c,a,s)
        self.assertTrue(all(row['raw_vol'][0]==row['raw_vol'][1]==0. for row in z['event_history']))
        self.assertTrue(all(row['vol_domain'][0] for row in z['event_history']))
        np.testing.assert_array_equal(v['vol_rank'][:,0],v['vol_rank'][:,1])
        a['eligible'][:]=False;m['etfs']=m['symbols'][0]
        vv,zz=k.build_features(p,m,c,a,s)
        for key in v:np.testing.assert_array_equal(vv[key],v[key])
        self.assertEqual(zz,z)

    def test_no_new_core_warmup_requirement(self):
        p,m,c,a,s=fixture();p['adj'].iloc[0,0]=np.nan
        # No original core reconstruction or common-support gate is introduced.
        v,z=k.build_features(p,m,c,a,s)
        self.assertEqual(z['source_vol_finite_cells'],18)
        self.assertTrue(np.isfinite(v['unit_delta']).all())

    def test_official_missing_day_and_precalendar_history(self):
        p,m,c,a,s=fixture();q=copy.deepcopy(p);day=q['adj'].index[260]
        q['adj']=q['adj'].drop(day);full=q['adj'].reindex(p['adj'].index);aa,ss=sources(full,m)
        v,z=k.build_features(q,m,c,aa,ss)
        self.assertTrue(all(not x for x in z['event_history'][0]['vol_domain']))
        self.assertEqual(z['source_vol_missing_neutral_cells'],18)
        # Starting the official calendar after the already-cached warmup does
        # not truncate the original rolling estimator's prefix.
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c[100:],a,s)
        np.testing.assert_array_equal(v['vol_rank'],s['vol'])

    def test_all_prefixes_empty_and_future_price_padding(self):
        p,m,c,a,s=fixture();full,z=k.build_features(p,m,c,a,s)
        fd,fz=k.build_deltas(full,a,s)
        for n in range(4):
            v,q=k.build_features(p,m,c,a,s,n);d,da=k.build_deltas(v,a,s,n)
            for key in full:np.testing.assert_array_equal(v[key],full[key][:n])
            for key in fd:np.testing.assert_array_equal(d[key],fd[key][:n])
            self.assertEqual(q['event_history'],z['event_history'][:n])
        p['adj']=p['adj'].astype(object)
        p['adj'].loc[p['adj'].index>pd.Timestamp(m['dates'][0])]='future poison'
        for container in (a,s):
            for name,x in list(container.items()):
                if x.dtype.kind!='b':
                    container[name]=x.astype(object);container[name][1:]='future poison'
        v,q=k.build_features(p,m,c,a,s,1)
        for key in full:np.testing.assert_array_equal(v[key],full[key][:1])
        d,da=k.build_deltas(v,a,s,1)
        for x in d.values():np.testing.assert_array_equal(x,0.)

    def test_wrong_rank_weight_and_state_are_rejected(self):
        p,m,c,a,s=fixture()
        for key in ('vol','vol_weight'):
            bad=copy.deepcopy(s);bad[key][0,0]+=.001
            with self.assertRaises(ValueError):k.build_features(p,m,c,a,bad)
        bad=copy.deepcopy(s);bad['state_highvol'][0]^=True
        with self.assertRaises(ValueError):k.build_features(p,m,c,a,bad)
        bad=copy.deepcopy(a);bad['state_calm10'][0]^=True
        with self.assertRaises(ValueError):k.build_features(p,m,c,bad,s)

    def test_invalid_observed_prices_and_arithmetic_overflow(self):
        p,m,c,a,s=fixture()
        for bad in (0.,-1.,np.inf,'bad'):
            q=copy.deepcopy(p);q['adj']=q['adj'].astype(object);q['adj'].iloc[20,0]=bad
            with self.assertRaises(ValueError):k.build_features(q,m,c,a,s)
        q=copy.deepcopy(p);q['adj'].iloc[20,0]=1e-300;q['adj'].iloc[21,0]=1e300
        with self.assertRaises(FloatingPointError):k.build_features(q,m,c,a,s)
        q=copy.deepcopy(p);q['adj'].iloc[200,0]=1.;q['adj'].iloc[201,0]=1e200
        with self.assertRaises(FloatingPointError):k.build_features(q,m,c,a,s)

    def test_invalid_axes_and_source_shapes(self):
        p,m,c,a,s=fixture()
        for lim in (-1,4,True,1.5):
            with self.assertRaises(ValueError):k.build_features(p,m,c,a,s,lim)
        for key in ('dates','symbols','etfs'):
            mm=copy.deepcopy(m);del mm[key]
            with self.assertRaises(ValueError):k.build_features(p,mm,c,a,s)
        for key in ('baseline','core','core_weight'):
            aa=copy.deepcopy(a);aa[key]=aa[key][:-1]
            with self.assertRaises(ValueError):k.build_features(p,m,c,aa,s)
        bad=copy.deepcopy(s);bad['vol']=bad['vol'].astype(object);bad['vol'][0,0]='bad'
        with self.assertRaises(ValueError):k.build_features(p,m,c,a,bad)

    def test_inputs_byte_and_frame_unchanged(self):
        p,m,c,a,s=fixture();before=copy.deepcopy((p,m,c,a,s))
        v,z=k.build_features(p,m,c,a,s);k.build_deltas(v,a,s)
        pd.testing.assert_frame_equal(p['adj'],before[0]['adj']);self.assertEqual(m,before[1]);self.assertEqual(c,before[2])
        for got,want in ((a,before[3]),(s,before[4])):
            for name in got:
                self.assertEqual(got[name].dtype,want[name].dtype);self.assertEqual(got[name].tobytes(),want[name].tobytes())


class DeltaTests(unittest.TestCase):
    def test_fixed_doses_zero_identity_and_partial_initial(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s);d,az=k.build_deltas(v,a,s)
        self.assertEqual(set(d),set(k.LAMBDAS))
        self.assertTrue(np.any(v['unit_delta'][0]!=0.))
        np.testing.assert_array_equal(k.scaled_delta(v['unit_delta'],0.),0.)
        for name,amount in k.LAMBDAS.items():
            np.testing.assert_array_equal(d[name][0],0.)
            np.testing.assert_array_equal(d[name][1:],amount*v['unit_delta'][1:])
            self.assertEqual(az['changed_cells'][name],np.count_nonzero(d[name]))
        for bad in (.075,-.025,.11,True,np.nan):
            with self.assertRaises(ValueError):k.scaled_delta(v['unit_delta'],bad)

    def test_original_nonvol_budget_reallocation_algebra(self):
        # Multiple distinct nonvol terms: no re-ranking, new information, or
        # donor selection. This compares the original component arithmetic.
        for calm,high in ((True,False),(False,False),(False,True)):
            cw=.5 if calm else .52;vw=.2 if calm else .219375*(1.-(.10 if high else 0.)/.48)
            donor=1.-cw-vw;w=np.array([.2,.3,.5])*donor
            f=np.array([[-.4,.3,.1],[.5,-.2,.4],[-.1,.2,-.3]])
            v=np.array([.5,-.25,.1]);core=np.array([-.2,.0,.2])
            baseline=core+vw*v+f@w
            unit=(1.-cw)*v-(baseline-core)
            for amount in k.LAMBDAS.values():
                explicit=core+(vw+amount*donor)*v+f@((1.-amount)*w)
                np.testing.assert_allclose(baseline+amount*unit,explicit,rtol=0,atol=1e-16)
                np.testing.assert_allclose(amount*unit,amount*np.sum(w*(v[:,None]-f),axis=1),rtol=0,atol=1e-16)
                self.assertTrue(np.all((.85-amount)*w>0.))

    def test_weight_broadcast_three_shapes_and_all_columns_checked(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s);expected,_=k.build_deltas(v,a,s)
        for mode in ('vector','wide'):
            aa,ss=copy.deepcopy(a),copy.deepcopy(s)
            for dest,key in ((aa,'core_weight'),(ss,'core_weight'),(ss,'vol_weight')):
                x=dest[key];dest[key]=x[:,0] if mode=='vector' else np.broadcast_to(x,a['baseline'].shape).copy()
            got,_=k.build_deltas(v,aa,ss)
            for name in got:np.testing.assert_array_equal(got[name],expected[name])
        aa=copy.deepcopy(a);aa['core_weight']=np.broadcast_to(aa['core_weight'],a['baseline'].shape).copy()
        aa['core_weight'][1,-1]+=.001
        with self.assertRaises(ValueError):k.build_deltas(v,aa,s)

    def test_tampering_bad_components_and_no_eligibility_dependency(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s);expected,_=k.build_deltas(v,a,s)
        for key in v:
            bad=copy.deepcopy(v);bad[key][1,0]+=.0001
            with self.assertRaises(ValueError):k.build_deltas(bad,a,s)
        for key in ('baseline','core'):
            for value in (np.nan,np.inf):
                aa=copy.deepcopy(a);aa[key][1,0]=value
                with self.assertRaises(ValueError):k.build_deltas(v,aa,s)
        a['eligible']='not inspected';a['state_weak_narrow']='not inspected'
        got,_=k.build_deltas(v,a,s)
        for name in got:np.testing.assert_array_equal(got[name],expected[name])


if __name__=='__main__':unittest.main()
