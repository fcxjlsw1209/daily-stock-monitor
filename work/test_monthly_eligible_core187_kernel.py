"""Synthetic rank-domain checks; never read stored market panels or returns."""
import copy
import unittest

import numpy as np
import pandas as pd

import monthly_eligible_core187_kernel as k


def sources(px, meta):
    r = px.pct_change(fill_method=None)
    sigma = r.rolling(63).std(ddof=1)
    calm = (sigma.SPY*np.sqrt(252.) < .10).loc[meta['dates']].to_numpy()
    mom = px.shift(42)/px.shift(252)-1.
    raw = (px.shift(42)/px.shift(294)-1.)/sigma.replace(0., np.nan)
    good = px.notna().rolling(295).sum().eq(295) & px.gt(0.) & mom.replace([np.inf,-np.inf],np.nan).gt(0.)
    rank = lambda x: x.replace([np.inf,-np.inf],np.nan).rank(method='average',pct=True).fillna(.5).to_numpy()-.5
    u = np.stack([rank((mom if calm[i] else raw).loc[day, ss])
                  for i, (day, ss) in enumerate(zip(meta['dates'], meta['symbols']))])
    eligible = np.stack([good.loc[day, ss].to_numpy() for day, ss in zip(meta['dates'], meta['symbols'])])
    cw = np.where(calm,.5,.52)[:,None]
    original = dict(core=cw*u, core_weight=cw, state_calm10=calm, eligible=eligible)
    source = dict(core=original['core'].copy(), core_weight=cw.copy())
    return original, source


def fixture(width=6, rows=3):
    dates = pd.bdate_range('2010-01-04', periods=520)
    t = np.arange(len(dates),dtype=float)
    names = ['S'+str(j) for j in range(width)]
    data = {s: (50.+j)*np.cumprod(1.+(.0004*(j+1) if j%3 != 1 else -.0008)
                +(.003+.0008*j)*np.sin(t*(.27+j*.03)+j)) for j,s in enumerate(names)}
    data['SPY'] = 100.*np.cumprod(1.+.0002+.013*np.sin(t*.31))
    px = pd.DataFrame(data,index=dates)
    signals = [dates[294+21*i] for i in range(rows)]
    meta = dict(dates=[str(d.date()) for d in signals],symbols=[names[:] for _ in signals],etfs=names[-1:])
    original, source = sources(px,meta)
    return dict(adj=px),meta,dates.strftime('%Y-%m-%d').tolist(),original,source


class RankTests(unittest.TestCase):
    def test_non_affine_removed_rank_gap(self):
        u = np.array([-.3,-.1,.1,.3,.5])
        c,a,m,z = k.eligible_comparisons(u,np.ones(5,bool),np.array([1,1,0,0,1],bool))
        np.testing.assert_allclose(c[m],[-1/6,1/6,.5],rtol=0,atol=1e-16)
        np.testing.assert_allclose(a[m],[-1/6,0.,.5],rtol=0,atol=1e-16)
        self.assertTrue(z['row_ready']);self.assertFalse(z['exact_equivalent'])
        self.assertEqual(a[0],c[0]);self.assertEqual(a[4],c[4])

    def test_two_distinct_levels_exactcopy_with_unequal_ties(self):
        u=np.array([-.4,-.1,-.1,-.1,.5,.5]);g=np.array([0,1,1,1,1,1],bool)
        c,a,m,z=k.eligible_comparisons(u,np.ones(6,bool),g)
        np.testing.assert_array_equal(c,a)
        np.testing.assert_array_equal(c[m],[2/5-.5,2/5-.5,2/5-.5,4.5/5-.5,4.5/5-.5])
        self.assertTrue(z['control_two_level_copy']);self.assertTrue(z['exact_equivalent'])

    def test_zero_rank_is_valid_but_missing_rank_is_not(self):
        u=np.array([-.4,0.,.2,.5]);g=np.array([1,1,0,1],bool)
        c,a,m,z=k.eligible_comparisons(u,np.ones(4,bool),g)
        self.assertTrue(m[1]);self.assertEqual(z['missing_eligible_core_count'],0)
        finite=np.array([1,0,1,1],bool)
        c,a,m,z=k.eligible_comparisons(u,finite,g)
        self.assertFalse(m.any());self.assertTrue(np.isnan(c).all())
        self.assertEqual(z['reason'],'missing_eligible_core')

    def test_noop_policies_and_initial_priority(self):
        for u,g,reason in [([-.2,.1,.5],[1,1,1],'all_eligible'),
                            ([-.2,.1,.5],[0,0,0],'fewer_than_two_eligible'),
                            ([-.2,.1,.5],[0,1,0],'fewer_than_two_eligible'),
                            ([-.2,.1,.1],[0,1,1],'equal_eligible_core')]:
            c,a,m,z=k.eligible_comparisons(u,np.ones(3,bool),np.array(g,bool))
            self.assertFalse(m.any());self.assertTrue(np.isnan(a).all());self.assertEqual(z['reason'],reason)
        c,a,m,z=k.eligible_comparisons([0.,.5],[False,True],[True,True],True)
        self.assertEqual(z['reason'],'initial_allocation');self.assertFalse(m.any())

    def test_permutation_and_no_name_rules(self):
        u=np.array([-.4,-.2,0.,.1,.3,.5]);g=np.array([1,0,1,1,0,1],bool);f=np.ones(6,bool)
        c,a,m,z=k.eligible_comparisons(u,f,g);order=[4,2,0,5,1,3]
        cc,aa,mm,zz=k.eligible_comparisons(u[order],f[order],g[order])
        np.testing.assert_array_equal(cc,c[order]);np.testing.assert_array_equal(aa,a[order])
        self.assertEqual(z,zz)

    def test_invalid_vector_types_and_nonfinite_rank_fail(self):
        for u,f,g,init in [([0.,np.inf],[True,True],[True,False],False),
                           ([0.,.1],[1,1],[True,False],False),
                           ([0.,.1],[True,True],[1,0],False),
                           ([[0.,.1]],[[True,True]],[[True,False]],False),
                           ([0.,.1],[True,True],[True,False],1)]:
            with self.assertRaises(ValueError):k.eligible_comparisons(u,f,g,init)


class SourceTests(unittest.TestCase):
    def test_original_formula_rank_eligibility_parity(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s)
        self.assertEqual(z['events'],3);self.assertEqual(z['core_rank_reconstructions'],3)
        self.assertEqual(z['predictive_model_fits'],0);self.assertGreater(z['ready_rows'],0)
        self.assertFalse(v['available'][0].any())
        for i,row in enumerate(z['event_history']):
            canonical=pd.Series(row['raw_core'],dtype=float).rank(method='average',pct=True).fillna(.5).to_numpy()-.5
            np.testing.assert_array_equal(row['core_rank'],canonical)
            np.testing.assert_allclose(canonical*a['core_weight'][i],a['core'][i],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(row['eligible'],a['eligible'][i])
        self.assertFalse((v['available']&~a['eligible']).any())

    def test_calm_source_and_full_eligible_noop(self):
        p,m,c,a,s=fixture();p['adj']['SPY']=100.*np.cumprod(1.+.0002+.002*np.sin(np.arange(520)*.31))
        for j,name in enumerate(m['symbols'][0]):
            p['adj'][name]=100.*np.exp((.0004+.0001*j)*np.arange(520))
        a,s=sources(p['adj'],m);v,z=k.build_features(p,m,c,a,s)
        self.assertTrue(a['state_calm10'].all());self.assertTrue(a['eligible'].all())
        self.assertFalse(v['available'].any());self.assertEqual(z['reason_counts']['all_eligible'],2)

    def test_source_missing_keeps_whole_original_eligibility(self):
        p,m,c,a,s=fixture();p['adj'].iloc[100,0]=np.nan;a,s=sources(p['adj'],m)
        v,z=k.build_features(p,m,c,a,s)
        self.assertFalse(a['eligible'][1,0]);self.assertFalse(v['available'][1,0])
        changed=copy.deepcopy(a);changed['eligible'][1,0]=True
        with self.assertRaisesRegex(ValueError,'eligibility parity'):k.build_features(p,m,c,changed,s)

    def test_endpoint_raw_does_not_add_complete_formation_gate(self):
        p,m,c,a,s=fixture();p['adj'].iloc[100,0]=np.nan;a,s=sources(p['adj'],m)
        v,z=k.build_features(p,m,c,a,s)
        # The raw endpoint/risk value remains in the original full-pool CDF,
        # although the inherited 295-session eligibility gate rejects it.
        self.assertTrue(z['event_history'][1]['core_valid'][0])
        self.assertFalse(z['event_history'][1]['eligible'][0])

    def test_infinite_endpoint_raw_is_missing_before_eligibility(self):
        p,m,c,a,s=fixture()
        p['adj']['S0']=1e-300
        p['adj'].loc[p['adj'].index[273]:,'S0']=1e300
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            a,s=sources(p['adj'],m)
            v,z=k.build_features(p,m,c,a,s)
        self.assertFalse(a['eligible'][1,0])
        self.assertFalse(z['event_history'][1]['core_valid'][0])
        self.assertIsNone(z['event_history'][1]['raw_core'][0])

    def test_prefix_futurepadding_and_precalendar_history(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s)
        for n in range(4):
            vv,zz=k.build_features(p,m,c,a,s,n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(zz['event_history'],z['event_history'][:n])
        pp=copy.deepcopy(p);pp['adj']=pp['adj'].astype(object)
        pp['adj'].loc[pp['adj'].index>pd.Timestamp(m['dates'][1])]='FUTURE'
        aa,ss=copy.deepcopy(a),copy.deepcopy(s)
        for obj,key in ((aa,'core'),(aa,'core_weight'),(ss,'core'),(ss,'core_weight')):
            obj[key]=obj[key].astype(object);obj[key][2:]='FUTURE'
        vv,zz=k.build_features(pp,m,c,aa,ss,2)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:2])
        vv,zz=k.build_features(p,m,c[10:],a,s)
        for key in v:np.testing.assert_array_equal(vv[key],v[key])

    def test_official_gap_is_not_compressed(self):
        p,m,c,a,s=fixture();missing=p['adj'].index[302]
        p['adj']=p['adj'].drop(index=missing)
        with self.assertRaisesRegex(ValueError,'eligibility parity'):
            k.build_features(p,m,c,a,s)
        filled=p['adj'].reindex(pd.DatetimeIndex(c));a,s=sources(filled,m)
        v,z=k.build_features(p,m,c,a,s)
        self.assertFalse(v['available'][1].any())

    def test_source_unused_fields_and_etf_labels_do_not_affect_ranks(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s)
        s.update(vol=object(),vol_weight=object(),state_highvol=object())
        m['etfs']=m['symbols'][0][:]
        vv,zz=k.build_features(p,m,c,a,s)
        for key in v:np.testing.assert_array_equal(vv[key],v[key])
        self.assertEqual(z,zz)

    def test_illegal_prices_metadata_and_parity_fail(self):
        p,m,c,a,s=fixture()
        for bad in (0.,-1.,np.inf,'bad'):
            pp=copy.deepcopy(p);pp['adj']=pp['adj'].astype(object);pp['adj'].iloc[10,0]=bad
            with self.assertRaises(ValueError):k.build_features(pp,m,c,a,s)
        for key in ('core','core_weight'):
            ss=copy.deepcopy(s);ss[key][0,0]+=.001
            with self.assertRaises(ValueError):k.build_features(p,m,c,a,ss)
        aa=copy.deepcopy(a);aa['state_calm10'][1]^=True
        with self.assertRaises(ValueError):k.build_features(p,m,c,aa,s)
        for bad in (True,-1,4):
            with self.assertRaises(ValueError):k.build_features(p,m,c,a,s,bad)
        mm=copy.deepcopy(m);mm['symbols'][0][1]=mm['symbols'][0][0]
        with self.assertRaises(ValueError):k.build_features(p,mm,c,a,s)

    def test_immutable_inputs(self):
        p,m,c,a,s=fixture();before=copy.deepcopy((p,m,c,a,s))
        v,z=k.build_features(p,m,c,a,s);k.build_deltas(v,a,s)
        pd.testing.assert_frame_equal(p['adj'],before[0]['adj']);self.assertEqual(m,before[1]);self.assertEqual(c,before[2])
        for obj,want in ((a,before[3]),(s,before[4])):
            for key in obj:np.testing.assert_array_equal(obj[key],want[key])

    def test_independent_crosscheck(self):
        import monthly_eligible_core187_independent as independent
        from test_monthly_eligible_core187_independent import fixture as other_fixture
        for make in (fixture,other_fixture):
            p,m,c,a,s=make();v,z=k.build_features(p,m,c,a,s)
            self.assertTrue(independent.verify_features(p,m,c,a,s,v,z)['passed'])
            d,dz=k.build_deltas(v,a,s)
            self.assertTrue(independent.verify_deltas(v,a,s,d,dz)['passed'])


class DeltaTests(unittest.TestCase):
    def test_formula_preserves_half_core_and_initial(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s);d,az=k.build_deltas(v,a,s)
        cw=np.broadcast_to(a['core_weight'],a['core'].shape)
        for key in d:
            want=np.zeros(a['core'].shape)
            want[v['available']]=.5*(cw[v['available']]*v[key][v['available']]-a['core'][v['available']])
            np.testing.assert_array_equal(want,d[key]);np.testing.assert_array_equal(d[key][0],0.)
            self.assertFalse((d[key][~a['eligible']]!=0.).any())
            np.testing.assert_allclose((a['core']+d[key])[v['available']],
                (.5*a['core']+.5*cw*v[key])[v['available']],rtol=0,atol=1e-16)
        self.assertFalse(az['episode_applied'])

    def test_shapes_prefix_and_invalid_availability(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s);d,az=k.build_deltas(v,a,s)
        for mode in ('vector','matrix'):
            aa,ss=copy.deepcopy(a),copy.deepcopy(s)
            for obj in (aa,ss):
                obj['core_weight']=obj['core_weight'][:,0] if mode=='vector' else np.broadcast_to(obj['core_weight'],(3,6)).copy()
            dd,_=k.build_deltas(v,aa,ss)
            for key in d:np.testing.assert_array_equal(dd[key],d[key])
        for n in range(4):
            small={key:x[:n] for key,x in v.items()};dd,_=k.build_deltas(small,a,s,n)
            for key in d:np.testing.assert_array_equal(dd[key],d[key][:n])
        with self.assertRaises(ValueError):k.build_deltas({key:x[:2] for key,x in v.items()},a,s)
        for row,col in ((0,0),(1,int(np.flatnonzero(~a['eligible'][1])[0]))):
            bad=copy.deepcopy(v);bad['available'][row,col]=True;bad['candidate'][row,col]=bad['control'][row,col]=0.
            with self.assertRaises(ValueError):k.build_deltas(bad,a,s)
        bad=copy.deepcopy(v);ids=np.flatnonzero(v['available'][1]);bad['available'][1,ids[0]]=False
        bad['candidate'][1,ids[0]]=bad['control'][1,ids[0]]=np.nan
        with self.assertRaises(ValueError):k.build_deltas(bad,a,s)


if __name__ == '__main__':
    unittest.main()
