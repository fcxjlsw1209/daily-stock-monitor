"""Pure synthetic tests. No stored feature panels or strategy returns are read."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

import monthly_core_peer184_kernel as k


def fixture(width=6, rows=3):
    dates = pd.bdate_range('2010-01-04', periods=520)
    t = np.arange(len(dates), dtype=float)
    names = ['S'+str(j) for j in range(width)]
    data = {s: (50.+j)*np.cumprod(1.+.0003*(j+1)+(.003+.0008*j)*np.sin(t*(.27+j*.03)+j))
            for j, s in enumerate(names)}
    data['SPY'] = 100.*np.cumprod(1.+.0002+.004*np.sin(t*.31))
    px = pd.DataFrame(data, index=dates)
    selected = [dates[294+21*i] for i in range(rows)]
    meta = dict(dates=[str(d.date()) for d in selected], symbols=[names[:] for _ in selected], etfs=names[-1:])
    original, source = sources(px, meta)
    return dict(adj=px), meta, dates.strftime('%Y-%m-%d').tolist(), original, source


def sources(px, meta):
    r = px.pct_change(fill_method=None)
    s63, s126 = r.rolling(63).std(), r.rolling(126).std()
    calm = (s63.SPY*np.sqrt(252.) < .10).loc[meta['dates']].to_numpy()
    high = (s63.SPY > s63.SPY.rolling(252).median()).loc[meta['dates']].to_numpy()
    c = px.shift(42)/px.shift(252)-1.
    o = (px.shift(42)/px.shift(294)-1.)/s63.replace(0., np.nan)
    def rank(x): return x.replace([np.inf,-np.inf], np.nan).rank(method='average', pct=True).fillna(.5).to_numpy()-.5
    u = np.stack([rank((c if calm[i] else o).loc[day, ss]) for i,(day,ss) in enumerate(zip(meta['dates'],meta['symbols']))])
    v = np.stack([rank(s126.loc[day,ss]) for day,ss in zip(meta['dates'],meta['symbols'])])
    cw = np.where(calm,.5,.52)[:,None]; vw = np.where(calm,.2,.219375*(1.-np.where(high,.10,0.)/.48))[:,None]
    shape = u.shape
    a = dict(core=cw*u, core_weight=cw, state_calm10=calm,
             state_weak_narrow=np.arange(len(calm))%2 == 1,
             extension_available=np.ones(shape,bool), forecast_available=np.ones(shape,bool),
             eligible=np.ones(shape,bool), rank_log_excess=np.zeros(shape))
    source = dict(vol=v,vol_weight=vw,state_highvol=high,core=a['core'].copy(),core_weight=cw.copy())
    return a, source


def vector_rank(x):
    return pd.Series(x).replace([np.inf,-np.inf],np.nan).rank(method='average',pct=True).fillna(.5).to_numpy()-.5


class PairTests(unittest.TestCase):
    def measure(self,u,s):return k.peer_comparisons(np.array(u,float),np.array(s,float),vector_rank(s),np.ones(len(u),bool))

    def test_symmetric_fixedN_and_matched_mass(self):
        u=[-.4,-.3,.1,.2];s=[3.,1.,4.,2.]
        c,d,mask,a=self.measure(u,s)
        self.assertTrue(mask.all());self.assertEqual(a['unordered_pair_comparisons'],6)
        self.assertEqual(a['nonzero_unordered_pair_comparisons'],2)
        self.assertAlmostEqual(a['mean_peer_weight'],1./6.)
        np.testing.assert_allclose(c,[.1875,.0625,.1875,.0625],atol=1e-15)
        np.testing.assert_allclose(d,.125+(vector_rank(s)-.125)/6.,atol=1e-15)
        self.assertAlmostEqual(c.sum(),.5);self.assertAlmostEqual(d.sum(),.5)

    def test_ties_exactcopy_and_zero_variance_valid(self):
        for u,s,branch in [([.1]*3,[1.,3.,2.],'all_core_ties'),([-.4,0.,.4],[0.]*3,'all_vol_ties')]:
            c,d,m,a=self.measure(u,s)
            np.testing.assert_array_equal(c,vector_rank(s));np.testing.assert_array_equal(d,vector_rank(s))
            self.assertEqual(a['identity_branch'],branch);self.assertTrue(m.all())

    def test_no_neighbors_shrinks_to_common_offset(self):
        c,d,m,a=self.measure([-.5,0.,.5],[1.,3.,2.])
        np.testing.assert_array_equal(c,np.full(3,.5/3));np.testing.assert_array_equal(d,c)
        self.assertEqual(a['mean_peer_weight'],0.)

    def test_full_domain_fallback_and_missingoutside(self):
        sig=np.array([1.,np.nan,3.,2.]);u=np.array([-.4,0.,.3,.2]);v=vector_rank(sig)
        c,d,m,a=k.peer_comparisons(u,sig,v,np.array([True,False,True,True]))
        self.assertTrue(a['row_ready']);self.assertFalse(m[1]);self.assertTrue(np.isnan(c[1]))
        c,d,m,a=k.peer_comparisons(u,sig,v,np.array([False,True,True,True]))
        self.assertFalse(m.any());self.assertTrue(np.isnan(c).all());self.assertEqual(a['reason'],'missing_core_in_vol_domain')
        for s in ([np.nan]*4,[np.nan,1.,np.nan,np.nan]):
            c,d,m,a=k.peer_comparisons(u,s,vector_rank(s),np.ones(4,bool))
            self.assertFalse(m.any());self.assertEqual(a['reason'],'insufficient_vol_domain')

    def test_permutation_and_rankconservation(self):
        u=np.linspace(-.4,.4,9);s=np.array([2,5,1,1,4,7,3,6,2.])
        c,d,m,a=self.measure(u,s);perm=np.array([3,0,8,1,7,2,6,4,5])
        cc,dd,mm,aa=self.measure(u[perm],s[perm])
        np.testing.assert_array_equal(cc,c[perm]);np.testing.assert_array_equal(dd,d[perm])
        for x in (c,d):self.assertAlmostEqual(x.sum(),.5,14)
        self.assertEqual(a['mean_peer_weight'],aa['mean_peer_weight'])

    def test_wrong_rankdomain_rejected(self):
        with self.assertRaises(ValueError):k.peer_comparisons([0.,.1],[1.,2.],[0.,0.],[True,True])


class SourceTests(unittest.TestCase):
    def test_independent_full_audit_on_both_synthetic_fixtures(self):
        import monthly_core_peer184_independent as independent
        from test_monthly_core_peer184_independent import fixture as other_fixture
        for make in (fixture,other_fixture):
            p,m,c,a,s=make()
            v,z=k.build_features(p,m,c,a,s)
            receipt=independent.verify_features(p,m,c,a,s,v,z)
            self.assertTrue(receipt['passed'])
            d,dz=k.build_deltas(v,a,s)
            self.assertTrue(independent.verify_deltas(v,a,s,d,dz)['passed'])

    def test_raw_source_and_original_rank_parity(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s)
        self.assertEqual(z['checked_cells'],18);self.assertEqual(z['predictive_model_fits'],0)
        self.assertTrue(v['available'].all());self.assertEqual(z['ready_rows'],3)
        self.assertEqual(z['unordered_pair_comparisons'],45)
        self.assertEqual(z['source_vol_finite_cells'],18)
        for i,row in enumerate(z['event_history']):
            raw=pd.Series(row['raw_core'],dtype=float)
            want=(raw.rank(method='average',pct=True).fillna(.5).to_numpy()-.5)+.5
            np.testing.assert_array_equal(row['core_rank'],want)

    def test_foreign_etf_labels_and_eligibility_do_not_change_pair_domain(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s)
        m['etfs']=m['symbols'][0];a['eligible'][:]=False
        vv,zz=k.build_features(p,m,c,a,s)
        for key in v:np.testing.assert_array_equal(v[key],vv[key])
        self.assertEqual(z,zz)

    def test_current_zero_vol_is_available_and_missing_core_fallsback(self):
        p,m,c,a,s=fixture();p['adj']['S0']=100.;a,s=sources(p['adj'],m)
        v,z=k.build_features(p,m,c,a,s)
        self.assertTrue(v['available'][:,0].all())
        self.assertTrue(all(x['raw_vol'][0] == 0. for x in z['event_history']))
        p,m,c,a,s=fixture();p['adj'].iloc[0,0]=np.nan;a,s=sources(p['adj'],m)
        # Noncalm first row needs the t-294 endpoint, despite valid recent vol.
        p['adj']['SPY']=100.*np.cumprod(1.+.02*np.sin(np.arange(len(p['adj']))*.31))
        a,s=sources(p['adj'],m);v,z=k.build_features(p,m,c,a,s)
        self.assertFalse(v['available'][0].any());self.assertEqual(z['event_history'][0]['reason'],'missing_core_in_vol_domain')

    def test_prefix_and_futurepoison(self):
        p,m,c,a,s=fixture();full,audit=k.build_features(p,m,c,a,s)
        for limit in range(4):
            v,z=k.build_features(p,m,c,a,s,limit)
            for key in full:np.testing.assert_array_equal(v[key],full[key][:limit])
            self.assertEqual(z['event_history'],audit['event_history'][:limit])
        q=copy.deepcopy(p);q['adj'].loc[q['adj'].index>pd.Timestamp(m['dates'][0])]=np.inf
        v,z=k.build_features(q,m,c,a,s,1)
        for key in full:np.testing.assert_array_equal(v[key],full[key][:1])

    def test_nan_support_keeps_original_rank_domain(self):
        p,m,c,a,s=fixture();p['adj'].loc[p['adj'].index[200:280],'S0']=np.nan
        a,s=sources(p['adj'],m);v,z=k.build_features(p,m,c,a,s)
        self.assertFalse(z['event_history'][0]['vol_domain'][0]);self.assertTrue(np.isnan(v['candidate'][0,0]))
        self.assertEqual(z['event_history'][0]['domain_count'],5)

    def test_parity_price_calendar_and_metadata_failures(self):
        p,m,c,a,s=fixture()
        for key in ('vol','vol_weight'):
            changed=copy.deepcopy(s);changed[key][0,0]+=.001
            with self.assertRaises(ValueError):k.build_features(p,m,c,a,changed)
        changed=copy.deepcopy(s);changed['state_highvol'][0]^=True
        with self.assertRaises(ValueError):k.build_features(p,m,c,a,changed)
        for bad in (0.,-1.,np.inf,'bad'):
            q=copy.deepcopy(p);q['adj']=q['adj'].astype(object);q['adj'].iloc[10,0]=bad
            with self.assertRaises(ValueError):k.build_features(q,m,c,a,s)
        with self.assertRaises(ValueError):k.build_features(p,m,c,a,s,True)
        mm=copy.deepcopy(m);del mm['etfs']
        with self.assertRaises(ValueError):k.build_features(p,mm,c,a,s)

    def test_inputs_unchanged(self):
        p,m,c,a,s=fixture();before=copy.deepcopy((p,m,c,a,s))
        v,z=k.build_features(p,m,c,a,s);k.build_deltas(v,a,s)
        pd.testing.assert_frame_equal(p['adj'],before[0]['adj']);self.assertEqual(m,before[1]);self.assertEqual(c,before[2])
        for got,want in ((a,before[3]),(s,before[4])):
            for key in got:np.testing.assert_array_equal(got[key],want[key])


class DeltaTests(unittest.TestCase):
    def test_half_slot_scope_masks_and_initial(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s)
        d,az=k.build_deltas(v,a,s);np.testing.assert_array_equal(d['candidate'][0],0.)
        for key in d:
            want=.5*np.array(az['effective_vol_weights'])*(v[key]-s['vol']);want[0]=0.
            np.testing.assert_array_equal(d[key],want)
        for key in ('extension_available','forecast_available','eligible'):
            aa=copy.deepcopy(a);aa[key][:]=False
            _,zz=k.build_deltas(v,aa,s);self.assertFalse(np.array(zz['scope_active']).any())
        aa=copy.deepcopy(a);aa['rank_log_excess'][:]=np.nan
        _,zz=k.build_deltas(v,aa,s);self.assertFalse(np.array(zz['scope_active']).any())

    def test_all_weight_shapes_prefix_and_badmask(self):
        p,m,c,a,s=fixture();v,z=k.build_features(p,m,c,a,s);d,az=k.build_deltas(v,a,s)
        for mode in ('vector','matrix'):
            aa=copy.deepcopy(a);ss=copy.deepcopy(s)
            for obj,key in ((aa,'core_weight'),(ss,'core_weight'),(ss,'vol_weight')):
                obj[key]=obj[key][:,0] if mode=='vector' else np.broadcast_to(obj[key],(3,6)).copy()
            dd,_=k.build_deltas(v,aa,ss)
            for key in d:np.testing.assert_array_equal(dd[key],d[key])
        for n in range(4):
            dd,zz=k.build_deltas(v,a,s,n)
            for key in d:np.testing.assert_array_equal(dd[key],d[key][:n])
        bad=copy.deepcopy(v);bad['candidate'][1,0]=np.nan
        with self.assertRaises(ValueError):k.build_deltas(bad,a,s)

    def test_seven_effective_coefficients_against_original_functions(self):
        root=Path(__file__).resolve().parents[1]
        def extract(path,name,namespace):
            tree=ast.parse((root/path).read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
            return namespace[name]
        prior=extract('outputs/monthly_state_rank_research/state_rank.py','score',{'np':np})
        C=np.array([1,1,1,0,0,0,0],bool);W=np.array([0,0,1,0,1,0,1],bool);H=np.array([0,0,0,0,0,1,1],bool)
        cw=np.where(C,.5,.52)[:,None];vw=np.where(C,.2,.219375*(1.-np.where(H,.10,0.)/.48))[:,None]
        a=dict(core=np.full((7,2),.04),core_weight=cw,state_calm10=C,state_weak_narrow=W,
               extension_available=np.ones((7,2),bool),forecast_available=np.ones((7,2),bool),
               eligible=np.ones((7,2),bool),rank_log_excess=np.full((7,2),.2),
               residual=np.full((7,2),.03),ml=np.full((7,2),-.1))
        a['extension_available'][0]=False
        spec=dict(base='incumbent67',scope='calm_nonweak',budget=.01)
        ps=[dict(amount=.025,state='calm10',key='residual'),dict(amount=.05,state='weak_narrow',key='ml')]
        adapter=SimpleNamespace(_state=lambda x,key,full,n:x[key][:n],_matrix=lambda value,shape,n,label,**kw:value[:n])
        upstream=SimpleNamespace(configs=lambda:list(range(6)),modified_pre_score=lambda spec,x,n:prior(ps,x,n))
        scope=extract('outputs/monthly_model_scope70/study.py','modified_pre_score',
                      dict(np=np,configs=lambda:[spec],adapter=adapter,upstream=upstream))
        def old(V):
            x=dict(a,baseline=a['core']+vw*V+.03)
            return scope(spec,x,7)
        measured=old(.25)-old(-.25)
        expected=np.array([.19,.186,.17,.219375,.1965234375,.173671875,.1555810546875])
        np.testing.assert_allclose(measured/.5,np.broadcast_to(expected[:,None],(7,2)),rtol=0,atol=3e-16)
        s=dict(vol=np.zeros((7,2)),vol_weight=vw,state_highvol=H)
        values=dict(candidate=np.zeros((7,2)),control=np.zeros((7,2)),available=np.ones((7,2),bool))
        _,audit=k.build_deltas(values,a,s)
        np.testing.assert_allclose(audit['effective_vol_weights'],measured/.5,rtol=0,atol=3e-16)


if __name__ == '__main__':
    unittest.main()
