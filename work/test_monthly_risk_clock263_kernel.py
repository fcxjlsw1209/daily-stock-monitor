"""Synthetic-only263 source, full-history core, arithmetic, and adapter tests."""
from copy import deepcopy
from fractions import Fraction
import hashlib,json,math,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_risk_clock263_kernel as k
import test_monthly_downside_basket258_kernel as base


def make_review(meta,calendar):
    return dict(schema=k.SOURCE_SCHEMA,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,panel_first_session=k.source.ORIGIN,listing_bounds=deepcopy(k.BOUNDS),identity_cells=k.source.identities(meta,k.BOUNDS,len(meta['dates'])),clock_events=[k.source.clock(q,d,meta['symbols'][q],calendar) for q,d in enumerate(meta['dates'])])


def sync_core(args):
    s,a,m,p,c,r=args;px=p['adj'];sigma=px.pct_change(fill_method=None).rolling(63).std();raw210=px.shift(42)/px.shift(252)-1.;raw252=(px.shift(42)/px.shift(294)-1.)/sigma.replace(0,np.nan)
    for q,d in enumerate(m['dates']):
        if d==k.INITIAL_PARTIAL:continue
        raw=(raw210 if a['state_calm10'][q] else raw252).loc[d,m['symbols'][q]].replace([np.inf,-np.inf],np.nan);rank=raw.rank(pct=True).fillna(.5).to_numpy()-.5;a['core'][q]=float(a['core_weight'][q,0])*rank
    a['baseline']=a['core']+.15*(1.-a['core_weight']);return args


def fixture():
    s,a,m,_,_,_=base.fixture();ds=pd.bdate_range('2008-01-02','2016-11-02');x=np.arange(len(ds),dtype=float);names=m['symbols'][0]
    p={'adj':pd.DataFrame({v:100.+.035*(j+1)*x+.7*np.sin(x/(j+4.))+.3*np.cos(x/(11+j)) for j,v in enumerate(names)},index=ds)};c=ds.strftime('%Y-%m-%d').tolist();r=make_review(m,c)
    return sync_core((s,a,m,p,c,r))


def physical(args,n):
    s,a,m,p,c,r=args;aa={key:np.asarray(v)[:n].copy() for key,v in a.items()};mm={key:deepcopy(m[key][:n]) for key in ('dates','ends','symbols')};mm['etfs']=m['etfs'].copy();last=mm['dates'][-1] if n else None
    return s,aa,mm,{'adj':p['adj'].loc[:last].copy() if n else p['adj'].iloc[:0].copy()},c[:c.index(last)+2] if n else [],deepcopy(r)


def run(args,n=None,**kw):return k.prepare(*args,n=n,**kw)


def prices():return [100.+.12*i+.8*math.sin(i/3.) for i in range(107)]


def assert_outputs(test,a,b):
    k._exact(a[0],b[0],'orders');k._exact(a[3],b[3],'audit')
    for role in k.ARMS:
        test.assertEqual(a[1][role].tobytes(),b[1][role].tobytes());test.assertEqual(a[2][role].tobytes(),b[2][role].tobytes())


class Mathematics(unittest.TestCase):
    def test_direct_m_signed_and_107_offsets(self):
        p=prices();p[0]=1000.;z=k.measure(p);self.assertEqual(tuple(z),k.STAT_KEYS);self.assertLess(z['momentum'],0.);self.assertEqual(z['momentum'],p[64]/p[0]-1.)
        self.assertEqual(z['current_returns'],[p[j]/p[j-1]-1. for j in range(44,107)]);self.assertEqual(z['lag_returns'],[p[j]/p[j-1]-1. for j in range(2,65)]);self.assertEqual(z['current_returns'][:21],z['lag_returns'][-21:]);self.assertIsNone(z['current_ratio']);self.assertIsNone(z['lag_ratio'])
    def test_sample_sd_order_and_x_times_x(self):
        z=k.measure(prices())
        for name in ('current','lag'):
            xs=z[name+'_returns'];r=z[name+'_risk'];mu=math.fsum(xs)/63.;self.assertEqual(r['mean'],mu);self.assertEqual(r['squares'],[(x-mu)*(x-mu) for x in xs]);self.assertEqual(r['variance'],math.fsum(r['squares'])/62.);self.assertEqual(r['sd'],math.sqrt(r['variance']))
    def test_constant_nonzero_exact_zero(self):
        r=k.risk63([.1]*63);self.assertTrue(r['constant']);self.assertEqual(r['mean'],.1)
        for key in ('square_sum','variance','sd'):self.assertEqual(r[key].hex(),'0x0.0p+0')
    def test_current_lag_both_constant_zero_valid(self):
        for start,end,key in ((43,107,'current'),(1,65,'lag'),(1,107,'current')):
            p=prices();p[start:end]=[100.]*(end-start);z=k.measure(p);self.assertEqual(z[key+'_risk']['sd'],0.)
    def test_one_ulp_nearconstant_not_epsilon(self):
        xs=[.1]*63;xs[-1]=math.nextafter(.1,math.inf);r=k.risk63(xs);self.assertFalse(r['constant']);self.assertGreater(r['sd'],0.)
    def test_arithmetic_underflow_overflow_hard(self):
        for f,args in ((k._square,(1e-200,'x')),(k._div,(math.ulp(0.),62.,'x')),(k._sum,([1e308,1e308],'x'))):
            with self.assertRaises(ArithmeticError):f(*args)
        with self.assertRaises(ArithmeticError):k.risk63([0.]*62+[1e-200])
    def test_nonzero_conversion_and_types_hard(self):
        for bad in (True,np.bool_(True),'1',None,0.,-1.,math.inf,math.nan,Fraction(1,10**1000)):
            p=prices();p[10]=bad
            with self.subTest(kind=type(bad).__name__),self.assertRaises(ValueError):k.measure(p)
        with self.assertRaises(ValueError):k.risk63([Fraction(1,10**1000)]*63)
    def test_shape_strict(self):
        with self.assertRaises(ValueError):k.measure(prices()[:-1])
        with self.assertRaises(ValueError):k.risk63([0.]*64)
    def test_rank_exact_three_step_ties_n1_n0(self):
        xs=[0.,1.,1.,3.,4.,5.]
        for v in xs:self.assertEqual(k.rank(v,xs),(float(2*sum(x<v for x in xs)+sum(x==v for x in xs)+1)/2.)/6.-.5)
        self.assertEqual(k.rank(0.,[0.]),.5)
        with self.assertRaises(ValueError):k.rank(0.,[])
    def test_one_ulp_rank_order_not_collapsed(self):
        a=1.;b=math.nextafter(a,math.inf);self.assertEqual(k.rank(a,[a,b]),0.);self.assertEqual(k.rank(b,[a,b]),.5)
    def test_power_two_rescaling_exact(self):
        p=prices();k._exact(k.measure(p),k.measure([2.*v for v in p]),'binary rescale')


class Sources(unittest.TestCase):
    def test_source_all_U0_before_E_including_ETF(self):
        args=fixture();args[1]['eligible'][:]=False;fs=k.build_features(args[3],args[2],args[4],args[1],args[5]);self.assertEqual(fs[1]['active_indices'],list(range(6)));self.assertEqual(len(fs[1]['cells'][5]['observations']),107);self.assertEqual(fs[1]['required_dates'][64],fs[1]['numerator_dates'][1]);self.assertIsNone(fs[1]['current_target'])
    def test_U0_not_zero_core_or_eligible(self):
        args=fixture();args[3]['adj']['B']=args[3]['adj']['A'];sync_core(args);f=run(args)[0][1];self.assertEqual(f['u0_indices'],list(range(6)));self.assertEqual(f['cells'][0]['original_rank'],f['cells'][1]['original_rank'])
    def test_missing_lag_for_Efalse_but_U0true_hard(self):
        args=fixture();d=args[5]['clock_events'][1]['lag_dates'][10];args[3]['adj'].at[pd.Timestamp(d),'A']=np.nan;args[1]['eligible'][1,0]=False;sync_core(args)
        with self.assertRaisesRegex(ValueError,'Required'):run(args,2)
    def test_fullhistory_missing_row_not_compressed(self):
        args=fixture();args[3]['adj'].drop(args[3]['adj'].index[100],inplace=True)
        with self.assertRaisesRegex(ValueError,'Complete original official'):run(args,2)
    def test_original_core_parity_hard(self):
        args=fixture();args[1]['core'][1,0]+=2e-12;args[1]['baseline'][1,0]+=2e-12
        with self.assertRaisesRegex(ValueError,'core parity'):run(args,2)
    def test_original_nan_history_retained_outside_required(self):
        args=fixture();args[3]['adj'].iloc[100,0]=np.nan;sync_core(args);run(args,2)
    def test_calm_no_new_source_arithmetic(self):
        args=fixture();args[1]['state_calm10'][:]=True;args[1]['core_weight'][:]=.5;sync_core(args)
        with patch.object(k,'measure',side_effect=AssertionError('calm new arithmetic')):out=run(args)
        self.assertEqual(out[4]['counts']['required_cells'],0);self.assertEqual(out[4]['counts']['calm_events'],2)
    def test_partial_and_n0_no_numeric(self):
        args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].iloc[:,:]=base.Poison()
        with patch.object(k,'_old_number',side_effect=AssertionError('partial history')):
            for n in (0,1):out=run(args,n);self.assertEqual(out[4]['counts']['original_core_cells'],0)
    def test_outside_U0_no_new107(self):
        args=fixture();args[3]['adj']['A']=np.nan;sync_core(args);f=run(args,2)[0][1];self.assertNotIn(0,f['u0_indices']);self.assertEqual(f['cells'][0]['observations'],[])
    def test_empty_U0_exactcopy(self):
        args=fixture();args[3]['adj'].iloc[:,:]=100.;sync_core(args);out=run(args,2);self.assertEqual(out[0][1]['branch'],'empty_domain')
        for arm in k.ARMS[1:]:self.assertEqual(out[3][arm].tobytes(),out[3][k.ARMS[0]].tobytes());k._exact(out[1][arm],out[1][k.ARMS[0]],'empty orders')
    def test_singleton_U0_rank_half(self):
        args=fixture();args[3]['adj'].loc[:,list('BCDEF')]=np.nan;sync_core(args);f=run(args,2)[0][1];self.assertEqual(f['u0_indices'],[0]);self.assertEqual(f['current_target'],[.5,None,None,None,None,None])
    def test_zero_lag_wholequery_after_allrisk(self):
        args=fixture();ds=args[5]['clock_events'][1]['lag_dates'];args[3]['adj'].loc[pd.to_datetime(ds),'A']=120.;sync_core(args);out=run(args,2);f=out[0][1];self.assertIn(0,f['active_indices']);self.assertEqual(f['branch'],'zero_risk');self.assertEqual(f['zero_lag_indices'],[0]);self.assertTrue(all(c['statistics'] is not None for c in f['cells']))
        self.assertTrue(all(c['statistics']['current_ratio'] is None for c in f['cells']))
        for arm in k.ARMS[1:]:self.assertEqual(out[3][arm].tobytes(),out[3][k.ARMS[0]].tobytes())
    def test_earlier_zero_does_not_hide_later_badsource(self):
        args=fixture();ds=args[5]['clock_events'][1]['lag_dates'];args[3]['adj'].loc[pd.to_datetime(ds),'A']=120.;args[3]['adj'].at[pd.Timestamp(ds[3]),'F']=np.nan;sync_core(args)
        with patch.object(k,'measure',side_effect=AssertionError('math before complete source')),self.assertRaises(ValueError):run(args,2)
    def test_earlier_zero_does_not_hide_later_arithmetic(self):
        args=fixture();ds=args[5]['clock_events'][1]['lag_dates'];args[3]['adj'].loc[pd.to_datetime(ds),'A']=120.;sync_core(args);original=k.measure;calls=[]
        def measure(p):
            calls.append(1)
            if len(calls)==2:raise ArithmeticError('Later arithmetic')
            return original(p)
        with patch.object(k,'measure',measure),self.assertRaisesRegex(ArithmeticError,'Later arithmetic'):run(args,2)
        self.assertEqual(len(calls),2)
    def test_identity_query_visible_conflict_and_no_early_override(self):
        args=fixture();m=args[2];m['symbols']=[['SPCX']+row[1:] for row in m['symbols']];args[3]['adj'].rename(columns={'A':'SPCX'},inplace=True);m['dates'][1]='2026-06-30'
        # Fixed policy itself never borrows a later first-trade override.
        self.assertEqual(k._bound('SPCX','2026-06-30')['not_before'],'2026-06-08');self.assertIsNone(k._bound('SPCX','2026-06-08'))
        with self.assertRaisesRegex(ValueError,'prelisting'):k._scan(args[3]['adj'],'SPCX',['2016-06-01'],k._bound('SPCX','2026-06-30'))
    def test_source_clock_identity_and_bound_tampering(self):
        for key in ('clock','bound','origin'):
            args=fixture()
            if key=='clock':args[5]['clock_events'][1]['required_dates'][2]='2010-01-01'
            elif key=='bound':args[5]['listing_bounds'][0]['not_before']='2026-06-12'
            else:args[5]['panel_first_session']='2010-01-04'
            with self.subTest(key=key),self.assertRaises(ValueError):run(args,2)


class Integration(unittest.TestCase):
    def test_all_logical_physical_cached_prefixes_exact(self):
        args=fixture();full=run(args);json.dumps(full[0],allow_nan=False);json.dumps(full[4],allow_nan=False)
        for n in range(4):
            left=run(args,n);right=run(physical(args,n));cached=k.decisions(args[0],args[1],args[2],full[0],n);k._exact(left[0],full[0][:n],'features prefix');k._exact(left[0],right[0],'physical source');assert_outputs(self,left[1:],right[1:]);assert_outputs(self,left[1:],cached)
    def test_fullfeatures_physicalarrays_prefix(self):
        args=fixture();full=run(args)
        for n in (0,1,2):
            small=physical(args,n);out=k.decisions(small[0],small[1],small[2],full[0],n);assert_outputs(self,run(small)[1:],out)
    def test_gate_order_and_threeepisodes(self):
        args=fixture();events=[];orig=args[0].ep.select_boundary
        def select(*a,**kw):events.append('episode');return orig(*a,**kw)
        def ig(*a):self.assertEqual(events,['episode']);events.append('inc');return {'passed':True,'scope':'inc'}
        def sg(fs):self.assertEqual(events,['episode','inc']);self.assertTrue(all(f['current_target'] is None and f['lag_target'] is None for f in fs));events.append('source');return {'passed':True,'scope':'source'}
        with patch.object(args[0].ep,'select_boundary',select):out=run(args,incumbent_validator=ig,source_validator=sg)
        self.assertEqual(events,['episode','inc','source','episode','episode']);self.assertEqual(out[4]['source_gate'],{'passed':True,'scope':'source'})
    def test_failed_inc_gate_blocks_stats(self):
        with patch.object(k,'measure',side_effect=AssertionError('premature math')),self.assertRaises(ValueError):run(fixture(),incumbent_validator=lambda *x:{'passed':False})
    def test_failed_source_gate_blocks_ranks(self):
        with patch.object(k,'rank',side_effect=AssertionError('premature rank')),self.assertRaises(ValueError):run(fixture(),source_validator=lambda *x:{'passed':False})
    def test_mutating_callback_rejected_before_candidate(self):
        args=fixture();calls=[];orig=args[0].ep.select_boundary
        def selector(*x,**kw):calls.append(1);return orig(*x,**kw)
        def sg(fs):args[1]['effects'][0,0]+=.1;return {'passed':True}
        with patch.object(args[0].ep,'select_boundary',selector),self.assertRaises(ValueError):run(args,source_validator=sg)
        self.assertEqual(len(calls),1)
    def test_original_inputs_unchanged(self):
        args=fixture();snap=k.old._snapshot(args[1]);m=deepcopy(args[2]);p=args[3]['adj'].copy(deep=True);run(args);self.assertEqual(snap,k.old._snapshot(args[1]));self.assertEqual(m,args[2]);pd.testing.assert_frame_equal(p,args[3]['adj'])
    def test_consumed_only_adapter_future_poison(self):
        args=fixture();full=run(args);poison=base.Poison();a=args[1];m=args[2];f=deepcopy(full[0])
        for v in a.values():
            pass
        for key,v in list(a.items()):
            if v.dtype.kind!='b':a[key]=v.astype(object);a[key][2:]=poison
        for key in ('dates','ends','symbols'):m[key][2]=poison
        f[2]=poison
        for n in (0,1,2):
            left=run(args,n);right=k.decisions(args[0],a,m,f,n);assert_outputs(self,left[1:],right)
    def test_future_panel_values_not_consumed(self):
        args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc['2016-09-01':,:]=base.Poison();run(args,2)
    def test_fullcore_delta_original_support_and_calm_copy(self):
        args=fixture();out=run(args);f=out[0][1]
        for arm,key in zip(k.ARMS[1:],('current','lag')):
            expected=out[3][k.ARMS[0]].copy()
            for j,t in enumerate(f[key+'_target']):
                d=float(float(args[1]['core_weight'][1,0]*t)-args[1]['core'][1,j]);self.assertEqual(out[4]['events'][1][key+'_deltas'][j],d)
                if d!=0.:expected[1,j]+=d
            self.assertEqual(out[3][arm].tobytes(),expected.tobytes());self.assertEqual(out[2][arm][2].tobytes(),out[2][k.ARMS[0]][2].tobytes())
    def test_prefix_n_strict(self):
        for n in (True,np.int64(1),-1,4,1.0):
            with self.subTest(n=n),self.assertRaises(ValueError):run(fixture(),n)
    def test_cached_stat_U0_rank_clock_corruption(self):
        args=fixture();out=run(args)
        for which in ('stat','domain','rank','clock'):
            fs=deepcopy(out[0]);f=fs[1]
            if which=='stat':f['cells'][0]['statistics']['momentum']=math.nextafter(f['cells'][0]['statistics']['momentum'],math.inf)
            elif which=='domain':f['u0_indices']=f['u0_indices'][1:]
            elif which=='rank':f['current_target'][0]=math.nextafter(f['current_target'][0],math.inf)
            else:f['required_dates'][2]='2008-01-02'
            with self.subTest(which=which),self.assertRaises(ValueError):k.decisions(args[0],args[1],args[2],fs)
    def test_exact_expected_incumbent_orders(self):
        args=fixture();out=run(args);expected=deepcopy(out[1][k.ARMS[0]]);expected[0]['symbols']=[]
        with self.assertRaises(ValueError):run(args,expected_incumbent_orders=expected)
    def test_descriptive_measure_call_accounting(self):
        args=fixture();orig=k.measure
        with patch.object(k,'measure',wraps=orig) as spy:out=run(args);self.assertEqual(spy.call_count,out[4]['counts']['required_cells'])
        for n in range(4):
            with patch.object(k,'measure',wraps=orig) as spy:result=k.decisions(args[0],args[1],args[2],out[0],n);self.assertEqual(spy.call_count,result[3]['counts']['required_cells'])
    def test_n0_three_episode_calls_zero_numeric(self):
        args=fixture();calls=[];orig=args[0].ep.select_boundary
        def select(*a,**kw):calls.append(1);return orig(*a,**kw)
        with patch.object(args[0].ep,'select_boundary',select),patch.object(k,'measure',side_effect=AssertionError('n0 measure')):out=run(args,0)
        self.assertEqual(len(calls),3);self.assertEqual(out[4]['counts']['price_observations'],0)


if __name__=='__main__':unittest.main(verbosity=2)
