"""Synthetic-only264 gain survival, original full-history core, and adapter tests."""
from copy import deepcopy
from fractions import Fraction
import hashlib,json,math,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_gain_survival264_kernel as k
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


def prices():return [100.+.12*i+.8*math.sin(i/3.) for i in range(253)]


def assert_outputs(test,a,b):
    k._exact(a[0],b[0],'orders');k._exact(a[3],b[3],'audit')
    for role in k.ARMS:
        test.assertEqual(a[1][role].tobytes(),b[1][role].tobytes());test.assertEqual(a[2][role].tobytes(),b[2][role].tobytes())


class Mathematics(unittest.TestCase):
    def test_early_late_jump_and_end_gain(self):
        early=k.measure([1.]+[2.]*252,3.);late=k.measure([1.]*252+[2.],3.)
        self.assertEqual(early['positive_anchor_indices'],[1]);self.assertEqual(early['survival_ages'],[251]);self.assertEqual(early['survival_quality'],251./252.)
        self.assertEqual(early['endpoint_quality'],early['survival_quality']);self.assertEqual(late['gain_sum'],1.);self.assertEqual(late['normalizer'],252.)
        self.assertEqual(late['survival_quality'],0.);self.assertEqual(late['endpoint_quality'],0.);self.assertEqual(late['survival_raw'],0.)
    def test_real_duration_bound_does_not_reject_canonical_rounding(self):
        p=[1.]+[1.0081]*252;z=k.measure(p,1.);g=1.0081/1.-1.;canonical=math.fsum([g*251.])/(252.*math.fsum([g]))
        self.assertGreater(canonical,251./252.)
        for key in ('survival','endpoint'):self.assertEqual(z[key+'_quality'].hex(),canonical.hex())
        for path in (p,prices(),[1.]*252+[2.]):
            z=k.measure(path,1.)
            for duration,age in zip(z['survival_ages'],z['potential_ages']):self.assertTrue(type(duration) is int and type(age) is int and 0<=duration<=age<=251)
    def test_breach_is_terminal_despite_recovery(self):
        p=[1.]+[2.]*252;p[100]=1.5;z=k.measure(p,3.)
        self.assertEqual(z['first_breach_indices'][0],100);self.assertEqual(z['survival_ages'][0],98);self.assertTrue(z['endpoint_retained'][0]);self.assertLess(z['survival_quality'],z['endpoint_quality'])
        p=[1.]+[2.]*250+[1.5]*2;z=k.measure(p,3.)
        self.assertGreater(z['survival_quality'],z['endpoint_quality']);self.assertEqual(z['endpoint_quality'],0.)
    def test_later_closes_only_ties_and_no_reset(self):
        p=[1.]+[2.]*252;p[5]=1.5;z=k.measure(p,1.)
        self.assertEqual(z['survival_ages'][0],3);self.assertEqual(z['first_breach_indices'][0],5);self.assertTrue(z['endpoint_retained'][0])
        p[5]=2.;z=k.measure(p,1.);self.assertEqual(z['survival_ages'],[251]);self.assertEqual(z['first_breach_indices'],[None]);self.assertEqual(z['survival_comparisons'],251)
    def test_quotient_then_subtract_and_chronological_weighted_sums(self):
        p=prices();z=k.measure(p,2.);indices=[i for i in range(1,253) if p[i]/p[i-1]-1.>0.]
        self.assertEqual(tuple(z),k.STAT_KEYS);self.assertEqual(z['positive_anchor_indices'],indices);self.assertEqual(z['positive_gains'],[p[i]/p[i-1]-1. for i in indices])
        self.assertEqual(z['potential_ages'],[252-i for i in indices]);self.assertEqual(z['gain_sum'],math.fsum(z['positive_gains']))
        for key,ages in (('survival',z['survival_ages']),('endpoint',[a if r else 0 for a,r in zip(z['potential_ages'],z['endpoint_retained'])])):
            terms=[g*float(a) for g,a in zip(z['positive_gains'],ages)];self.assertEqual(z[key+'_weighted_terms'],terms);self.assertEqual(z[key+'_weighted_sum'],math.fsum(terms));self.assertEqual(z[key+'_quality'],math.fsum(terms)/(252.*math.fsum(z['positive_gains'])))
    def test_no_positives_identity_and_negative_signed_raw(self):
        for p in ([100.]*253,[float(300-i) for i in range(253)]):
            for raw in (3.,-3.,0.,-0.):
                z=k.measure(p,raw);self.assertTrue(z['no_positive_gains']);self.assertEqual(z['positive_gains'],[]);self.assertEqual(z['gain_sum'],0.);self.assertEqual(z['normalizer'],0.)
                for key in ('survival','endpoint'):self.assertEqual(z[key+'_quality'],1.);self.assertEqual(z[key+'_raw'].hex(),raw.hex())
        for raw in (-3.,0.,-0.):
            z=k.measure(prices(),raw)
            for key in ('survival','endpoint'):self.assertEqual(z[key+'_raw'].hex(),raw.hex())
    def test_arithmetic_overflow_underflow_and_legitimate_zero(self):
        for f,args in ((k._mul,(1e-200,1e-200,'x')),(k._div,(math.ulp(0.),252.,'x')),(k._sum,([1e308,1e308],'x'))):
            with self.assertRaises(ArithmeticError):f(*args)
        for p,raw in (([1e-308,1e308]+[1.]*251,1.),([1e308,1e-308]+[1.]*251,1.),([1.]+[1e308]*252,1.),([1.]*251+[2.]*2,math.ulp(0.))):
            with self.assertRaises(ArithmeticError):k.measure(p,raw)
        self.assertEqual(k._mul(math.ulp(0.),0.,'exactzero'),0.);self.assertEqual(k.measure([1.]*252+[2.],math.ulp(0.))['survival_raw'],0.)
    def test_nonzero_conversion_and_types_hard(self):
        for bad in (True,np.bool_(True),'1',None,0.,-1.,math.inf,math.nan,Fraction(1,10**1000)):
            p=prices();p[10]=bad
            with self.subTest(kind=type(bad).__name__),self.assertRaises(ValueError):k.measure(p,1.)
        for raw in (True,None,math.inf,math.nan,Fraction(1,10**1000)):
            with self.assertRaises(ValueError):k.measure(prices(),raw)
    def test_shape_strict(self):
        with self.assertRaises(ValueError):k.measure(prices()[:-1],1.)
    def test_rank_exact_three_step_ties_n1_n0(self):
        xs=[0.,1.,1.,3.,4.,5.]
        for v in xs:self.assertEqual(k.rank(v,xs),(float(2*sum(x<v for x in xs)+sum(x==v for x in xs)+1)/2.)/6.-.5)
        self.assertEqual(k.rank(0.,[0.]),.5)
        with self.assertRaises(ValueError):k.rank(0.,[])
    def test_one_ulp_rank_order_not_collapsed(self):
        a=1.;b=math.nextafter(a,math.inf);self.assertEqual(k.rank(a,[a,b]),0.);self.assertEqual(k.rank(b,[a,b]),.5)
    def test_power_two_rescaling_exact(self):
        p=prices();k._exact(k.measure(p,2.),k.measure([2.*v for v in p],2.),'binary rescale')


class Sources(unittest.TestCase):
    def test_source_all_U0_before_E_including_ETF(self):
        args=fixture();args[1]['eligible'][:]=False;fs=k.build_features(args[3],args[2],args[4],args[1],args[5]);self.assertEqual(fs[1]['active_indices'],list(range(6)));self.assertEqual(len(fs[1]['cells'][5]['observations']),253);self.assertEqual(fs[1]['required_dates'][252],fs[1]['numerator_dates'][1]);self.assertIsNone(fs[1]['endpoint_target'])
    def test_U0_not_zero_core_or_eligible(self):
        args=fixture();args[3]['adj']['B']=args[3]['adj']['A'];sync_core(args);f=run(args)[0][1];self.assertEqual(f['u0_indices'],list(range(6)));self.assertEqual(f['cells'][0]['original_rank'],f['cells'][1]['original_rank'])
    def test_missing_formation_for_Efalse_but_U0true_hard(self):
        args=fixture();d=args[5]['clock_events'][1]['formation_dates'][10];args[3]['adj'].at[pd.Timestamp(d),'A']=np.nan;args[1]['eligible'][1,0]=False;sync_core(args)
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
    def test_outside_U0_no_new253(self):
        args=fixture();args[3]['adj']['A']=np.nan;sync_core(args);f=run(args,2)[0][1];self.assertNotIn(0,f['u0_indices']);self.assertEqual(f['cells'][0]['observations'],[])
    def test_empty_U0_exactcopy(self):
        args=fixture();args[3]['adj'].iloc[:,:]=100.;sync_core(args);out=run(args,2);self.assertEqual(out[0][1]['branch'],'empty_domain')
        for arm in k.ARMS[1:]:self.assertEqual(out[3][arm].tobytes(),out[3][k.ARMS[0]].tobytes());k._exact(out[1][arm],out[1][k.ARMS[0]],'empty orders')
    def test_singleton_U0_rank_half(self):
        args=fixture();args[3]['adj'].loc[:,list('BCDEF')]=np.nan;sync_core(args);f=run(args,2)[0][1];self.assertEqual(f['u0_indices'],[0]);self.assertEqual(f['endpoint_target'],[.5,None,None,None,None,None])
    def test_no_positive_does_not_hide_later_badsource(self):
        args=fixture();ds=args[5]['clock_events'][1]['formation_dates'];args[3]['adj'].loc[pd.to_datetime(ds),'A']=120.;args[3]['adj'].at[pd.Timestamp(ds[3]),'F']=np.nan;sync_core(args)
        with patch.object(k,'measure',side_effect=AssertionError('math before complete source')),self.assertRaises(ValueError):run(args,2)
    def test_no_positive_does_not_hide_later_arithmetic(self):
        args=fixture();ds=args[5]['clock_events'][1]['formation_dates'];args[3]['adj'].loc[pd.to_datetime(ds),'A']=120.;sync_core(args);original=k.measure;calls=[]
        def measure(p,r):
            calls.append(1)
            if len(calls)==2:raise ArithmeticError('Later arithmetic')
            return original(p,r)
        with patch.object(k,'measure',measure),self.assertRaisesRegex(ArithmeticError,'Later arithmetic'):run(args,2)
        self.assertEqual(len(calls),2)
    def test_missing_formation_ETF_hard_even_Efalse(self):
        args=fixture();d=args[5]['clock_events'][1]['formation_dates'][100];args[3]['adj'].at[pd.Timestamp(d),'F']=np.nan;args[1]['eligible'][1,5]=False;sync_core(args)
        self.assertIn('F',args[2]['etfs'])
        with self.assertRaisesRegex(ValueError,'Required'):run(args,2)
    def test_nonpositive_raw_still_requires_complete_source(self):
        args=fixture();ds=args[5]['clock_events'][1]['formation_dates'];args[3]['adj'].loc[pd.to_datetime(ds),'A']=np.linspace(200.,100.,253);args[3]['adj'].at[pd.Timestamp(ds[100]),'A']=np.nan;sync_core(args)
        with self.assertRaisesRegex(ValueError,'Required'):run(args,2)

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
        def sg(fs):self.assertEqual(events,['episode','inc']);self.assertTrue(all(f['endpoint_target'] is None and f['survival_target'] is None for f in fs));events.append('source');return {'passed':True,'scope':'source'}
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
        for arm,key in zip(k.ARMS[1:],('endpoint','survival')):
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
            if which=='stat':f['cells'][0]['statistics']['survival_quality']=math.nextafter(f['cells'][0]['statistics']['survival_quality'],math.inf)
            elif which=='domain':f['u0_indices']=f['u0_indices'][1:]
            elif which=='rank':f['endpoint_target'][0]=math.nextafter(f['endpoint_target'][0],math.inf)
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

    def test_nonpositive_raw_whole_vector_identity_copies_bytes(self):
        args=fixture();ds=args[5]['clock_events'][1]['formation_dates']
        for symbol in args[2]['symbols'][1]:args[3]['adj'].loc[pd.to_datetime(ds),symbol]=np.linspace(200.,100.,253)
        sync_core(args);args[1]['core'][1,0]+=5e-13;args[1]['baseline'][1,0]+=5e-13;out=run(args,2);f=out[0][1]
        for key,arm in zip(('endpoint','survival'),k.ARMS[1:]):
            self.assertTrue(f[key+'_identity']);self.assertEqual(out[4]['events'][1][key+'_deltas'],[0.]*6);self.assertTrue(all(t is not None for t in f[key+'_target']))
            for index in (2,3):self.assertEqual(out[index][arm].tobytes(),out[index][k.ARMS[0]].tobytes())
            k._exact(out[1][arm],out[1][k.ARMS[0]],'identity orders')
        self.assertEqual(out[4]['counts']['nonpositive_original_cells'],6);self.assertEqual(out[4]['counts']['positive_raw_multiplications'],0)
    def test_all_flat_formation_zero_original_identity(self):
        args=fixture();ds=args[5]['clock_events'][1]['formation_dates'];args[3]['adj'].loc[pd.to_datetime(ds),:]=100.;sync_core(args);out=run(args,2)
        self.assertEqual(out[4]['counts']['no_positive_gain_cells'],6);self.assertEqual(out[4]['counts']['quality_ratio_evaluations'],0)
        for key in ('endpoint','survival'):self.assertTrue(out[0][1][key+'_identity'])
    def test_count_schema_and_operations(self):
        out=run(fixture());c=out[4]['counts'];f=out[0][1];stats=[x['statistics'] for x in f['cells']]
        self.assertEqual(tuple(c),k.COUNT_FIELDS);self.assertEqual(c['price_observations'],253*6);self.assertEqual(c['gain_return_observations'],252*6);self.assertEqual(c['gain_statistics'],6)
        self.assertEqual(c['positive_gain_anchors'],sum(len(s['positive_gains']) for s in stats));self.assertEqual(c['endpoint_comparisons'],c['positive_gain_anchors']);self.assertEqual(c['survival_comparisons'],sum(s['survival_comparisons'] for s in stats))
        self.assertEqual(c['quality_ratio_evaluations'],2*sum(not s['no_positive_gains'] for s in stats));self.assertEqual(c['positive_raw_multiplications'],2*sum(s['original_raw']>0. for s in stats))
        self.assertEqual(c['endpoint_ranked_cells'],6);self.assertEqual(c['survival_ranked_cells'],6);self.assertEqual(c['new_model_fits'],0);self.assertEqual(c['predictive_model_fits'],0)
    def test_cached_identity_flag_tamper_hard(self):
        args=fixture();out=run(args);fs=deepcopy(out[0]);fs[1]['endpoint_identity']=not fs[1]['endpoint_identity']
        with self.assertRaises(ValueError):k.decisions(args[0],args[1],args[2],fs)



if __name__=='__main__':unittest.main(verbosity=2)
