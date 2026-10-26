"""Synthetic265 selection/source integration; no actual data or lifecycle calls."""
from copy import deepcopy
import hashlib,json,math,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_gain_concentration265_selection as k
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
    s,a,m,_,_,_=base.fixture()
    # Tests instrument only these local namespaces, never a frozen module's
    # globals. Function objects still execute the unchanged pinned algorithms.
    s=SimpleNamespace(**vars(s));s.ep=SimpleNamespace(**vars(s.ep))
    ds=pd.bdate_range('2008-01-02','2016-11-02');x=np.arange(len(ds),dtype=float);names=m['symbols'][0]
    p={'adj':pd.DataFrame({v:100.+.035*(j+1)*x+.7*np.sin(x/(j+4.))+.3*np.cos(x/(11+j)) for j,v in enumerate(names)},index=ds)};c=ds.strftime('%Y-%m-%d').tolist();r=make_review(m,c)
    return sync_core((s,a,m,p,c,r))

def physical(args,n):
    s,a,m,p,c,r=args;aa={key:np.asarray(v)[:n].copy() for key,v in a.items()};mm={key:deepcopy(m[key][:n]) for key in ('dates','ends','symbols')};mm['etfs']=m['etfs'].copy();last=mm['dates'][-1] if n else None
    return s,aa,mm,{'adj':p['adj'].loc[:last].copy() if n else p['adj'].iloc[:0].copy()},c[:c.index(last)+2] if n else [],deepcopy(r)

def run(args,n=None,**kw):return k.prepare(*args,n=n,**kw)

def assert_outputs(test,a,b):
    k._exact(a[0],b[0],'orders');k._exact(a[3],b[3],'audit')
    for role in k.ARMS:
        test.assertEqual(a[1][role].tobytes(),b[1][role].tobytes());test.assertEqual(a[2][role].tobytes(),b[2][role].tobytes())

class Sources(unittest.TestCase):
    def test_source_all_U0_before_E_including_ETF(self):
            args=fixture();args[1]['eligible'][:]=False;fs=k.build_features(args[3],args[2],args[4],args[1],args[5]);self.assertEqual(fs[1]['active_indices'],list(range(6)));self.assertEqual(len(fs[1]['cells'][5]['observations']),253);self.assertEqual(fs[1]['required_dates'][252],fs[1]['numerator_dates'][1]);self.assertIsNone(fs[1]['daily_target'])

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

    def test_outside_U0_no_new253(self):
            args=fixture();args[3]['adj']['A']=np.nan;sync_core(args);f=run(args,2)[0][1];self.assertNotIn(0,f['u0_indices']);self.assertEqual(f['cells'][0]['observations'],[])

    def test_empty_U0_exactcopy(self):
            args=fixture();args[3]['adj'].iloc[:,:]=100.;sync_core(args);out=run(args,2);self.assertEqual(out[0][1]['branch'],'empty_domain')
            for arm in k.ARMS[1:]:self.assertEqual(out[3][arm].tobytes(),out[3][k.ARMS[0]].tobytes());k._exact(out[1][arm],out[1][k.ARMS[0]],'empty orders')

    def test_singleton_U0_rank_half(self):
            args=fixture();args[3]['adj'].loc[:,list('BCDEF')]=np.nan;sync_core(args);f=run(args,2)[0][1];self.assertEqual(f['u0_indices'],[0]);self.assertEqual(f['daily_target'],[.5,None,None,None,None,None])

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
            def sg(fs):self.assertEqual(events,['episode','inc']);self.assertTrue(all(f['daily_target'] is None and f['block_target'] is None for f in fs));events.append('source');return {'passed':True,'scope':'source'}
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
            for arm,key in zip(k.ARMS[1:],('daily','block')):
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
                if which=='stat':f['cells'][0]['statistics']['block_quality']=math.nextafter(f['cells'][0]['statistics']['block_quality'],math.inf)
                elif which=='domain':f['u0_indices']=f['u0_indices'][1:]
                elif which=='rank':f['daily_target'][0]=math.nextafter(f['daily_target'][0],math.inf)
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
            for key,arm in zip(('daily','block'),k.ARMS[1:]):
                self.assertTrue(f[key+'_identity']);self.assertEqual(out[4]['events'][1][key+'_deltas'],[0.]*6);self.assertTrue(all(t is not None for t in f[key+'_target']))
                for index in (2,3):self.assertEqual(out[index][arm].tobytes(),out[index][k.ARMS[0]].tobytes())
                k._exact(out[1][arm],out[1][k.ARMS[0]],'identity orders')
            self.assertEqual(out[4]['counts']['nonpositive_original_cells'],6);self.assertEqual(out[4]['counts']['positive_raw_multiplications'],0)

    def test_cached_identity_flag_tamper_hard(self):
            args=fixture();out=run(args);fs=deepcopy(out[0]);fs[1]['daily_identity']=not fs[1]['daily_identity']
            with self.assertRaises(ValueError):k.decisions(args[0],args[1],args[2],fs)

class ConcentrationIntegration(unittest.TestCase):
    def formation(self,values):
        args=fixture();dates=pd.to_datetime(args[5]['clock_events'][1]['formation_dates'])
        for symbol in args[2]['symbols'][1]:args[3]['adj'].loc[dates,symbol]=values
        return sync_core(args)

    def jumps(self,indices):
        values=[1.];price=1.
        for i in range(1,253):
            if i in indices:price*=2.
            values.append(price)
        return values

    def test_partial_and_n0_do_not_measure_or_convert_poison(self):
        args=fixture();args[3]['adj']=args[3]['adj'].astype(object)
        args[3]['adj'].iloc[:,:]=base.Poison()
        with patch.object(k,'measure',side_effect=AssertionError('inactive arithmetic')):
            for n in (0,1):
                out=run(args,n)
                self.assertEqual(out[4]['counts']['original_core_cells'],0)
                self.assertEqual(out[4]['counts']['quality_float_conversions'],0)

    def test_zero_mass_is_neutral_but_still_has_two_quality_conversions(self):
        args=self.formation([100.]*253);out=run(args,2);c=out[4]['counts']
        self.assertEqual(c['required_cells'],6);self.assertEqual(c['no_positive_gain_cells'],6)
        self.assertEqual(c['quality_float_conversions'],12)
        self.assertEqual(c['concentration_evaluations'],0)
        self.assertEqual(c['quality_normalization_evaluations'],0)
        self.assertEqual(c['positive_gain_days'],0)
        for role in k.ROLES:self.assertTrue(out[0][1][role+'_identity'])

    def test_zero_quality_keeps_full_domain_without_fallback(self):
        args=self.formation(self.jumps({1}));out=run(args,2);f=out[0][1];c=out[4]['counts']
        self.assertEqual(f['branch'],'available');self.assertEqual(f['u0_indices'],list(range(6)))
        for role in k.ROLES:
            self.assertFalse(f[role+'_identity'])
            self.assertEqual(c[role+'_zero_quality_cells'],6)
            self.assertEqual(c[role+'_ranked_cells'],6)
            self.assertEqual(len(set(f[role+'_target'])),1)
            self.assertTrue(all(cell['statistics'][role+'_raw']==0. for cell in f['cells']))
        self.assertEqual(c['positive_raw_multiplications'],12)

    def test_positive_uniform_gains_produce_whole_vector_raw_identity(self):
        args=self.formation(self.jumps(set(range(1,253))))
        args[1]['core'][1,0]+=5e-13;args[1]['baseline'][1,0]+=5e-13
        out=run(args,2)
        for role,arm in zip(k.ROLES,k.ARMS[1:]):
            self.assertTrue(out[0][1][role+'_identity'])
            self.assertEqual(out[4]['events'][1][role+'_deltas'],[0.]*6)
            self.assertEqual(out[2][arm].tobytes(),out[2][k.ARMS[0]].tobytes())
            self.assertEqual(out[3][arm].tobytes(),out[3][k.ARMS[0]].tobytes())
            k._exact(out[1][arm],out[1][k.ARMS[0]],'Uniform positive identity orders')

    def test_rank_unchanged_is_descriptive_not_a_copy_shortcut(self):
        args=self.formation(self.jumps(set(range(21,253,21))))
        args[1]['core'][1,0]+=5e-13;args[1]['baseline'][1,0]+=5e-13
        out=run(args,2);f=out[0][1];event=out[4]['events'][1];counts=out[4]['counts']
        self.assertTrue(f['block_identity']);self.assertFalse(f['daily_identity'])
        self.assertTrue(event['daily_rank_unchanged']);self.assertEqual(counts['daily_rank_unchanged_events'],1)
        self.assertNotEqual(event['daily_deltas'][0],0.)
        self.assertNotEqual(out[3][k.ARMS[1]][1].tobytes(),out[3][k.ARMS[0]][1].tobytes())
        self.assertEqual(event['block_deltas'],[0.]*6)
        self.assertEqual(out[3][k.ARMS[2]].tobytes(),out[3][k.ARMS[0]].tobytes())

    def test_zero_quality_on_nonpositive_raw_is_counted_without_multiplication(self):
        args=self.formation([2.,4.]+[1.]*251);out=run(args,2);c=out[4]['counts']
        self.assertEqual(c['nonpositive_original_cells'],6);self.assertEqual(c['positive_raw_multiplications'],0)
        for role in k.ROLES:
            self.assertEqual(c[role+'_zero_quality_cells'],6)
            self.assertTrue(out[0][1][role+'_identity'])

    def test_exact_concentration_schema_and_actual_operation_scope(self):
        out=run(fixture());c=out[4]['counts'];f=out[0][1];stats=[cell['statistics'] for cell in f['cells']]
        self.assertEqual(tuple(c),k.COUNT_FIELDS);self.assertEqual(tuple(out[4]['events'][1]),k.DECISION_EVENT_KEYS)
        self.assertEqual(tuple(f),k.EVENT_KEYS)
        self.assertTrue(all(tuple(s)==k.STAT_KEYS and s['schema']==k.STAT_SCHEMA for s in stats))
        self.assertEqual(c['price_observations'],253*6);self.assertEqual(c['gain_return_observations'],252*6)
        self.assertEqual(c['gain_statistics'],6);self.assertEqual(c['quality_float_conversions'],12)
        self.assertEqual(c['positive_gain_days'],sum(len(s['positive_gain_indices']) for s in stats))
        nz=sum(not s['no_positive_gains'] for s in stats)
        self.assertEqual(c['concentration_evaluations'],2*nz);self.assertEqual(c['quality_normalization_evaluations'],2*nz)
        self.assertEqual(c['positive_raw_multiplications'],2*sum(s['original_raw']>0. for s in stats))
        for role,arm in zip(k.ROLES,k.ARMS[1:]):
            self.assertEqual(c[role+'_ranked_cells'],6)
            self.assertEqual(c[role+'_zero_quality_cells'],sum(s[role+'_quality']==0. for s in stats))
            self.assertEqual(c[role+'_rank_unchanged_events'],sum(e[role+'_rank_unchanged'] for e in out[4]['events']))
            self.assertEqual(c[role+'_changed_holdings'],sum(set(a['symbols'])!=set(b['symbols']) for a,b in zip(out[1][arm],out[1][k.ARMS[0]])))
        self.assertEqual(c['candidate_control_changed_holdings'],sum(set(a['symbols'])!=set(b['symbols']) for a,b in zip(out[1][k.ARMS[1]],out[1][k.ARMS[2]])))
        self.assertEqual(c['new_model_fits'],0);self.assertEqual(c['predictive_model_fits'],0)
        self.assertFalse(any('survival' in name or 'endpoint' in name for name in c))

    def test_parent_and_pure_kernel_remain_pinned_and_unmodified(self):
        for path,want in ((k.ORIGINAL_PATH,k.ORIGINAL_SHA),(k.MATH_PATH,k.MATH_SHA)):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),want)
        self.assertEqual(k.source.SOURCE_SCHEMA,'gain_concentration265_metadata_source_contract_v1')
        self.assertIs(k._old_data,k.original._old_data)
        self.assertIs(k._old_cells,k.original._old_cells)
        self.assertIs(k._scan,k.original._scan)
        self.assertEqual(k.original.ARMS,('incumbent_local67','endpoint_gain_age_control','continuous_gain_survival_candidate'))
        args=fixture()
        self.assertIsInstance(args[0],SimpleNamespace)
        self.assertIsInstance(args[0].ep,SimpleNamespace)


if __name__=='__main__':unittest.main(verbosity=2)

