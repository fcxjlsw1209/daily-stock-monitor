"""Synthetic-only264 independent source/path/selection/capability checks."""
import ast
import copy
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_gain_survival264_independent as ind

FIXTURE=ind.ROOT/'work/test_monthly_calendar_clock248_v2_independent.py'
FIXTURE_SHA='1213e1bad7836293547a76c9d27c60c6392b573c7eeeabd45ff60225c4885b84'
base=ind.load_pinned(FIXTURE,FIXTURE_SHA,'gain264_fake_original')
def review(meta,calendar,arrays):
    b=ind.core.original_base(arrays,meta,len(meta['dates']));sessions=[pd.Timestamp(x) for x in calendar]
    events=[ind.event_clock(q,b,sessions) for q in range(len(meta['dates']))];ids=[]
    for e in events:
        for j,s in enumerate(e['symbols']):
            if s in ind.peer.BOUNDS:
                z=ind.peer.BOUNDS[s];ids.append(dict(query_index=e['query_index'],column_index=j,date=e['date'],symbol=s,issuer_key=z['issuer_key'],not_before=z['not_before'],available_from=z['available_from'],bound_visible=z['available_from']<=e['date']))
    return dict(schema=ind.SOURCE_SCHEMA,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,panel_first_session='2008-01-02',listing_bounds=copy.deepcopy(list(ind.peer.BOUNDS.values())),clock_events=[ind.clock_record(e) for e in events],identity_cells=ids)
def fixture():
    p,m,c,a,r=base.fixture();names=sorted({s for row in m['symbols'] for s in row});axis=pd.bdate_range('2008-01-02',max(m['ends']));x=np.arange(len(axis),dtype=float)
    p={'adj':pd.DataFrame({s:100.+(.018+j*.004)*x+1.5*np.sin(x/(6.+j)) for j,s in enumerate(names)},index=axis)};c=axis.strftime('%Y-%m-%d').tolist();base.set_core((p,m,c,a,{}));return p,m,c,a,review(m,c,a)
def physical(args,n):
    p,m,c,a,r=args;aa={k:np.asarray(v)[:n].copy() for k,v in a.items()};mm={k:copy.deepcopy(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    if n:t=c.index(mm['dates'][-1]);cc=c[:t+2];pp={'adj':p['adj'].loc[:mm['dates'][-1]].copy()}
    else:cc=[];pp={'adj':p['adj'].iloc[:0].copy()}
    return pp,mm,cc,aa,r
def refresh(args):
    p,m,c,a,r=args;base.set_core(args);return p,m,c,a,review(m,c,a)
def proof(args,n=None):
    p,m,c,a,r=args;receipt=ind.verify_original_source(p,m,c,a,n);return receipt,ind.validated_original_proof(receipt,n=n)
def run(args,n=None,cap=None):
    p,m,c,a,r=args
    if cap is None:_,cap=proof(args,n)
    raw=ind.independent_features(p,m,c,a,r,n,original_proof=cap);f=ind.finalize(raw,a['eligible'].shape[1]);s,ch,pre,audit=ind.selection_oracle(a,m,f,n)
    return (f,ind.orders_from_choices(s,ch,m),s,pre,audit),cap
def verify(args,out,cap,n=None):
    p,m,c,a,r=args;f,o,s,pre,audit=out;return ind.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=audit,n=n,original_proof=cap)
def pricepath():return [100.+.05*k+math.sin(k/7.) for k in range(253)]

class MathTests(unittest.TestCase):
    def test_no_gain_neutral_signed_raw(self):
        for p in ([100.]*253,[400.-k for k in range(253)]):
            for raw in (-2.,-0.,0.,2.):
                s=ind.measure(p,raw);self.assertTrue(s['no_positive_gains']);self.assertEqual((s['survival_quality'],s['endpoint_quality']),(1.,1.));self.assertEqual(s['survival_raw'].hex(),raw.hex());self.assertEqual(s['endpoint_raw'].hex(),raw.hex())
    def test_last_gain_denominator_only(self):
        s=ind.measure([100.]*252+[110.],2.);self.assertEqual(s['positive_anchor_indices'],[252]);self.assertEqual(s['potential_ages'],[0]);self.assertEqual(s['survival_ages'],[0]);self.assertEqual(s['gain_sum'],110./100.-1.);self.assertEqual(s['survival_quality'],0.);self.assertEqual(s['endpoint_quality'],0.);self.assertEqual(s['survival_comparisons'],0)
    def test_first_breach_excludes_anchor_and_breach(self):
        p=[100.,110.,110.,105.,115.]+[115.]*248;s=ind.measure(p,2.);i=s['positive_anchor_indices'].index(1)
        self.assertEqual(s['first_breach_indices'][i],3);self.assertEqual(s['survival_ages'][i],1);self.assertTrue(s['endpoint_retained'][i]);self.assertEqual(s['potential_ages'][i],251)
    def test_late_breach_not_recovered_endpoint(self):
        p=[100.,110.]+[110.]*249+[105.,105.];s=ind.measure(p,1.)
        self.assertGreater(s['survival_quality'],s['endpoint_quality']);self.assertEqual(s['endpoint_quality'],0.)
    def test_recovery_does_not_reset(self):
        p=[100.,110.,105.]+[110.]*250;s=ind.measure(p,1.);i=s['positive_anchor_indices'].index(1)
        self.assertEqual(s['survival_ages'][i],0);self.assertEqual(s['first_breach_indices'][i],2);self.assertLess(s['survival_quality'],s['endpoint_quality'])
    def test_literal_aggregation(self):
        p=pricepath();s=ind.measure(p,3.);g=s['positive_gains'];L=s['survival_ages'];a=s['potential_ages'];keep=s['endpoint_retained'];D=252.*math.fsum(g)
        self.assertEqual(s['survival_quality'].hex(),(math.fsum(x*float(y) for x,y in zip(g,L))/D).hex());self.assertEqual(s['endpoint_quality'].hex(),(math.fsum(x*float(y if k else 0) for x,y,k in zip(g,a,keep))/D).hex());self.assertEqual(s['survival_raw'].hex(),(3.*s['survival_quality']).hex())
        self.assertTrue(all(0<=d<=age<=251 for d,age in zip(L,a)))
    def test_nonpositive_raw_never_attenuated(self):
        for raw in (-4.,-0.,0.):
            s=ind.measure(pricepath(),raw);self.assertEqual(s['survival_raw'].hex(),raw.hex());self.assertEqual(s['endpoint_raw'].hex(),raw.hex())
    def test_invalid_source_before_no_gain(self):
        for value in (True,None,'100',math.inf,math.nan,0.,-1.,Fraction(1,10**1000)):
            p=[100.]*253;p[240]=value
            with self.subTest(value=str(value)[:20]),self.assertRaises((ValueError,ArithmeticError)):ind.measure(p,-1.)
    def test_arithmetic_underflow_overflow(self):
        with self.assertRaises(ArithmeticError):ind.product(5e-324,.1)
        with self.assertRaises(ArithmeticError):ind.product(1e308,252.)
        with self.assertRaises(ArithmeticError):ind.quotient(5e-324,2.)
        p=[1.]*253;p[1]=1e308
        with self.assertRaises(ArithmeticError):ind.measure(p,1.)
        self.assertEqual(ind.product(1.,0.),0.)
    def test_exact_scaling(self):
        s=ind.measure(pricepath(),2.);ind.record_check(s,ind.measure([4.*x for x in pricepath()],2.),'Power2 scale')
    def test_rank_n0_n1_n6_exact(self):
        self.assertEqual(ind.ranks([]),[]);self.assertEqual(ind.ranks([1.]),[.5]);x=[0.,1.,1.,3.,4.,5.];self.assertEqual(ind.ranks(x),ind.peer.average_percentile(x))
    def test_binary64_upper_bound_counterexample(self):
        upper=1.0081;g=upper-1.;q=(g*251.)/(252.*g)
        self.assertGreater(q,251./252.)
        s=ind.measure([1.]+[upper]*252,1.);self.assertEqual(s['survival_quality'],q);self.assertEqual(s['endpoint_quality'],q)

class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.out,cls.cap=run(cls.args)
    def test_full_receipts_no_repeated_original_history(self):
        p,m,c,a,r=self.args
        with patch.object(ind.old,'original_evidence',side_effect=AssertionError('Repeated original source')):
            x=verify(self.args,self.out,self.cap);raw=ind.unranked_prefix(self.out[0],3);y=ind.verify_features(p,m,c,a,r,raw,original_proof=self.cap)
        self.assertEqual(x['original_source_cells'],0);self.assertEqual(y['original_source_price_observations'],0);self.assertEqual(x['original_source_proof_sha256'],self.cap._certificate_sha256);self.assertEqual(x['original_episode_calls'],3)
    def test_capability_requires_issued_exact_object(self):
        receipt,cap=proof(self.args)
        with self.assertRaises(ValueError):ind.validated_original_proof(copy.deepcopy(receipt))
        receipt['original_events'][2]['original_raw'][0]+=1.
        with self.assertRaises(ValueError):ind.validated_original_proof(receipt)
        for invalid in ({},self.cap._certificate_sha256):
            with self.assertRaises(ValueError):run(self.args,cap=invalid)
        with self.assertRaises(TypeError):cap._events=()
    def test_capability_prefix_no_future_body(self):
        receipt,cap=proof(self.args)
        class Poison:
            def __str__(self):raise AssertionError('Future body')
        object.__setattr__(cap,'_events',cap._events[:2]+(Poison(),))
        object.__setattr__(cap,'_bindings',cap._bindings[:2]+(Poison(),))
        for n in (0,2):
            p,m,c,a,r=physical(self.args,n);run((self.args[0],m,c,a,r),cap=cap)
    def test_capability_rejects_replacement_panel(self):
        p,m,c,a,r=self.args
        for changed in ({'adj':p['adj']},copy.deepcopy(p)):
            with self.assertRaises(ValueError):ind.independent_features(changed,m,c,a,r,original_proof=self.cap)
    def test_same_panel_recent63_mutation_rejected_before_gain_math(self):
        args=copy.deepcopy(self.args);p,m,c,a,r=args;receipt,cap=proof(args);t=c.index(m['dates'][-1]);day=pd.Timestamp(c[t-10]);symbol=m['symbols'][-1][0]
        p['adj'].at[day,symbol]*=1.01
        with patch.object(ind,'measure',side_effect=AssertionError('Snapshot failure precedes new statistics')):
            with self.assertRaisesRegex(ValueError,'snapshot values unchanged'):run(args,cap=cap)
    def test_snapshot_only_consumed_dates_and_columns(self):
        args=copy.deepcopy(self.args);p,m,c,a,r=args;receipt,cap=proof(args)
        p['adj'].loc[p['adj'].index>pd.Timestamp(m['dates'][1]),:]='future poison';p['adj']['UNCONSUMED']='unused poison'
        out,_=run(args,2,cap=cap);self.assertEqual(len(out[0]),2)
        self.assertEqual(ind.proof_fields(cap,2)['original_source_snapshot_price_observations'],(c.index(m['dates'][1])+1)*len(set(m['symbols'][1])))
    def test_snapshot_no_numeric_for_n0_or_partial_only(self):
        args=copy.deepcopy(self.args);p,m,c,a,r=args;receipt,cap=proof(args);p['adj'].iloc[:,:]='unread numeric'
        for n in (0,1):
            out,_=run(args,n,cap=cap);check=verify(args,out,cap,n)
            self.assertEqual(check['original_source_snapshot_price_observations'],0);self.assertEqual(check['original_source_snapshot_hashes'],0)
    def _snapshot_invalid_missing_replacement(self,replacement):
        args=copy.deepcopy(self.args);p,m,c,a,r=args;p['adj']=p['adj'].astype(object)
        p['adj'].iat[0,0]=float('nan');args=refresh(args);receipt,cap=proof(args)
        ind.validate_source_snapshot(cap,p,m,c,len(m['dates']))
        p['adj'].iat[0,0]=replacement
        with patch.object(ind,'measure',side_effect=AssertionError('Invalid snapshot type must precede gain mathematics')):
            with self.assertRaisesRegex(ValueError,'Snapshot original real nonboolean'):run(args,cap=cap)
    def test_snapshot_history_nan_cannot_become_NaT(self):
        self._snapshot_invalid_missing_replacement(pd.NaT)
    def test_snapshot_history_nan_cannot_become_complex_nan(self):
        self._snapshot_invalid_missing_replacement(complex(float('nan'),0.))
    def test_consumed_identity_core_tamper(self):
        args=copy.deepcopy(self.args);args[1]['symbols'][2][0]='OTHER'
        with self.assertRaises(ValueError):run(args,cap=self.cap)
        args=copy.deepcopy(self.args);args[3]['core'][2,0]+=5e-13
        with self.assertRaises(ValueError):run(args,cap=self.cap)
    def test_physical_and_fullbundle_prefixes(self):
        for n in range(4):
            args=physical(self.args,n);out,cap=run(args);verify(args,out,cap)
            wanted,_=run(self.args,n)
            for k in (0,1,4):ind.record_check(out[k],wanted[k],'Physical prefix')
            s,ch,pre,a=ind.selection_oracle(args[3],args[1],self.out[0],n);ind.verify_selection(args[3],args[1],self.out[0],ind.orders_from_choices(s,ch,args[1]),scores=s,pre_scores=pre,audit=a,n=n)
    def test_calm_partial_exact(self):
        f,o,s,p,a=self.out
        for q in (0,1):
            self.assertEqual(f[q]['active_indices'],[])
            for arm in ind.ARMS[1:]:ind.exact_array(s[arm][q],s[ind.ARMS[0]][q],'Inactive final');ind.exact_array(p[arm][q],p[ind.ARMS[0]][q],'Inactive pre');self.assertEqual(o[arm][q],o[ind.ARMS[0]][q])
    def test_nonpositive_whole_vector_identity(self):
        args=copy.deepcopy(self.args);p,m,c,a,r=args;t=c.index(m['dates'][2])
        for s in m['symbols'][2]:p['adj'].at[pd.Timestamp(c[t-294]),s]=100000.
        args=refresh(args);out,cap=run(args);verify(args,out,cap);e=out[0][2]
        self.assertTrue(e['endpoint_identity']);self.assertTrue(e['survival_identity'])
        for arm in ind.ARMS[1:]:ind.exact_array(out[3][arm][2],out[3][ind.ARMS[0]][2],'Identity pre');self.assertEqual(out[1][arm][2],out[1][ind.ARMS[0]][2])
    def test_full_delta_uses_frozen_core_without_snapping(self):
        args=copy.deepcopy(self.args);args[3]['core'][2,0]+=5e-13;out,cap=run(args);verify(args,out,cap);e=out[0][2];b=ind.core.original_base(args[3],args[1],3)
        for arm,role in zip(ind.ARMS[1:],('endpoint','survival')):
            self.assertFalse(e[role+'_identity'])
            for j in e['u0_indices']:self.assertEqual(out[4]['events'][2][role+'_deltas'][j].hex(),float(float(b['cw'][2]*e[role+'_target'][j])-b['core'][2,j]).hex())
    def test_Efalse_negative_source_still_required(self):
        args=copy.deepcopy(self.args);p,m,c,a,r=args;t=c.index(m['dates'][2]);s=m['symbols'][2][0];a['eligible'][2,0]=False;p['adj'].at[pd.Timestamp(c[t-294]),s]=100000.;p['adj'].at[pd.Timestamp(c[t-200]),s]=math.nan;args=refresh(args)
        with self.assertRaises(ValueError):run(args)
    def test_source_scan_before_any_measurement(self):
        args=copy.deepcopy(self.args);p,m,c,a,r=args;t=c.index(m['dates'][2]);s=m['symbols'][2][-1];p['adj'].at[pd.Timestamp(c[t-180]),s]=0.;args=refresh(args)
        with patch.object(ind,'measure',side_effect=AssertionError('Premature measurement')):
            with self.assertRaises(ValueError):run(args)
    def test_raw_source_gate_no_rank(self):
        p,m,c,a,r=self.args;raw=ind.unranked_prefix(self.out[0],3)
        with patch.object(ind,'ranks',side_effect=AssertionError('Premature ranks')):
            ind.verify_features(p,m,c,a,r,raw,original_proof=self.cap);ind.verify_features_cached(a,m,raw)
        with self.assertRaises(ValueError):ind.verify_features_cached(a,m,self.out[0])
    def test_future_features_and_meta_not_read(self):
        class Poison(dict):
            def __deepcopy__(self,memo):raise AssertionError('Future deepcopy')
            def __eq__(self,other):raise AssertionError('Future equality')
        f=copy.deepcopy(self.out[0]);f[2]=Poison(blocked=object());ind.selection_oracle(self.args[3],self.args[1],f,2)
        args=physical(self.args,1);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].iloc[:]='partial poison';run(args)
    def test_discrete_and_numeric_tamper(self):
        for mutate in (lambda f:f[2]['cells'][0]['statistics']['survival_ages'].__setitem__(0,999),lambda f:f[2].__setitem__('endpoint_identity',True),lambda f:f[2]['cells'][0]['statistics'].__setitem__('survival_raw',math.nextafter(f[2]['cells'][0]['statistics']['survival_raw'],math.inf)),lambda f:f[2]['cells'][0]['observations'][0].__setitem__('date',f[2]['date'])):
            f=copy.deepcopy(self.out[0]);mutate(f)
            with self.assertRaises(ValueError):ind.cached_features(self.args[3],self.args[1],f)
    def test_source_and_input_immutable(self):
        snapshot=pickle.dumps(self.args);verify(self.args,self.out,self.cap);self.assertEqual(snapshot,pickle.dumps(self.args))
    def test_actual_measure_calls_and_work(self):
        p,m,c,a,r=self.args;raw=ind.unranked_prefix(self.out[0],3);original=ind.measure
        with patch.object(ind,'measure',wraps=original) as calls:x=ind.verify_features_cached(a,m,raw)
        self.assertEqual(calls.call_count,x['feature_counts']['required_cells']);self.assertEqual(x['return_evaluations'],252*calls.call_count);self.assertEqual(x['statistic_evaluations'],calls.call_count);self.assertEqual(x['momentum_evaluations'],0);self.assertEqual(x['rank_evaluations'],0)
    def test_no_producer_or_host_dependency(self):
        source=Path(ind.__file__).read_text();tree=ast.parse(source)
        names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(x and ('264_kernel' in x or '264_host' in x) for x in names));self.assertFalse(any('264_kernel' in str(x) or '264_host' in str(x) for x in ind.dependency_files()))

class BilateralCross(unittest.TestCase):
    def test_complete_host_source_gates_and_all_synthetic_prefixes(self):
        # Producer/host are imported only by this integration test; the oracle
        # above has no producer/host dependency or shared gain implementation.
        import monthly_gain_survival264_kernel as kernel
        import monthly_gain_survival264_host as host
        import test_monthly_gain_survival264_kernel as fixture_module
        for path in ([1.]+[1.0081]*252,pricepath(),[100.]*253,[100.,110.,105.]+[110.]*250):
            for raw in (-2.,-0.,0.,2.):ind.record_check(kernel.measure(path,raw),ind.measure(path,raw),'Independent canonical path statistics')
        study,arrays,meta,panel,calendar,source=fixture_module.fixture();n=len(meta['dates']);width=arrays['eligible'].shape[1]
        b=ind.core.original_base(arrays,meta,n);original_scores,original_choices=ind.core.episode(b['P'],b)
        expected=ind.orders_from_choices({a:original_scores for a in ind.ARMS},{a:original_choices for a in ind.ARMS},meta)[ind.ARMS[0]]
        full,certificate=host.prepare_one(study,arrays,meta,panel,calendar,source,n,expected,width=width)
        features,orders,scores,pres,audit=full
        self.assertEqual(certificate['counts'],audit['counts']);self.assertEqual(certificate['original_source']['u0_indices'],[f['u0_indices'] for f in features])
        for check in (certificate['gates']['source'],certificate['full']):
            self.assertEqual(check['original_source_proof_sha256'],host.digest(certificate['original_source']));self.assertEqual(check['original_source_proof_events'],n)
            self.assertEqual(check['original_source_price_observations'],0);self.assertEqual(check['original_source_cells'],0);self.assertEqual(check['momentum_evaluations'],0)
        for count in range(n+1):
            aa,mm,pp,cc=host.physical_inputs(arrays,meta,panel,calendar,count)
            physical_result,physical_proof=host.prepare_one(study,aa,mm,pp,cc,source,count,expected[:count],width=width)
            cached,cached_proof=host.decisions(study,aa,mm,features,count,expected[:count],certificate['gates']['source'],original_certificate=certificate['original_source'],original_anchor=certificate['original_source_anchor'],width=width)
            ind.record_check(physical_result[0],features[:count],'Full/physical source identity')
            for got in (physical_result[1:],cached):
                ind.record_check(got[0],{a:orders[a][:count] for a in ind.ARMS},'Full/physical ordered choices')
                for arm in ind.ARMS:
                    ind.exact_array(got[1][arm],scores[arm][:count],'Full/physical final scores');ind.exact_array(got[2][arm],pres[arm][:count],'Full/physical pre-scores')
                self.assertEqual(got[3]['events'],audit['events'][:count])
            self.assertEqual(physical_proof['full']['cached_slot_validations'],0);self.assertEqual(cached_proof['selection']['original_source_price_observations'],0)
            self.assertEqual(cached_proof['original_source_certificate_sha256'],certificate['original_source_anchor']['certificate_sha256'])
            projection=host.original_prefix_certificate(certificate['original_source'],certificate['original_source_anchor'],count)
            self.assertEqual(cached_proof['consumed_original_source_sha256'],host.digest(projection))
            self.assertEqual(physical_proof['original_source_anchor']['checked_events'],count)
        with patch.object(kernel,'measure',side_effect=AssertionError('Incumbent failure must precede statistics')):
            wrong=copy.deepcopy(expected);wrong[0]['symbols']=['wrong']
            with self.assertRaises(ValueError):host.prepare_one(study,arrays,meta,panel,calendar,source,n,wrong,width=width)
        class Poison(dict):
            def __deepcopy__(self,memo):raise AssertionError('Future body copied')
            def __eq__(self,other):raise AssertionError('Future body compared')
            def items(self):raise AssertionError('Future body serialized')
        poisoned=copy.deepcopy(features);poisoned[-1]=Poison(blocked=object())
        anchor=certificate['original_source_anchor']
        for count in (0,n-1):
            aa,mm,pp,cc=host.physical_inputs(arrays,meta,panel,calendar,count)
            bad_future=copy.deepcopy(certificate['original_source'])
            for key in ('original_events','u0_indices'):bad_future[key][count:]=[Poison(blocked=object()) for _ in range(n-count)]
            with self.assertRaises((TypeError,AssertionError)):host.digest(bad_future)
            got,check=host.decisions(study,aa,mm,poisoned,count,expected[:count],certificate['gates']['source'],original_certificate=bad_future,original_anchor=anchor,width=width)
            ind.record_check(got[0],{a:orders[a][:count] for a in ind.ARMS},'Future certificate must not affect orders')
            self.assertEqual(check['original_source_certificate_sha256'],anchor['certificate_sha256'])
        count=n-1;aa,mm,pp,cc=host.physical_inputs(arrays,meta,panel,calendar,count)
        with patch.object(kernel,'decisions',side_effect=AssertionError('Consumed certificate must fail before decisions')):
            for mode in ('event','u0','missing'):
                bad=copy.deepcopy(certificate['original_source'])
                if mode=='event':bad['original_events'][count-1]['date']='1999-01-01'
                elif mode=='u0':bad['u0_indices'][count-1]=[width]
                else:bad['original_events']=[]
                with self.assertRaises(ValueError):host.decisions(study,aa,mm,features,count,expected[:count],certificate['gates']['source'],original_certificate=bad,original_anchor=anchor,width=width)
            with self.assertRaises(ValueError):host.decisions(study,aa,mm,features,count,expected[:count],certificate['gates']['source'],original_certificate=certificate['original_source'],original_anchor=dict(anchor),width=width)

if __name__=='__main__':unittest.main(verbosity=2)
