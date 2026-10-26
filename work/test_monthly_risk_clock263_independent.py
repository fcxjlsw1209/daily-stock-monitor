"""263 synthetic source, risk-clock and integration checks; no actual inputs."""
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
import monthly_risk_clock263_independent as ind

FIXTURE=ind.ROOT/'work/test_monthly_calendar_clock248_v2_independent.py'
FIXTURE_SHA='1213e1bad7836293547a76c9d27c60c6392b573c7eeeabd45ff60225c4885b84'
base=ind.load_pinned(FIXTURE,FIXTURE_SHA,'risk_clock263_fake_original')

def review(meta,calendar,arrays):
    b=ind.core.original_base(arrays,meta,len(meta['dates']));sessions=[pd.Timestamp(x) for x in calendar]
    events=[ind.event_clock(q,b,sessions) for q in range(len(meta['dates']))];identities=[]
    for e in events:
        for j,s in enumerate(e['symbols']):
            if s in ind.peer.BOUNDS:
                bound=ind.peer.BOUNDS[s];identities.append(dict(query_index=e['query_index'],column_index=j,date=e['date'],symbol=s,issuer_key=bound['issuer_key'],not_before=bound['not_before'],available_from=bound['available_from'],bound_visible=bound['available_from']<=e['date']))
    return dict(schema=ind.SOURCE_SCHEMA,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,panel_first_session='2008-01-02',listing_bounds=copy.deepcopy(list(ind.peer.BOUNDS.values())),clock_events=[ind.clock_record(e) for e in events],identity_cells=identities)

def fixture():
    p,m,c,a,r=base.fixture();names=sorted({s for row in m['symbols'] for s in row});axis=pd.bdate_range('2008-01-02',max(m['ends']));x=np.arange(len(axis),dtype=float)
    frame=pd.DataFrame({s:100.+(.018+j*.004)*x+1.5*np.sin(x/(6.+j)) for j,s in enumerate(names)},index=axis)
    p={'adj':frame};c=axis.strftime('%Y-%m-%d').tolist();args=base.set_core((p,m,c,a,{}));return p,m,c,a,review(m,c,a)

def physical(args,n):
    p,m,c,a,r=args;aa={k:np.asarray(v)[:n].copy() for k,v in a.items()};mm={k:copy.deepcopy(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    if n:t=c.index(mm['dates'][-1]);cc=c[:t+2];pp={'adj':p['adj'].loc[:mm['dates'][-1]].copy()}
    else:cc=[];pp={'adj':p['adj'].iloc[:0].copy()}
    return pp,mm,cc,aa,r

def refresh(args):
    p,m,c,a,r=args;base.set_core(args);return p,m,c,a,review(m,c,a)

def run(args,n=None):
    p,m,c,a,r=args;raw=ind.independent_features(p,m,c,a,r,n);f=ind.finalize(raw,a['eligible'].shape[1]);s,ch,pre,audit=ind.selection_oracle(a,m,f,n)
    return f,ind.orders_from_choices(s,ch,m),s,pre,audit

def verify(args,out,n=None):
    p,m,c,a,r=args;f,o,s,pre,audit=out;return ind.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=audit,n=n)

def prices():return [90.+.13*k+.7*math.sin(k/3.) for k in range(107)]

class MathTests(unittest.TestCase):
    def test_constant_and_near_constant(self):
        for v in (0.,-0.,.1,-.25):
            result=ind.risk63([v]*63);self.assertTrue(result['constant']);self.assertEqual(result['mean'].hex(),v.hex());self.assertTrue(all(x.hex()==0.0.hex() for x in result['squares']));self.assertEqual(result['sd'].hex(),0.0.hex())
        x=[.1]*63;x[-1]=math.nextafter(.1,math.inf);self.assertGreater(ind.risk63(x)['sd'],0.)
    def test_canonical_two_pass_exact(self):
        x=[.03*math.sin(k) for k in range(63)];r=ind.risk63(x);mean=math.fsum(x)/63.;center=[v-mean for v in x];ss=math.fsum(v*v for v in center)
        self.assertEqual(r['mean'].hex(),mean.hex());self.assertEqual(r['sd'].hex(),math.sqrt(ss/62.).hex())
    def test_underflow_and_invalid(self):
        with self.assertRaises(ArithmeticError):ind.risk63([0.]*62+[1e-200])
        with self.assertRaises(ArithmeticError):ind.quotient(5e-324,62.)
        for v in (True,None,'1',math.inf,math.nan,0.,-1.,Fraction(1,10**1000)):
            x=prices();x[5]=v
            with self.subTest(v=str(v)[:30]),self.assertRaises((ValueError,ArithmeticError)):ind.measure(x)
    def test_107_clock_and_21_return_overlap(self):
        p=prices();s=ind.measure(p)
        self.assertEqual(s['momentum'],p[64]/p[0]-1.);self.assertEqual(s['current_returns'],[p[k]/p[k-1]-1. for k in range(44,107)]);self.assertEqual(s['lag_returns'],[p[k]/p[k-1]-1. for k in range(2,65)])
        self.assertEqual(s['current_returns'][:21],s['lag_returns'][-21:]);self.assertIsNone(s['current_ratio']);self.assertIsNone(s['lag_ratio'])
    def test_direct_momentum_not_reconstructed_gross(self):
        p=prices();p[0]=1e25;s=ind.measure(p);self.assertEqual(s['momentum'],-1.)
        p[0]=p[64];self.assertEqual(ind.measure(p)['momentum'],0.)
    def test_exact_scale_and_recent_clock_effect(self):
        p=prices();s=ind.measure(p);ind.record_check(s,ind.measure([4.*v for v in p]))
        p[-1]*=1.01;t=ind.measure(p);self.assertEqual(s['momentum'],t['momentum']);self.assertEqual(s['lag_risk'],t['lag_risk']);self.assertNotEqual(s['current_risk'],t['current_risk'])
    def test_constant_positive_returns_zero_risk(self):
        p=[1.]+[2.**k for k in range(106)];s=ind.measure(p);self.assertEqual(s['current_risk']['sd'],0.);self.assertEqual(s['lag_risk']['sd'],0.);self.assertEqual(s['current_risk']['mean'],1.)
    def test_canonical_n0_n1_n6_ties(self):
        self.assertEqual(ind.ranks([]),[]);self.assertEqual(ind.ranks([-1.]),[.5]);x=[0.,1.,1.,3.,4.,5.];r=ind.ranks(x);self.assertEqual(r,ind.peer.average_percentile(x))
        collapsed=[(2*sum(y<v for y in x)+sum(y==v for y in x)+1-len(x))/(2.*len(x)) for v in x];self.assertTrue(any(a.hex()!=b.hex() for a,b in zip(r,collapsed)))

class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.out=run(cls.args)
    def test_original_full_evidence_and_receipts(self):
        p,m,c,a,r=self.args;x=ind.verify_original_source(p,m,c,a);self.assertEqual(x['u0_indices'],[e['u0_indices'] for e in self.out[0]])
        self.assertEqual(x['original_source_price_observations'],sum((c.index(d)+1)*a['eligible'].shape[1] for d in m['dates'] if d!=ind.INITIAL));self.assertEqual(x['statistic_evaluations'],0)
        proof=verify(self.args,self.out);self.assertTrue(all(proof[k] for k in ind.PREPARE_FLAGS));self.assertEqual(proof['original_episode_calls'],3)
    def test_rawgate_no_ranking_and_immutable_source(self):
        p,m,c,a,r=self.args;snapshot=pickle.dumps(self.args);raw=ind.unranked_prefix(self.out[0],len(m['dates']))
        with patch.object(ind,'ranks',side_effect=AssertionError('Early rank')):
            full=ind.verify_features(p,m,c,a,r,raw);cached=ind.verify_features_cached(a,m,raw)
        self.assertEqual(pickle.dumps(self.args),snapshot);self.assertEqual(full['full_source_slot_inspections'],cached['cached_slot_validations']);self.assertEqual(cached['original_source_cells'],0)
        with self.assertRaises(ValueError):ind.verify_features_cached(a,m,self.out[0])
    def test_prefix_physical_small_fullfeatures(self):
        for n in range(4):
            part=physical(self.args,n);got=run(part);wanted=run(self.args,n);verify(part,got)
            for k in (0,1,4):ind.record_check(got[k],wanted[k])
            for k in (2,3):
                for arm in ind.ARMS:ind.exact_array(got[k][arm],wanted[k][arm],'Physical prefix')
            s,ch,pre,a=ind.selection_oracle(part[3],part[1],self.out[0],n);ind.verify_selection(part[3],part[1],self.out[0],ind.orders_from_choices(s,ch,part[1]),scores=s,pre_scores=pre,audit=a,n=n)
    def test_calm_and_partial_exact(self):
        f,o,s,p,a=self.out
        for q in (0,1):
            self.assertEqual(f[q]['active_indices'],[])
            for arm in ind.ARMS[1:]:ind.exact_array(s[arm][q],s[ind.ARMS[0]][q],'Inactive final');ind.exact_array(p[arm][q],p[ind.ARMS[0]][q],'Inactive pre');self.assertEqual(o[arm][q],o[ind.ARMS[0]][q])
    def test_full_core_delta_uses_frozen_C(self):
        args=copy.deepcopy(self.args);args[3]['core'][2,0]+=5e-13;out=run(args);verify(args,out);f,o,s,p,a=out;b=ind.core.original_base(args[3],args[1],3)
        for arm,key in zip(ind.ARMS[1:],('current_target','lag_target')):
            for j in f[2]['u0_indices']:
                delta=float(float(b['cw'][2]*f[2][key][j])-b['core'][2,j]);self.assertEqual(a['events'][2]['current_deltas' if key=='current_target' else 'lag_deltas'][j].hex(),delta.hex())
    def test_source_before_E_and_negative_M(self):
        p,m,c,a,r=copy.deepcopy(self.args);q=2;j=0;s=m['symbols'][q][j];t=c.index(m['dates'][q]);a['eligible'][q,j]=False;p['adj'].at[pd.Timestamp(c[t-294]),s]=10000.;args=refresh((p,m,c,a,r));out=run(args);e=out[0][q]
        self.assertIn(j,e['active_indices']);self.assertLess(e['cells'][j]['statistics']['momentum'],0.);verify(args,out)
    def test_required_lag_hole_hardfails_despite_E_false(self):
        p,m,c,a,r=copy.deepcopy(self.args);q=2;j=0;t=c.index(m['dates'][q]);a['eligible'][q,j]=False;p['adj'].at[pd.Timestamp(c[t-100]),m['symbols'][q][j]]=math.nan;args=refresh((p,m,c,a,r))
        with self.assertRaises(ValueError):run(args)
    def test_whole_query_zero_and_later_invalid(self):
        p,m,c,a,r=copy.deepcopy(self.args);q=2;t=c.index(m['dates'][q]);s=m['symbols'][q][0];p['adj'].loc[c[t-105]:c[t-42],s]=100.;args=refresh((p,m,c,a,r));out=run(args);self.assertEqual(out[0][q]['branch'],'zero_risk')
        for arm in ind.ARMS[1:]:ind.exact_array(out[3][arm][q],out[3][ind.ARMS[0]][q],'Whole query zero pre')
        self.assertTrue(all(x['statistics'] is None or x['statistics']['current_ratio'] is None for x in out[0][q]['cells']));verify(args,out)
        p['adj'].at[pd.Timestamp(c[t-100]),m['symbols'][q][1]]=math.nan;args=refresh((p,m,c,a,r))
        with self.assertRaises(ValueError):run(args)
    def test_empty_U0_and_singleton(self):
        args=copy.deepcopy(self.args);args[0]['adj'].iloc[:]=100.;args=refresh(args);out=run(args);self.assertEqual(out[0][2]['branch'],'empty_domain');verify(args,out)
        args=copy.deepcopy(self.args);keep=args[1]['symbols'][2][0]
        for s in args[0]['adj'].columns:
            if s!=keep:args[0]['adj'][s]=100.
        args=refresh(args);out=run(args);self.assertEqual(out[0][2]['u0_indices'],[0]);self.assertEqual(out[0][2]['lag_target'][0],.5)
    def test_original_grid_missing_older_row_fails(self):
        args=copy.deepcopy(self.args);args[0]['adj']=args[0]['adj'].drop(args[0]['adj'].index[20])
        with self.assertRaises(ValueError):run(args)
    def test_past_NaN_outside_new_window_is_original_semantics(self):
        args=copy.deepcopy(self.args);args[0]['adj'].iloc[20,0]=math.nan;args=refresh(args);verify(args,run(args))
    def test_native_types_and_tampered_saved_moments(self):
        for mutate in (lambda f:f[2].__setitem__('query_index',True),lambda f:f[2]['cells'][0]['statistics']['current_risk'].__setitem__('sd',math.nextafter(f[2]['cells'][0]['statistics']['current_risk']['sd'],math.inf)),lambda f:f[2]['cells'][0]['observations'][0].__setitem__('date',f[2]['date']),lambda f:f[2].__setitem__('u0_indices',[])):
            f=copy.deepcopy(self.out[0]);mutate(f)
            with self.assertRaises(ValueError):ind.cached_features(self.args[3],self.args[1],f)
    def test_visible_bounds_and_identity_required(self):
        b=ind.peer.BOUNDS['SPCX'];self.assertIsNone(ind._bound('SPCX','2026-06-08'));self.assertEqual(ind._bound('SPCX','2026-06-30'),b)
        with self.assertRaises(ValueError):ind._check_prices([{'date':'2026-06-05','value':1.}],['2026-06-05'],b)
        with self.assertRaises(ValueError):ind._check_prices([{'date':'2026-06-12','value':math.nan}],['2026-06-12'],b)
    def test_future_bodies_not_read(self):
        class Poison(dict):
            def __deepcopy__(self,memo):raise AssertionError('Future deepcopy')
            def __eq__(self,other):raise AssertionError('Future equality')
        args=copy.deepcopy(self.args);args[1]['symbols'][2]=Poison();args[4]['clock_events'][2]=Poison(query_index=2);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>args[1]['dates'][1]]='future poison';run(args,2)
        features=copy.deepcopy(self.out[0]);features[2]=Poison();ind.selection_oracle(self.args[3],self.args[1],features,2)
        args=physical(self.args,1);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].iloc[:]='partial poison';run(args)
    def test_actual_work_matches_receipt(self):
        p,m,c,a,r=self.args;raw=ind.unranked_prefix(self.out[0],3)
        with patch.object(ind,'measure',wraps=ind.measure) as measures,patch.object(ind,'risk63',wraps=ind.risk63) as risks,patch.object(ind,'ranks',wraps=ind.ranks) as ranks,patch.object(ind.core,'episode',wraps=ind.core.episode) as episodes:
            receipt=ind.verify_selection(a,m,self.out[0],self.out[1],scores=self.out[2],pre_scores=self.out[3],audit=self.out[4])
        self.assertEqual(measures.call_count,receipt['momentum_evaluations']);self.assertEqual(risks.call_count,receipt['statistic_evaluations']);self.assertEqual(ranks.call_count,receipt['rank_evaluations']);self.assertEqual(episodes.call_count,receipt['original_episode_calls'])
    def test_no_new_producer_or_host_dependency(self):
        text=Path(ind.__file__).read_text();tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):self.assertNotIn('risk_clock263_kernel',ast.get_source_segment(text,node));self.assertNotIn('risk_clock263_host',ast.get_source_segment(text,node))
        self.assertTrue(all('risk_clock263_kernel' not in str(p) for p in ind.dependency_files()))

class BilateralCross(unittest.TestCase):
    def test_complete_host_source_gates_and_all_synthetic_prefixes(self):
        # New producer/host imports are confined to this integration test. All
        # expected source/math/choices above are independently reconstructed.
        import monthly_risk_clock263_kernel as kernel
        import monthly_risk_clock263_host as host
        import test_monthly_risk_clock263_kernel as fixture_module
        study,arrays,meta,panel,calendar,source=fixture_module.fixture();n=len(meta['dates']);width=arrays['eligible'].shape[1]
        b=ind.core.original_base(arrays,meta,n);original_scores,original_choices=ind.core.episode(b['P'],b)
        expected=ind.orders_from_choices({a:original_scores for a in ind.ARMS},{a:original_choices for a in ind.ARMS},meta)[ind.ARMS[0]]
        full,certificate=host.prepare_one(study,arrays,meta,panel,calendar,source,n,expected,width=width)
        features,orders,scores,pres,audit=full
        self.assertEqual(certificate['counts'],audit['counts']);self.assertEqual(certificate['original_source']['u0_indices'],[f['u0_indices'] for f in features])
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
            self.assertEqual(physical_proof['full']['cached_slot_validations'],0)
            self.assertEqual(cached_proof['selection']['original_source_price_observations'],0)
            self.assertEqual(cached_proof['original_source_certificate_sha256'],certificate['original_source_anchor']['certificate_sha256'])
            projection=host.original_prefix_certificate(certificate['original_source'],certificate['original_source_anchor'],count)
            self.assertEqual(cached_proof['consumed_original_source_sha256'],host.digest(projection))
            self.assertEqual(physical_proof['original_source_anchor']['checked_events'],count)
        with patch.object(kernel,'measure',side_effect=AssertionError('Incumbent failure must precede statistics')):
            wrong=copy.deepcopy(expected);wrong[0]['symbols']=['wrong']
            with self.assertRaises(ValueError):host.prepare_one(study,arrays,meta,panel,calendar,source,n,wrong,width=width)
        class Poison(dict):
            def __deepcopy__(self,memo):raise AssertionError('Future feature body copied')
            def __eq__(self,other):raise AssertionError('Future feature body compared')
        poisoned=copy.deepcopy(features);poisoned[-1]=Poison()
        for count in (0,n-1):
            aa,mm,pp,cc=host.physical_inputs(arrays,meta,panel,calendar,count)
            host.decisions(study,aa,mm,poisoned,count,expected[:count],certificate['gates']['source'],original_certificate=certificate['original_source'],original_anchor=certificate['original_source_anchor'],width=width)

        class CertificatePoison(dict):
            def __iter__(self):raise AssertionError('Future certificate body iterated')
            def items(self):raise AssertionError('Future certificate body serialized')
            def __deepcopy__(self,memo):raise AssertionError('Future certificate body copied')
        anchor=certificate['original_source_anchor']
        for count in (0,n-1):
            aa,mm,pp,cc=host.physical_inputs(arrays,meta,panel,calendar,count)
            bad_future=copy.deepcopy(certificate['original_source'])
            for key in ('original_events','u0_indices'):
                bad_future[key][count:]=[CertificatePoison(blocked=object()) for _ in range(n-count)]
            with self.assertRaises((TypeError,AssertionError)):host.digest(bad_future)
            got,proof=host.decisions(study,aa,mm,poisoned,count,expected[:count],certificate['gates']['source'],original_certificate=bad_future,original_anchor=anchor,width=width)
            ind.record_check(got[0],{a:orders[a][:count] for a in ind.ARMS},'Future certificate must not affect orders')
            self.assertEqual(proof['original_source_certificate_sha256'],anchor['certificate_sha256'])
            clean_prefix=host.original_prefix_certificate(certificate['original_source'],anchor,count)
            self.assertEqual(proof['consumed_original_source_sha256'],host.digest(clean_prefix))
        count=n-1
        aa,mm,pp,cc=host.physical_inputs(arrays,meta,panel,calendar,count)
        with patch.object(kernel,'decisions',side_effect=AssertionError('Consumed certificate rejection must precede producer decisions')):
            for mode in ('event','u0','missing'):
                tampered=copy.deepcopy(certificate['original_source'])
                if mode=='event':tampered['original_events'][count-1]['date']='1999-01-01'
                elif mode=='u0':tampered['u0_indices'][count-1]=[width]
                else:tampered['original_events']=[]
                with self.assertRaises(ValueError):host.decisions(study,aa,mm,features,count,expected[:count],certificate['gates']['source'],original_certificate=tampered,original_anchor=anchor,width=width)
            with self.assertRaises(ValueError):host.decisions(study,aa,mm,features,count,expected[:count],certificate['gates']['source'],original_certificate=certificate['original_source'],original_anchor=dict(anchor),width=width)

if __name__=='__main__':unittest.main(verbosity=2)
