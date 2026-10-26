"""262 invented source/risk/selection proofs. Producer imported only in bilateral test."""
import ast
import copy
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_tracking_risk262_independent as ind

OLD_TEST=ind.ROOT/'work/test_monthly_calendar_clock248_v2_independent.py'
OLD_TEST_SHA='1213e1bad7836293547a76c9d27c60c6392b573c7eeeabd45ff60225c4885b84'
base=ind.load_pinned(OLD_TEST,OLD_TEST_SHA,'tracking262_fake_original')

def review(meta,calendar):
    rows=[];clocks=[];ids=[]
    for q,(d,ss) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=calendar.index(d);stock=[j for j,s in enumerate(ss) if s not in meta['etfs']]
        e=ind.event_clock(q,d,ss,stock,t,calendar);clocks.append(e)
        for j in stock:
            s=ss[j];b=ind.peer.BOUNDS.get(s);issuer=b['issuer_key'] if b else 'issuer-'+s
            sector=None if s=='SPCX' else 'XLY'
            r=dict(query_index=q,column_index=j,date=d,symbol=s,issuer_key=issuer,original_sector=sector,sector=sector,mapping_origin='query_date_sector_panel',mapping_repair='unresolved' if sector is None else 'unchanged',mapping_evidence=None);rows.append(r)
            ob,pb=ind.bounds_for(s,issuer,d,e['initial_partial'])
            if ob:
                ids.append(dict(query_index=q,column_index=j,date=d,symbol=s,issuer_key=issuer,original_history_bound=ob,path_history_bound=pb,path_bound_proof='262_early_actual_confirmation' if pb!=ob else 'original239_bound',structural_warmup=bool(e['numerator_dates'] and e['numerator_dates'][0]<ob['not_before']),required_dates_before_bound=[x for x in e['required_dates'] if x<pb['not_before']],required_dates_at_or_after_bound=[x for x in e['required_dates'] if x>=pb['not_before']]))
    return dict(schema=ind.SOURCE_SCHEMA,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,mapping_records=rows,clock_events=clocks,listing_identity_cells=ids,listing_bounds=copy.deepcopy(list(ind.peer.BOUNDS.values())),early_listing_policy=dict(sha256=ind.POLICY_SHA,policy=ind.listing_policy()),listing_policy252_v2=dict(policy=ind.old.listing_policy()),factor_listing_bounds=[dict(available_from=None,evidence_clock_certified=False,inception_date=d,symbol=s) for s,d in ind.FACTOR_BOUNDS.items()])

def fixture():
    p,m,c,a,_=base.fixture();names=sorted({s for row in m['symbols'] for s in row});m['etfs']=[names[-1]]
    axis=pd.bdate_range('2008-01-02',max(m['ends']));x=np.arange(len(axis),dtype=float)
    frame=pd.DataFrame({s:100.+(.018+j*.004)*x+1.5*np.sin(x/(6.+j)) for j,s in enumerate(names)},index=axis)
    frame['XLY']=80.+.023*x+.7*np.sin(x/17.)
    p={'adj':frame};c=axis.strftime('%Y-%m-%d').tolist();r=review(m,c)
    base.set_core((p,m,c,a,{}));return p,m,c,a,r

def physical(args,n):
    p,m,c,a,r=args;aa={k:np.asarray(v)[:n].copy() for k,v in a.items()};mm={k:copy.deepcopy(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    if n:t=c.index(mm['dates'][-1]);cc=c[:t+2];pp={'adj':p['adj'].loc[:mm['dates'][-1]].copy()}
    else:cc=[];pp={'adj':p['adj'].iloc[:0].copy()}
    return pp,mm,cc,aa,r

def warm_fixture():
    p,m,c,a,r=fixture();names=m['symbols'][0].copy();old=names[0];names[0]='SPCX'
    m['dates']=['2026-06-30','2026-07-31','2026-08-31'];m['ends']=['2026-07-31','2026-08-31','2026-09-30'];m['symbols']=[names.copy() for _ in range(3)]
    axis=pd.bdate_range('2024-01-01','2026-10-02');x=np.arange(len(axis),dtype=float)
    p={'adj':pd.DataFrame({s:100.+.1*x+.5*np.sin(x/(3+j)) for j,s in enumerate(names+['XLY'])},index=axis)}
    p['adj'].loc[:'2026-06-11','SPCX']=np.nan;c=axis.strftime('%Y-%m-%d').tolist();r=review(m,c)
    return p,m,c,a,r

def run(args,n=None):
    p,m,c,a,r=args;raw=ind.independent_features(p,m,c,a,r,n);f=ind.finalize(raw,a['eligible'].shape[1]);s,ch,pre,d=ind.selection_oracle(a,m,f,n)
    return f,ind.orders_from_choices(s,ch,m),s,pre,d

def verify(args,out,n=None):
    p,m,c,a,r=args;f,o,s,pre,d=out
    return ind.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n)

def prices():return [100.+.17*i+.8*math.sin(i/3.) for i in range(65)],[90.+.09*i+.4*math.cos(i/7.) for i in range(65)]

class MathTests(unittest.TestCase):
    def test_constant_nonzero_and_signed_zero(self):
        for value in (.1,-.25,-0.):
            r=ind.risk63([value]*63);self.assertTrue(r['constant']);self.assertEqual(r['mean'].hex(),value.hex())
            self.assertTrue(all(x.hex()=='0x0.0p+0' for x in r['centered']));self.assertEqual(r['sd'].hex(),'0x0.0p+0')
    def test_near_constant_is_not_floored(self):
        x=[.1]*63;x[-1]=math.nextafter(.1,math.inf);r=ind.risk63(x);self.assertFalse(r['constant']);self.assertGreater(r['sd'],0.)
    def test_square_and_divide_underflow(self):
        with self.assertRaises(ArithmeticError):ind.risk63([0.]*62+[1e-200])
        with self.assertRaises(ArithmeticError):ind.quotient(5e-324,62.)
    def test_invalid_source_and_cast_underflow(self):
        a,b=prices()
        for bad in (True,False,'3',None,0.,-1.,math.nan,math.inf,Fraction(1,10**1000)):
            z=a.copy();z[4]=bad
            with self.subTest(bad=type(bad).__name__),self.assertRaises((ValueError,ArithmeticError)):ind.measure(z,b)
    def test_gross_reconstruction_cancellation_is_error(self):
        a,b=prices();a[22]=1e-20
        with self.assertRaises(ValueError):ind.measure(a,b)
    def test_exact_relative_negative_and_zero(self):
        a,b=prices();a[0]=10000.;s=ind.measure(a,b);self.assertLess(s['relative_momentum'],0.);self.assertLess(s['tracking_ratio'],0.)
        a[22]=a[0];b[22]=b[0];s=ind.measure(a,b);self.assertTrue(s['supported']);self.assertEqual(s['total_ratio'],0.)
    def test_zero_either_denominator_shared_unavailable(self):
        a,b=prices()
        for x,y in ((a,a),([100.]*65,b)):
            s=ind.measure(x,y);self.assertFalse(s['supported']);self.assertIsNone(s['total_ratio']);self.assertIsNone(s['tracking_ratio'])
    def test_flat_sector_same_risk_and_exact_power2_scale(self):
        a,_=prices();s=ind.measure(a,[80.]*65);ind.record_check(s['total_risk'],s['tracking_risk'])
        ind.record_check(s,ind.measure([v*4. for v in a],[40.]*65))
    def test_ranks_n0_n1_and_old_n6_operation(self):
        self.assertEqual(ind.ranks([]),[]);self.assertEqual(ind.ranks([0.]),[.5])
        x=[0.,1.,1.,3.,4.,5.];r=ind.ranks(x)
        self.assertEqual(r,ind.peer.average_percentile(x))
        shortcut=[float(2*sum(y<v for y in x)+sum(y==v for y in x)+1-len(x))/(2*len(x)) for v in x]
        self.assertTrue(any(a.hex()!=b.hex() for a,b in zip(r,shortcut)))
    def test_recent_price_change_does_change_denominator(self):
        a,b=prices();old=ind.measure(a,b);a[-1]*=1.01;new=ind.measure(a,b)
        self.assertEqual(old['relative_momentum'],new['relative_momentum']);self.assertNotEqual(old['total_ratio'],new['total_ratio'])

class SourceAndSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.out=run(cls.args)
    def test_full_source_receipt_and_original_core(self):
        p,m,c,a,r=self.args;proof=verify(self.args,self.out);self.assertTrue(proof['passed']);self.assertEqual(proof['original_episode_calls'],3)
        self.assertEqual(proof['statistic_evaluations'],2*proof['counts']['mature_cells']);self.assertEqual(proof['cached_slot_validations'],0)
        source=ind.verify_original_source(p,m,c,a);self.assertEqual(source['episode_calls'],0);self.assertTrue(source['passed'])
    def test_raw_source_gate_and_work_counters(self):
        p,m,c,a,r=self.args;raw=ind.unranked_prefix(self.out[0],len(m['dates']))
        with patch.object(ind,'ranks',side_effect=AssertionError('Early rank')):
            x=ind.verify_features(p,m,c,a,r,raw);y=ind.verify_features_cached(a,m,raw)
        self.assertEqual(x['full_source_slot_inspections'],y['cached_slot_validations']);self.assertEqual(x['rank_evaluations'],0)
        with self.assertRaises(ValueError):ind.verify_features_cached(a,m,self.out[0])
    def test_logical_physical_and_fullbundle_smallarray(self):
        for n in range(4):
            got=run(self.args,n);part=physical(self.args,n);other=run(part,n);verify(part,other,n)
            ind.record_check(got[0],other[0]);ind.record_check(got[1],other[1])
            s,ch,pre,d=ind.selection_oracle(part[3],part[1],self.out[0],n)
            ind.verify_selection(part[3],part[1],self.out[0],ind.orders_from_choices(s,ch,part[1]),scores=s,pre_scores=pre,audit=d,n=n)
    def test_source_before_E_zero_or_mapping(self):
        p,m,c,a,r=copy.deepcopy(self.args);a['eligible'][:]=False;p['adj'][m['symbols'][0][0]]=100.
        day=r['clock_events'][1]['risk_dates'][5];p['adj'].at[pd.Timestamp(day),m['symbols'][0][1]]=math.nan
        with self.assertRaises(ValueError):ind.independent_features(p,m,c,a,r,2)
    def test_missing_required_row_is_not_clock_compression(self):
        p,m,c,a,r=copy.deepcopy(self.args);p['adj'].drop(pd.Timestamp(r['clock_events'][1]['risk_dates'][5]),inplace=True)
        with self.assertRaises(ValueError):ind.independent_features(p,m,c,a,r,2)
    def test_nonrequired_poison_and_partial_unconsumed(self):
        p,m,c,a,r=copy.deepcopy(self.args);p['adj']=p['adj'].astype(object);t=c.index(m['dates'][1]);p['adj'].at[pd.Timestamp(c[t-200]),m['symbols'][0][0]]='POISON'
        ind.independent_features(p,m,c,a,r,2);p['adj'].iloc[:,:]='POISON';ind.independent_features(p,m,c,a,r,1)
    def test_future_feature_and_metadata_bodies_unconsumed(self):
        class Poison(dict):
            def __deepcopy__(self,memo):raise AssertionError('Future copy')
            def __eq__(self,other):raise AssertionError('Future comparison')
        p,m,c,a,r=copy.deepcopy(self.args);f=copy.deepcopy(self.out[0])
        for i in range(1,3):f[i]=Poison();m['symbols'][i]=Poison();r['clock_events'][i]=Poison(query_index=i)
        r['mapping_records']=[x if x['query_index']==0 else Poison(query_index=x['query_index']) for x in r['mapping_records']]
        ind.independent_features(p,m,c,a,r,1);ind.cached_features(a,m,f,1)
    def test_warmup_early_evidence_scope(self):
        args=warm_fixture();f=run(args)[0];cell=f[0]['cells'][0]
        self.assertEqual(cell['history_status'],'official_pre_inception');self.assertEqual(cell['history_bound']['not_before'],'2026-06-08');self.assertEqual(cell['path_history_bound']['not_before'],'2026-06-12');self.assertEqual(len(cell['own_observations']),65)
        issuer=ind.peer.BOUNDS['SPCX']['issuer_key'];self.assertEqual(ind.bounds_for('SPCX',issuer,'2026-06-15')[1]['not_before'],'2026-06-08')
        self.assertEqual(ind.bounds_for('SPCX',issuer,'2026-06-16')[1]['not_before'],'2026-06-12')
    def test_warmup_positive_prebound_and_missing_postbound(self):
        for day,value in (('2026-06-10',1.),('2026-06-12',math.nan),('2026-06-10',-1.),('2026-06-10',True)):
            args=warm_fixture();args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].at[pd.Timestamp(day),'SPCX']=value
            with self.subTest(day=day,value=value),self.assertRaises(ValueError):ind.independent_features(*args[:3],args[3],args[4],1)
    def test_known_sector_warmup_scan_and_wrong_issuer(self):
        p,m,c,a,r=warm_fixture();r['mapping_records'][0].update(original_sector='XLY',sector='XLY',mapping_repair='unchanged');d=r['clock_events'][0]['risk_dates'][2];p['adj'].at[pd.Timestamp(d),'XLY']=math.nan
        with self.assertRaises(ValueError):ind.independent_features(p,m,c,a,r,1)
        p,m,c,a,r=warm_fixture();r['mapping_records'][0]['issuer_key']='old-fund'
        with self.assertRaises(ValueError):ind.independent_features(p,m,c,a,r,1)
    def test_metadata_clock_identity_and_native_type(self):
        for mutation in ('clock','identity','issuer'):
            p,m,c,a,r=copy.deepcopy(self.args)
            if mutation=='clock':r['clock_events'][1]['query_position']+=1
            elif mutation=='identity':r['mapping_records'][0]['symbol']='WRONG'
            else:r['mapping_records'][1]['issuer_key']=r['mapping_records'][0]['issuer_key']
            with self.assertRaises(ValueError):ind.independent_features(p,m,c,a,r)
        f=copy.deepcopy(self.out[0]);f[0]['query_index']=False
        with self.assertRaises(ValueError):ind.cached_features(self.args[3],self.args[1],f)
    def test_ulp_stats_rank_and_unavailable_tamper(self):
        for mode in ('stat','rank','support'):
            f=copy.deepcopy(self.out[0])
            if mode=='stat':s=f[1]['cells'][0]['statistics'];s['total_risk']['sd']=math.nextafter(s['total_risk']['sd'],math.inf)
            elif mode=='rank':f[1]['total_target'][0]=math.nextafter(f[1]['total_target'][0],math.inf)
            else:f[1]['supported_indices']=f[1]['supported_indices'][1:]
            with self.subTest(mode=mode),self.assertRaises(ValueError):ind.selection_oracle(self.args[3],self.args[1],f)
    def test_complete_audit_and_gate_payload(self):
        p,m,c,a,r=self.args;f,o,s,pre,d=copy.deepcopy(self.out)
        d['incumbent_gate']=ind.verify_incumbent(a,m,o[ind.ARMS[0]],s[ind.ARMS[0]],pre[ind.ARMS[0]])
        d['source_gate']=ind.verify_features(p,m,c,a,r,ind.unranked_prefix(f,3));verify(self.args,(f,o,s,pre,d))
        d['source_gate']['full_source_slot_inspections']-=1
        with self.assertRaises(ValueError):verify(self.args,(f,o,s,pre,d))
    def test_original_P_F_order_and_no_E_rerank(self):
        p,m,c,a,r=copy.deepcopy(self.args);a['eligible'][1:,0]=False;out=run((p,m,c,a,r))
        self.assertEqual(out[0][1]['total_target'],self.out[0][1]['total_target'])
        for field in (2,3):
            bad=list(copy.deepcopy(out));bad[field][ind.ARMS[2]][1,0]+=1e-8
            with self.assertRaises(ValueError):verify((p,m,c,a,r),bad)
    def test_actual_function_calls_match_each_receipt(self):
        p,m,c,a,r=self.args;f,o,sc,pre,d=self.out;raw=ind.unranked_prefix(f,3)
        jobs=[lambda:ind.verify_features(p,m,c,a,r,raw),lambda:ind.verify_features_cached(a,m,raw),lambda:ind.verify_selection(a,m,f,o,scores=sc,pre_scores=pre,audit=d),lambda:verify(self.args,self.out)]
        for job in jobs:
            with patch.object(ind,'risk63',wraps=ind.risk63) as risk,patch.object(ind,'ranks',wraps=ind.ranks) as rank,patch.object(ind.core,'episode',wraps=ind.core.episode) as ep:
                receipt=job()
                self.assertEqual(risk.call_count,receipt['statistic_evaluations'])
                self.assertEqual(rank.call_count,receipt['rank_evaluations'])
                self.assertEqual(ep.call_count,receipt['original_episode_calls'])
    def test_no_new_producer_or_host_import(self):
        tree=ast.parse(Path(ind.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                text=ast.dump(node);self.assertNotIn('tracking_risk262_kernel',text);self.assertNotIn('tracking_risk262_host',text)


class BilateralCross(unittest.TestCase):
    def test_producer_and_host_full_cached_physical(self):
        # This is integration only: expected source/math/selection above do not
        # import or call the new producer. Producer fixture supplies fake data.
        import monthly_tracking_risk262_kernel as k
        import test_monthly_tracking_risk262_kernel as kt
        import monthly_tracking_risk262_host as h
        s,a,m,p,c,_=kt.fixture();r=review(m,c);args=(s,a,m,p,c,r)
        original=ind.core.original_base(a,m,len(m['dates']));f,ch=ind.core.episode(original['P'],original)
        expected=ind.orders_from_choices({x:f for x in ind.ARMS},{x:ch for x in ind.ARMS},m)[ind.ARMS[0]]
        # Host original-source gate requires unchanged original fullhistory.
        # Extend the invented history and recompute its frozen-control source
        # cache in the fixture only, preserving all other synthetic inputs.
        axis=pd.bdate_range('2008-01-02',max(c));x=np.arange(len(axis),dtype=float)
        p={'adj':pd.DataFrame({z:100.+(.018+j*.003)*x+.8*np.sin(x/(5.+j)) for j,z in enumerate(list(p['adj'].columns))},index=axis)}
        c=axis.strftime('%Y-%m-%d').tolist();r=review(m,c);base.set_core((p,m,c,a,{}))
        original=ind.core.original_base(a,m,len(m['dates']));f,ch=ind.core.episode(original['P'],original)
        expected=ind.orders_from_choices({x:f for x in ind.ARMS},{x:ch for x in ind.ARMS},m)[ind.ARMS[0]]
        out,item=h.prepare_one(s,a,m,p,c,r,len(m['dates']),expected,width=a['eligible'].shape[1])
        features,orders,scores,pres,audit=out
        fullproof=ind.verify_prepared(p,m,c,a,r,features,orders,scores=scores,pre_scores=pres,audit=audit)
        h.full_check(fullproof,len(m['dates']),ind,width=a['eligible'].shape[1])
        for n in range(len(m['dates'])+1):
            aa,mm,pp,cc=h.physical_inputs(a,m,p,c,n)
            po,pi=h.prepare_one(s,aa,mm,pp,cc,r,n,expected[:n],width=a['eligible'].shape[1])
            got,proof=h.decisions(s,aa,mm,features,n,expected[:n],item['gates']['source'],width=a['eligible'].shape[1])
            ind.record_check(po[0],features[:n],'Physical feature parity');ind.record_check(po[1],got[0],'Cached ordered choices')
            for arm in ind.ARMS:ind.exact_array(po[2][arm],got[1][arm],'Cached final score');ind.exact_array(po[3][arm],got[2][arm],'Cached pre')
        for aa,bb in (prices(),([100.]*65,prices()[1])):
            ind.record_check(k.measure(aa,bb),ind.measure(aa,bb),'Bilateral mathematical evidence')
        s,a,m,p,c,_=kt.warm_fixture();r=review(m,c);out=k.prepare(s,a,m,p,c,r)
        ind.verify_prepared(p,m,c,a,r,out[0],out[1],scores=out[2],pre_scores=out[3],audit=out[4])

if __name__=='__main__':unittest.main(verbosity=2)
