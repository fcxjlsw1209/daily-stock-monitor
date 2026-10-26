"""Synthetic254 source, common-maturity, estimator and exact-selector checks."""
import ast
import copy
from functools import lru_cache
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_calendar_memory254_independent as audit

@lru_cache(None)
def helper():
    return audit.bridge.prior.load_pinned(audit.bridge.PRIOR_TEST,audit.bridge.PRIOR_TEST_SHA,'calendar254_frozen_fixture')

@lru_cache(None)
def _fixture():
    _,m,_,a,_=helper().fixture()
    names=m['symbols'][0];end=pd.Timestamp(m['dates'][-1])+pd.offsets.MonthEnd(3)
    idx=pd.bdate_range('2013-01-02',end);t=np.arange(len(idx),dtype=float)
    needed=sorted(set(s for ss in m['symbols'] for s in ss)|{'SPY','QQQ'})
    adj=pd.DataFrame({s:50*np.exp((.0005+j*.000004)*t+.025*np.sin(.14*t+j*.1)) for j,s in enumerate(needed)},index=idx)
    # Low close-location makes every good month-end a signal2 or3 anchor.
    p=dict(adj=adj,close=adj*.985,high=adj*1.02,low=adj*.98,volume=pd.DataFrame(1000.+(t[:,None]%31)+np.arange(len(needed)),index=idx,columns=needed),open=adj*.995)
    return p,idx.strftime('%Y-%m-%d').tolist(),m,a

def fixture():return copy.deepcopy(_fixture())
@lru_cache(None)
def complete():
    p,c,m,a=_fixture();d,sa=audit.independent_source(p,c,m)
    return d,sa

def run(a,m,d,n=None):
    s,ch,p,ef,au=audit.selection_oracle(a,m,d,n)
    return audit.orders_from_choices(s,ch,m),s,p,ef,au

def verify(a,m,d,out,n=None):
    o,s,p,e,au=out
    return audit.verify_selection(a,m,d,o,scores=s,pre_scores=p,effects=e,audit=au,n=n)

def physical(args,n):
    p,c,m,a=copy.deepcopy(args);a={k:v[:n].copy() for k,v in a.items()}
    m={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    p={k:(v.loc[:m['dates'][-1]].copy() if n else v.iloc[:0].copy()) for k,v in p.items()}
    return p,c,m,a

def memory_fixture():
    d=audit._empty_data(1);count=8
    d['event_symbol']=np.array(['A']*count,dtype='U64')
    d['event_dates']=np.array(['2019-01-31','2019-02-28','2019-03-29','2019-04-30','2019-05-31','2019-06-28','2019-07-31','2019-08-30'],dtype='datetime64[D]')
    d['event_common_maturity']=d['event_dates']+np.timedelta64(35,'D')
    d['event_signal']=np.full(count,2,np.int64);d['event_common_valid']=np.ones(count,bool)
    d['event_utility_calendar']=np.linspace(.1,.2,count);d['event_utility21']=np.linspace(.15,.25,count)
    return d

class SourceTests(unittest.TestCase):
    def test_schema_counts_and_complete_independent_gate(self):
        p,c,m,a=fixture();d,sa=copy.deepcopy(complete())
        proof=audit.verify_source(p,c,m,d,sa)
        self.assertEqual(tuple(d),audit.DATA_KEYS);self.assertEqual(tuple(sa['counts']),audit.SOURCE_COUNT_FIELDS)
        self.assertTrue(all(proof[k] for k in audit.SOURCE_FLAGS));self.assertEqual(proof['episode_calls'],0)
        self.assertFalse(proof['memory_effects_computed']);self.assertEqual(d['query_symbols'].shape,(len(m['dates']),49))
        self.assertGreater(len(d['event_dates']),49)

    def test_monthends_no_lockout_and_exit_clocks(self):
        p,c,m,a=fixture();d,_=complete();cal=pd.DatetimeIndex(c)
        for i,day in enumerate(d['event_dates']):
            pos=int(d['event_anchor_index'][i]);self.assertEqual(np.datetime64(cal[pos],'D'),day)
            self.assertNotEqual(cal[pos].month,cal[pos+1].month)
            self.assertEqual(d['event_entry'][i],np.datetime64(cal[pos+1],'D'))
            self.assertEqual(d['event_exit21'][i],np.datetime64(cal[pos+21],'D'))
            month=cal[pos].to_period('M')+1
            self.assertEqual(d['event_exit_calendar'][i],np.datetime64(cal[cal.to_period('M')==month][-1],'D'))
        self.assertTrue((np.diff(d['event_anchor_index'][d['event_symbol']==d['event_symbol'][0]])<21).any())

    def test_missing_future_does_not_remove_anchor_or_truncate_horizon(self):
        args=physical(fixture(),2);p,c,m,a=args;d,sa=audit.independent_source(p,c,m)
        latest=d['event_dates']==np.datetime64(m['dates'][-1]);self.assertTrue(latest.any())
        self.assertFalse(d['event_common_valid'][latest].any());self.assertTrue(np.isnan(d['event_utility_calendar'][latest]).all())
        self.assertTrue((d['event_exit_calendar'][latest]>np.datetime64(m['dates'][-1])).all())

    def test_path_adjusted_open_excess_and_adverse_close(self):
        dates=pd.bdate_range('2020-01-02',periods=4);ss=['A','SPY']
        px=pd.DataFrame([[10,10],[9,11],[12,12],[11,10]],index=dates,columns=ss)
        p={k:px.copy() for k in audit.FRAME_KEYS};p['open'].iloc[0]=[8,10]
        x,u=audit.path_label(p,list(dates),'A',dates[0],dates[-1])
        self.assertEqual(x,11/8-1);self.assertEqual(u,x)
        p['open'].iloc[0]=[10,10];x,u=audit.path_label(p,list(dates),'A',dates[0],dates[-1])
        self.assertEqual(x,11/10-1);self.assertEqual(u,x-.5*(1-9/10))

    def test_missing_official_path_row_is_invalid_not_compressed(self):
        p,c,m,a=fixture();d,_=complete();i=0;s=str(d['event_symbol'][i]);entry=pd.Timestamp(d['event_entry'][i]);end=pd.Timestamp(d['event_exit_calendar'][i])
        dates=[pd.Timestamp(v) for v in c];omit=dates[dates.index(entry)+3]
        p={k:v.drop(omit) for k,v in p.items()};x,u=audit.path_label(p,dates,s,entry,end)
        self.assertTrue(np.isnan(x) and np.isnan(u))

    def test_missing_internal_row_equals_explicit_official_nan_row(self):
        p,c,m,a=fixture();day=p['adj'].index[400]
        missing={k:v.drop(day) for k,v in p.items()};explicit=copy.deepcopy(p)
        for f in explicit.values():f.loc[day]=np.nan
        left,la=audit.independent_source(missing,c,m);right,ra=audit.independent_source(explicit,c,m)
        for k in left:audit.exact_array(left[k],right[k],k)
        self.assertEqual(la,ra)

    def test_separate_validity_when_only_longer_exit_missing(self):
        p,c,m,a=fixture();d,_=complete();ids=np.flatnonzero(d['event_exit_calendar']!=d['event_exit21'])
        i=next(int(j) for j in ids if d['event_common_valid'][j]);s=str(d['event_symbol'][i]);late=pd.Timestamp(d['event_common_maturity'][i]);p['adj'].at[late,s]=np.nan
        sessions=list(pd.DatetimeIndex(c));entry=pd.Timestamp(d['event_entry'][i])
        v=[audit.path_label(p,sessions,s,entry,pd.Timestamp(d[k][i])) for k in ('event_exit21','event_exit_calendar')]
        self.assertEqual(sum(np.isfinite(x[0]) for x in v),1)

    def test_full_source_future_values_and_future_meta_unconsumed(self):
        p,c,m,a=fixture();d,sa=audit.independent_source(p,c,m,2)
        for k in p:p[k]=p[k].astype(object);p[k].loc[p[k].index>m['dates'][1]]='future'
        class Poison:
            def __iter__(self):raise AssertionError('future meta')
        for i in range(2,len(m['dates'])):m['dates'][i]=Poison();m['symbols'][i]=Poison()
        other,oa=audit.independent_source(p,c,m,2)
        for k in d:audit.exact_array(other[k],d[k],k)
        self.assertEqual(oa,sa)

    def test_axes_nonreal_query_and_incomplete_calendar_fail(self):
        args=fixture()
        for kind in ('axis','nonreal','query','calendar'):
            p,c,m,a=copy.deepcopy(args)
            if kind=='axis':p['volume']=p['volume'].iloc[1:]
            elif kind=='nonreal':p['close']=p['close'].astype(object);p['close'].iloc[0,0]=True
            elif kind=='query':p={k:v.drop(pd.Timestamp(m['dates'][1])) for k,v in p.items()}
            else:c=c[:c.index(m['dates'][-1])+2]
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.independent_source(p,c,m)

    def test_source_tamper_and_no_episode_before_source(self):
        p,c,m,a=fixture();d,sa=copy.deepcopy(complete())
        with patch.object(audit,'selection_oracle',side_effect=AssertionError('selection')):
            audit.verify_source(p,c,m,d,sa)
        for kind in ('query','exit','valid','count'):
            dd,aa=copy.deepcopy((d,sa))
            if kind=='query':dd['query_vol'][1,0]+=1.
            elif kind=='exit':dd['event_common_maturity'][0]+=np.timedelta64(1,'D')
            elif kind=='valid':dd['event_valid21'][0]=not dd['event_valid21'][0]
            else:aa['counts']['anchors']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.verify_source(p,c,m,dd,aa)

    def test_zero_prefix_physical_and_logical(self):
        p,c,m,a=fixture();d,sa=audit.independent_source(p,c,m,0);pp,cc,mm,aa=physical(fixture(),0)
        e,ea=audit.independent_source(pp,[],mm)
        self.assertEqual(sa,ea)
        for k in d:audit.exact_array(d[k],e[k],k)
        out=run(a,m,d,0);verify(a,m,d,out,0);self.assertEqual(out[4]['counts']['candidate_episode_calls'],1)

class EstimateTests(unittest.TestCase):
    def test_constant_labels_analytic_shrinkage(self):
        d=memory_fixture();d['event_utility_calendar'][:]=.125;day=np.datetime64('2020-01-31')
        effect,info=audit.estimate(d,'A',day,2,True,.02,'_calendar')
        age=(day-d['event_dates']).astype('timedelta64[D]').astype(float)/30.4375;weights=2**(-age/24.);mass=float(weights.sum())
        expected=.5*np.tanh((.125*mass/(mass+6))/max(.02*np.sqrt(21),.01))
        self.assertAlmostEqual(effect,expected,15);self.assertEqual(info['observations'],8);self.assertGreaterEqual(info['effective_samples'],3)

    def test_future_unmatured_wrong_security_and_lookback_ignored(self):
        d=memory_fixture();day=np.datetime64('2020-01-31');expected=audit.estimate(d,'A',day,2,True,.02,'_calendar')
        d['event_utility_calendar']=d['event_utility_calendar'].astype(object)
        class Poison:
            def __float__(self):raise AssertionError('unmature label consumed')
        # Duplicated future row has valid source identity but is not mature.
        for k in tuple(d):
            if k.startswith('event_') and len(d[k])==8:d[k]=np.concatenate([d[k],d[k][-1:]])
        d['event_dates'][-1]=np.datetime64('2020-01-31');d['event_common_maturity'][-1]=np.datetime64('2020-02-28');d['event_utility_calendar'][-1]=Poison()
        self.assertEqual(audit.estimate(d,'A',day,2,True,.02,'_calendar'),expected)
        self.assertEqual(audit.estimate(d,'B',day,2,True,.02,'_calendar')[0],0.)
        self.assertEqual(audit.estimate(d,'A','2030-01-31',2,False,.02,'_calendar')[0],0.)

    def test_exact_signal_no_evidence_neff_and_common_mask(self):
        d=memory_fixture()
        for signal,eligible in ((0,True),(2,False),(1,True)):
            value,detail=audit.estimate(d,'A','2020-01-31',signal,eligible,.02,'21');self.assertEqual(value,0.)
        d['event_common_valid'][2:]=False
        value,detail=audit.estimate(d,'A','2020-01-31',2,True,.02,'21');self.assertEqual(value,0.);self.assertLess(detail['effective_samples'],3)

    def test_boundary_maturity_and_self_anchor_exclusion(self):
        d=memory_fixture();day=np.datetime64('2020-01-31');d['event_common_maturity'][-1]=day
        _,info=audit.estimate(d,'A',day,2,True,.02,'21');self.assertEqual(info['observations'],8)
        d['event_dates'][-1]=day
        _,info=audit.estimate(d,'A',day,2,True,.02,'21');self.assertEqual(info['observations'],7)

    def test_sign_symmetry_and_shared_sqrt21(self):
        d=memory_fixture();d['event_utility_calendar']=d['event_utility21'].copy()
        a=audit.estimate(d,'A','2020-01-31',2,True,.02,'_calendar')[0];b=audit.estimate(d,'A','2020-01-31',2,True,.02,'21')[0];self.assertEqual(a,b)
        d['event_utility_calendar']*=-1
        self.assertEqual(audit.estimate(d,'A','2020-01-31',2,True,.02,'_calendar')[0],-a)

class SelectionTests(unittest.TestCase):
    def test_full_effects_scores_pre_orders_audit_original_inputs(self):
        p,c,m,a=fixture();d,_=copy.deepcopy(complete());before={k:v.tobytes() for k,v in a.items()}
        out=run(a,m,d);proof=verify(a,m,d,out)
        self.assertTrue(all(proof[k] for k in audit.SELECTION_FLAGS));self.assertEqual(proof['ordered_choices_checked'],3*len(m['dates']))
        self.assertEqual(tuple(out[4]['counts']),audit.COUNT_FIELDS)
        for key in a:self.assertEqual(a[key].tobytes(),before[key])
        for arm in audit.ARMS[1:]:self.assertEqual(out[2][arm].tobytes(),out[2][audit.ARMS[0]].tobytes())

    def test_partial_exact_all_effects_and_full_scores(self):
        p,c,m,a=fixture();d,_=complete();out=run(a,m,d)
        for arm in audit.ARMS[1:]:
            self.assertEqual(out[0][arm][0],out[0][audit.ARMS[0]][0])
            for artifact in (out[1],out[2],out[3]):self.assertEqual(artifact[arm][0].tobytes(),artifact[audit.ARMS[0]][0].tobytes())
        only=run(a,m,d,1)
        for key in ('query_signal_cells','query_finite_cells','calendar_evidence_cells','fixed21_evidence_cells'):
            self.assertEqual(only[4]['counts'][key],0)

    def test_incumbent_gate_does_not_estimate(self):
        p,c,m,a=fixture();d,_=complete();out=run(a,m,d);k=audit.ARMS[0]
        with patch.object(audit,'estimate',side_effect=AssertionError('estimate before incumbent')):
            proof=audit.verify_incumbent(a,m,out[0][k],out[1][k],out[2][k])
        self.assertFalse(proof['candidate_selection_computed']);self.assertEqual(proof['ordered_choices_checked'],len(m['dates']))

    def test_all_logical_physical_prefix_predictions_equal(self):
        p,c,m,a=fixture();d,_=complete()
        for n in range(len(m['dates'])+1):
            pp,cc,mm,aa=physical(fixture(),n);dd,_=audit.independent_source(pp,cc,mm)
            logical=run(a,m,d,n);physicalout=run(aa,mm,dd);self.assertEqual(logical[0],physicalout[0])
            for artifact in (1,2,3):
                for arm in audit.ARMS:audit.exact_array(logical[artifact][arm],physicalout[artifact][arm],'Prefix predictions')
            self.assertEqual(logical[4],physicalout[4]);verify(a,m,d,logical,n)

    def test_malformed_query_effect_and_order_audit_rejected(self):
        p,c,m,a=fixture();d,_=complete();out=run(a,m,d)
        for kind in ('effect','pre','score','order','count','boundary'):
            o,s,pr,e,au=copy.deepcopy(out);arm=audit.ARMS[1]
            if kind=='effect':e[arm][1,0]=np.nextafter(e[arm][1,0],np.inf)
            elif kind=='pre':pr[arm][1,0]+=1.
            elif kind=='score':s[arm][1,0]+=1.
            elif kind=='order':o[arm][1]['symbols'].reverse()
            elif kind=='count':au['counts']['anchors']+=1
            else:au['events'][1]['boundary_candidates']=[]
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.verify_selection(a,m,d,o,scores=s,pre_scores=pr,effects=e,audit=au)

    def test_query_missing_no_extra_good_eligibility_and_cash_two(self):
        p,c,m,a=fixture();d,_=copy.deepcopy(complete())
        d['query_shape'][1,0,0]=np.nan;out=run(a,m,d)
        self.assertEqual(out[3][audit.ARMS[1]][1,0],0.)
        for count in (0,1,2):
            aa=copy.deepcopy(a);aa['eligible'][1:]=False;aa['eligible'][1:,:count]=True;out=run(aa,m,d)
            for arm in audit.ARMS:self.assertTrue(all(len(o['symbols'])==(2 if count==2 else 0) for o in out[0][arm][1:]))

    def test_no_new_producer_import_and_pinned_dependencies(self):
        tree=ast.parse(audit.Path(audit.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any('254_kernel' in (s or '') or '254_host' in (s or '') for s in names))
        self.assertIn(audit.EP,audit.dependency_files())

class AccountingTests(unittest.TestCase):
    def test_nine_paths_fortyfive_slices_eighteen_groups_zero_exact(self):
        old=audit.bridge.prior.old;h=audit.bridge.prior.load_pinned(old.ACCOUNT_TEST,old.ACCOUNT_TEST_SHA,'calendar254_account_fixture')
        px,op,orders,exports,results,m=h.account_fixture();key=next(iter(orders))
        args=(px,op,{a:copy.deepcopy(orders[key]) for a in audit.ARMS},{a:copy.deepcopy(exports[key]) for a in audit.ARMS},{a:copy.deepcopy(results[key]) for a in audit.ARMS},m)
        for arm in audit.ARMS:
            for path,_,_ in audit.PATHS:args[4][arm][path]['slices']['full122']['fees_paid']=args[4][arm][path]['metrics']['fees_paid']
        r=audit.audit_outcomes(*args);self.assertEqual((r['checked_paths'],r['checked_slices'],r['checked_training_groups']),(9,45,18))
        args[4][audit.ARMS[1]]['same_close_zero']['metrics']['fees_paid']=1e-16
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)

class BilateralCross(unittest.TestCase):
    def test_full_source_effects_scores_orders_and_host_gates_all_prefixes(self):
        # This is the sole cross-implementation test; neither implementation
        # supplies arithmetic to the other independent implementation.
        import monthly_calendar_memory254_kernel as producer
        import test_monthly_calendar_memory254_kernel as fixtures
        import monthly_calendar_memory254_host as host
        self.assertEqual(audit.hashlib.sha256(audit.Path(producer.__file__).read_bytes()).hexdigest(),
          'bc13337fd877c8b56ea5f2a51cf44d27323ceb1a49eae97195f9d87bbf297338')
        self.assertEqual(audit.hashlib.sha256(audit.Path(fixtures.__file__).read_bytes()).hexdigest(),
          '6a3989f1920d92020a73ff789cd83ce98b38ab462c586e404665bc88d8f14468')
        study,aa,mm,pp,cc=fixtures.fixture()
        p,c,m,a=fixture()
        for arrays,meta,panel,calendar in ((aa,mm,pp,cc),(a,m,p,c)):
            full,sourceaudit=producer.build_features(panel,calendar,meta)
            expected,indsource=audit.independent_source(panel,calendar,meta)
            self.assertEqual(sourceaudit,indsource)
            for key in audit.DATA_KEYS:audit.exact_array(full[key],expected[key],'Bilateral full source/'+key)
            wanted=run(arrays,meta,expected)
            for n in range(len(meta['dates'])+1):
                ma={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in meta.items()}
                ar={k:v[:n].copy() for k,v in arrays.items()}
                pa={k:(v.loc[:ma['dates'][-1]].copy() if n else v.iloc[:0].copy()) for k,v in panel.items()}
                data,sa=producer.build_features(pa,calendar,ma,n=n)
                sg,ig,gates=host.gate_callbacks(audit,ar,ma,pa,calendar,data,sa,n)
                actual=producer.decisions(study,ar,ma,data,n=n,source_validator=sg,incumbent_validator=ig,
                  expected_incumbent_orders=wanted[0][audit.ARMS[0]][:n])
                host.pre_gate_check(gates,n,audit)
                receipt=verify(ar,ma,data,actual,n);host.receipt_check(receipt,n,audit.SELECTION_FLAGS,'selection')
                cached=producer.decisions(study,arrays,meta,full,n=n,
                  expected_incumbent_orders=wanted[0][audit.ARMS[0]][:n])
                verify(arrays,meta,full,cached,n)
                self.assertEqual(actual[0],cached[0]);self.assertEqual(actual[4],cached[4])
                for artifact in (1,2,3):
                    for arm in audit.ARMS:
                        audit.exact_array(actual[artifact][arm],cached[artifact][arm],'Physical/cache parity')
                        audit.exact_array(actual[artifact][arm],wanted[artifact][arm][:n],'Independent full prefix')
            # Labels outside common maturity can change without changing a
            # past forecast. Keep dtype fixed while making values adversarial.
            n=2;poison=copy.deepcopy(full);future=poison['event_common_maturity']>poison['query_dates'][n-1]
            for key in ('event_excess21','event_excess_calendar','event_utility21','event_utility_calendar'):
                poison[key][future]=np.inf
            changed=producer.decisions(study,arrays,meta,poison,n=n,
              expected_incumbent_orders=wanted[0][audit.ARMS[0]][:n])
            verify(arrays,meta,poison,changed,n)
            self.assertEqual(changed[0],{arm:wanted[0][arm][:n] for arm in audit.ARMS})

if __name__=='__main__':unittest.main(verbosity=2)
