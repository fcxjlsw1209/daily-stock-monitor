"""Synthetic49-member histories and scores only; no actual source or cache."""
import ast
import copy
from functools import lru_cache
import math
import pickle
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_volume_breadth249_v2_independent as audit

def helper():
    return audit.load_pinned(audit.PRIOR_TEST,audit.PRIOR_TEST_SHA,'volume249_old_fixture')

def review_for(panel,meta,calendar,arrays):
    base=audit.original_base(arrays,meta,len(meta['dates']))
    sessions=[pd.Timestamp(d) for d in calendar]
    return dict(schema='monthly_volume_breadth249_v2_metadata_source_review_v1',
      panel_first_session=str(panel['adj'].index[0].date()),listing_bounds=[],identity_cells=[],
      clock_events=[audit.clock_row(i,base,meta,sessions) for i in range(len(meta['dates']))],
      transition_records=[],undated_loader_rules=[],automatic_known_unresolved_transition_fallback_ids=[])

@lru_cache(None)
def _fixture():
    prior=helper();a,m=prior.helper().fixture();a=copy.deepcopy(a);m=copy.deepcopy(m)
    oldw=len(m['symbols'][0]);names=['S%02d'%j for j in range(49)];take=np.arange(49)%oldw
    for k,v in a.items():
        if v.ndim==2 and v.shape[1]==oldw:a[k]=v[:,take].copy()
    m['symbols']=[names.copy() for _ in m['dates']];m['etfs']=names[-17:]
    n=len(m['dates']);a['state_calm10']=np.array([i%2==1 for i in range(n)])
    a['core_weight']=np.where(a['state_calm10'],.5,.52)[:,None]
    idx=pd.bdate_range('2008-01-02',max(m['ends']));t=np.arange(len(idx),dtype=float)
    adj=pd.DataFrame({s:30*np.exp((.00005+j*.000003)*t+.08*np.sin((.013+j*.0002)*t+j)) for j,s in enumerate(names)},index=idx)
    vol=pd.DataFrame({s:10.+((np.arange(len(idx))*(j%7+1)+j)%97).astype(float) for j,s in enumerate(names)},index=idx)
    for i,day in enumerate(m['dates']):
        vals,_=audit.old.original_values(adj,pd.Timestamp(day),names,str(idx[0].date()))
        raw=[x[0 if a['state_calm10'][i] else 1] for x in vals]
        ranks,_=audit.old.independent_rank(raw);a['core'][i]=float(a['core_weight'][i,0])*ranks
    a['baseline']=a['core']+(1.-a['core_weight'])*np.linspace(-.18,.18,49)
    p=dict(adj=adj,volume=vol);c=idx.strftime('%Y-%m-%d').tolist();r=review_for(p,m,c,a)
    return p,m,c,a,r

def fixture():return copy.deepcopy(_fixture())

def physical(args,n):
    p,m,c,a,r=copy.deepcopy(args);a={k:v[:n].copy() for k,v in a.items()}
    m={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    if n:
        c=c[:c.index(m['dates'][-1])+2]
        p={k:v.loc[:m['dates'][-1]].copy() for k,v in p.items()}
    else:c=[];p={k:v.iloc[:0].copy() for k,v in p.items()}
    r['clock_events']=r['clock_events'][:n];r['identity_cells']=[v for v in r['identity_cells'] if v['query_index']<n]
    return p,m,c,a,r

def run(args,n=None):
    p,m,c,a,r=args;f=audit.independent_features(p,m,c,a,r,n,include_ranks=True)
    s,ch,pre,d=audit.selection_oracle(a,m,f,n)
    return f,audit.orders_from_choices(s,ch,m),s,pre,d

def verify(args,out,n=None):
    p,m,c,a,r=args;f,o,s,pre,d=out
    return audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n)

@lru_cache(None)
def full():return run(_fixture())

def bound(args,j,not_before,available_from):
    p,m,c,a,r=args;name=m['symbols'][0][j]
    b=dict(symbol=name,issuer_key='synthetic-'+name,not_before=not_before,
      available_from=available_from,kind='synthetic',evidence_id='synthetic-bound')
    r['listing_bounds']=[b];r['identity_cells']=[]
    for i,e in enumerate(r['clock_events']):
        t=c.index(e['date'])
        r['identity_cells'].append(dict(query_index=i,column_index=j,date=e['date'],symbol=name,
          issuer_key=b['issuer_key'],not_before=not_before,available_from=available_from,bound_visible=available_from<=e['date'],
          price_start=e['price_start'],price_end=e['price_end'],volume_start=e['volume_start'],volume_end=e['volume_end'],
          price_window_dates_before_bound=sum(d<not_before for d in c[t-294:t-41]),
          volume_window_dates_before_bound=sum(d<not_before for d in c[t-356:t-41])))

class SourceTests(unittest.TestCase):
    def test_full_exact_receipts_and_input_immutability(self):
        args=fixture();before=pickle.dumps(args);out=copy.deepcopy(full());r=verify(args,out)
        self.assertEqual(pickle.dumps(args),before);self.assertTrue(all(r[k] for k in audit.PREPARE_FLAGS))
        self.assertEqual(r['ordered_choices_checked'],3*len(args[1]['dates']))
        self.assertEqual((r['C_source_atol'],r['C_source_rtol']),(1e-12,0.))
        self.assertEqual(r['counts']['original_episode_calls'],3)

    def test_width49_and_zero_logical_physical_prefix(self):
        args=fixture()
        for cut,n in ((args,0),(physical(args,0),None)):
            out=run(cut,n);r=verify(cut,out,n)
            self.assertEqual(out[0],[]);self.assertEqual(r['counts']['symbols_per_event'],49)
            self.assertEqual(r['counts']['original_episode_calls'],3)
        args=list(fixture());args[3]['eligible']=args[3]['eligible'][:,:48]
        with self.assertRaises(ValueError):run(tuple(args),0)

    def test_source_gate_has_zero_episode_and_no_candidate_selection(self):
        args=fixture();p,m,c,a,r=args
        source=copy.deepcopy(full()[0])
        for e in source:e['breadth_rank']=None;e['mean_rank']=None
        with patch.object(audit,'funded_episode',side_effect=AssertionError('no decisions in source gate')), patch.object(audit.exactmath,'rank49_oracle',side_effect=AssertionError('no new target ranks before incumbent')):
            receipt=audit.verify_source_features(p,m,c,a,r,source)
        self.assertFalse(receipt['new_target_ranks_computed'])
        self.assertEqual(receipt['episode_calls'],0);self.assertFalse(receipt['candidate_selection_computed'])
        for key in audit.SOURCE_GATE_FLAGS:self.assertTrue(receipt[key])

    def test_source_gate_rejects_premature_new_target_rank(self):
        p,m,c,a,r=fixture()
        with self.assertRaises(ValueError):audit.verify_source_features(p,m,c,a,r,copy.deepcopy(full()[0]))

    def test_logical_future_poison_and_physical_prefixes(self):
        args=fixture();ref=copy.deepcopy(full())
        for n in range(len(args[1]['dates'])+1):
            cut=physical(args,n);out=run(cut);self.assertEqual(out[0],ref[0][:n]);verify(cut,out)
            audit.verify_selection(args[3],args[1],ref[0],{a:ref[1][a][:n] for a in audit.ARMS},
              scores={a:ref[2][a][:n] for a in audit.ARMS},pre_scores={a:ref[3][a][:n] for a in audit.ARMS},
              audit=audit.selection_oracle(args[3],args[1],ref[0],n)[3],n=n)
        p,m,c,a,r=fixture();n=2;day=m['dates'][n-1]
        p['volume']=p['volume'].astype(object);p['volume'].loc[p['volume'].index>day]='future poison'
        m['symbols'][2]=['broken future'];r['clock_events'][2]={'query_index':2,'future':'poison'}
        r['identity_cells'].append({'query_index':2,'future':'poison'})
        f=audit.independent_features(p,m,c,a,r,n,include_ranks=True);self.assertEqual(f,ref[0][:n])

    def test_ineligible_etf_source_missing_whole_event(self):
        args=physical(fixture(),2);p,m,c,a,r=args;a['eligible'][1,-1]=False
        p['volume'].at[pd.Timestamp(r['clock_events'][1]['volume_start']),m['symbols'][1][-1]]=np.nan
        out=run(args);e=out[0][1]
        self.assertEqual(e['branch'],'missing_source');self.assertEqual(e['missing_indices'],[48]);self.assertIsNone(e['breadth_rank'])
        for arm in audit.ARMS[1:]:
            self.assertEqual(out[1][arm],out[1][audit.ARMS[0]])
            self.assertEqual(out[2][arm].tobytes(),out[2][audit.ARMS[0]].tobytes())

    def test_missing_cannot_hide_other_invalid(self):
        args=physical(fixture(),2);p,m,c,a,r=args;day=pd.Timestamp(r['clock_events'][1]['volume_start'])
        p['volume'].at[day,m['symbols'][1][0]]=np.nan;p['volume'].at[day,m['symbols'][1][-1]]=-1.
        with self.assertRaisesRegex(ValueError,'positive finite'):run(args)

    def test_original_axis_fail_is_not_feature_fallback(self):
        args=physical(fixture(),2);p,m,c,a,r=args
        p['adj']=p['adj'].drop(pd.Timestamp(r['clock_events'][1]['price_start']))
        with self.assertRaisesRegex(ValueError,'last295'):run(args)
        args=physical(fixture(),2);p,m,c,a,r=args
        p['volume']=p['volume'].drop(pd.Timestamp(r['clock_events'][1]['volume_start']))
        out=run(args);self.assertEqual(out[0][1]['branch'],'missing_source')
        self.assertEqual(len(out[0][1]['missing_indices']),49)
        self.assertEqual(out[0][1]['cells'][0]['volume_gaps'][0]['reason'],'missing_row')

    def test_original_core_error_hard_and_fixed_tolerance(self):
        args=physical(fixture(),2);args[3]['core'][1,0]+=1e-8
        with self.assertRaisesRegex(ValueError,'core source'):run(args)
        args=physical(fixture(),2);args[3]['core'][1,0]+=5e-13
        out=run(args);verify(args,out)
        self.assertGreater(out[0][1]['cells'][0]['core_abs_error'],0.)

    def test_longer_volume_listing_gate_and_future_evidence(self):
        args=physical(fixture(),2);p,m,c,a,r=args
        first=r['clock_events'][1]['price_start'];bound(args,0,first,'2016-08-01')
        with self.assertRaisesRegex(ValueError,'visible listing'):run(args)
        bound(args,0,first,'2017-01-01')
        out=run(args);self.assertEqual(out[0][1]['branch'],'available')

    def test_partial_no_numeric_reads(self):
        args=physical(fixture(),1);p,m,c,a,r=args
        p['adj'][:]=np.inf;p['volume']=p['volume'].astype(object);p['volume'][:]='poison'
        out=run(args);verify(args,out);self.assertEqual(out[0][0]['cells'],[])
        for arm in audit.ARMS:self.assertEqual(out[2][arm].tobytes(),out[2][audit.ARMS[0]].tobytes())

    def test_transition_boundary_visibility_unknown_and_missing_precedence(self):
        args=physical(fixture(),2);p,m,c,a,r=args;e=r['clock_events'][1]
        row=dict(id='synthetic-unit',symbols=[m['symbols'][1][0]],effective_date=e['volume_start'],
          affected_fields=['volume'],available_from='2016-01-01',
          known_unresolved_fallback_eligible=True,unresolved_unit_transition=True)
        r['transition_records']=[row];r['automatic_known_unresolved_transition_fallback_ids']=[row['id']]
        r['undated_loader_rules']=[dict(id='unknown-unit',symbols=[m['symbols'][1][1]],effective_date=None)]
        records={row['id']:row};unknown={v['id']:v for v in r['undated_loader_rules']}
        self.assertEqual(audit.transition_ids(e,records,[row['id']],unknown),([],['unknown-unit']))
        row['effective_date']=c[c.index(e['volume_start'])+1]
        out=run(args);self.assertEqual(out[0][1]['branch'],'known_transition')
        self.assertEqual(out[0][1]['continuity_unknown_ids'],['unknown-unit'])
        self.assertTrue(all(x['source_available'] for x in out[0][1]['cells']))
        p['volume'].at[pd.Timestamp(e['volume_start']),m['symbols'][1][0]]=np.nan
        out=run(args);self.assertEqual(out[0][1]['branch'],'missing_source')
        self.assertEqual(out[0][1]['transition_ids'],['synthetic-unit'])

    def test_metadata_and_gap_corruption_rejected(self):
        args=physical(fixture(),2);args[4]['clock_events'][1]['volume_count']=314
        with self.assertRaises(ValueError):run(args)
        args=fixture();out=copy.deepcopy(full());out[0][1]['query_position']=True
        with self.assertRaises(ValueError):verify(args,out)

class SelectionTests(unittest.TestCase):
    def test_complete_internal_fraction_audit_and_rank_tampering(self):
        args=fixture()
        for edit in ('covariance','high_flags','rank'):
            out=copy.deepcopy(full())
            if edit=='covariance':out[0][1]['cells'][0]['measurement']['blocks'][0]['covariance']={'numerator':'0','denominator':'1'}
            elif edit=='high_flags':out[0][1]['cells'][0]['measurement']['high_flags'][0]=True
            else:out[0][1]['breadth_rank']['targets'][0]+=1e-10
            with self.assertRaises(ValueError):
                audit.verify_selection(args[3],args[1],out[0],out[1],scores=out[2],pre_scores=out[3],audit=out[4])

    def test_funding_pre_and_budget_exact_all_states(self):
        args=fixture();base=audit.original_base(args[3],args[1],len(args[1]['dates']))
        for calm,weak in ((False,False),(False,True),(True,False),(True,True)):
            b={k:copy.deepcopy(v) for k,v in base.items()}
            n=len(b['P']);b['calm'][:]=calm;b['weak'][:]=weak;b['cw'][:]=.5 if calm else .52
            b['budget']=np.minimum(.05,np.maximum(0.,.10-(.025*b['calm']+.05*b['weak'])))
            q=np.tile(np.linspace(-.5,.5,49),(n,1));on=np.ones(n,bool)
            scores,ch,pre,events=audit.funded_episode(b,q,on)
            coef=.01/(1.-b['cw']);delta=.01*q-coef[:,None]*(b['P']-b['core'])
            expected=b['P'].copy();nz=delta!=0;expected[nz]+=delta[nz]
            self.assertEqual(pre.tobytes(),expected.tobytes())
            for i,e in enumerate(events):
                self.assertEqual(e['requested_episode_budget'],(1.-coef[i])*b['budget'][i])
                self.assertEqual(e['original_support'],b['support'][i].tolist())

    def test_zero_targets_fund_and_only_zero_correction_skips_addition(self):
        p,m,c,a,r=fixture();base=audit.original_base(a,m,len(m['dates']))
        q=np.zeros_like(base['P']);on=np.ones(len(q),bool)
        _,_,pre,_=audit.funded_episode(base,q,on);self.assertFalse(np.array_equal(pre,base['P']))
        base['P'][:]=0.;base['P'][0,0]=-0.;base['core'][:]=0.;base['effects'][:]=0.;base['support'][:]=0.
        _,_,pre,_=audit.funded_episode(base,q,on);self.assertEqual(pre.tobytes(),base['P'].tobytes())

    def test_inactive_bypass_and_incumbent_gate(self):
        args=fixture();out=copy.deepcopy(full());p,m,c,a,r=args;inc=audit.ARMS[0]
        with patch.object(audit,'selection_oracle',side_effect=AssertionError('candidate forbidden')):
            receipt=audit.verify_incumbent(a,m,out[1][inc],out[2][inc],out[3][inc])
        self.assertTrue(all(receipt[k] for k in audit.INCUMBENT_GATE_FLAGS))
        self.assertFalse(receipt['candidate_selection_computed'])
        out[2][inc][1,0]+=1e-15
        with self.assertRaises(ValueError):audit.verify_incumbent(a,m,out[1][inc],out[2][inc],out[3][inc])

    def test_full_scores_unselected_cell_and_audit_count_corruption(self):
        args=fixture()
        for which in ('pre','final','count','budget'):
            out=copy.deepcopy(full())
            if which=='pre':out[3][audit.ARMS[1]][1,-1]+=1e-12
            elif which=='final':out[2][audit.ARMS[1]][1,-1]+=1e-12
            elif which=='count':out[4]['counts']['block_covariances']+=1
            else:out[4]['events'][1]['arms'][audit.ARMS[1]]['effective_episode_budget']+=1e-12
            with self.assertRaises(ValueError):
                audit.verify_selection(args[3],args[1],out[0],out[1],scores=out[2],pre_scores=out[3],audit=out[4])

    def test_whole49_source_domain_independent_of_eligibility(self):
        args=fixture();args[3]['eligible'][:]=False
        f=audit.independent_features(args[0],args[1],args[2],args[3],args[4],include_ranks=True)
        for i in range(1,len(f)):
            self.assertEqual(f[i]['breadth_rank'],full()[0][i]['breadth_rank'])
            self.assertEqual(len(f[i]['cells']),49)
        s,ch,pre,d=audit.selection_oracle(args[3],args[1],f)
        for arm in audit.ARMS:self.assertTrue(all(x==[] for x in ch[arm]))

    def test_readonly_arrays_and_no_new_producer_imports(self):
        args=fixture();before=pickle.dumps(args)
        for v in args[3].values():v.setflags(write=False)
        out=copy.deepcopy(full());verify(args,out)
        for v in args[3].values():v.setflags(write=True)
        self.assertEqual(pickle.dumps(args),before)
        tree=ast.parse(audit.Path(audit.__file__).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any('249_kernel' in x or '249_v2_kernel' in x or '249_adapter' in x for x in imports))
        self.assertTrue(audit.MATH in audit.dependency_files())

class ZeroVolumeTests(unittest.TestCase):
    def test_native_signed_zero_explicit_no_measure_and_exact_fallback(self):
        args=physical(fixture(),2);p,m,c,a,r=args;name=m['symbols'][1][-1]
        day=pd.Timestamp(r['clock_events'][1]['volume_start']);a['eligible'][1,-1]=False
        p['volume'].at[day,name]=-0.;before=copy.deepcopy(args)
        original=audit.exactmath.measure_oracle
        def guarded(prices,volumes):
            self.assertFalse(any(v is not None and v==0 for v in volumes))
            return original(prices,volumes)
        with patch.object(audit.exactmath,'measure_oracle',side_effect=guarded) as guard:out=run(args)
        self.assertEqual(guard.call_count,48)
        for key in p:pd.testing.assert_frame_equal(p[key],before[0][key],check_exact=True)
        for key in a:
            self.assertEqual((a[key].dtype.str,a[key].shape,a[key].tobytes()),(before[3][key].dtype.str,before[3][key].shape,before[3][key].tobytes()))
        self.assertEqual((m,c,r),(before[1],before[2],before[4]))
        e=out[0][1];self.assertEqual(e['branch'],'zero_volume');self.assertEqual(e['zero_indices'],[48]);self.assertEqual(e['missing_indices'],[])
        cell=e['cells'][48];self.assertIsNone(cell['measurement']);self.assertFalse(cell['source_available'])
        self.assertEqual(cell['volume_zeros'],[dict(offset=0,date=str(day.date()),value=-0.,reason='unresolved_zero_volume')])
        self.assertEqual(cell['volume_zeros'][0]['value'].hex(),'-0x0.0p+0')
        for arm in audit.ARMS[1:]:
            self.assertEqual(out[1][arm],out[1][audit.ARMS[0]])
            self.assertEqual(out[2][arm].tobytes(),out[2][audit.ARMS[0]].tobytes())
            self.assertEqual(out[3][arm].tobytes(),out[3][audit.ARMS[0]].tobytes())
        q=out[4]['counts'];self.assertEqual((q['source_missing_cells'],q['source_zero_volume_cells'],q['source_unavailable_cells'],q['source_zero_volume_observations']),(0,1,1,1))
        self.assertEqual((q['zero_volume_events'],q['missing_source_events'],q['ranked_cells'],q['block_covariances']),(1,0,0,48*12))
        verify(args,out)

    def test_zero_missing_transition_priorities_and_union_counts(self):
        args=physical(fixture(),2);p,m,c,a,r=args;e=r['clock_events'][1];day=pd.Timestamp(e['volume_start']);nextday=p['volume'].index[p['volume'].index.get_loc(day)+1]
        p['volume'].at[day,m['symbols'][1][0]]=0.;p['volume'].at[nextday,m['symbols'][1][0]]=np.nan
        p['volume'].at[day,m['symbols'][1][1]]=np.nan
        row=dict(id='unit',symbols=[m['symbols'][1][0]],effective_date=str(nextday.date()),affected_fields=['volume'],available_from='2016-01-01',known_unresolved_fallback_eligible=True,unresolved_unit_transition=True)
        r['transition_records']=[row];r['automatic_known_unresolved_transition_fallback_ids']=['unit']
        out=run(args);event=out[0][1]
        self.assertEqual(event['zero_indices'],[0]);self.assertEqual(event['missing_indices'],[0,1]);self.assertEqual(event['transition_ids'],['unit']);self.assertEqual(event['branch'],'zero_volume')
        self.assertIsNone(event['cells'][0]['measurement']);self.assertIsNotNone(event['cells'][1]['measurement'])
        q=out[4]['counts'];self.assertEqual((q['source_missing_cells'],q['source_zero_volume_cells'],q['source_unavailable_cells']),(2,1,2))
        self.assertEqual((q['zero_volume_events'],q['missing_source_events'],q['known_transition_events']),(1,0,0));verify(args,out)

    def test_zero_never_hides_negative_type_inf_price_or_underflow(self):
        from fractions import Fraction
        for bad in (-1.,float('inf'),True,'0',Fraction(1,10**1000)):
            args=physical(fixture(),2);p,m,c,a,r=args;day=pd.Timestamp(r['clock_events'][1]['volume_start'])
            p['volume']=p['volume'].astype(object);p['volume'].at[day,m['symbols'][1][0]]=0.;p['volume'].at[day,m['symbols'][1][-1]]=bad
            with self.assertRaises(ValueError):run(args)
        idx=pd.DatetimeIndex(['2010-01-04']);frame=pd.DataFrame({'S':[0.]},index=idx)
        with self.assertRaisesRegex(ValueError,'positive finite price'):audit.extract(frame,'S',list(idx),None,'2016-08-31')
        for x in (Fraction(1,10**1000),Fraction(-1,10**1000)):
            frame=pd.DataFrame({'S':pd.Series([x],index=idx,dtype=object)})
            with self.assertRaisesRegex(ValueError,'underflow'):audit.extract(frame,'S',list(idx),None,'2016-08-31',is_volume=True)

    def test_prelisting_zero_unavailable_positive_contradiction_still_hard(self):
        args=physical(fixture(),2);p,m,c,a,r=args;start=r['clock_events'][1]['price_start'];name=m['symbols'][1][0]
        bound(args,0,start,'2016-08-01')
        lo=r['clock_events'][1]['volume_start'];p['volume'].loc[(p['volume'].index>=lo)&(p['volume'].index<start),name]=0.
        out=run(args);self.assertEqual(out[0][1]['branch'],'zero_volume');self.assertEqual(len(out[0][1]['cells'][0]['volume_zeros']),62);verify(args,out)
        p['volume'].at[pd.Timestamp(lo),name]=1.
        with self.assertRaisesRegex(ValueError,'visible listing'):run(args)

    def test_zero_evidence_signed_bits_counts_and_lists_tampering(self):
        args=physical(fixture(),2);p,m,c,a,r=args;day=pd.Timestamp(r['clock_events'][1]['volume_start'])
        p['volume'].at[day,m['symbols'][1][0]]=-0.;reference=run(args)
        for kind in ('value','reason','offset','overlap','zero_indices','count','measurement'):
            out=copy.deepcopy(reference);e=out[0][1];z=e['cells'][0]['volume_zeros'][0]
            if kind=='value':z['value']=0.
            elif kind=='reason':z['reason']='missing_value'
            elif kind=='offset':z['offset']=True
            elif kind=='overlap':e['cells'][0]['volume_gaps']=[dict(offset=0,date=str(day.date()),reason='missing_value')]
            elif kind=='zero_indices':e['zero_indices']=[]
            elif kind=='count':out[4]['counts']['source_zero_volume_observations']+=1
            else:e['cells'][0]['measurement']=copy.deepcopy(full()[0][1]['cells'][0]['measurement'])
            with self.assertRaises(ValueError):verify(args,out)

    def test_all49_zero_source_gates_no_rank_math_or_partial_reads(self):
        args=physical(fixture(),2);p,m,c,a,r=args;day=pd.Timestamp(r['clock_events'][1]['volume_start']);p['volume'].loc[day,:]=0.
        with patch.object(audit.exactmath,'measure_oracle',side_effect=AssertionError('zero never enters positive math')),patch.object(audit.exactmath,'rank49_oracle',side_effect=AssertionError('fallback never ranks')):
            out=run(args);receipt=verify(args,out)
        self.assertEqual(out[4]['counts']['source_zero_volume_cells'],49);self.assertEqual(out[4]['counts']['block_covariances'],0)
        self.assertTrue(receipt['unresolved_zero_volume_fallback_checked'])
        args=physical(args,1)
        with patch.object(audit,'extract',side_effect=AssertionError('partial no numeric reads')):out=run(args)
        self.assertEqual(out[0][0]['zero_indices'],[])

    def test_zero_logical_physical_prefix_and_no_new_core_funding_math(self):
        args=fixture();p,m,c,a,r=args;day=pd.Timestamp(r['clock_events'][1]['volume_start']);p['volume'].at[day,m['symbols'][1][0]]=0.
        reference=run(args)
        for n in range(len(m['dates'])+1):
            result=run(args,n);cut=physical(args,n);other=run(cut)
            self.assertEqual(result[0],reference[0][:n]);self.assertEqual(other[0],result[0]);verify(cut,other)
        original=ast.parse(audit.SOURCE_BASE.read_text());current=ast.parse(audit.Path(audit.__file__).read_text())
        for name in ('funded_episode','validate_measurement','audit_outcomes'):
            before=next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name==name)
            after=next(x for x in current.body if isinstance(x,ast.FunctionDef) and x.name==name)
            self.assertEqual(ast.dump(before),ast.dump(after))

class AccountingTests(unittest.TestCase):
    @staticmethod
    def args():
        h=audit.load_pinned(audit.old.ACCOUNT_TEST,audit.old.ACCOUNT_TEST_SHA,'volume249_account_fixture')
        p,o,orders,exports,results,m=h.account_fixture();first=next(iter(orders))
        return p,o,{k:copy.deepcopy(orders[first]) for k in audit.ARMS},{k:copy.deepcopy(exports[first]) for k in audit.ARMS},{k:copy.deepcopy(results[first]) for k in audit.ARMS},m

    def test_nine_paths_fortyfive_slices_eighteen_training_groups(self):
        r=audit.audit_outcomes(*self.args());self.assertEqual((r['checked_paths'],r['checked_slices'],r['checked_training_groups']),(9,45,18))

    def test_zero_cost_and_cost_diagnostic_corruptions_rejected(self):
        args=list(self.args());args[3][audit.ARMS[1]]['same_close_zero']['HOLDINGS'][0]['fee']=1e-15
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args());args[4][audit.ARMS[2]]['next_open_cost']['slices']['later34']['cagr']+=.1
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*args)

class BilateralCross(unittest.TestCase):
    def test_both_fixtures_full_artifacts_and_all_logical_physical_prefixes(self):
        import monthly_volume_breadth249_v2_kernel as producer
        import test_monthly_volume_breadth249_v2_kernel as fixtures
        own=fixture();study=helper().helper().real_study()
        own_args=(study,own[3],own[1],own[0],own[2],own[4])
        for args in (fixtures.fixture(),own_args):
            study,a,m,p,c,r=args
            fullout=producer.prepare(*args,len(m['dates']),
              source_validator=lambda f:audit.verify_source_features(p,m,c,a,r,f),
              incumbent_validator=lambda o,s,pre:audit.verify_incumbent(a,m,o,s,pre))
            f,o,s,pre,d=fullout;audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d)
            for n in range(len(m['dates'])+1):
                f,o,s,pre,d=producer.prepare(*args,n,
                  source_validator=lambda f:audit.verify_source_features(p,m,c,a,r,f,n),
                  incumbent_validator=lambda o,s,pre:audit.verify_incumbent(a,m,o,s,pre,n))
                audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n)
                o,s,pre,d=producer.decisions(study,a,m,fullout[0],n,
                  incumbent_validator=lambda o,s,pre:audit.verify_incumbent(a,m,o,s,pre,n))
                audit.verify_selection(a,m,fullout[0],o,scores=s,pre_scores=pre,audit=d,n=n)
                pa=fixtures.physical(args,n);ps,aa,mm,pp,cc,rr=pa
                f,o,s,pre,d=producer.prepare(*pa,n,
                  source_validator=lambda f:audit.verify_source_features(pp,mm,cc,aa,rr,f,n),
                  incumbent_validator=lambda o,s,pre:audit.verify_incumbent(aa,mm,o,s,pre,n))
                audit.verify_prepared(pp,mm,cc,aa,rr,f,o,scores=s,pre_scores=pre,audit=d,n=n)

        # The only new policy is also cross-checked end-to-end, preserving zeros.
        for with_gap in (False,True):
            own=physical(fixture(),2);pp,mm,cc,aa,rr=own
            day=pd.Timestamp(rr['clock_events'][1]['volume_start']);symbol=mm['symbols'][1][-1]
            pp['volume'].at[day,symbol]=-0.
            if with_gap:
                nextday=pp['volume'].index[pp['volume'].index.get_loc(day)+1]
                pp['volume'].at[nextday,symbol]=np.nan
            pa=(study,aa,mm,pp,cc,rr)
            f,o,scores,pres,d=producer.prepare(*pa,2,
              source_validator=lambda f:audit.verify_source_features(pp,mm,cc,aa,rr,f,2),
              incumbent_validator=lambda o,s,pre:audit.verify_incumbent(aa,mm,o,s,pre,2))
            audit.verify_prepared(pp,mm,cc,aa,rr,f,o,scores=scores,pre_scores=pres,audit=d,n=2)
            self.assertEqual(f[1]['branch'],'zero_volume')
            self.assertEqual(f[1]['cells'][-1]['volume_zeros'][0]['value'].hex(),'-0x0.0p+0')
            oo,ss,ps,dd=producer.decisions(study,aa,mm,f,2,
              incumbent_validator=lambda o,s,pre:audit.verify_incumbent(aa,mm,o,s,pre,2))
            audit.verify_selection(aa,mm,f,oo,scores=ss,pre_scores=ps,audit=dd,n=2)
        own=physical(fixture(),2);pp,mm,cc,aa,rr=own
        day=pd.Timestamp(rr['clock_events'][1]['volume_start']);pp['volume'].at[day,mm['symbols'][1][0]]=0.;pp['volume'].at[day,mm['symbols'][1][-1]]=-1.
        with self.assertRaises(ValueError):producer.prepare(study,aa,mm,pp,cc,rr,2)
        with self.assertRaises(ValueError):audit.independent_features(pp,mm,cc,aa,rr,2)

if __name__=='__main__':
    if '--self-only' in sys.argv:
        sys.argv.remove('--self-only')
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (SourceTests,SelectionTests,ZeroVolumeTests,AccountingTests))
        result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
    unittest.main(verbosity=2)

