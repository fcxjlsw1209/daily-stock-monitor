"""Synthetic source/core/funding tests; no historical arrays or replay."""
from copy import deepcopy
from fractions import Fraction
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_reference_gain250_kernel as kernel
import test_monthly_calendar_clock248_v2_kernel as previous


def make_review(meta,calendar,first,bounds=None,records=None,rules=None,automatic=None):
    clocks=[];identities=[];lookup={d:i for i,d in enumerate(calendar)}
    bounds=deepcopy(bounds or [])
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=lookup[day];partial=day==kernel.INITIAL_PARTIAL
        e=dict(query_index=i,date=day,initial_partial=partial,numeric_evaluation_required=not partial,symbols=names.copy(),query_position=t,next_official_session=calendar[t+1],
          price_start=calendar[t-294],price_end=calendar[t-42],price_count=253,
          volume_start=calendar[t-294],volume_end=calendar[t-43],volume_count=252,historical_count=252,
          historical_price_start=calendar[t-294],historical_price_end=calendar[t-43],reference_date=calendar[t-42],reference_volume_consumed=False)

        clocks.append(e)
        for j,s in enumerate(names):
            for b in bounds:
                if s==b['symbol']:
                    identities.append(dict(query_index=i,column_index=j,date=day,symbol=s,issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=day,
                      price_start=e['price_start'],price_end=e['price_end'],volume_start=e['volume_start'],volume_end=e['volume_end'],
                      price_window_dates_before_bound=sum(d<b['not_before'] for d in calendar[t-294:t-41]),volume_window_dates_before_bound=sum(d<b['not_before'] for d in calendar[t-294:t-42])))
    return dict(schema=kernel.SOURCE_SCHEMA,panel_first_session=first,clock_events=clocks,listing_bounds=bounds,identity_cells=identities,
      transition_records=deepcopy(records or []),undated_loader_rules=deepcopy(rules or []),automatic_known_unresolved_transition_fallback_ids=list(automatic or []))


def fixture():
    s,a,m,p,c,_=previous.fixture();names=['S'+str(j).zfill(2) for j in range(49)];shape=(3,49)
    for k,v in list(a.items()):
        if np.shape(v)==(3,6):a[k]=np.asarray(v)[:,np.arange(49)%6].copy()
    m['symbols']=[names.copy() for _ in range(3)];m['etfs']=names[-17:]
    x=np.arange(len(p['adj']),dtype=float);ix=p['adj'].index
    p={'adj':pd.DataFrame({s:90+(j+1)*.003*x+2*np.sin(x/(11+j))+1.1*np.cos(x/(17+j)) for j,s in enumerate(names)},index=ix),
       'volume':pd.DataFrame({s:1000.+(j+1)*2+100.*np.sin(x/(3+j%7))+50.*np.cos(x/(5+j%5)) for j,s in enumerate(names)},index=ix)}
    r=make_review(m,c,c[0]);args=(s,a,m,p,c,r);return previous.refresh_core(args)


def physical(args,n):
    s,a,m,p,c,r=args;mm=deepcopy(m)
    for k in ('dates','ends','symbols'):mm[k]=mm[k][:n]
    aa={k:np.asarray(v)[:n].copy() for k,v in a.items()}
    if n:
        last=mm['dates'][-1];pp={k:v.loc[:last].copy() for k,v in p.items()};cc=c[:c.index(last)+2]
    else:pp={k:v.iloc[:0].copy() for k,v in p.items()};cc=[]
    return s,aa,mm,pp,cc,deepcopy(r)


def run(args,n=None,**kw):return kernel.prepare(*args,len(args[2]['dates']) if n is None else n,**kw)
def day(args,row=1,offset=42):return args[4][args[4].index(args[2]['dates'][row])-offset]
def set_review(args,**kw):args[5].clear();args[5].update(make_review(args[2],args[4],args[4][0],**kw))
def same(case,a,b):
    case.assertEqual(tuple(a),tuple(b))
    for k in a:
        case.assertEqual(a[k].dtype,b[k].dtype);case.assertEqual(a[k].shape,b[k].shape);case.assertEqual(a[k].tobytes(),b[k].tobytes())


def transition(args,effective=None,available='2014-01-01'):
    return dict(id='synthetic_units',symbols=['S00'],effective_date=effective or day(args,1,100),available_from=available,
      affected_fields=['volume'],known_unresolved_fallback_eligible=True,unresolved_unit_transition=True)


class KernelTests(unittest.TestCase):
    def test_complete_schema_single_episode_allarms(self):
        args=fixture()
        with patch.object(kernel.funding.ep,'select_boundary',wraps=kernel.funding.ep.select_boundary) as ep:
            f,o,s,p,a=run(args)
        self.assertEqual(ep.call_count,3);self.assertEqual(tuple(o),kernel.ARMS);self.assertEqual(tuple(a['counts']),kernel.COUNT_FIELDS)
        self.assertTrue(all(tuple(e)==kernel.EVENT_KEYS for e in f));self.assertTrue(all(tuple(c)==kernel.CELL_KEYS for e in f for c in e['cells']))
        self.assertTrue(all(tuple(e)==kernel.AUDIT_EVENT_KEYS for e in a['events']))
        self.assertEqual(a['counts']['source_checked_cells'],98);self.assertEqual(a['counts']['source_price_observations'],98*253)
        self.assertEqual(a['counts']['source_volume_observations'],98*252);self.assertEqual(a['counts']['weighted_reference_estimates'],98);self.assertEqual(a['counts']['equal_reference_estimates'],98)
        self.assertEqual(a['counts']['ranked_cells'],196);self.assertEqual(a['counts']['active_events'],2)
        self.assertEqual(a['counts']['new_model_fits'],0);json.dumps([f,o,a],allow_nan=False)

    def test_original_control_exact_and_partial_allarms(self):
        args=fixture();f,o,s,p,a=run(args);st,aa,m,*_=args
        ch,post=st.choices(st.configs()[4],aa,m,3,policy=dict(st.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],st.rp.make_orders(ch,post,m));self.assertEqual(s[kernel.ARMS[0]].tobytes(),post.tobytes())
        self.assertEqual(p[kernel.ARMS[0]].tobytes(),st.modified_pre_score(st.configs()[4],aa,3).tobytes())
        for arm in kernel.ARMS[1:]:
            self.assertEqual(o[arm][0],o[kernel.ARMS[0]][0]);self.assertEqual(s[arm][0].tobytes(),s[kernel.ARMS[0]][0].tobytes())

    def test_exact_math_and_all49_domain(self):
        args=fixture();f,*_=run(args,2);i=1;t=args[4].index(args[2]['dates'][i])
        for j in (0,32,48):
            s=args[2]['symbols'][i][j]
            expected=kernel.exact.measure(args[3]['adj'].loc[args[4][t-294]:args[4][t-42],s].tolist(),args[3]['volume'].loc[args[4][t-294]:args[4][t-43],s].tolist())
            self.assertEqual(f[i]['cells'][j]['measurement'],expected)
        self.assertEqual(f[i]['volume_rank'],kernel.exact.rank49([c['measurement']['raw_volume_gain'] for c in f[i]['cells']]))
        args[1]['eligible'][:,32:]=False
        changed=run(args,2)[0];self.assertEqual(changed,f)

    def test_nonfinite_original_domain_does_not_remove_newmember(self):
        args=fixture();args[3]['adj'].loc[day(args,1,10),'S48']=np.nan;previous.refresh_core(args)
        f,*_=run(args,2);self.assertNotIn(48,f[1]['u0_indices']);self.assertTrue(f[1]['cells'][48]['source_available']);self.assertEqual(f[1]['branch'],'available')

    def test_funding_allaux_and_episode_budget(self):
        args=fixture();f,o,s,p,a=run(args)
        for i in (1,2):
            cw=args[1]['core_weight'][i,0];coef=.02/(1-cw)
            for arm in kernel.ARMS[1:]:
                e=a['events'][i]['arms'][arm];q=np.asarray(e['target']);delta=.02*q-coef*(p[kernel.ARMS[0]][i]-args[1]['core'][i])
                expected=p[kernel.ARMS[0]][i].copy();nz=delta!=0;expected[nz]+=delta[nz]
                self.assertEqual(expected.tobytes(),p[arm][i].tobytes());self.assertEqual(e['alpha'],1-coef)
                self.assertEqual(e['requested_episode_budget'],(1-coef)*a['events'][i]['arms'][kernel.ARMS[0]]['effective_episode_budget'])
                self.assertEqual(e['original_support'],a['events'][i]['arms'][kernel.ARMS[0]]['original_support'])

    def test_missing_volume_column_entire_event_exact_fallback(self):
        args=fixture();args[3]['volume']=args[3]['volume'].drop(columns='S48');f,o,s,p,a=run(args)
        self.assertEqual(f[1]['missing_indices'],[48]);self.assertEqual(f[1]['branch'],'missing_source');self.assertIsNone(f[1]['volume_rank'])
        for arm in kernel.ARMS[1:]:self.assertEqual(o[arm],o[kernel.ARMS[0]]);self.assertEqual(s[arm].tobytes(),s[kernel.ARMS[0]].tobytes())
        self.assertEqual(f[1]['cells'][48]['volume_gaps'][0]['reason'],'missing_symbol')

    def test_missing_volume_row_and_nan_recorded(self):
        args=fixture();d=day(args,1,294);args[3]['volume']=args[3]['volume'].drop(index=d);args[3]['volume'].loc[day(args,1,43),'S48']=np.nan
        f,*_=run(args,2);self.assertEqual(f[1]['missing_indices'],list(range(49)))
        self.assertEqual(f[1]['cells'][0]['volume_gaps'],[dict(offset=0,date=d,reason='missing_row')])
        self.assertEqual(f[1]['cells'][48]['volume_gaps'][-1]['reason'],'missing_value')

    def test_missing_price_nan_is_fallback_with_core_parity(self):
        args=fixture();args[3]['adj'].loc[day(args,1,110),'S00']=np.nan;previous.refresh_core(args)
        f,*_=run(args,2);self.assertEqual(f[1]['branch'],'missing_source');self.assertEqual(f[1]['missing_indices'],[0]);self.assertTrue(f[1]['cells'][1]['source_available'])

    def test_missing_does_not_hide_invalid_other_member(self):
        for bad in (-1.,float('inf'),True,'bad'):
            args=fixture();args[3]['volume']=args[3]['volume'].astype(object);args[3]['volume'].loc[day(args,1,43),'S00']=None;args[3]['volume'].loc[day(args,1,43),'S48']=bad
            with self.assertRaises(ValueError):run(args,2)

    def test_all_required_slots_scan_before_missing(self):
        args=fixture();args[3]['volume']=args[3]['volume'].astype(object);args[3]['volume'].loc[day(args,1,294),'S00']=None;args[3]['volume'].loc[day(args,1,43),'S48']=float('inf')
        with self.assertRaisesRegex(ValueError,'finite'):run(args,2)

    def test_visible_listing_positive_volume_not_hidden_by_missingprice(self):
        args=fixture();bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2016-01-02',evidence_id='listing',kind='new_issue')
        set_review(args,bounds=[bound]);args[3]['adj'].loc[day(args,1,294),'S00']=np.nan;previous.refresh_core(args)
        with self.assertRaisesRegex(ValueError,'Visible listing contradiction'):run(args,2)

    def test_future_listing_evidence_not_used(self):
        args=fixture();bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2027-01-02',evidence_id='listing',kind='new_issue');set_review(args,bounds=[bound])
        f,*_=run(args,2);self.assertEqual(f[1]['branch'],'available')

    def test_listing_all_missing_is_not_positive_contradiction(self):
        args=fixture();bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2016-01-02',evidence_id='listing',kind='new_issue');set_review(args,bounds=[bound])
        args[3]['adj'].loc[:'2015-12-31','S48']=np.nan;args[3]['volume'].loc[:'2015-12-31','S48']=np.nan;previous.refresh_core(args)
        f,*_=run(args,2);self.assertEqual(f[1]['branch'],'missing_source')

    def test_identity_proof_missing_wrong_or_duplicate_rejected(self):
        for kind in ('missing','issuer','duplicate'):
            args=fixture();b=dict(symbol='S48',issuer_key='issuer48',not_before='2010-01-01',available_from='2010-01-02',evidence_id='listing',kind='new_issue');set_review(args,bounds=[b])
            if kind=='missing':args[5]['identity_cells'].pop(1)
            elif kind=='issuer':args[5]['identity_cells'][1]['issuer_key']='wrong'
            else:args[5]['identity_cells'].append(deepcopy(args[5]['identity_cells'][1]))
            with self.assertRaises(ValueError):run(args,2)

    def test_transition_crossing_and_missing_precedence(self):
        args=fixture();x=transition(args);set_review(args,records=[x],automatic=[x['id']]);f,o,s,p,a=run(args,2)
        self.assertEqual(f[1]['branch'],'known_transition');self.assertEqual(f[1]['transition_ids'],[x['id']])
        self.assertEqual(o[kernel.ARMS[1]],o[kernel.ARMS[0]])
        args[3]['volume'].loc[day(args,1,43),'S01']=np.nan;f,*_=run(args,2)
        self.assertEqual(f[1]['branch'],'missing_source');self.assertEqual(f[1]['transition_ids'],[x['id']])

    def test_transition_first_new_unit_day_boundary(self):
        args=fixture();x=transition(args,effective=day(args,1,294));set_review(args,records=[x],automatic=[x['id']]);self.assertEqual(run(args,2)[0][1]['branch'],'available')
        x['effective_date']=day(args,1,43);set_review(args,records=[x],automatic=[x['id']]);self.assertEqual(run(args,2)[0][1]['branch'],'known_transition')

    def test_future_transition_unknown_bridge_are_not_imputed(self):
        args=fixture();x=transition(args,available='2027-01-01');rule=dict(id='undated',symbols=['S00'],effective_date=None)
        set_review(args,records=[x],rules=[rule],automatic=[x['id']]);f,*_=run(args,2)
        self.assertEqual(f[1]['branch'],'available');self.assertEqual(f[1]['continuity_unknown_ids'],['undated'])

    def test_unknown_automatic_transition_rejected(self):
        args=fixture();set_review(args,automatic=['bad'])
        with self.assertRaisesRegex(ValueError,'Unknown automatic'):run(args,2)

    def test_original_source_priority_clock_column_core(self):
        for mode in ('column','row','query','core','origin'):
            args=fixture();args[3]['volume']=args[3]['volume'].drop(columns='S48')
            if mode=='column':args[3]['adj']=args[3]['adj'].drop(columns='S00')
            elif mode=='row':args[3]['adj']=args[3]['adj'].drop(index=day(args,1,100))
            elif mode=='query':args[3]['adj']=args[3]['adj'].drop(index=args[2]['dates'][1])
            elif mode=='core':args[1]['core'][1,0]+=.01
            else:args[5]['panel_first_session']='2010-01-04'
            with self.assertRaises(ValueError):run(args,2)

    def test_full_history_not_reset_sigma(self):
        args=fixture()
        with patch.object(kernel.old,'_rank',wraps=kernel.old._rank) as rank:
            f,*_=run(args,2)
        self.assertGreater(rank.call_count,0)
        raw=args[3]['adj'].pct_change(fill_method=None).rolling(63).std().loc[args[2]['dates'][1],'S00']
        self.assertEqual(f[1]['cells'][0]['sigma63'],raw)

    def test_partial_no_numeric_reads(self):
        args=fixture()
        for k in args[3]:args[3][k]=args[3][k].astype(object);args[3][k].loc[:,:]='bad'
        f,o,*_=run(args,1);self.assertEqual(f[0]['cells'],[]);self.assertEqual(o[kernel.ARMS[0]],o[kernel.ARMS[1]])

    def test_n0_empty_physical(self):
        f,o,s,p,a=run(physical(fixture(),0));self.assertEqual(f,[]);self.assertEqual(o,{k:[] for k in kernel.ARMS});self.assertEqual(a['counts']['original_episode_calls'],3)
        self.assertTrue(all(v.shape==(0,49) for v in s.values()))

    def test_all_logical_physical_cached_prefixes(self):
        args=fixture();full=run(args)
        for n in range(4):
            left=run(args,n);right=run(physical(args,n));self.assertEqual(left[0],right[0]);self.assertEqual(left[0],full[0][:n]);self.assertEqual(left[1],right[1]);self.assertEqual(left[4],right[4]);same(self,left[2],right[2]);same(self,left[3],right[3])
            with patch.object(kernel,'_features',side_effect=AssertionError('No source reread')),patch.object(kernel.exact,'measure',side_effect=AssertionError('No measurement rerun')):
                cached=kernel.decisions(args[0],args[1],args[2],full[0],n)
            self.assertEqual(cached[0],left[1]);same(self,cached[1],left[2]);same(self,cached[2],left[3]);self.assertEqual(cached[3],left[4])

    def test_future_numeric_and_metadata_unconsumed(self):
        args=fixture();base=run(args,2);cut=args[2]['dates'][1]
        for k in args[3]:args[3][k]=args[3][k].astype(object);args[3][k].loc[args[3][k].index>cut,:]='futurebad'
        args[5]['clock_events'][2]={'query_index':2,'bad':'future'};args[2]['symbols'][2]=['BAD'];args[1]['baseline'][2,:]=np.nan
        now=run(args,2);self.assertEqual(base[0],now[0]);self.assertEqual(base[1],now[1]);same(self,base[2],now[2])

    def test_source_gate_before_any_episode(self):
        args=fixture();calls=[]
        def gate(f):calls.append('source');self.assertEqual(len(f),3);return {'passed':False}
        with patch.object(kernel.funding,'score_and_choose',side_effect=AssertionError('No episode')):
            with self.assertRaisesRegex(ValueError,'original-source gate'):run(args,source_validator=gate)
        self.assertEqual(calls,['source'])

    def test_incumbent_gate_before_candidate_episode(self):
        args=fixture();calls=[]
        def source(f):calls.append('source');return {'passed':True}
        def inc(o,s,p):calls.append('inc');return {'passed':False}
        with patch.object(kernel.funding,'score_and_choose',wraps=kernel.funding.score_and_choose) as adapter:
            with self.assertRaisesRegex(ValueError,'incumbent gate'):run(args,source_validator=source,incumbent_validator=inc)
            self.assertEqual(adapter.call_count,1)
        self.assertEqual(calls,['source','inc'])

    def test_rank49_only_after_source_and_incumbent_gates(self):
        args=fixture();history=[]
        def source(features):
            self.assertTrue(all(e['volume_rank'] is None and e['equal_rank'] is None for e in features));history.append('source');return {'passed':True}
        def incumbent(orders,scores,pres):history.append('incumbent');return {'passed':True}
        real_rank=kernel.exact.rank49
        def rank(values):
            self.assertEqual(history[:2],['source','incumbent']);history.append('rank');return real_rank(values)
        with patch.object(kernel.exact,'rank49',side_effect=rank):
            f,*_=run(args,2,source_validator=source,incumbent_validator=incumbent)
        self.assertEqual(history,['source','incumbent','rank','rank'])
        bad=deepcopy(f);bad[1]['volume_rank']='malformed target not consumed before gate'
        def reject(orders,scores,pres):return {'passed':False}
        with patch.object(kernel.exact,'rank49',side_effect=AssertionError('No ranks before failed gate')):
            with self.assertRaisesRegex(ValueError,'incumbent gate'):kernel.decisions(args[0],args[1],args[2],bad,2,incumbent_validator=reject)
        with patch.object(kernel.funding,'score_and_choose',wraps=kernel.funding.score_and_choose) as adapter:
            with self.assertRaisesRegex(ValueError,'Exact common49 ranks'):kernel.decisions(args[0],args[1],args[2],bad,2)
            self.assertEqual(adapter.call_count,1)

    def test_expected_incumbent_orders_before_candidate(self):
        args=fixture();expected=run(args)[1][kernel.ARMS[0]];expected=deepcopy(expected);expected[1]['symbols']=[]
        with patch.object(kernel.funding,'score_and_choose',wraps=kernel.funding.score_and_choose) as adapter:
            with self.assertRaisesRegex(ValueError,'Frozen incumbent'):run(args,expected_incumbent_orders=expected)
            self.assertEqual(adapter.call_count,1)

    def test_cached_tamper_rank_measurement_core_branch_rejected(self):
        args=fixture();f=run(args)[0]
        for mode in ('rank','raw','core','branch','count'):
            z=deepcopy(f)
            if mode=='rank':z[1]['volume_rank']['targets'][0]+=.01
            elif mode=='raw':z[1]['cells'][0]['measurement']['raw_equal_gain']['numerator']='999'
            elif mode=='core':z[1]['cells'][0]['frozen_core']+=.01
            elif mode=='branch':z[1]['branch']='missing_source'
            else:z[1]['cells'][0]['measurement']['historical_count']=True
            with self.assertRaises(ValueError):kernel.decisions(args[0],args[1],args[2],z,3)

    def test_equal_price_gains_available_alltie_not_neutral(self):
        args=fixture();args[3]['adj'].loc[:,:]=100.;previous.refresh_core(args)
        f,o,s,p,a=run(args)
        self.assertEqual(f[1]['branch'],'available');self.assertTrue(all(c['measurement']['raw_volume_gain']=={'numerator':'0','denominator':'1'} for c in f[1]['cells']))
        self.assertEqual(f[1]['volume_rank']['targets'],[float(Fraction(1,98))]*49)
        self.assertNotEqual(p[kernel.ARMS[1]][1].tobytes(),p[kernel.ARMS[0]][1].tobytes())
        self.assertEqual(p[kernel.ARMS[1]].tobytes(),p[kernel.ARMS[2]].tobytes())

    def test_constant_volume_matches_equal_reference_control(self):
        args=fixture();args[3]['volume'].loc[:,:]=1000.;f,o,s,p,a=run(args)
        self.assertEqual(f[1]['volume_rank'],f[1]['equal_rank'])
        self.assertEqual(p[kernel.ARMS[1]].tobytes(),p[kernel.ARMS[2]].tobytes());self.assertEqual(o[kernel.ARMS[1]],o[kernel.ARMS[2]])

    def test_input_nonmutation(self):
        args=fixture();a=kernel.old._snapshot(args[1]);m=deepcopy(args[2]);p={k:v.copy(deep=True) for k,v in args[3].items()};r=deepcopy(args[5]);run(args)
        self.assertEqual(a,kernel.old._snapshot(args[1]));self.assertEqual(m,args[2]);self.assertEqual(r,args[5])
        for k in p:pd.testing.assert_frame_equal(p[k],args[3][k])

    def test_strict_width49_axes_and_clock_receipt(self):
        for mode in ('width','volumeaxis','clock'):
            args=fixture()
            if mode=='width':
                for k,v in list(args[1].items()):
                    if v.shape==(3,49):args[1][k]=v[:,:48]
                args[2]['symbols']=[r[:48] for r in args[2]['symbols']]
            elif mode=='volumeaxis':args[3]['volume']=args[3]['volume'].iloc[::-1]
            else:args[5]['clock_events'][1]['price_count']=True
            with self.assertRaises(ValueError):run(args,2)

    def test_future_aux_values_after_tminus42_not_consumed(self):
        args=fixture();base=run(args,2);cut=day(args,1,43)
        args[3]['volume']=args[3]['volume'].astype(object);args[3]['volume'].loc[args[3]['volume'].index>cut,:]='bad'
        self.assertEqual(run(args,2)[0],base[0])


class ZeroVolumePolicyTests(unittest.TestCase):
    def test_native_signed_zero_retained_whole_event_fallback_all49(self):
        args=fixture();args[1]['eligible'][:,48]=False
        first=day(args,1,294);last=day(args,1,43)
        args[3]['volume'].loc[first,'S48']=0.;args[3]['volume'].loc[last,'S48']=-0.
        before=args[3]['volume'].copy(deep=True)
        f,o,sc,p,a=run(args,2);e=f[1];cell=e['cells'][48]
        self.assertEqual(e['branch'],'zero_volume');self.assertEqual(e['zero_indices'],[48]);self.assertEqual(e['missing_indices'],[])
        self.assertEqual(cell['volume_zeros'],[dict(offset=0,date=first,value=0.,reason='unresolved_zero_volume'),dict(offset=251,date=last,value=-0.,reason='unresolved_zero_volume')])
        self.assertEqual(cell['volume_zeros'][1]['value'].hex(),(-0.).hex());self.assertEqual(cell['volume_gaps'],[])
        self.assertFalse(cell['source_available']);self.assertIsNone(cell['measurement']);self.assertIsNone(e['volume_rank']);self.assertIsNone(e['equal_rank'])
        self.assertTrue(e['cells'][0]['source_available']);self.assertIsNotNone(e['cells'][0]['measurement'])
        for arm in kernel.ARMS[1:]:
            self.assertEqual(o[arm],o[kernel.ARMS[0]]);self.assertEqual(sc[arm].tobytes(),sc[kernel.ARMS[0]].tobytes());self.assertEqual(p[arm].tobytes(),p[kernel.ARMS[0]].tobytes())
        counts=a['counts'];self.assertEqual(len(counts),30)
        for k,v in {'source_available_cells':48,'source_zero_volume_cells':1,'source_missing_cells':0,'source_unavailable_cells':1,'source_zero_volume_observations':2,'zero_volume_events':1,'missing_source_events':0,'active_events':0,'ranked_cells':0,'weighted_reference_estimates':48,'equal_reference_estimates':48}.items():self.assertEqual(counts[k],v,k)
        pd.testing.assert_frame_equal(before,args[3]['volume']);json.dumps([f,o,a],allow_nan=False)

    def test_zero_and_missing_retained_separately_with_union_count(self):
        args=fixture();d=day(args,1,43);args[3]['volume'].loc[d,'S00']=0.;args[3]['volume'].loc[day(args,1,44),'S00']=np.nan;args[3]['volume'].loc[d,'S01']=np.nan
        f,o,sc,p,a=run(args,2);e=f[1];c=a['counts']
        self.assertEqual(e['zero_indices'],[0]);self.assertEqual(e['missing_indices'],[0,1]);self.assertEqual(e['branch'],'zero_volume')
        self.assertIsNone(e['cells'][0]['measurement']);self.assertEqual(e['cells'][0]['volume_gaps'][0]['offset'],250)
        self.assertEqual(e['cells'][1]['measurement']['reason'],'missing_source')
        for k,v in {'source_missing_cells':2,'source_zero_volume_cells':1,'source_unavailable_cells':2,'source_available_cells':47,'zero_volume_events':1,'missing_source_events':0}.items():self.assertEqual(c[k],v)

    def test_zero_precedes_missing_and_known_transition_but_keeps_evidence(self):
        args=fixture();x=transition(args);set_review(args,records=[x],automatic=[x['id']]);args[3]['volume'].loc[day(args,1,43),'S00']=0.;args[3]['volume'].loc[day(args,1,43),'S01']=np.nan
        e=run(args,2)[0][1];self.assertEqual(e['branch'],'zero_volume');self.assertEqual(e['transition_ids'],[x['id']]);self.assertEqual(e['missing_indices'],[1]);self.assertEqual(e['zero_indices'],[0])

    def test_zero_cannot_mask_hard_invalid_last_member(self):
        for bad in (-1.,float('inf'),float('-inf'),True,'bad',Fraction(1,10**1000)):
            with self.subTest(bad=type(bad).__name__):
                args=fixture();args[3]['volume']=args[3]['volume'].astype(object);args[3]['volume'].loc[day(args,1,294),'S00']=0.;args[3]['volume'].loc[day(args,1,43),'S48']=bad
                with patch.object(kernel.exact,'measure',side_effect=AssertionError('No measurements before hard error')):
                    with self.assertRaises(ValueError):run(args,2)

    def test_price_zero_still_hard_and_native_zero_not_cast_underflow(self):
        days=['2015-01-02'];index=pd.to_datetime(days)
        for label,x in [('price',0.),('price',-0.),('volume',Fraction(1,10**1000)),('volume',Fraction(-1,10**1000))]:
            with self.subTest(label=label,x=type(x).__name__):
                frame=pd.DataFrame({'S':[x]},index=index,dtype=object)
                values,gaps,zeros,errors=kernel._window(frame,'S',days,None,'2016-01-01',label)
                self.assertTrue(errors);self.assertEqual(zeros,[]);self.assertEqual(gaps,[])
        with self.assertRaises(ValueError):kernel.exact.measure([1.]*253,[0.]*252)

    def test_zero_cannot_mask_positive_prelisting_other_member(self):
        args=fixture();bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2016-01-02',evidence_id='listing',kind='new_issue');set_review(args,bounds=[bound]);args[3]['volume'].loc[day(args,1,43),'S00']=0.
        with self.assertRaisesRegex(ValueError,'Visible listing contradiction'):run(args,2)

    def test_prelisting_zero_never_supplies_history(self):
        args=fixture();bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2016-01-02',evidence_id='listing',kind='new_issue');set_review(args,bounds=[bound])
        args[3]['adj'].loc[:'2015-12-31','S48']=np.nan;args[3]['volume'].loc[:'2015-12-31','S48']=0.;previous.refresh_core(args)
        e=run(args,2)[0][1];self.assertEqual(e['branch'],'zero_volume');self.assertEqual(e['zero_indices'],[48]);self.assertEqual(e['missing_indices'],[48]);self.assertFalse(e['cells'][48]['source_available']);self.assertIsNone(e['cells'][48]['measurement'])
        self.assertTrue(all(x['date']<bound['not_before'] for x in e['cells'][48]['volume_zeros']))

    def test_zero_source_gate_no_target_ranks_and_failed_incumbent_blocks_candidates(self):
        args=fixture();args[3]['volume'].loc[day(args,1,43),'S48']=0.;seen=[]
        def source(features):
            seen.append('source');self.assertEqual(features[1]['zero_indices'],[48]);self.assertTrue(all(e['volume_rank'] is None and e['equal_rank'] is None for e in features));return {'passed':True}
        def inc(o,s,p):seen.append('inc');return {'passed':False}
        with patch.object(kernel.exact,'rank49',side_effect=AssertionError('No rank on failed incumbent')),patch.object(kernel.funding,'score_and_choose',wraps=kernel.funding.score_and_choose) as adapter:
            with self.assertRaisesRegex(ValueError,'incumbent gate'):run(args,2,source_validator=source,incumbent_validator=inc)
            self.assertEqual(adapter.call_count,1)
        self.assertEqual(seen,['source','inc'])

    def test_zero_all_logical_physical_cached_prefixes(self):
        args=fixture();args[3]['volume'].loc[day(args,1,43),'S48']=0.;full=run(args)
        for n in range(4):
            left=run(args,n);right=run(physical(args,n));self.assertEqual(left[0],right[0]);self.assertEqual(left[0],full[0][:n]);self.assertEqual(left[1],right[1]);self.assertEqual(left[4],right[4]);same(self,left[2],right[2]);same(self,left[3],right[3])
            with patch.object(kernel,'_features',side_effect=AssertionError('No source reread')),patch.object(kernel.exact,'measure',side_effect=AssertionError('No new measurement')):
                cached=kernel.decisions(args[0],args[1],args[2],full[0],n)
            self.assertEqual(cached[0],left[1]);self.assertEqual(cached[3],left[4]);same(self,cached[1],left[2]);same(self,cached[2],left[3])

    def test_cached_zero_schema_and_gap_union_tamper_rejected(self):
        args=fixture();args[3]['volume'].loc[day(args,1,43),'S48']=0.;f=run(args,2)[0]
        for mode in ('value','bool','reason','offset','duplicate','overlap','measurement','available','zero_indices','missing_indices','branch','rank'):
            with self.subTest(mode=mode):
                z=deepcopy(f);c=z[1]['cells'][48];zero=c['volume_zeros'][0]
                if mode=='value':zero['value']=1.
                elif mode=='bool':zero['value']=False
                elif mode=='reason':zero['reason']='missing_value'
                elif mode=='offset':zero['offset']=252
                elif mode=='duplicate':c['volume_zeros'].append(deepcopy(zero))
                elif mode=='overlap':c['volume_gaps'].append(dict(offset=zero['offset'],date=zero['date'],reason='missing_value'))
                elif mode=='measurement':c['measurement']=deepcopy(z[1]['cells'][0]['measurement'])
                elif mode=='available':c['source_available']=True
                elif mode=='zero_indices':z[1]['zero_indices']=[]
                elif mode=='missing_indices':z[1]['missing_indices']=[48]
                elif mode=='branch':z[1]['branch']='available'
                else:z[1]['volume_rank']={'malformed':'rank must be None'}
                with self.assertRaises(ValueError):kernel.decisions(args[0],args[1],args[2],z,2)

    def test_future_zero_unconsumed_partial_bypass(self):
        args=fixture();base=run(args,2);cut=day(args,1,43);args[3]['volume'].loc[args[3]['volume'].index>cut,:]=0.
        self.assertEqual(run(args,2)[0],base[0])
        args[3]['volume'].loc[:,:]=0.;f,o,*_=run(args,1);self.assertEqual(f[0]['zero_indices'],[]);self.assertEqual(f[0]['cells'],[]);self.assertEqual(o[kernel.ARMS[1]],o[kernel.ARMS[0]])


class UnequalWindowTests(unittest.TestCase):
    def test_reference_volume_all_invalid_forms_not_consumed(self):
        for bad in (None,np.nan,0.,-1.,True,'bad',float('inf')):
            args=fixture();base=run(args,2);d=day(args,1,42)
            args[3]['volume']=args[3]['volume'].astype(object);args[3]['volume'].loc[d,:]=bad
            now=run(args,2);self.assertEqual(base[0],now[0]);self.assertEqual(base[1],now[1]);same(self,base[2],now[2])
        args=fixture();base=run(args,2);args[3]['volume']=args[3]['volume'].drop(index=day(args,1,42))
        self.assertEqual(run(args,2)[0],base[0])

    def test_reference_day_transition_depends_on_affected_field(self):
        args=fixture();x=transition(args,effective=day(args,1,42));set_review(args,records=[x],automatic=[x['id']])
        self.assertEqual(run(args,2)[0][1]['branch'],'available')
        x['affected_fields']=['adj'];set_review(args,records=[x],automatic=[x['id']])
        self.assertEqual(run(args,2)[0][1]['branch'],'known_transition')

    def test_reference_volume_outside_listing_scan(self):
        args=fixture();d=day(args,1,42);bound=dict(symbol='S48',issuer_key='issuer48',not_before=day(args,1,41),available_from=day(args,1,40),evidence_id='listing',kind='new_issue')
        set_review(args,bounds=[bound]);args[3]['adj'].loc[:d,'S48']=np.nan;args[3]['volume'].loc[:day(args,1,43),'S48']=np.nan
        # The only positive prelisting volume is the excluded reference date.
        args[3]['volume'].loc[d,'S48']=1.;previous.refresh_core(args)
        e=run(args,2)[0][1];self.assertEqual(e['branch'],'missing_source')
        self.assertEqual(len(e['cells'][48]['price_gaps']),253);self.assertEqual(len(e['cells'][48]['volume_gaps']),252)

    def test_reference_price_is_consumed_and_clock_binding_exact(self):
        args=fixture();args[3]['adj'].loc[day(args,1,42),'S48']=np.nan;previous.refresh_core(args)
        e=run(args,2)[0][1];self.assertEqual(e['missing_indices'],[48]);self.assertEqual(e['cells'][48]['price_gaps'][-1]['offset'],252)
        for k,v in [('reference_volume_consumed',True),('volume_count',253),('historical_price_end',day(args,1,42))]:
            args=fixture();args[5]['clock_events'][1][k]=v
            with self.assertRaisesRegex(ValueError,'Source clock binding'):run(args,2)

    def test_cached_exact_reference_algebra_tamper_rejected(self):
        args=fixture();f=run(args,2)[0]
        for key in kernel.exact.MEASURE_KEYS[-8:]:
            z=deepcopy(f);z[1]['cells'][0]['measurement'][key]={'numerator':'0','denominator':'1'}
            with self.assertRaises(ValueError):kernel.decisions(args[0],args[1],args[2],z,2)



if __name__=='__main__':unittest.main(verbosity=2)
