"""Synthetic basket-domain, clocks and two early gates; no research inputs."""
from copy import deepcopy
from fractions import Fraction
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_basket_core251_kernel as kernel
import test_monthly_calendar_clock248_v2_kernel as previous


def make_review(meta,calendar,first,bounds=None):
    clocks=[];identities=[];bounds=deepcopy(bounds or []);lookup={d:i for i,d in enumerate(calendar)}
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=lookup[day];partial=day==kernel.INITIAL_PARTIAL
        e=dict(query_index=i,date=day,initial_partial=partial,numeric_evaluation_required=not partial,symbols=names.copy(),query_position=t,next_official_session=calendar[t+1],price_start=calendar[t-294],price_end=calendar[t-42],source_start_position=t-294,source_end_position=t-42,price_count=253,return_count=252,block_count=12,block_sessions=21,block_boundary_dates=[calendar[t-294+21*b] for b in range(13)])
        clocks.append(e)
        for j,s in enumerate(names):
            for b in bounds:
                if s==b['symbol']:identities.append(dict(query_index=i,column_index=j,date=day,symbol=s,issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=day,price_start=e['price_start'],price_end=e['price_end'],price_window_dates_before_bound=sum(d<b['not_before'] for d in calendar[t-294:t-41])))
    return dict(schema=kernel.SOURCE_SCHEMA,panel_first_session=first,clock_events=clocks,listing_bounds=bounds,identity_cells=identities)


def fixture():
    s,a,m,p,c,_=previous.fixture();names=['S'+str(j).zfill(2) for j in range(49)]
    for k,v in list(a.items()):
        if np.shape(v)==(3,6):a[k]=np.asarray(v)[:,np.arange(49)%6].copy()
    # Preserve 27 cached arrays, including fields irrelevant to this mechanism.
    for j in range(27-len(a)):a['synthetic_unchanged_'+str(j)]=np.full((3,49),j,dtype=np.int64)
    m['symbols']=[names.copy() for _ in range(3)];m['etfs']=names[-17:]
    x=np.arange(len(p['adj']),dtype=float);ix=p['adj'].index
    p={'adj':pd.DataFrame({s:90.+(j+1)*.007*x+2*np.sin(x/(11+j))+1.1*np.cos(x/(17+j)) for j,s in enumerate(names)},index=ix)}
    return previous.refresh_core((s,a,m,p,c,make_review(m,c,c[0])))


def physical(args,n):
    s,a,m,p,c,r=args;mm=deepcopy(m)
    for k in ('dates','ends','symbols'):mm[k]=mm[k][:n]
    aa={k:np.asarray(v)[:n].copy() for k,v in a.items()}
    if n:last=mm['dates'][-1];pp={'adj':p['adj'].loc[:last].copy()};cc=c[:c.index(last)+2]
    else:pp={'adj':p['adj'].iloc[:0].copy()};cc=[]
    return s,aa,mm,pp,cc,deepcopy(r)


def run(args,n=None,**kw):return kernel.prepare(*args,len(args[2]['dates']) if n is None else n,**kw)
def day(args,row=1,offset=42):return args[4][args[4].index(args[2]['dates'][row])-offset]
def same(case,a,b):
    case.assertEqual(tuple(a),tuple(b))
    for k in a:case.assertEqual(a[k].dtype,b[k].dtype);case.assertEqual(a[k].shape,b[k].shape);case.assertEqual(a[k].tobytes(),b[k].tobytes())
def scope(args,names):
    args[1]['eligible'][1:]=False
    args[1]['eligible'][1:,names]=True
    return args

class KernelTests(unittest.TestCase):
    def test_schema_and_one_inc_episode(self):
        args=fixture()
        with patch.object(args[0].ep,'select_boundary',wraps=args[0].ep.select_boundary) as ep:f,o,s,p,a=run(args)
        self.assertEqual(ep.call_count,1);self.assertEqual(tuple(o),kernel.ARMS);self.assertEqual(tuple(a['counts']),kernel.COUNT_FIELDS)
        self.assertEqual(a['counts']['original_episode_calls'],1);self.assertEqual(a['counts']['experimental_episode_calls'],0)
        self.assertEqual(a['counts']['source_required_cells'],98);self.assertEqual(a['counts']['source_price_observations'],98*253)
        self.assertTrue(all(tuple(e)==kernel.EVENT_KEYS for e in f));self.assertTrue(all(tuple(c)==kernel.CELL_KEYS for e in f for c in e['cells']))
        self.assertTrue(all(tuple(e)==kernel.AUDIT_EVENT_KEYS for e in a['events']));json.dumps([f,o,a],allow_nan=False)
    def test_original_exact_and_partial(self):
        args=fixture();f,o,s,p,a=run(args);st,aa,m,*_=args;ch,post=st.choices(st.configs()[4],aa,m,3,policy=dict(st.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],st.rp.make_orders(ch,post,m));self.assertEqual(s[kernel.ARMS[0]].tobytes(),post.tobytes())
        self.assertEqual(p[kernel.ARMS[0]].tobytes(),st.modified_pre_score(st.configs()[4],aa,3).tobytes())
        for arm in kernel.ARMS[1:]:self.assertEqual(o[arm][0],o[kernel.ARMS[0]][0]);self.assertEqual(s[arm][0].tobytes(),s[kernel.ARMS[0]][0].tobytes());self.assertEqual(p[arm][0].tobytes(),p[kernel.ARMS[0]][0].tobytes())
        self.assertTrue(all(f[0][k] is None for k in kernel.CANDIDATE_KEYS))
    def test_source_inc_order_and_no_early_sort(self):
        args=fixture();marks=[]
        def source(f):
            self.assertTrue(all(all(e[k] is None for k in kernel.CANDIDATE_KEYS) for e in f));self.assertEqual(ep.call_count,0);self.assertEqual(final.call_count,0);marks.append('source');return {'passed':True}
        def inc(o,s,p):self.assertEqual(ep.call_count,1);self.assertEqual(final.call_count,0);marks.append('inc');return {'passed':True}
        with patch.object(args[0].ep,'select_boundary',wraps=args[0].ep.select_boundary) as ep,patch.object(kernel,'_finalize',wraps=kernel._finalize) as final:
            run(args,source_validator=source,incumbent_validator=inc)
        self.assertEqual(marks,['source','inc'])
    def test_source_gate_failure_zero_episode(self):
        args=fixture()
        with patch.object(args[0].ep,'select_boundary',side_effect=AssertionError('episode')),self.assertRaisesRegex(ValueError,'source gate'):run(args,source_validator=lambda f:{'passed':False})
    def test_inc_gate_failure_zero_basket(self):
        args=fixture()
        with patch.object(kernel.exact,'basket',side_effect=AssertionError('basket')),self.assertRaisesRegex(ValueError,'incumbent gate'):run(args,incumbent_validator=lambda *x:{'passed':False})
    def test_expected_order_failure_before_basket(self):
        args=fixture();expected=run(args)[1][kernel.ARMS[0]];expected[1]['symbols']=[]
        with patch.object(kernel.exact,'basket',side_effect=AssertionError('basket')),self.assertRaises(ValueError):run(args,expected_incumbent_orders=expected)
    def test_all_prefixes_and_cached(self):
        args=fixture();full=run(args)
        for n in range(4):
            logical=run(args,n);phys=run(physical(args,n));cached=kernel.decisions(args[0],args[1],args[2],full[0],n)
            self.assertEqual(logical[0],full[0][:n]);self.assertEqual(logical[0],phys[0]);self.assertEqual(logical[1],phys[1]);self.assertEqual(logical[4],phys[4]);same(self,logical[2],phys[2]);same(self,logical[3],phys[3])
            self.assertEqual(logical[1],cached[0]);self.assertEqual(logical[4],cached[3]);same(self,logical[2],cached[1]);same(self,logical[3],cached[2])
    def test_cached_no_source_and_gate_before_selection(self):
        args=fixture();f,*_=run(args)
        def inc(*x):self.assertEqual(final.call_count,0);return {'passed':True}
        with patch.object(kernel,'_features',side_effect=AssertionError('source')),patch.object(kernel,'_finalize',wraps=kernel._finalize) as final:kernel.decisions(args[0],args[1],args[2],f,3,incumbent_validator=inc)
    def test_partial_no_price_numbers(self):
        args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc[:,:]='bad';f,o,*_=run(args,1);self.assertEqual(f[0]['cells'],[]);self.assertEqual(o[kernel.ARMS[0]],o[kernel.ARMS[1]])
    def test_empty_prefix_width49(self):
        f,o,s,p,a=run(physical(fixture(),0));self.assertEqual(f,[]);self.assertEqual(a['counts']['symbols_per_event'],49);self.assertEqual(a['counts']['original_episode_calls'],1);self.assertEqual(s[kernel.ARMS[1]].shape,(0,49))
    def test_required_exact_E_and_no_other_new_window(self):
        args=scope(fixture(),[0,1,48]);args[3]['adj'].loc[day(args,1,110),'S47']=np.nan;previous.refresh_core(args);f,*_=run(args,2)
        self.assertEqual(f[1]['required_indices'],[0,1,48]);self.assertIsNone(f[1]['cells'][47]['measurement']);self.assertTrue(f[1]['cells'][48]['source_available'])
    def test_E_true_missing_cannot_drop_and_refill(self):
        args=fixture();args[3]['adj'].loc[day(args,1,110),'S48']=np.nan;previous.refresh_core(args)
        with patch.object(kernel.exact,'measure',side_effect=AssertionError('measure')),self.assertRaisesRegex(ValueError,'Required price'):run(args,2)
    def test_source_full_required_scan_before_M(self):
        args=fixture();args[3]['adj'].loc[day(args,1,110),'S00']=np.nan;args[3]['adj'].loc[day(args,1,111),'S48']=0;previous.refresh_core(args)
        with patch.object(kernel.exact,'measure',side_effect=AssertionError('measure')),self.assertRaisesRegex(ValueError,'Positive required price S48'):run(args,2)
    def test_invalid_required_prices(self):
        for value in (0.,-1.,float('inf'),True,'bad',None):
            args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc[day(args,1,110),'S48']=value
            with self.subTest(value=value),self.assertRaises(ValueError):run(args,2)
    def test_origin_clock_and_columns_hard(self):
        for mode in ('origin','row','column'):
            args=fixture()
            if mode=='origin':args[3]['adj']=args[3]['adj'].iloc[1:]
            elif mode=='row':args[3]['adj']=args[3]['adj'].drop(index=day(args,1,110))
            else:args[3]['adj']=args[3]['adj'].drop(columns='S48')
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(args,2)
    def test_core_parity_hard_before_measure(self):
        args=fixture();args[1]['core'][1,0]+=.001
        with patch.object(kernel.exact,'measure',side_effect=AssertionError('measure')),self.assertRaisesRegex(ValueError,'core source mismatch'):run(args,2)
    def test_listing_visible_vs_future(self):
        for visible in (True,False):
            args=fixture();bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2016-01-02' if visible else '2027-01-02',evidence_id='listing',kind='new_issue')
            args[5].update(make_review(args[2],args[4],args[4][0],[bound]))
            if visible:
                with self.assertRaisesRegex(ValueError,'Visible listing contradiction'):run(args,2)
            else:run(args,2)
    def test_listing_Efalse_no_new_consumption(self):
        args=scope(fixture(),[0,1]);bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2016-01-02',evidence_id='listing',kind='new_issue');args[5].update(make_review(args[2],args[4],args[4][0],[bound]));run(args,2)
    def test_source_metadata_tamper(self):
        for mode in ('clock','missing','duplicate','identity'):
            args=fixture()
            if mode=='clock':args[5]['clock_events'][1]['block_boundary_dates'][1]=args[5]['clock_events'][1]['block_boundary_dates'][2]
            elif mode=='missing':args[5]['clock_events'].pop(1)
            elif mode=='duplicate':args[5]['clock_events'].append(deepcopy(args[5]['clock_events'][1]))
            else:args[5]['identity_cells']=[dict(query_index=1,column_index=0)]
            with self.subTest(mode=mode),self.assertRaises(ValueError):run(args,2)
    def test_cash_zero_one_and_two(self):
        for count in (0,1,2):
            args=scope(fixture(),list(range(40,40+count)));f,o,s,p,a=run(args,2);expected='two_names' if count==2 else 'cash';self.assertEqual(f[1]['branch'],expected)
            for arm in kernel.ARMS[1:]:self.assertEqual(len(o[arm][1]['symbols']),count if count==2 else 0);self.assertEqual(f[1]['winners'][arm]['winning_trio_index'],None)
    def test_no_fixed_six_requirement(self):
        for count,trios in ((3,1),(4,4),(5,10),(6,20),(7,20)):
            args=scope(fixture(),list(range(40,40+count)));f,*_=run(args,2);self.assertEqual(len(f[1]['trios']),trios);self.assertEqual(len(f[1]['shortlist_indices']),min(6,count))
    def test_M_nonpositive_excluded_not_E_rewrite(self):
        args=scope(fixture(),[40,41,42]);t=args[4].index(args[2]['dates'][1]);p=args[3]['adj'];p.loc[args[4][t-42],'S40']=p.loc[args[4][t-294],'S40'];previous.refresh_core(args);before=args[1]['eligible'].copy();f,*_=run(args,2)
        self.assertEqual(f[1]['excluded_momentum_indices'],[40]);self.assertNotIn(40,f[1]['new_eligible_indices']);np.testing.assert_array_equal(before,args[1]['eligible'])
    def test_top6_exact_M_symbol_tie(self):
        args=fixture();p=args[3]['adj'];p.loc[:,:]=np.asarray(p['S48'])[:,None];previous.refresh_core(args);f,*_=run(args,2);self.assertEqual(f[1]['shortlist_indices'],list(range(6)))
        self.assertEqual(f[1]['winners'][kernel.ARMS[1]]['selected_symbols'],['S00','S01','S02']);self.assertEqual(f[1]['winners'][kernel.ARMS[1]]['tied_trio_indices'],list(range(20)))
    def test_zero_variance_shared_rawM_top3(self):
        args=fixture();x=np.arange(len(args[3]['adj']));values=np.exp2(x//21);args[3]['adj'].loc[:,:]=values[:,None];previous.refresh_core(args);f,o,*_=run(args,2)
        self.assertEqual(f[1]['branch'],'zero_variance');self.assertEqual(f[1]['common_trio_indices'],[])
        for arm in kernel.ARMS[1:]:self.assertEqual(o[arm][1]['symbols'],['S00','S01','S02']);self.assertIsNone(f[1]['winners'][arm]['winning_trio_index'])
    def test_objectives_direct_not_old_F_or_episode(self):
        args=fixture();first=run(args,2);args[1]['effects'][1]=np.linspace(-100,100,49);args[1]['pre_score'][1]=np.linspace(1,-1,49);second=run(args,2)
        self.assertEqual(first[0],second[0])
        for arm in kernel.ARMS[1:]:self.assertEqual(first[1][arm][1],second[1][arm][1])
    def test_display_only_selected_S_to1(self):
        args=fixture();f,o,s,p,a=run(args)
        for i in (1,2):
            S=len(f[i]['shortlist_indices'])
            for arm in kernel.ARMS[1:]:
                selected=f[i]['winners'][arm]['selected_indices'];expected=np.zeros(49)
                for j in selected:expected[j]=S-f[i]['shortlist_indices'].index(j)
                self.assertEqual(s[arm][i].tobytes(),expected.tobytes());self.assertEqual(p[arm][i].tobytes(),expected.tobytes())
    def test_cached_tamper_rejected_after_inc_gate(self):
        args=fixture();full=run(args);f=deepcopy(full[0]);f[1]['winners'][kernel.ARMS[1]]['selected_indices'].reverse();seen=[]
        with self.assertRaises(ValueError):kernel.decisions(args[0],args[1],args[2],f,3,incumbent_validator=lambda *x:(seen.append('inc') or {'passed':True}))
        self.assertEqual(seen,['inc'])
    def test_cached_measure_or_E_tamper(self):
        args=fixture();f=run(args)[0]
        for mode in ('M','E','type'):
            bad=deepcopy(f)
            if mode=='M':bad[1]['cells'][0]['measurement']['momentum']=kernel.exact._pair(0)
            elif mode=='E':bad[1]['required_indices'].pop()
            else:bad[1]['cells'][0]['original_eligible']=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):kernel.decisions(args[0],args[1],args[2],bad,3)
    def test_future_prefix_poison(self):
        args=fixture();expected=run(args,2);args[2]['dates'][2]='bad';args[2]['symbols'][2]=['bad'];args[2]['ends'][2]='bad'
        args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc['2016-09-01':,:]='bad'
        args[5]['clock_events'][2]={'query_index':2,'bad':True}
        for key in ('core','baseline','effects'):args[1][key]=args[1][key].astype(object);args[1][key][2]='bad'
        actual=run(args,2);self.assertEqual(expected[0],actual[0]);self.assertEqual(expected[1],actual[1]);self.assertEqual(expected[4],actual[4])
    def test_inputs_unchanged(self):
        args=fixture();aa=kernel.old._snapshot(args[1]);m=deepcopy(args[2]);r=deepcopy(args[5]);px=args[3]['adj'].copy(deep=True);run(args);self.assertEqual(aa,kernel.old._snapshot(args[1]));self.assertEqual(m,args[2]);self.assertEqual(r,args[5]);pd.testing.assert_frame_equal(px,args[3]['adj'])
    def test_width_and_extra_panel_rejected(self):
        args=previous.fixture()
        with self.assertRaisesRegex(ValueError,'49'):run(args)
        args=fixture();args[3]['volume']=args[3]['adj'].copy()
        with self.assertRaises(ValueError):run(args)

if __name__=='__main__':unittest.main(verbosity=2)
