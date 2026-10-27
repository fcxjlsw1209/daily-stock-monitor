"""Synthetic source integration only: never load real metadata,NPZ or prices."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_boundary_pair267_source as s


class Poison:
    def __getitem__(self,key):raise AssertionError('Unconsumed body accessed')
    def __iter__(self):raise AssertionError('Unconsumed body iterated')
    def __len__(self):raise AssertionError('Unconsumed body length')
    def __array__(self,*a,**k):raise AssertionError('Unconsumed body coerced')
    def __float__(self):raise AssertionError('Unconsumed body converted')


class GuardArray(np.ndarray):
    def __new__(cls,value,allowed):
        obj=np.asarray(value).view(cls);obj.allowed=set(allowed);return obj
    def __array_finalize__(self,parent):self.allowed=getattr(parent,'allowed',None)
    def __getitem__(self,key):
        k=key[0] if type(key) is tuple else key
        ids=list(range(len(self)))[k] if isinstance(k,slice) else [int(k)]
        if type(ids) is int:ids=[ids]
        if self.allowed is not None and not set(ids)<=self.allowed:raise AssertionError('Future numerical row accessed')
        # Return ordinary arrays: only the first source access is monitored.
        return np.asarray(super().__getitem__(key)) if isinstance(super().__getitem__(key),np.ndarray) else super().__getitem__(key)


def fixture():
    sessions=pd.bdate_range('2016-01-01','2020-02-28')
    def end(y,m):return pd.bdate_range(pd.Timestamp(y,m,1),pd.Timestamp(y,m,1)+pd.offsets.MonthEnd())[-1].date().isoformat()
    def months(y,m,count):
        ds=[];es=[]
        for i in range(count):
            yy=y+(m-1+i)//12;mm=(m-1+i)%12+1;ds.append(end(yy,mm));es.append(end(yy+(mm==12),mm%12+1))
        return ds,es
    td,te=months(2016,1,48);qd,qe=months(2016,8,32)
    qd=[s.INITIAL_PARTIAL]+qd;qe=['2016-08-31']+qe
    names=list('ABCDEF');w=len(names)
    meta=dict(monthly=dict(dates=qd,ends=qe,symbols=[names.copy() for _ in qd],etfs=[]),
              training=dict(dates=td,ends=te,symbols=[names.copy() for _ in td],pool_origins=td.copy()))
    def features(d):
        # Independent of training/query row number; identical same-date calls.
        return np.array([[((pd.Timestamp(d).day+j+k)%9-4)/10. for k in range(10)] for j in range(w)],float)
    train=dict(train_plain=np.vstack([features(d) for d in td]),event_plain=np.vstack([features(d) for d in qd]),
               train_eligible=np.ones(len(td)*w,bool),eligible=np.ones((len(qd),w),bool),offsets=np.arange(len(td)+1)*w)
    arrays=dict(baseline=np.zeros((len(qd),w)),eligible=train['eligible'].copy(),effects=np.zeros((len(qd),w)))
    pre=np.tile(np.array([.6,.5,.4,.39,.38,.1]),(len(qd),1))
    adj=pd.DataFrame(1.+np.arange(len(sessions))[:,None]/10000.+np.arange(w)[None,:]/100.,index=sessions,columns=names)
    join=s.build_join(meta,[d.date().isoformat() for d in sessions],expected_overlap=32)
    return meta,join,train,arrays,pre,dict(adj=adj)


def prepare(args,n=None):
    _,join,train,arrays,pre,panel=args
    return s.prepare_inputs(join,train,arrays,pre,panel,n)


class MetadataTests(unittest.TestCase):
    def test_date_identity_join_and_global_offsets(self):
        meta,j,*_=fixture();self.assertEqual(len(j['rows']),32)
        r=j['rows'][0];self.assertEqual((r['query_index'],r['training_index'],r['training_offset']),(1,7,42))
        self.assertEqual(r['signal'],'2016-08-31');self.assertEqual(r['end'],'2016-09-30')
        self.assertTrue(j['metadata_only']);self.assertFalse(j['source_files_authenticated'])
    def test_conflicting_shared_end_rejected(self):
        meta,j,*_=fixture();meta['training']['ends'][7]='2016-09-29'
        with self.assertRaisesRegex(ValueError,'conflicting maturity'):s.build_join(meta,pd.bdate_range('2016-01-01','2020-02-28').strftime('%Y-%m-%d').tolist())
    def test_identity_reordered_rejected(self):
        meta,*_=fixture();meta['training']['symbols'][7][0:2]=['B','A']
        with self.assertRaisesRegex(ValueError,'Sorted unique'):s.build_join(meta,pd.bdate_range('2016-01-01','2020-02-28').strftime('%Y-%m-%d').tolist())
    def test_identity_changed_rejected(self):
        meta,*_=fixture();meta['training']['symbols'][7][-1]='G'
        with self.assertRaisesRegex(ValueError,'Shared original'):s.build_join(meta,pd.bdate_range('2016-01-01','2020-02-28').strftime('%Y-%m-%d').tolist())
    def test_not_official_monthend_rejected(self):
        meta,*_=fixture();meta['monthly']['dates'][1]='2016-08-30'
        with self.assertRaisesRegex(ValueError,'monthend witness'):s.build_join(meta,pd.bdate_range('2016-01-01','2020-02-28').strftime('%Y-%m-%d').tolist())
    def test_calendar_endpoint_requires_next_session_witness(self):
        meta,*_=fixture();days=pd.bdate_range('2016-01-01','2016-09-30').strftime('%Y-%m-%d').tolist()
        with self.assertRaisesRegex(ValueError,'endpoint monthend'):s.build_join(meta,days)
    def test_overlap_count_not_silently_changed(self):
        meta,*_=fixture()
        with self.assertRaisesRegex(ValueError,'Expected joined'):s.build_join(meta,pd.bdate_range('2016-01-01','2020-02-28').strftime('%Y-%m-%d').tolist(),expected_overlap=113)
    def test_join_mutation_rejected(self):
        a=fixture();a[1]['rows'][0]['training_offset']+=6
        with self.assertRaisesRegex(ValueError,'join checksum'):prepare(a)


class SourceTests(unittest.TestCase):
    def test_mature_union_counts_and_kernel_bodies(self):
        p=prepare(fixture());self.assertEqual(p['audit']['consumed_years'],[2016,2017,2018,2019])
        self.assertEqual(len(p['audit']['consumed_month_indices']),28)
        c=p['audit']['counts'];self.assertEqual(c['training_feature_values_read'],28*6*10)
        self.assertEqual(c['feature_pairs_compared'],28*6*10)
        self.assertEqual(c['selected_signal_prices_read'],28*5);self.assertEqual(c['selected_end_prices_read'],28*5)
        self.assertEqual(set(p['months'][-1]),{'signal','end'})
        self.assertFalse(p['audit']['labels_computed']);self.assertFalse(p['audit']['npz_decoded'])
    def test_n0_and_initial_only_ignore_training_panel_bodies(self):
        for n in (0,1):
            a=fixture();a=list(a);a[2]=Poison();a[5]=Poison();p=prepare(a,n)
            self.assertEqual(p['audit']['consumed_month_indices'],[])
            self.assertEqual(p['audit']['counts']['training_feature_values_read'],0)
            self.assertTrue(all(set(m)=={'signal','end'} for m in p['months']))
    def test_no_mature_dates_does_not_read_training_keys_or_panel(self):
        a=list(fixture());a[2]={k:a[2][k] for k in ('event_plain','eligible')};a[5]=Poison()
        p=prepare(a,6);self.assertEqual(p['audit']['consumed_month_indices'],[])
    def test_maturity_precedes_old_and_future_training_rows(self):
        a=fixture();join=a[1];rows=join['rows'][:4];allowed=[j for r in rows for j in range(r['training_offset'],r['training_offset']+6)]
        for k in ('train_plain','train_eligible'):a[2][k]=GuardArray(a[2][k],allowed)
        p=prepare(a,7);self.assertEqual(p['audit']['consumed_month_indices'],list(range(4)))
    def test_unselected_price_column_never_consumed(self):
        a=fixture();a[5]['adj']['F']=pd.Series([Poison() for _ in range(len(a[5]['adj']))],index=a[5]['adj'].index,dtype=object)
        self.assertEqual(prepare(a)['audit']['counts']['selected_end_prices_read'],140)
    def test_future_price_rows_never_consumed(self):
        a=fixture();frame=a[5]['adj'].astype(object);frame.loc['2019-01-01':,:]=object();a[5]['adj']=frame
        self.assertEqual(prepare(a)['audit']['counts']['selected_end_prices_read'],140)
    def test_missing_selected_prices_preserved_no_replacement(self):
        a=fixture();r=a[1]['rows'][0];a[5]['adj'].at[pd.Timestamp(r['end']),'A']=np.nan
        p=prepare(a);m=p['months'][0]
        self.assertEqual(m['selected_indices'],[0,1,2,3,4]);self.assertTrue(np.isnan(m['end_prices'][0]));self.assertIsNone(m['end_prices'][5])
        # This endpoint is also the following month's signal: two actual reads.
        self.assertEqual(p['audit']['counts']['selected_price_slots_missing'],2)
    def test_missing_endpoint_axis_records_no_actual_value_read(self):
        a=fixture();a[5]['adj']=a[5]['adj'].drop(columns='A');p=prepare(a)
        self.assertEqual(p['audit']['counts']['selected_signal_prices_read'],28*4)
        self.assertEqual(p['audit']['counts']['selected_price_slots_missing'],28*2)
    def test_invalid_price_does_not_fill(self):
        a=fixture();r=a[1]['rows'][0];a[5]['adj'].at[pd.Timestamp(r['end']),'A']=-1.
        p=prepare(a);self.assertEqual(p['months'][0]['end_prices'][0],-1.)
        self.assertEqual(p['audit']['counts']['selected_price_slots_invalid'],2)
    def test_nat_complex_bool_are_not_missing(self):
        for value in (pd.NaT,complex(float('nan'),0),True):
            a=fixture();a[5]['adj']=a[5]['adj'].astype(object);r=a[1]['rows'][0]
            a[5]['adj'].at[pd.Timestamp(r['end']),'A']=value
            with self.assertRaisesRegex(ValueError,'real nonbool'):prepare(a)
    def test_fewer_than_four_no_price_values(self):
        a=fixture();a[3]['eligible'][:,3:]=False;a[2]['eligible'][:,3:]=False
        a[2]['train_eligible'].reshape(-1,6)[:,3:]=False;a[5]['adj']=a[5]['adj'].astype(object);a[5]['adj'].iloc[:,:]=object()
        p=prepare(a);self.assertEqual(p['audit']['counts']['selected_signal_prices_read'],0)
    def test_no_training_gap_filter(self):
        a=fixture();a[4][:,3]=.2;a[4][:,4]=.19
        self.assertEqual(len(prepare(a)['audit']['consumed_month_indices']),28)
    def test_training_query_float_dtype_mismatch(self):
        a=fixture();a[2]['train_plain']=a[2]['train_plain'].astype('float32')
        with self.assertRaisesRegex(ValueError,'shape/dtype'):prepare(a)
    def test_feature_one_ulp_mismatch(self):
        a=fixture();j=a[1]['rows'][0]['training_offset'];a[2]['train_plain'][j,0]=np.nextafter(a[2]['train_plain'][j,0],np.inf)
        with self.assertRaisesRegex(ValueError,'plain10 exact bits'):prepare(a)
    def test_signed_zero_mismatch(self):
        a=fixture();r=a[1]['rows'][0];a[2]['train_plain'][r['training_offset'],0]=-0.;a[2]['event_plain'][r['event_offset'],0]=0.
        with self.assertRaisesRegex(ValueError,'plain10 exact bits'):prepare(a)
    def test_equal_nan_payload_passes_unequal_payload_rejects(self):
        a=fixture();r=a[1]['rows'][0];v=np.array([0x7ff8000000000001],dtype=np.uint64).view(np.float64)[0]
        a[2]['train_plain'][r['training_offset'],0]=v;a[2]['event_plain'][r['event_offset'],0]=v
        prepare(a);a[2]['event_plain'][r['event_offset'],0]=np.nan
        with self.assertRaisesRegex(ValueError,'plain10 exact bits'):prepare(a)
    def test_training_eligibility_mismatch(self):
        a=fixture();a[2]['train_eligible'][a[1]['rows'][0]['training_offset']]=False
        with self.assertRaisesRegex(ValueError,'eligibility exact bits'):prepare(a)
    def test_query_eligibility_mismatch(self):
        a=fixture();a[2]['eligible'][1,0]=False
        with self.assertRaisesRegex(ValueError,'query E exact bits'):prepare(a)
    def test_only_adj_accepted_no_execution_price_substitution(self):
        a=list(fixture());a[5]={'px':a[5]['adj']}
        with self.assertRaisesRegex(ValueError,'Only original panel adj'):prepare(a)
    def test_boolean_negative_and_too_large_prefix_rejected(self):
        for n in (True,-1,34):
            with self.assertRaises(ValueError):prepare(fixture(),n)
    def test_physical_small_and_full_inputs_equivalent(self):
        a=fixture();full=prepare(a)
        for n in (0,1,7,19,31,33):
            small=list(a);small[3]={k:v[:n].copy() for k,v in a[3].items()};small[4]=a[4][:n].copy()
            fresh=prepare(small,n);cached=s.project_inputs(full,n)
            self.assertEqual(s._record_hash(fresh['months']),s._record_hash(cached['months']))
            self.assertEqual(s._record_hash(fresh['events']),s._record_hash(cached['events']))


class ProjectionTests(unittest.TestCase):
    def test_cached_future_poison_n0_and_nonzero(self):
        full=prepare(fixture())
        for n in (0,1,7):
            p=dict(full,months=list(full['months']),events=list(full['events']))
            consumed=set(s._mature(p['binding'].headers,sorted({int(d[:4]) for d in p['binding'].query_dates[:n] if d!=s.INITIAL_PARTIAL})))
            for i in range(len(p['months'])):
                if i not in consumed:p['months'][i]={'bad':Poison()}
            for i in range(n,len(p['events'])):p['events'][i]={'bad':Poison()}
            with self.assertRaises(ValueError):s._record_hash(dict(months=p['months'],events=p['events']))
            got=s.project_inputs(p,n);self.assertEqual(len(got['events']),n)
            self.assertEqual(got['audit']['counts']['selected_signal_prices_read'],0)
    def test_consumed_month_tamper_rejected(self):
        p=prepare(fixture());p['months'][0]['signal_prices'][0]+=1.
        with self.assertRaisesRegex(ValueError,'Consumed month'):s.project_inputs(p,7)
    def test_consumed_event_tamper_rejected(self):
        p=prepare(fixture());p['events'][0]['effects'][0]=.1
        with self.assertRaisesRegex(ValueError,'Consumed event'):s.project_inputs(p,1)
    def test_bare_or_copied_binding_rejected(self):
        p=prepare(fixture());b=p['binding']
        with self.assertRaises(TypeError):deepcopy(b)
        p['binding']={'anchor_sha256':b.anchor_sha256}
        with self.assertRaisesRegex(ValueError,'Issued source'):s.project_inputs(p,0)
    def test_no_returns_or_model_imports(self):
        self.assertFalse(hasattr(s,'fit_models'));self.assertFalse(hasattr(s,'load_arrays'))
        p=prepare(fixture());self.assertFalse(p['audit']['labels_computed']);self.assertEqual(p['audit']['models_fitted'],0)


class OriginalPrefixTests(unittest.TestCase):
    def test_fullP_not_upstream_cache_and_exact_reference(self):
        a=fixture();meta,_,_,arrays,pre,_=a;arrays['pre_score']=pre-.1
        class Study:
            def configs(self):return [None]*4+[dict(id=4,base='incumbent_scope67',mode='local_linear')]
            def modified_pre_score(self,spec,x,n):return pre[:n].copy()
        ref=dict(pre_scores=pre.copy(),eligible=arrays['eligible'].copy(),meta=meta['monthly'])
        p,e,proof=s.original_prefix(Study(),arrays,meta['monthly'],ref,7)
        self.assertEqual(p.tobytes(),pre[:7].tobytes());self.assertTrue(proof['full_local67_p_exact'])
        class Bad(Study):
            def modified_pre_score(self,spec,x,n):return x['pre_score'].copy()
        with self.assertRaisesRegex(ValueError,'P/reference exact bits'):s.original_prefix(Bad(),arrays,meta['monthly'],ref,7)
    def test_original_prefix_mutation_rejected(self):
        a=fixture();meta,_,_,arrays,pre,_=a
        class Study:
            def configs(self):return [None]*4+[dict(id=4,base='incumbent_scope67',mode='local_linear')]
            def modified_pre_score(self,spec,x,n):x['effects'][0,0]=1.;return pre[:n].copy()
        ref=dict(pre_scores=pre,eligible=arrays['eligible'],meta=meta['monthly'])
        with self.assertRaisesRegex(ValueError,'mutated'):s.original_prefix(Study(),arrays,meta['monthly'],ref,7)
        self.assertEqual(arrays['effects'][0,0],0.)
    def test_original_helper_only_receives_consumed_arrays(self):
        a=fixture();meta,_,_,arrays,pre,_=a
        class Study:
            def configs(self):return [None]*4+[dict(id=4,base='incumbent_scope67',mode='local_linear')]
            def modified_pre_score(self,spec,x,n):
                assert all(len(v)==n for v in x.values());return pre[:n].copy()
        for n in (0,1,7):
            guarded={k:GuardArray(v,range(n)) for k,v in arrays.items()}
            ref=dict(pre_scores=GuardArray(pre,range(n)),eligible=GuardArray(arrays['eligible'],range(n)),meta=meta['monthly'])
            s.original_prefix(Study(),guarded,meta['monthly'],ref,n)


class KernelSourceSeam(unittest.TestCase):
    def test_synthetic_kernel_consumes_exact_source_and_missing_month(self):
        import monthly_boundary_pair267_kernel as k
        a=fixture();p=prepare(a);co=k.build_cohort(p['months'],2019)
        self.assertEqual(co['common_months'],28);self.assertEqual(len(co['arms'][k.ARMS[0]]['y']),280)
        self.assertEqual(len(co['arms'][k.ARMS[1]]['y']),28)
        r=a[1]['rows'][0];a[5]['adj'].at[pd.Timestamp(r['end']),'A']=np.nan
        # A shared endpoint invalidates both adjacent months, without replacement.
        p=prepare(a);co=k.build_cohort(p['months'],2019);self.assertEqual(co['common_months'],26)
        self.assertEqual(co['months'][0]['selected_indices'],[0,1,2,3,4])
    def test_cached_prefix_kernel_does_not_touch_unmatured_body(self):
        import monthly_boundary_pair267_kernel as k
        full=prepare(fixture());prefix=s.project_inputs(full,7)
        self.assertEqual(k.build_cohort(prefix['months'],2017)['common_months'],4)
        partial=s.project_inputs(full,1)
        self.assertEqual(k.build_cohort(partial['months'],2016)['common_months'],0)


if __name__=='__main__':unittest.main(verbosity=2)
