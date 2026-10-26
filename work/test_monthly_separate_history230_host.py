"""Synthetic230 host wiring only: no actual cached arrays, forecasts or returns."""
from pathlib import Path
import ast
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import monthly_separate_history230_host as h
import monthly_separate_history230_model as model
import monthly_separate_history230_independent as oracle
import monthly_expanded_error228_adapter as adapter
import monthly_expanded_error228_independent as adapter_oracle
import test_monthly_expanded_error228_host as oldtests
from test_monthly_separate_history230_model import fixture as model_fixture, partial_fixture
from test_monthly_joint_error226_model import fixture as old_fixture


def snap(value):
    if isinstance(value,dict):return {k:snap(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [snap(v) for v in value]
    a=np.asarray(value)
    return a.dtype.str,a.shape,a.tobytes()


def bundle(value=None):
    args,current=copy.deepcopy(model_fixture() if value is None else value)
    tr,tm,m,hm,src,use,pre,E=args;n,w=E.shape;early_n=len(hm['dates'])-n
    late={key:np.asarray(v)[early_n*w:].copy() for key,v in src.items()}
    oldargs=(tr,tm,m,late,use[early_n:].copy() & E,pre,E)
    supplied,sm=oldtests.adapter_fixture(oldargs)
    adapted=adapter.adapt(*supplied,source_meta227=sm)
    ac=adapter_oracle.check_adapter(*supplied,adapted,source_meta227=sm)
    adapter_oracle.require_receipt(ac,adapted['audit'])
    labels=supplied[5];q=early_n*w
    em=dict(history=h.early_metadata_prefix(tm,early_n),current_query_axis_unchanged=True,
        training_reference=str(h.REBUILD/'MONTHLY_META.json'))
    early={key:np.asarray(src['raw_local_linear'])[:q].copy() for key in adapter.RAW_KEYS}
    early.update(available=np.asarray(src['available'])[:q].copy(),local_available=np.zeros(q,bool),
        query_eligible=adapted['train']['train_eligible'][:q].copy(),expansion_allowed=np.ones(q,bool))
    el={key:np.asarray(v)[:early_n+1 if key=='offsets' else q].copy() for key,v in labels.items()}
    joinargs=[tm,m,em,early,el,labels,adapted]
    joined=h.join_history(*joinargs)
    arrays,_,_=oldtests.original_fixture(oldargs)
    return dict(joinargs=joinargs,joined=joined,arrays=arrays,meta=m,adapter=adapted,
        adapter_check=ac,original=supplied,source_meta=sm)


def rebuilt(b):
    j=b['joined'];a=b['arrays'];m=b['meta'];study=h.local()
    pre=study.modified_pre_score(study.configs()[4],a,len(m['dates']))
    inputs=(b['adapter']['train'],b['joinargs'][0],m,j['history_meta'],j['history_source'],
        j['historical_use'],pre,a['eligible'])
    built=model.build(*inputs,current_available=j['current_available'])
    return inputs,j['current_available'],built


def selection_fixture():
    metrics,orders=oldtests.selection_fixture()
    rename=dict(zip(oldtests.h.NAMES,h.NAMES))
    return ({rename[k]:v for k,v in metrics.items()},{rename[k]:v for k,v in orders.items()})


def full_fixture():
    # Invented natural-month axes only. No real prices, labels or cache values.
    current=[((2016*12+7+i)//12,(2016*12+7+i)%12+1) for i in range(121)]
    tr,tm,hm,src,use,pre,E=old_fixture(n_history=67,start=(2011,1),current=current)
    width=49;oldwidth=6;syms=list('ABCDEF')+['X%02d'%i for i in range(43)]
    months=len(tm['dates'])
    for key in list(tr):
        if key in ('dates','ends','offsets'):continue
        arr=tr[key].reshape((months,oldwidth)+tr[key].shape[1:])
        extra=np.repeat(arr[:,:1],43,axis=1)
        tr[key]=np.concatenate((arr,extra),axis=1).reshape((months*width,)+arr.shape[2:])
    for meta in (tm,hm):meta['symbols']=[syms.copy() for _ in meta['dates']]
    tr['dates']=np.repeat(np.asarray(tm['dates']),width);tr['ends']=np.repeat(np.asarray(tm['ends']),width)
    tr['offsets']=np.arange(months+1,dtype=int)*width
    for key,value in src.items():
        arr=value.reshape(-1,oldwidth)
        src[key]=np.concatenate((arr,np.repeat(arr[:,:1],43,axis=1)),axis=1).reshape(-1)
    pre=np.c_[pre,np.full((len(pre),43),-1.)];E=np.c_[E,np.zeros((len(E),43),bool)]
    use=np.ones(pre.shape,bool)
    for key in ('dates','ends','symbols'):
        hm[key].insert(67,model.INITIAL if key=='dates' else '2016-08-31' if key=='ends' else syms.copy())
    for key,value in src.items():src[key]=np.insert(value,67*width,value[67*width:68*width])
    pre=np.insert(pre,67,pre[67],axis=0);E=np.insert(E,67,E[67],axis=0);use=np.insert(use,67,use[67],axis=0)
    m={key:copy.deepcopy(v[67:]) for key,v in hm.items()}
    args=[tr,tm,m,hm,src,use,pre[67:],E[67:]]
    return bundle((args,np.ones((122,width),bool)))


def full_receipt_fixture():
    b=full_fixture();inputs,current,built=rebuilt(b);a=b['arrays'];m=b['meta'];j=b['joined'];study=h.local()
    orders,scores,choices=h.decisions(m,a,built,122,study=study)
    for n in range(123):
        aa=h.slice_arrays(a,n);mm={k:v[:n] for k,v in m.items()}
        assert len(aa)==27 and all(v.shape[0]==n for v in aa.values())
        oo,ss,cc=h.decisions(mm,aa,h.slice_built(built,n),n,study=study)
        assert oo=={arm:orders[arm][:n] for arm in h.NAMES}
        for arm in h.NAMES:np.testing.assert_array_equal(ss[arm],scores[arm][:n])
    check=oracle.check_build(*inputs,built,current_available=current)
    selection=oracle.verify_selection(a,m,built,scores,choices);prefix=[]
    for n in h.MODEL_PREFIXES:
        base,ca=h.prefix_inputs(*inputs[:6],current,n);short=base+(inputs[6][:n],inputs[7][:n])
        small=model.build(*short,current_available=ca);aa=h.slice_arrays(a,n)
        oo,ss,cc=h.decisions(short[2],aa,small,n,study=study)
        assert oo=={arm:orders[arm][:n] for arm in h.NAMES}
        assert small['audit']['event_history']==built['audit']['event_history'][:n]
        for key in model.ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][key],built['arrays'][key][:n])
        prefix.append(dict(prefix=n,query_rows=n,training_rows=len(short[0]['dates']),history_rows=len(short[3]['dates']),
            history_last_date=short[3]['dates'][-1] if short[3]['dates'] else None,current_available_rows=int(ca.sum()),
            query_date=m['dates'][n-1] if n else None,cutoff='%04d-12-31'%(int(m['dates'][n-1][:4])-1) if n else None,
            latest_label_end=str(np.asarray(short[0]['ends'],dtype='datetime64[D]').max()) if len(short[0]['ends']) else None,
            source_history_starts_at_original_zero=True,counts=small['audit']['counts'],
            model=oracle.check_build(*short,small,current_available=ca),selection=oracle.verify_selection(aa,short[2],small,ss,cc)))
    audit=dict(model=check,selection=selection,prefixes=123,model_prefixes=list(h.MODEL_PREFIXES),
        model_prefix_evidence=prefix,original_array_count=27,model_counts=built['audit']['counts'])
    for key in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
        'model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact','original_arrays_unchanged',
        'original_source_arrays_unchanged','original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget',
        'historical_expansion_contract_checked','error_prediction_source_local_linear','separate_history_axis',
        'source229_labels_exact_copy','current_available_counts_separate'):audit[key]=True
    parity={key:True for key in ('original_training_cache_exact','original_identity_exact','cached206_source_vectors_exact',
        'cached206_control_constant_exact','source_availability_exact','historical_expansion_contract_checked','no_source_model_refits','no_fitted_residual_error_inputs')}
    parity.update(fresh_training_array_comparison=False,new_labels=False,new_raw_features=False,error_prediction_source='raw_local_linear',
        target_source='source227_original_center_extended_log_excess',risk_history_adapter=b['adapter']['audit'],independent_adapter=b['adapter_check'],
        original_ridge_training_unchanged=True,current_trading_eligibility_unchanged=True,history_join=j['audit'],history_meta=j['history_meta'],
        historical_mask_semantics='early229.available then originalU OR newlyavailable227; historical_original_universe_checked verifies the supplied mask')
    completion=dict(artifacts_sha256={key:'synthetic digest' for key in h.PREPARED_FILES},
        original_array_count=27,new_returns_computed=False,source_model_refits=0,model_counts=built['audit']['counts'])
    return completion,m,audit,parity,built


class JoinTests(unittest.TestCase):
    def test_exact_join_preserves_current_and_copies_all_outputs(self):
        b=bundle();args=b['joinargs'];before=snap(args);j=h.join_history(*args);w=j['audit']['width'];early=j['audit']['early_months']
        self.assertEqual(snap(args),before);self.assertEqual(j['audit']['history_months'],early+len(b['meta']['dates']))
        self.assertFalse(j['audit']['labels_appended']);self.assertFalse(j['audit']['source_forecasts_refitted'])
        np.testing.assert_array_equal(j['current_available'],b['adapter']['source']['available'].reshape(-1,w))
        for key in j['history_source']:
            np.testing.assert_array_equal(j['history_source'][key][early*w:],b['adapter']['source'][key])
            self.assertFalse(np.shares_memory(j['history_source'][key],b['adapter']['source'][key]))
        self.assertFalse(np.shares_memory(j['historical_use'],b['adapter']['historical_use']))
        self.assertFalse(np.shares_memory(j['current_available'],b['adapter']['source']['available']))
    def test_exact_metadata_identity_and_order_required(self):
        for field in ('dates','ends','symbols'):
            a=bundle()['joinargs']
            a[2]['history'][field][0]=(list(reversed(a[2]['history'][field][0])) if field=='symbols'
                else copy.deepcopy(a[2]['history'][field][1]))
            with self.subTest(field=field),self.assertRaises(ValueError):h.join_history(*a)
        a=bundle()['joinargs'];a[2]['current_query_axis_unchanged']=False
        with self.assertRaises(ValueError):h.join_history(*a)
    def test_every_early_label_field_requires_original_prefix_bytes(self):
        for key in adapter.LABEL_KEYS:
            a=bundle()['joinargs'];v=a[4][key]
            if v.dtype.kind=='b':v[0]=~v[0]
            elif v.dtype.kind in 'fiu':v[0]+=1
            else:v[0]='1999-01-01'
            with self.subTest(key=key),self.assertRaises(ValueError):h.join_history(*a)
        a=bundle()['joinargs'];a[4]['vol63']=a[4]['vol63'].astype('float32')
        with self.assertRaises(ValueError):h.join_history(*a)
    def test_vol63_matches_227_labels_not_untouched_original_training(self):
        a=bundle()['joinargs'];a[6]['train']['vol63']=np.full(len(a[6]['train']['dates']),99.)
        a[4]['vol63'][:]=2.;a[5]['vol63'][:len(a[4]['vol63'])]=2.
        result=h.join_history(*a);self.assertFalse(result['audit']['labels_appended'])
        self.assertTrue((a[6]['train']['vol63']==99.).all())
    def test_all_four_risk_targets_and_good_must_be_full227(self):
        for key in ('common_good',)+tuple('target_'+k for k in h.TARGETS):
            a=bundle()['joinargs'];v=a[6]['train'][key];v[-1]=not v[-1] if v.dtype.kind=='b' else v[-1]+1
            with self.subTest(key=key),self.assertRaises(ValueError):h.join_history(*a)
    def test_exact_forecast_schema_and_boolean_masks(self):
        for key in ('available','local_available','query_eligible','expansion_allowed'):
            a=bundle()['joinargs'];a[3][key]=a[3][key].astype(int)
            with self.subTest(key=key),self.assertRaises(ValueError):h.join_history(*a)
        a=bundle()['joinargs'];a[3]['unregistered']=np.zeros(1)
        with self.assertRaises(ValueError):h.join_history(*a)
        a=bundle()['joinargs'];a[3]['available'][0]=False
        for key in adapter.RAW_KEYS:a[3][key][0]=np.nan
        a[3]['local_available'][0]=True
        with self.assertRaises(ValueError):h.join_history(*a)
    def test_raw_support_nan_infinity_dtype_and_length_rejected(self):
        for key in adapter.RAW_KEYS:
            for change in ('nan','inf','dtype','length'):
                a=bundle()['joinargs']
                if change=='nan':a[3][key][0]=np.nan
                elif change=='inf':a[3][key][0]=np.inf
                elif change=='dtype':a[3][key]=a[3][key].astype('float32')
                else:a[3][key]=a[3][key][:-1]
                with self.subTest(key=key,change=change),self.assertRaises(ValueError):h.join_history(*a)
    def test_early_original_or_warm_domain_and_current_use_are_distinct(self):
        a=bundle()['joinargs'];a[3]['query_eligible'][0]=False;a[6]['train']['train_eligible'][0]=False
        j=h.join_history(*a);self.assertTrue(j['historical_use'][0,0])
        a[3]['expansion_allowed'][0]=False
        with self.assertRaisesRegex(ValueError,'union'):h.join_history(*a)
        a=bundle()['joinargs'];a[6]['historical_use'][0,0]=False
        j=h.join_history(*a);self.assertFalse(j['historical_use'][-2,0]);self.assertTrue(j['current_available'][0,0])
    def test_cold_early_source_not_imputed_and_current_count_not_shifted(self):
        a=bundle()['joinargs'];a[3]['available'][:]=False
        for key in adapter.RAW_KEYS:a[3][key][:]=np.nan
        j=h.join_history(*a);self.assertEqual(j['audit']['early_history_use_rows'],0)
        self.assertTrue(j['current_available'].all());self.assertFalse(j['historical_use'][:-2].any())
        a=bundle()['joinargs'];a[3]['query_eligible'][0]=False
        with self.assertRaisesRegex(ValueError,'eligibility'):h.join_history(*a)


class PrefixTests(unittest.TestCase):
    def test_history_is_date_prefix_not_current_count_and_labels_mature(self):
        b=bundle();inputs,current,full=rebuilt(b)
        for n in range(len(b['meta']['dates'])+1):
            base,ca=h.prefix_inputs(*inputs[:6],current,n);short=base+(inputs[6][:n],inputs[7][:n])
            small=model.build(*short,current_available=ca)
            self.assertEqual(len(base[2]['dates']),n);self.assertEqual(len(base[3]['dates']),18+n if n else 0)
            self.assertEqual(len(base[4]['available']),len(base[3]['dates'])*current.shape[1])
            for key in model.ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][key],full['arrays'][key][:n])
            self.assertEqual(small['audit']['event_history'],full['audit']['event_history'][:n])
            h.require_model_audit(oracle.check_build(*short,small,current_available=ca),small,n,len(base[3]['dates']))
            if n:self.assertTrue((np.asarray(base[0]['ends'])<=str(int(base[2]['dates'][-1][:4])-1)+'-12-31').all())
            else:self.assertEqual(base[0]['offsets'].tolist(),[0]);self.assertEqual(len(base[0]['dates']),0)
    def test_initial_partial_at_history67_and_terminal_partial_prefix(self):
        b=bundle(partial_fixture());inputs,current,full=rebuilt(b)
        base,ca=h.prefix_inputs(*inputs[:6],current,1)
        self.assertEqual(len(base[3]['dates']),68);self.assertEqual(base[3]['dates'][-1],model.INITIAL)
        small=model.build(*base,inputs[6][:1],inputs[7][:1],current_available=ca)
        self.assertEqual(small['audit']['event_history'][0]['status'],'initial_partial')
        self.assertFalse(small['arrays']['ready'][0]);self.assertEqual(full['audit']['history_months'],70)
    def test_future_poison_removed_from_history_and_mature_training(self):
        b=bundle();inputs,current,full=rebuilt(b);a=copy.deepcopy(inputs);n=1;hm=a[3]
        hn=sum(x<=a[2]['dates'][n-1] for x in hm['dates']);w=current.shape[1]
        for key in a[4]:a[4][key]=a[4][key].astype(object);a[4][key][hn*w:]='future poison'
        a=list(a);a[5]=a[5].astype(object);a[5][hn:]='future poison'
        future=np.asarray(a[0]['ends'])>str(int(a[2]['dates'][0][:4])-1)+'-12-31'
        for key in ('train_plain',)+tuple('target_'+k for k in h.TARGETS):
            a[0][key]=a[0][key].astype(object);a[0][key][future]='future poison'
        base,ca=h.prefix_inputs(*a[:6],current,n);small=model.build(*base,a[6][:n],a[7][:n],current_available=ca)
        for key in model.ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][key],full['arrays'][key][:n])
    def test_prefix_bad_axes_and_flat_training_identity_rejected(self):
        b=bundle();a,current,_=rebuilt(b)
        for key in ('offsets','dates'):
            bad=copy.deepcopy(a)
            if key=='offsets':bad[0][key][1]+=1
            else:bad[0][key][0]='1999-01-01'
            with self.subTest(key=key),self.assertRaises(ValueError):h.prefix_inputs(*bad[:6],current,1)
        for ca in (current.reshape(-1),current.astype(int)):
            with self.assertRaises(ValueError):h.prefix_inputs(*a[:6],ca,1)
    def test_slice_built_requires_explicit_physical_history_count(self):
        b=bundle();a,current,full=rebuilt(b)
        for n in range(3):
            base,ca=h.prefix_inputs(*a[:6],current,n);short=base+(a[6][:n],a[7][:n]);hn=len(base[3]['dates'])
            sliced=h.slice_built(full,n,history_months=hn);check=oracle.check_build(*short,sliced,current_available=ca)
            h.require_model_audit(check,sliced,n,hn)
        self.assertEqual(h.slice_built(full,1)['audit']['history_months'],20)
        with self.assertRaises(ValueError):h.slice_built(full,1,history_months=21)


class DecisionTests(unittest.TestCase):
    def test_real_frozen_selector_single_episode_and_independent_choices(self):
        b=bundle();a,current,built=rebuilt(b);study=h.local();n=len(b['meta']['dates'])
        with patch.object(study.ep,'select_boundary',wraps=study.ep.select_boundary) as episode:
            orders,scores,choices=h.decisions(b['meta'],b['arrays'],built,n,study=study)
        self.assertEqual(episode.call_count,3);self.assertNotEqual(choices[h.NAMES[1]][-1],choices[h.NAMES[2]][-1])
        h.require_selection_audit(oracle.verify_selection(b['arrays'],b['meta'],built,scores,choices),n)
        for arm in h.NAMES[1:]:self.assertTrue({0,1}<=set(choices[arm][-1]))
    def test_market_states_use_original_funded_budget_no_addition(self):
        for calm,weak in ((False,False),(True,False),(False,True),(True,True)):
            b=bundle();b['arrays']['state_calm10'][-1]=calm;b['arrays']['state_weak_narrow'][-1]=weak
            a,current,built=rebuilt(b);_,scores,choices=h.decisions(b['meta'],b['arrays'],built,len(b['meta']['dates']))
            h.require_selection_audit(oracle.verify_selection(b['arrays'],b['meta'],built,scores,choices),2)
            budget=min(.05,max(0.,.10-(.025*calm+.05*weak)));aa=b['arrays'];support=(aa['baseline']-aa['core'])/(1-aa['core_weight'])
            for arm,key in zip(h.NAMES[1:],('candidate','control')):
                want=a[6][-1].copy()
                for j in built['audit']['event_history'][-1]['J']:
                    want[j]+=budget*(np.clip(support[-1,j]+built['arrays'][key+'_effect'][-1,j],-.5,.5)-support[-1,j])
                np.testing.assert_array_equal(scores[arm][-1],want)
    def test_full_support_constant_effect_zero_is_not_incumbent_fallback(self):
        b=bundle();b['joined']['history_source']['raw_local_linear'][:]=1.
        a,current,built=rebuilt(b);self.assertTrue(built['arrays']['ready'].all())
        _,scores,choices=h.decisions(b['meta'],b['arrays'],built,2)
        for arm in h.NAMES[1:]:np.testing.assert_array_equal(scores[arm],a[6])
        self.assertNotEqual(scores[h.NAMES[0]][-1].tolist(),a[6][-1].tolist())
    def test_missing_common_history_joint_fallback_and_current_eligibility(self):
        b=bundle();b['joined']['historical_use'][0,4]=False
        a,current,built=rebuilt(b);self.assertFalse(built['arrays']['ready'].any())
        orders,scores,_=h.decisions(b['meta'],b['arrays'],built,2)
        for arm in h.NAMES[1:]:self.assertEqual(orders[arm],orders[h.NAMES[0]]);np.testing.assert_array_equal(scores[arm],scores[h.NAMES[0]])
        b=bundle();b['arrays']['eligible'][-1,4]=False;a,current,built=rebuilt(b)
        _,_,choices=h.decisions(b['meta'],b['arrays'],built,2)
        for arm in h.NAMES:self.assertNotIn(4,choices[arm][-1])
    def test_order_prefixes_and_all27_input_bytes_unchanged(self):
        b=bundle(partial_fixture());a,current,built=rebuilt(b);before=snap((b['arrays'],built));m=b['meta'];n=len(m['dates'])
        full=h.decisions(m,b['arrays'],built,n)
        for count in range(n+1):
            aa=h.slice_arrays(b['arrays'],count);mm={k:v[:count] for k,v in m.items()}
            self.assertEqual(len(aa),27);self.assertTrue(all(v.shape[0]==count for v in aa.values()))
            orders,scores,choices=h.decisions(mm,aa,h.slice_built(built,count),count)
            self.assertEqual(orders,{arm:full[0][arm][:count] for arm in h.NAMES})
            for arm in h.NAMES:np.testing.assert_array_equal(scores[arm],full[1][arm][:count])
        self.assertEqual(snap((b['arrays'],built)),before)
    def test_future_selector_data_not_consumed(self):
        b=bundle();a,current,built=rebuilt(b);arrays=b['arrays'];m=b['meta'];expected=h.decisions(m,arrays,built,1)
        for key,v in arrays.items():
            if v.dtype.kind=='b':arrays[key]=v.astype(object);arrays[key][1:]='future poison'
            else:v[1:]=np.inf
        for key in ('dates','ends','symbols'):m[key][1:]=[object()]*(len(m[key])-1)
        for key in model.ARRAY_KEYS:
            if key!='ready':built['arrays'][key][1:]=np.inf
        built['audit']['event_history'][1:]=[object()]
        got=h.decisions(m,arrays,built,1);self.assertEqual(got[0],expected[0])
        for arm in h.NAMES:np.testing.assert_array_equal(got[1][arm],expected[1][arm])


class SelectionGateTests(unittest.TestCase):
    def test_first88_only_and_no_report_only_later_influence(self):
        metrics,orders=selection_fixture();selected=h.select_first88(metrics,orders)
        self.assertEqual(selected['primary'],h.NAMES[1])
        for arm in h.NAMES:
            metrics[arm]['later34']={'cagr':1000.};metrics[arm]['full122']={'cagr':-1.}
            for i in range(88,122):orders[arm][i]['symbols']=[arm+str(i)]
        self.assertEqual(h.select_first88(metrics,orders),selected)
    def test_diagonal_control_never_promotes_and_cost_guard_required(self):
        for path,key in (('same_close_zero','close_vs_diagonal_error_control'),('next_open_cost','open_vs_diagonal_error_control')):
            metrics,orders=selection_fixture();metrics[h.NAMES[2]][path]['training']['cagr']=2.
            got=h.select_first88(metrics,orders)
            self.assertEqual(got['primary'],h.NAMES[0]);self.assertFalse(got['guards'][key])
    def test_candidate_must_differ_from_both_reference_sets_in_both_eras(self):
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        got=h.select_first88(metrics,orders);self.assertFalse(got['guards']['changed_support_vs_control'])
        metrics,orders=selection_fixture()
        for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['C','B','A']
        for i in (90,100,110):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
        got=h.select_first88(metrics,orders);self.assertFalse(got['guards']['changed_support']);self.assertEqual(got['changed_months'],[])
    def test_each_return_risk_and_era_guard_is_material(self):
        for path in ('same_close_zero','next_open_cost'):
            for era,key,value in (('training','cagr',.49),('training','sharpe',1.44),('training','max_drawdown',-.331),('early','cagr',.474),('recent','cagr',.474)):
                metrics,orders=selection_fixture();metrics[h.NAMES[1]][path][era][key]=value
                with self.subTest(path=path,era=era,key=key):self.assertEqual(h.select_first88(metrics,orders)['primary'],h.NAMES[0])
    def test_freeze_binds_plan_orders_training_and_preparation(self):
        metrics,orders=selection_fixture();frozen=dict(plan_sha256='plan',orders_sha256='digest',training_sha256='digest',prepared_sha256='digest',selection=h.select_first88(metrics,orders))
        def reader(p):return frozen if p.name=='FROZEN.json' else metrics
        with patch.object(h,'sha',return_value='digest'),patch.object(h,'read',side_effect=reader):
            self.assertEqual(h.frozen_selection('plan',orders),frozen)
            for key in ('plan_sha256','orders_sha256','training_sha256','prepared_sha256'):
                old=frozen[key];frozen[key]='changed'
                with self.subTest(key=key),self.assertRaises(ValueError):h.frozen_selection('plan',orders)
                frozen[key]=old




original_fixture=oldtests.original_fixture

class SourceAndIdentityTests(unittest.TestCase):
    def test_history_use_is_original_forecast_and_eligible_not_local_or_current_model(self):
        a,m,args=original_fixture();a['forecast_available'][1,0]=False;a['eligible'][1,1]=False
        a['model_available'][1,2]=False;a['local_available'][:]=False
        want=a['forecast_available']&a['eligible']
        np.testing.assert_array_equal(h.original_history_use(a,len(m['dates'])),want)
        self.assertTrue(h.original_history_use(a,2)[1,2])
        for n in (0,1,3):
            bad=copy.deepcopy(a)
            for k in ('forecast_available','eligible'):
                bad[k]=bad[k].astype(object);bad[k][n:]='future poison'
            np.testing.assert_array_equal(h.original_history_use(bad,n),want[:n])
    def test_consumed_nonboolean_history_masks_rejected(self):
        for key in ('forecast_available','eligible'):
            a,_,_=original_fixture();a[key]=a[key].astype(object);a[key][0,0]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.original_history_use(a,1)
    def test_cached_source_local_linear_cannot_be_constant_or_wrong_identity_width(self):
        a,m,args=original_fixture();a['raw_local_constant']+=.5
        source={key:np.zeros(a['baseline'].size) for key in h.SOURCE_KEYS}
        for key in ('source_log','local_linear','local_constant'):source['raw_'+key]=a['raw_'+key].reshape(-1).copy()
        source['raw_control']=source['raw_local_constant'].copy()
        source['available']=a['model_available'].reshape(-1).copy()
        source['local_available']=a['local_available'].reshape(-1).copy()
        self.assertEqual(h.require_cached_source(source,a)['error_prediction_source'],'raw_local_linear')
        bad=copy.deepcopy(source);bad['raw_local_linear']=source['raw_control'].copy()
        with self.assertRaises(AssertionError):h.require_cached_source(bad,a)
        bad=copy.deepcopy(source);bad['available']=bad['available'].astype(int)
        with self.assertRaises(ValueError):h.require_cached_source(bad,a)
        bad=copy.deepcopy(source);bad['raw_local_linear']=bad['raw_local_linear'][:-1]
        with self.assertRaises(ValueError):h.require_cached_source(bad,a)
        bad=copy.deepcopy(source);bad['unknown']=np.zeros(a['baseline'].size)
        with self.assertRaises(ValueError):h.require_cached_source(bad,a)
    def test_source_review_early_scope_gate_before_cache_access(self):
        import monthly_joint_error226_host as frozen226
        good=dict(passed=True,source_only=True,metadata_only=True,source_npz_decoded=False,
            actual_support_computed=False,actual_returns_computed=False,observers_modified=False)
        flags=list(good);evidence={'work/synthetic':'digest'}
        good.update(context_sha256=h.CONTEXT_SHA,evidence_sha256=evidence)
        original=dict(input_sha256={'original-pin':'original-digest'})
        def reader(path):
            if path==h.SOURCE_REVIEW:return review
            if path==h.CONTEXT:return dict(evidence_sha256=evidence)
            raise AssertionError('Actual source cache accessed before bad receipt rejected')
        with patch.object(frozen226,'model_source_binding',return_value=original) as oldbind,patch.object(h,'sha',return_value=h.SOURCE_REVIEW_SHA),patch.object(h,'read',side_effect=reader):
            for k in flags:
                for value in (not good[k],None,1 if good[k] else 0):
                    review=copy.deepcopy(good);review[k]=value
                    with self.subTest(key=k,value=value),self.assertRaises(ValueError):h.model_source_binding({})
            for k,value in (('context_sha256','wrong'),('evidence_sha256',{})):
                review=copy.deepcopy(good);review[k]=value
                with self.subTest(key=k),self.assertRaisesRegex(ValueError,'source-context'):h.model_source_binding({})
            self.assertEqual(oldbind.call_count,3*len(flags)+2)
        with patch.object(frozen226,'model_source_binding',side_effect=ValueError('old source changed')),patch.object(h,'read') as rd:
            with self.assertRaisesRegex(ValueError,'old source changed'):h.model_source_binding({})
            rd.assert_not_called()
    def test_pin_conflicts_external_allowlist_and_mutable_central(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'synthetic';file.write_bytes(b'first')
            pins={};h.bind(pins,file);expected=dict(pins);h.bind(pins,file)
            file.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Conflicting inherited'):h.bind(pins,file)
            self.assertEqual(pins,expected)
        runtime,digest=next(iter(h.EXTERNAL_RUNTIME.items()))
        with patch.object(h,'sha',return_value=digest),patch.object(Path,'is_file',return_value=True):
            h.verify({runtime:digest})
            for p,d in ((runtime,'wrong'),('/tmp/unregistered.py',digest)):
                with self.subTest(path=p),self.assertRaises(ValueError):h.verify({p:d})
        with self.assertRaisesRegex(ValueError,'Mutable central'):h.verify({str(h.CENTRAL):'x'})



class ReceiptTests(unittest.TestCase):
    def test_real_model_receipt_all25_flags_counts_and_scope(self):
        b=bundle();a,current,built=rebuilt(b);check=oracle.check_build(*a,built,current_available=current);n=len(a[2]['dates'])
        h.require_model_audit(check,built,n,20)
        self.assertEqual(len(h.MODEL_FLAGS),25);self.assertEqual(len(h.MODEL_COUNT_FIELDS),21)
        for key in h.MODEL_FLAGS:
            for value in (False,None,1):
                bad=copy.deepcopy(check);bad[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):h.require_model_audit(bad,built,n)
        for key,value in (('query_rows',n+1),('history_rows',n),('numerical_tolerance',1e-12),('gram_relative_tolerance',1e-6),
            ('producer_modules_imported',True),('computes_returns',True),('source_model_refits',1),('new_ridge_fits',1),('new_tree_fits',1),
            ('parameter_estimates',0),('new_statistical_estimation_queries',0)):
            bad=copy.deepcopy(check);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,built,n)
        for key in h.MODEL_COUNT_FIELDS:
            bad=copy.deepcopy(check);bad['counts'][key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(bad,built,n)
    def test_estimation_algebra_and_support_not_only_receipt_equality(self):
        b=bundle();a,current,built=rebuilt(b);check=oracle.check_build(*a,built,current_available=current)
        for key in ('parameter_estimates','security_mean_estimates','joint_variance_estimates'):
            z=copy.deepcopy(built);c=copy.deepcopy(check)
            z['audit']['event_history'][-1]['counts'][key]+=1;z['audit']['counts'][key]+=1;c['counts'][key]+=1
            if key=='parameter_estimates':z['audit'][key]+=1;c[key]+=1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Parameter accounting'):h.require_model_audit(c,z,2)
        for key in ('candidate_risk','control_risk','candidate_effect','control_effect'):
            z=copy.deepcopy(built);z['arrays'][key][0,0]=1.
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(check,z,2)
        z=copy.deepcopy(built);z['audit']['separate_history_axis']=False
        with self.assertRaises(ValueError):h.require_model_audit(check,z,2)
    def test_selection_receipt_flags_scope_and_coverage(self):
        b=bundle();a,current,built=rebuilt(b);scores,choices=oracle.selection_oracle(b['arrays'],b['meta'],built)
        check=oracle.verify_selection(b['arrays'],b['meta'],built,scores,choices);h.require_selection_audit(check,2)
        self.assertEqual(len(h.SELECTION_FLAGS),9)
        for key in h.SELECTION_FLAGS:
            c=dict(check);c[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(c,2)
        for key,value in (('ordered_choices_checked',5),('numerical_tolerance',1e-12),('producer_modules_imported',True),('computes_returns',True)):
            c=dict(check);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(c,2)
    def test_complete_122x49_receipt_all123orders_and_five_physical_history_prefixes(self):
        values=full_receipt_fixture();h.verify_prepared_evidence(*values)
        self.assertEqual([x['history_rows'] for x in values[2]['model_prefix_evidence']],[0,68,120,155,189])
        cases=[(0,('artifacts_sha256','EXTRA.json'),'x'),(0,('original_array_count',),26),(0,('new_returns_computed',),True),
            (2,('prefixes',),122),(2,('model_prefix_sources_truncated',),False),(2,('separate_history_axis',),False),
            (2,('source229_labels_exact_copy',),False),(2,('current_available_counts_separate',),False),
            (2,('model_prefix_evidence',0,'training_rows'),1),(2,('model_prefix_evidence',1,'history_rows'),1),
            (2,('model_prefix_evidence',1,'history_last_date'),'2011-01-31'),
            (2,('model_prefix_evidence',1,'current_available_rows'),9999),
            (2,('model_prefix_evidence',-1,'latest_label_end'),'2027-01-01'),
            (3,('historical_mask_semantics',),'originalU'),(3,('independent_adapter','passed'),False),
            (3,('history_join','labels_appended'),True),(3,('history_join','early_rows'),3284),
            (3,('history_join','current_available_rows'),9261),(3,('history_join','history_use_rows'),1),
            (3,('history_join','current_adapted_history_unchanged'),False),
            (3,('error_prediction_source',),'raw_control'),(3,('new_labels',),True)]
        for part,path,value in cases:
            bad=copy.deepcopy(values);obj=bad[part]
            for key in path[:-1]:obj=obj[key]
            obj[path[-1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError):h.verify_prepared_evidence(*bad)


class PhaseTests(unittest.TestCase):
    def test_manifest_contract_and_prospective_context(self):
        self.assertEqual(len(h.PREPARED_FILES),7);self.assertEqual(len(h.RESULT_FILES),38)
        self.assertEqual(h.MODEL_PREFIXES,(0,1,53,88,122));self.assertEqual(h.PRIOR.name,'monthly_expanded_error228')
        self.assertEqual(h.EARLY.name,'monthly_early_history229')
        for change in ('missing','extra'):
            bad={key:'digest' for key in h.RESULT_FILES}
            if change=='missing':bad.pop(next(iter(bad)))
            else:bad['COMPLETION.json']='bad'
            with self.subTest(change=change),self.assertRaises(ValueError):h.require_result_manifest(bad)
        c=h.contract()
        self.assertTrue(c['replace_original_episode_effects']);self.assertTrue(c['no_new_score_budget']);self.assertFalse(c['source_models_refitted'])
        self.assertEqual((c['history_months'],c['current_query_months'],c['early_history_months']),(189,122,67))
        self.assertEqual(c['minimum_common_months'],18);self.assertEqual(c['third_gap'],.02)
        files=h.code_files();self.assertEqual(len(set(files)),len(files));self.assertEqual(files[0],Path(h.__file__))
    def test_phase_writeonce_and_cas_precede_actual_inputs(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
            loader.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            receipt=Path(directory)/'tests';receipt.write_text('keep')
            with patch.object(h,'TEST_RECEIPT',receipt):
                with self.assertRaisesRegex(ValueError,'write-once'):h.tests()
            self.assertEqual(receipt.read_text(),'keep')
            central=Path(directory)/'central';central.write_bytes(b'changed')
            with patch.object(h,'CENTRAL',central):
                with self.assertRaisesRegex(ValueError,'Concurrent central'):h.commit_state(b'prior',{},'synthetic')
            self.assertEqual(central.read_bytes(),b'changed')
    def test_state_requires225_1192_and_closed229_not_only228(self):
        rows=[dict(trials=1,directory='old') for _ in range(225)]
        rows[0]['trials']=968;rows[-1]['directory']=h.PRIOR.name
        source=dict(directory=h.EARLY.name,strategy_trials=0,completion=dict(sha256=h.EARLY_EXPECTED['COMPLETION.json']))
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'),source_experiments=[source])
        with patch.object(h,'sha',return_value=h.EARLY_EXPECTED['COMPLETION.json']):
            h.state_check(state)
            for kind in ('missing','source','strategy','digest','trials'):
                bad=copy.deepcopy(state)
                if kind=='missing':bad.pop('source_experiments')
                elif kind=='source':bad['source_experiments'][0]['directory']='monthly_ineligible_source227'
                elif kind=='strategy':bad['completed'][-1]['directory']='other'
                elif kind=='digest':bad['source_experiments'][0]['completion']['sha256']='wrong'
                else:bad['completed'][0]['trials']+=1
                with self.subTest(kind=kind),self.assertRaises(ValueError):h.state_check(bad)
    def test_closed_early_source_scope_and_manifest_must_pass_before_cache(self):
        good=dict(status='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED',source_only=True,strategy_trials=0,new_portfolio_returns=False,
            original_strategy_unchanged=True,protected_files=21444,observers_modified=False)
        for key in good:
            changed=dict(good);changed[key]='wrong'
            with patch.object(h,'bind'),patch.object(h,'read',return_value=changed):
                with self.subTest(key=key),self.assertRaises(ValueError):h.early_source_binding({}, {})
    def test_static_phase_dataflow_and_no_refitting(self):
        text=Path(h.__file__).read_text();tree=ast.parse(text)
        functions={x.name:ast.get_source_segment(text,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        for name in ('model_source_binding(raw)','code_review_binding()','len(protected)==21444'):
            self.assertLess(functions['register'].index(name),functions['register'].index('OUT.mkdir()'))
        self.assertEqual(functions['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn('Selection escaped first88 cutoff',functions['select'])
        self.assertLess(functions['prepare'].index('plan=gate(expected)'),functions['prepare'].index('study=local()'))
        self.assertIn('join_history(',functions['prepare']);self.assertIn('current_available=current',functions['prepare'])
        self.assertIn("eligible=slice_arrays(a,122)['eligible']",functions['prepare'])
        self.assertIn('history_rows=len(smallinputs[3]',functions['prepare'])
        self.assertIn('for n in range(123)',functions['prepare'])
        self.assertIn("day<=qm['dates'][-1]",functions['prefix_inputs'])
        for name in ('annual_forecasts(','build_labels(','forecasts('):self.assertNotIn(name,functions['prepare'])


if __name__=='__main__':unittest.main()
