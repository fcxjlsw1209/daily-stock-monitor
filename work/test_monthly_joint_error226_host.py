"""Synthetic host integration/gates only; never read actual cached arrays or returns."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tested_joint_error226_host',P/'monthly_joint_error226_host.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
import monthly_joint_error226_model as model
import monthly_joint_error226_independent as oracle
from test_monthly_joint_error226_model import fixture as model_fixture


def selection_fixture():
    metric={era:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for era in ('training','early','recent')}
    metrics={name:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')} for name in h.NAMES}
    orders={name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in h.NAMES}
    for path in metrics[h.NAMES[1]]:metrics[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return metrics,orders


def original_fixture(args=None):
    """Legal original selector inputs tied to a wholly synthetic source history."""
    args=model_fixture(initial=True) if args is None else args
    tr,tm,meta,src,use,pre,eligible=copy.deepcopy(args)
    n,w=eligible.shape;cw=np.full((n,1),.52)
    support=np.tile(np.linspace(-.1,.1,w),(n,1))
    core=cw*np.tile(np.linspace(.45,.2,w),(n,1))
    arrays=dict(baseline=core+(1-cw)*support,core=core,core_weight=cw,
        eligible=eligible.copy(),state_calm10=np.zeros(n,bool),state_weak_narrow=np.zeros(n,bool),
        pre_score=pre.copy(),effects=np.tile(np.r_[np.array([0.,0.,-.1,.05,.15,.5])[:w],np.zeros(max(0,w-6))],(n,1)),
        forecast_available=use.copy(),extension_available=np.ones((n,w),bool),
        old_ml4=np.zeros((n,w)),rank_log_excess=np.zeros((n,w)),
        model_available=src['available'].reshape(n,w).copy(),local_available=np.zeros((n,w),bool),
        raw_source_log=src['raw_local_linear'].reshape(n,w).copy(),
        raw_local_constant=src['raw_local_linear'].reshape(n,w).copy(),
        raw_local_linear=src['raw_local_linear'].reshape(n,w).copy())
    # Exactly the 27 original cache keys, including inherited unused target ranks.
    for name in h.TARGETS:
        arrays['raw_'+name]=np.zeros((n,w))
        arrays.setdefault('rank_'+name,np.zeros((n,w)))
    arrays['rank_local_constant']=np.zeros((n,w));arrays['rank_local_linear']=np.zeros((n,w))
    arrays['paired_month_ready']=np.ones(n,bool)
    assert len(arrays)==27
    return arrays,meta,(tr,tm,meta,src,use,pre,eligible)


def rebuild(arrays,args):
    tr,tm,meta,src,use,_,_=args
    study=h.local();pre=study.modified_pre_score(study.configs()[4],arrays,len(meta['dates']))
    inputs=(tr,tm,meta,src,use,pre,arrays['eligible'])
    return inputs,model.build(*inputs)


def full_receipt_fixture():
    """122×49 entirely synthetic axes, using real producer/oracle and frozen selector."""
    tr,tm,m,src,use,pre,eligible=model_fixture(n_history=120,start=(2010,1),current=((2020,1),(2020,6)))
    rows=len(m['dates']);width=49;old_width=6
    syms=list('ABCDEF')+['X%02d'%i for i in range(width-old_width)]
    for meta in (tm,m):meta['symbols']=[syms.copy() for _ in meta['dates']]
    months=len(tm['dates'])
    for key in list(tr):
        if key in ('dates','ends','offsets'):continue
        a=tr[key].reshape((months,old_width)+tr[key].shape[1:])
        extra=np.repeat(a[:,:1],width-old_width,axis=1)
        tr[key]=np.concatenate([a,extra],axis=1).reshape((months*width,)+a.shape[2:])
    tr['dates']=np.repeat(np.asarray(tm['dates']),width);tr['ends']=np.repeat(np.asarray(tm['ends']),width)
    tr['offsets']=np.arange(months+1)*width
    for key,value in src.items():
        a=value.reshape(rows,old_width);extra=np.repeat(a[:,:1],width-old_width,axis=1)
        src[key]=np.concatenate([a,extra],axis=1).reshape(-1)
    use=np.c_[use,np.zeros((rows,width-old_width),bool)]
    eligible=np.c_[eligible,np.zeros((rows,width-old_width),bool)]
    pre=np.c_[pre,np.full((rows,width-old_width),-1.)]
    arrays,m,args=original_fixture((tr,tm,m,src,use,pre,eligible))
    inputs,built=rebuild(arrays,args);study=h.local()
    orders,scores,choices=h.decisions(m,arrays,built,rows,study=study)
    for n in range(rows+1):
        aa=h.slice_arrays(arrays,n);mm={k:v[:n] for k,v in m.items()}
        assert len(aa)==27 and all(v.shape[0]==n for v in aa.values())
        oo,ss,cc=h.decisions(mm,aa,h.slice_built(built,n),n,study=study)
        assert oo=={arm:orders[arm][:n] for arm in h.NAMES}
        for arm in h.NAMES:np.testing.assert_array_equal(ss[arm],scores[arm][:n])
    check=oracle.check_build(*inputs,built)
    selection=oracle.verify_selection(arrays,m,built,scores,choices)
    prefix=[]
    for n in h.MODEL_PREFIXES:
        short=h.prefix_inputs(*inputs[:5],n)+(inputs[5][:n],inputs[6][:n])
        small=model.build(*short);aa=h.slice_arrays(arrays,n)
        oo,ss,cc=h.decisions(short[2],aa,small,n,study=study)
        assert oo=={arm:orders[arm][:n] for arm in h.NAMES}
        prefix.append(dict(prefix=n,query_rows=n,training_rows=len(short[0]['dates']),
            query_date=m['dates'][n-1] if n else None,
            cutoff='%04d-12-31'%(int(m['dates'][n-1][:4])-1) if n else None,
            latest_label_end=str(np.asarray(short[0]['ends'],dtype='datetime64[D]').max()) if len(short[0]['ends']) else None,
            source_history_starts_at_original_zero=True,counts=small['audit']['counts'],
            model=oracle.check_build(*short,small),selection=oracle.verify_selection(aa,short[2],small,ss,cc)))
    audit=dict(model=check,selection=selection,prefixes=123,model_prefixes=list(h.MODEL_PREFIXES),
        model_prefix_evidence=prefix,original_array_count=27,model_counts=built['audit']['counts'])
    for key in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
        'model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact','original_arrays_unchanged',
        'original_source_arrays_unchanged','original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget',
        'historical_original_universe_exact','error_prediction_source_local_linear'):audit[key]=True
    parity={key:True for key in ('original_training_cache_exact','original_identity_exact','cached206_source_vectors_exact',
        'cached206_control_constant_exact','source_availability_exact','historical_original_universe_exact','no_source_model_refits','no_fitted_residual_error_inputs')}
    parity.update(fresh_training_array_comparison=False,new_labels=False,new_raw_features=False,
        error_prediction_source='raw_local_linear',target_source='original169_target_log_excess_without_recentring')
    completion=dict(artifacts_sha256={key:'synthetic digest' for key in h.PREPARED_FILES},
        original_array_count=27,new_returns_computed=False,source_model_refits=0,model_counts=built['audit']['counts'])
    return completion,m,audit,parity,built


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
        evidence={'work/fake':'pin'}
        review=dict(passed=True,source_only=True,context_sha256=h.CONTEXT_SHA,coverage_sha256=h.COVERAGE_SHA,
            evidence_sha256=evidence,current_and_historical_prediction_key='raw_local_linear',target_key='target_log_excess',
            historical_use='originalU AND206available',actual_new_estimates=0,matched_formal_months=113,source_annual=[{}]*11)
        for k in ('source_admission_metadata_only','cached_forecasts_are_prior_annual_asof_replays','not_realtime_archival_vintages',
                  'source_bytes_reverified','metadata_axes_verified','calendar_clock_verified','annual_cutoff_calendar_verified',
                  'raw_source_hashes_verified','raw_source_map_matches_original'):review[k]=True
        for k in ('uses_fitted_residuals_as_historical_errors','source_npz_decoded','npz_decoded','training_npz_decoded',
                  'actual_joint_history_support_computed','local_available_filter','actual_new_features','actual_new_orders',
                  'actual_new_returns','observers_modified'):review[k]=False
        def reader(path):
            if path==h.SOURCE_REVIEW:return review
            if path==h.CONTEXT:return {'evidence_sha256':evidence}
            raise AssertionError('Cache read before rejecting bad source receipt')
        cases=[('context_sha256','bad'),('coverage_sha256','bad'),('actual_new_estimates',True),
            ('actual_new_estimates',1),('current_and_historical_prediction_key','raw_control'),('target_key','new_target'),
            ('historical_use','eligible AND available'),('matched_formal_months',112)]
        cases += [(k,not review[k]) for k,v in review.items() if type(v) is bool]
        with patch.object(h,'sha',return_value=h.SOURCE_REVIEW_SHA),patch.object(h,'read',side_effect=reader):
            for k,v in cases:
                old=review[k];review[k]=v
                with self.subTest(key=k,value=v),self.assertRaises(ValueError):h.model_source_binding({})
                review[k]=old
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


class PrefixTests(unittest.TestCase):
    def test_physical_prefix_removes_future_label_values_before_model(self):
        args=model_fixture();tr,tm,m,src,use,pre,eligible=args
        full=model.build(*args)
        for n in range(len(m['dates'])+1):
            part=h.prefix_inputs(tr,tm,m,src,use,n)
            small=model.build(*part,pre[:n].copy(),eligible[:n].copy())
            self.assertEqual(len(part[2]['dates']),n)
            self.assertTrue(all(len(v)==n*use.shape[1] for v in part[3].values()))
            for k in model.ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][k],full['arrays'][k][:n])
            if n:
                cutoff=np.datetime64('%04d-12-31'%(int(m['dates'][n-1][:4])-1))
                self.assertTrue((np.asarray(part[0]['ends'],dtype='datetime64[D]')<=cutoff).all())
            else:
                self.assertEqual(part[0]['offsets'].tolist(),[0]);self.assertEqual(len(part[0]['dates']),0)
    def test_future_values_poisoned_and_source_id_reordering_survives_prefix(self):
        tr,tm,m,src,use,pre,eligible=model_fixture()
        n=len(m['dates'])-1;cutoff=str(int(m['dates'][n-1][:4])-1)+'-12-31'
        future=np.asarray(tr['ends'])>cutoff
        tr['train_plain']=tr['train_plain'].astype(object);tr['train_plain'][future]='future poison'
        for key in h.TARGETS:
            tr['target_'+key]=tr['target_'+key].astype(object);tr['target_'+key][future]='future poison'
        for k in src:src[k]=src[k].astype(object);src[k][n*use.shape[1]:]='future poison'
        use=use.astype(object);use[n:]='future poison'
        part=h.prefix_inputs(tr,tm,m,src,use,n)
        built=model.build(*part,pre[:n],eligible[:n])
        self.assertEqual(built['audit']['query_months'],n)
        bad=copy.deepcopy(tr);bad['offsets'][1]+=1
        with self.assertRaisesRegex(ValueError,'offsets'):h.prefix_inputs(bad,tm,m,src,use,n)
        bad=copy.deepcopy(tr);bad['dates'][0]='2017-06-29'
        with self.assertRaisesRegex(ValueError,'flat date identity'):h.prefix_inputs(bad,tm,m,src,use,n)
    def test_slice_built_recomputes_prefix_counts_and_removes_future_events(self):
        args=model_fixture();full=model.build(*args);before=copy.deepcopy(full)
        for n in range(len(args[2]['dates'])+1):
            small=h.slice_built(full,n)
            self.assertEqual(len(small['audit']['event_history']),n)
            part=h.prefix_inputs(*args[:5],n)+(args[5][:n],args[6][:n])
            h.require_model_audit(oracle.check_build(*part,small),small,n)
            self.assertEqual(small['audit']['query_months'],n)
            for k in model.ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][k],full['arrays'][k][:n])
            for k in model.COUNT_FIELDS:
                self.assertEqual(small['audit']['counts'][k],sum(e['counts'][k] for e in full['audit']['event_history'][:n]))
            self.assertEqual(small['audit']['parameter_estimates'],small['audit']['counts']['parameter_estimates'])
        self.assertEqual(before['audit'],full['audit'])
        for k in model.ARRAY_KEYS:np.testing.assert_array_equal(before['arrays'][k],full['arrays'][k])
        for n in (True,-1,len(args[2]['dates'])+1):
            with self.subTest(n=n),self.assertRaises(ValueError):h.slice_built(full,n)


class DecisionTests(unittest.TestCase):
    def test_real_frozen_selector_joint_vs_diagonal_and_single_episode(self):
        a,m,args=original_fixture();inputs,built=rebuild(a,args);study=h.local()
        with patch.object(h,'local',return_value=study),patch.object(study.ep,'select_boundary',wraps=study.ep.select_boundary) as episode:
            orders,scores,choices=h.decisions(m,a,built,len(m['dates']))
        check=oracle.verify_selection(a,m,built,scores,choices)
        h.require_selection_audit(check,len(m['dates']))
        self.assertEqual(episode.call_count,3)
        self.assertEqual(set(orders),set(h.NAMES))
        self.assertNotEqual(choices[h.NAMES[1]][-1],choices[h.NAMES[2]][-1])
        for arm in h.NAMES[1:]:
            self.assertEqual(choices[arm][0],choices[h.NAMES[0]][0])
            np.testing.assert_array_equal(scores[arm][0],scores[h.NAMES[0]][0])
            self.assertTrue({0,1}<=set(choices[arm][-1]))
        # A second episode or additive old effects changes these exact values.
        support=(a['baseline']-a['core'])/(1-a['core_weight'])
        pre=inputs[5];i=len(m['dates'])-1;J=built['audit']['event_history'][i]['J']
        for arm,key in zip(h.NAMES[1:],('candidate','control')):
            want=pre[i].copy()
            for j in J:want[j]+=.05*(np.clip(support[i,j]+built['arrays'][key+'_effect'][i,j],-.5,.5)-support[i,j])
            np.testing.assert_array_equal(scores[arm][i],want)
    def test_all_market_states_use_existing_funded_episode_budget(self):
        for calm,weak in ((False,False),(True,False),(False,True),(True,True)):
            a,m,args=original_fixture();a['state_calm10'][-1]=calm;a['state_weak_narrow'][-1]=weak
            inputs,built=rebuild(a,args);orders,scores,choices=h.decisions(m,a,built,len(m['dates']))
            self.assertTrue(built['arrays']['ready'][-1])
            h.require_selection_audit(oracle.verify_selection(a,m,built,scores,choices),len(m['dates']))
            budget=min(.05,max(0.,.10-(.025*calm+.05*weak)))
            support=(a['baseline']-a['core'])/(1-a['core_weight'])
            for arm,key in zip(h.NAMES[1:],('candidate','control')):
                want=inputs[5][-1].copy()
                for j in built['audit']['event_history'][-1]['J']:
                    want[j]+=budget*(np.clip(support[-1,j]+built['arrays'][key+'_effect'][-1,j],-.5,.5)-support[-1,j])
                with self.subTest(calm=calm,weak=weak,arm=arm):np.testing.assert_array_equal(scores[arm][-1],want)
    def test_supported_zero_effect_replaces_old_episode_not_fallback(self):
        a,m,args=original_fixture();args[3]['raw_local_linear'][:]=1.
        a['raw_local_linear'][:]=a['raw_local_constant'][:]=a['raw_source_log'][:]=1.
        inputs,built=rebuild(a,args)
        self.assertTrue(built['arrays']['ready'][-1])
        orders,scores,choices=h.decisions(m,a,built,len(m['dates']))
        for arm in h.NAMES[1:]:
            np.testing.assert_array_equal(scores[arm][-1],inputs[5][-1])
            self.assertEqual(choices[arm][-1],[0,1,2])
        self.assertNotEqual(scores[h.NAMES[0]][-1].tolist(),inputs[5][-1].tolist())
    def test_same_common_history_missing_one_boundary_id_falls_back_both(self):
        args=model_fixture(n_history=18);args[4][0,4]=False
        a,m,args=original_fixture(args);inputs,built=rebuild(a,args)
        self.assertFalse(built['arrays']['ready'][-1])
        orders,scores,choices=h.decisions(m,a,built,len(m['dates']))
        for arm in h.NAMES[1:]:
            self.assertEqual(orders[arm],orders[h.NAMES[0]])
            np.testing.assert_array_equal(scores[arm],scores[h.NAMES[0]])
    def test_physical_order_prefixes_and_input_immutability(self):
        a,m,args=original_fixture();inputs,built=rebuild(a,args);original=copy.deepcopy((a,m,built))
        self.assertEqual(len(a),27)
        all_orders,all_scores,all_choices=h.decisions(m,a,built,len(m['dates']))
        for n in range(len(m['dates'])+1):
            aa=h.slice_arrays(a,n);mm={k:v[:n] for k,v in m.items()}
            self.assertEqual(len(aa),27);self.assertTrue(all(v.shape[0]==n for v in aa.values()))
            orders,scores,choices=h.decisions(mm,aa,h.slice_built(built,n),n)
            for arm in h.NAMES:
                self.assertEqual(orders[arm],all_orders[arm][:n]);self.assertEqual(choices[arm],all_choices[arm][:n])
                np.testing.assert_array_equal(scores[arm],all_scores[arm][:n])
        self.assertEqual(m,original[1]);self.assertEqual(built['audit'],original[2]['audit'])
        for k in a:np.testing.assert_array_equal(a[k],original[0][k])
    def test_future_selector_rows_poisoned_before_prefix(self):
        a,m,args=original_fixture();inputs,built=rebuild(a,args);n=3
        expected=h.decisions(m,a,built,n)
        for k,v in a.items():
            if v.dtype.kind=='b':a[k]=v.astype(object);a[k][n:]='future poison'
            else:a[k][n:]=np.inf
        for k in ('dates','ends','symbols'):m[k][n:]=[object()]*(len(m[k])-n)
        for k in model.ARRAY_KEYS:
            if k=='ready':built['arrays'][k][n:]=~built['arrays'][k][n:]
            else:built['arrays'][k][n:]=np.inf
        built['audit']['event_history'][n:]=[object()]*(len(built['audit']['event_history'])-n)
        orders,scores,choices=h.decisions(m,a,built,n)
        self.assertEqual(orders,expected[0]);self.assertEqual(choices,expected[2])
        for arm in h.NAMES:np.testing.assert_array_equal(scores[arm],expected[1][arm])


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


class ReceiptAndPhaseTests(unittest.TestCase):
    def test_model_receipt_flags_counts_estimates_and_scope(self):
        args=model_fixture();built=model.build(*args);check=oracle.check_build(*args,built);n=len(args[2]['dates'])
        h.require_model_audit(check,built,n)
        self.assertEqual(len(h.MODEL_FLAGS),23);self.assertEqual(len(h.MODEL_COUNT_FIELDS),21)
        for k in h.MODEL_FLAGS:
            for value in (False,1,None):
                bad=copy.deepcopy(check);bad[k]=value
                with self.subTest(flag=k,value=value),self.assertRaises(ValueError):h.require_model_audit(bad,built,n)
        for k,value in (('computes_returns',True),('producer_modules_imported',True),('numerical_tolerance',1e-12),('gram_relative_tolerance',1e-6),('query_rows',n+1),('source_model_refits',1),('new_ridge_fits',1),('new_tree_fits',1),('parameter_estimates',0),('new_statistical_estimation_queries',0)):
            bad=copy.deepcopy(check);bad[k]=value
            with self.subTest(field=k),self.assertRaises(ValueError):h.require_model_audit(bad,built,n)
        for k in h.MODEL_COUNT_FIELDS:
            bad=copy.deepcopy(check);bad['counts'][k]=-1
            with self.subTest(count=k),self.assertRaises(ValueError):h.require_model_audit(bad,built,n)
    def test_model_receipt_count_algebra_cannot_accept_fabricated_same_totals(self):
        args=model_fixture();built=model.build(*args);check=oracle.check_build(*args,built);n=len(args[2]['dates'])
        for k in ('query_months','supported_queries','parameter_estimates','security_mean_estimates','security_variance_estimates','joint_variance_estimates','diagonal_variance_estimates','attempted_history_months'):
            bad=copy.deepcopy(check);bad['counts'][k]+=1
            with self.subTest(count=k),self.assertRaises(ValueError):h.require_model_audit(bad,built,n)
    def test_model_receipt_mirrored_corruption_still_rejected_by_estimate_algebra(self):
        args=model_fixture();built=model.build(*args);check=oracle.check_build(*args,built);n=len(args[2]['dates'])
        for key in ('parameter_estimates','security_mean_estimates','joint_variance_estimates'):
            b=copy.deepcopy(built);c=copy.deepcopy(check)
            b['audit']['event_history'][-1]['counts'][key]+=1;b['audit']['counts'][key]+=1
            c['counts'][key]+=1
            if key=='parameter_estimates':b['audit'][key]+=1;c[key]+=1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Parameter accounting'):h.require_model_audit(c,b,n)
        for key in ('candidate_risk','control_risk','candidate_effect','control_effect'):
            b=copy.deepcopy(built);b['arrays'][key][0,0]=1.
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_model_audit(check,b,n)
    def test_complete_prepared_receipt_and_maturity_corruptions(self):
        values=full_receipt_fixture();h.verify_prepared_evidence(*values)
        cases=[(0,('artifacts_sha256','EXTRA.json'),'extra'),
            (0,('original_array_count',),26),(0,('new_returns_computed',),True),
            (2,('prefixes',),122),(2,('model_prefix_sources_truncated',),False),
            (2,('model_prefix_evidence',0,'training_rows'),1),
            (2,('model_prefix_evidence',-1,'latest_label_end'),'2020-01-01'),
            (2,('model_prefix_evidence',-1,'source_history_starts_at_original_zero'),False),
            (3,('error_prediction_source',),'raw_control'),(3,('new_labels',),True)]
        for part,path,value in cases:
            bad=copy.deepcopy(values);container=bad[part]
            for key in path[:-1]:container=container[key]
            container[path[-1]]=value
            with self.subTest(part=part,path=path),self.assertRaises(ValueError):h.verify_prepared_evidence(*bad)
    def test_selection_receipt_strict_scope_flags_and_rows(self):
        a,m,args=original_fixture();inputs,built=rebuild(a,args)
        scores,choices=oracle.selection_oracle(a,m,built)
        receipt=oracle.verify_selection(a,m,built,scores,choices);n=len(m['dates'])
        h.require_selection_audit(receipt,n);self.assertEqual(len(h.SELECTION_FLAGS),9)
        for k in h.SELECTION_FLAGS:
            bad=dict(receipt);bad[k]=1
            with self.subTest(flag=k),self.assertRaises(ValueError):h.require_selection_audit(bad,n)
        for k,value in (('ordered_choices_checked',3*n-1),('numerical_tolerance',1e-12),('producer_modules_imported',True),('computes_returns',True)):
            bad=dict(receipt);bad[k]=value
            with self.subTest(key=k),self.assertRaises(ValueError):h.require_selection_audit(bad,n)
    def test_manifest_sets_and_prospective_contract(self):
        self.assertEqual(len(h.PREPARED_FILES),7);self.assertEqual(len(h.RESULT_FILES),38)
        self.assertEqual(h.MODEL_PREFIXES,(0,1,53,88,122));self.assertEqual(h.PRIOR.name,'monthly_relative_duration225')
        for kind in ('missing','extra'):
            bad={name:'digest' for name in h.RESULT_FILES}
            if kind=='missing':bad.pop(next(iter(bad)))
            else:bad['COMPLETION.json']='bad'
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.require_result_manifest(bad)
        h.require_result_manifest({name:'digest' for name in h.RESULT_FILES})
        c=h.contract();self.assertTrue(c['replace_original_episode_effects']);self.assertTrue(c['no_new_score_budget'])
        self.assertEqual(c['minimum_common_months'],18);self.assertEqual(c['third_gap'],.02)
    def test_phase_and_writeonce_guards_precede_actual_inputs(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
            loader.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            receipt=Path(directory)/'tests';receipt.write_text('keep')
            with patch.object(h,'TEST_RECEIPT',receipt):
                with self.assertRaisesRegex(ValueError,'write-once'):h.tests()
            self.assertEqual(receipt.read_text(),'keep')
            central=Path(directory)/'state';central.write_bytes(b'changed')
            with patch.object(h,'CENTRAL',central):
                with self.assertRaisesRegex(ValueError,'Concurrent central'):h.commit_state(b'prior',{},'test')
            self.assertEqual(central.read_bytes(),b'changed')
    def test_exact_source_expected_counters_and_no_live_data_imports_in_test(self):
        rows=[dict(trials=1,directory='old') for _ in range(223)]
        rows[0]['trials']=964;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state);state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
        text=Path(h.__file__).read_text();tree=ast.parse(text)
        functions={x.name:ast.get_source_segment(text,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        self.assertLess(functions['register'].index('model_source_binding(raw)'),functions['register'].index('OUT.mkdir()'))
        self.assertLess(functions['register'].index('code_review_binding()'),functions['register'].index('OUT.mkdir()'))
        self.assertLess(functions['register'].index('len(protected)==21444'),functions['register'].index('OUT.mkdir()'))
        self.assertEqual(functions['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn('Selection escaped first88 cutoff',functions['select'])


if __name__=='__main__':unittest.main()
