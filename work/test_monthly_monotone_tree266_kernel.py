"""Direct synthetic266 tests; no actual source/forecast/return loaders."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
import monthly_monotone_tree266_kernel as k
import test_monthly_local_distance256_kernel as fixtures


def fixture():return fixtures.fixture()
def clone(args):return (args[0],)+deepcopy(args[1:])
def fit(args,n=None,**kw):return k.fit_models(args[1],args[2],args[3],n,**kw)
def decide(args,b,n=None,**kw):return k.decisions(args[0],args[3],args[2]['monthly'],b,n,**kw)
def refresh(args):return fixtures.refresh_original(args)

class Poison:
    def __float__(self):raise AssertionError('Future numeric consumed')
    def __deepcopy__(self,memo):raise AssertionError('Future body copied')
    def __len__(self):raise AssertionError('Future body inspected')
    def __iter__(self):raise AssertionError('Future body iterated')


def tiny_tree(reference,role='monotone_candidate'):
    t=deepcopy(reference);t['role']=role;t['parameters']=k.parameters(role)
    def node(depth,leaf,value=0.):
        r={name:float(0).hex() if k.PREDICTOR_RECORD_DTYPE[name].kind=='f' else 0 for name in k.PREDICTOR_RECORD_DTYPE.names}
        r.update(depth=depth,is_leaf=leaf,count=100,value=float(value).hex());return r
    root=node(0,0);root.update(left=1,right=2,gain=1.0.hex())
    t['baseline']=k._pack(np.zeros((1,1)));t['bin_thresholds']=[k._pack(np.array([0.]))]+[k._pack(np.array([])) for _ in range(9)];t['n_bins_non_missing']=[2]+[1]*9
    for r in t['trees']:r['nodes']=[node(0,1)]
    t['trees'][0]['nodes']=[root,node(1,1,1.),node(1,1,2.)]
    t['shape_certificate']=k._shape_certificate(t);k.validate_tree(t,role);return t


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=fixture();cls.full=fit(cls.args);cls.reference=k.legacy.fit_models(cls.args[1],cls.args[2],cls.args[3])
    def test_exact_schema_and_actual_paired_fit_counts(self):
        b=self.full;c=b['audit']['counts'];self.assertEqual(tuple(b),k.BUNDLE_KEYS);self.assertEqual(tuple(c),k.MODEL_COUNT_FIELDS);self.assertEqual(tuple(b['audit']),k.MODEL_AUDIT_KEYS)
        self.assertEqual((c['matched_ridge_fits'],c['unconstrained_tree_fits'],c['monotone_tree_fits'],c['actual_model_fits']),(2,2,2,6))
        self.assertEqual(c['original_other_target_fits'],0);self.assertEqual(c['original_local_fits'],0)
        for short in ('unconstrained','monotone'):
            self.assertEqual(c[short+'_trees'],200);self.assertEqual(c[short+'_boosting_iterations'],200);self.assertEqual(c[short+'_manual_tree_traversals'],100*c['prediction_rows'])
        for role in k.ROLES:
            self.assertEqual(tuple(b[role]),k.FORECAST_KEYS)
            for row in b[role]['audit']['annual']:self.assertEqual(tuple(row),k.RIDGE_ANNUAL_KEYS if role=='calibration_ridge' else k.TREE_ANNUAL_KEYS)
    def test_frozen261_unconstrained_forecasts_structures_exact(self):
        for new,old in (('calibration_ridge','control'),('unconstrained_control','candidate')):
            for key in ('raw','available','rank'):self.assertEqual(self.full[new][key].tobytes(),self.reference[old][key].tobytes())
        for a,b in zip(self.full['unconstrained_control']['audit']['annual'],self.reference['candidate']['audit']['annual']):
            for key in k.TREE_ANNUAL_KEYS:
                self.assertEqual(k.legacy_unconstrained_tree(a[key]) if key=='tree' and a[key] is not None else a[key],b[key])
    def test_frozen261_unconstrained_strategy_exact(self):
        old=k.legacy.decisions(self.args[0],self.args[3],self.args[2]['monthly'],self.reference)
        new=decide(self.args,self.full)
        for na,oa in ((k.ARMS[0],k.legacy.ARMS[0]),(k.ARMS[1],k.legacy.ARMS[2])):
            self.assertEqual(new[0][na],old[0][oa]);self.assertEqual(new[1][na].tobytes(),old[1][oa].tobytes());self.assertEqual(new[2][na].tobytes(),old[2][oa].tobytes())
    def test_source_and_all_four_labels_unchanged(self):
        inp,_,_,_=k.model_inputs(self.args[1],self.args[2],self.args[3]);lp=self.full['label_preservation'];self.assertEqual(tuple(lp),k.LABEL_KEYS)
        self.assertEqual(self.full['label_values'].tobytes(),inp[4]['log_excess'].tobytes())
        for name in k.TARGETS:self.assertEqual(lp['target_sha256'][name],k._digest(inp[4][name]))
        self.assertEqual(self.full['calibration_ridge']['raw'].tobytes(),self.args[3]['raw_source_log'].reshape(-1).tobytes())
    def test_callbacks_complete_order_and_strategy_before_candidate(self):
        calls=[];rf=k.fit_ridge;tf=k._fit_tree
        def ridge(*a):calls.append('ridge');return rf(*a)
        def tree(*a):calls.append(a[-1]);return tf(*a)
        def calibration(v):
            self.assertIsNone(v['unconstrained_control']);self.assertIsNone(v['monotone_candidate']);self.assertIsNone(v['calibration_gate']);self.assertIsNone(v['control_gate'])
            self.assertEqual(v['audit']['counts']['actual_model_fits'],2)
            original=k.original_decisions(self.args[0],self.args[3],self.args[2]['monthly']);got=k.calibration_decisions(self.args[0],self.args[3],self.args[2]['monthly'],v)
            self.assertEqual(original[0],got[0]);self.assertEqual(original[1].tobytes(),got[1].tobytes());self.assertEqual(original[2].tobytes(),got[2].tobytes());calls.append('calibration_gate');return {'passed':True,'synthetic':True}
        def control(v):
            self.assertIsNone(v['monotone_candidate']);self.assertIsNone(v['control_gate']);self.assertTrue(v['calibration_gate']['passed']);self.assertEqual(v['audit']['counts']['actual_model_fits'],4)
            got=k.control_decisions(self.args[0],self.args[3],self.args[2]['monthly'],v);old=k.legacy.decisions(self.args[0],self.args[3],self.args[2]['monthly'],self.reference)
            self.assertEqual(got[0],old[0][k.legacy.ARMS[2]]);self.assertEqual(got[1].tobytes(),old[1][k.legacy.ARMS[2]].tobytes());self.assertEqual(got[2].tobytes(),old[2][k.legacy.ARMS[2]].tobytes());calls.append('control_gate');return {'passed':True,'synthetic':True}
        with patch.object(k,'fit_ridge',side_effect=ridge),patch.object(k,'_fit_tree',side_effect=tree):b=fit(self.args,calibration_validator=calibration,control_validator=control)
        self.assertEqual(calls,['ridge','ridge','calibration_gate','unconstrained_control','unconstrained_control','control_gate','monotone_candidate','monotone_candidate'])
        self.assertEqual(b['control_gate'],{'passed':True,'synthetic':True})
    def test_failed_calibration_prevents_all_trees(self):
        with patch.object(k,'_fit_tree') as tf,self.assertRaisesRegex(ValueError,'gate failed'):fit(self.args,calibration_validator=lambda v:{'passed':False})
        self.assertEqual(tf.call_count,0)
    def test_failed_control_prevents_candidate(self):
        calls=[];original=k._fit_tree
        def tree(*a):calls.append(a[-1]);return original(*a)
        with patch.object(k,'_fit_tree',side_effect=tree),self.assertRaisesRegex(ValueError,'gate failed'):fit(self.args,control_validator=lambda v:{'passed':False})
        self.assertEqual(calls,['unconstrained_control','unconstrained_control'])
    def test_callback_view_mutation_rejected(self):
        for stage in ('calibration_validator','control_validator'):
            def callback(v):v['calibration_ridge']['rank'][1,0]+=.01;return {'passed':True}
            with self.assertRaisesRegex(ValueError,'mutated supplied view'):fit(self.args,**{stage:callback})
    def test_callback_training_query_and_original_mutations_rejected(self):
        for stage in ('calibration_validator','control_validator'):
            for field in ('train_plain','event_plain','target_log_excess','common_good','baseline','metadata'):
                args=clone(self.args)
                def callback(v):
                    if field=='baseline':args[3][field][1,0]+=.01
                    elif field=='metadata':args[2]['monthly']['etfs'].append('CHANGED')
                    elif field=='common_good':args[1][field][0]=False
                    elif field=='target_log_excess':args[1][field][0]+=.01
                    else:args[1][field][0,0]+=.01
                    return {'passed':True}
                with self.assertRaises((ValueError,AssertionError)):fit(args,**{stage:callback})
    def test_bad_later_original_year_prevents_all_treefits(self):
        args=clone(self.args);args[3]['raw_source_log'][3,0]=np.nextafter(args[3]['raw_source_log'][3,0],np.inf)
        with patch.object(k,'_fit_tree') as tf,self.assertRaises(AssertionError):fit(args)
        self.assertEqual(tf.call_count,0)
    def test_n0_partial_no_numeric_body_or_fit(self):
        for n in (0,1):
            args=clone(self.args)
            for key in ('train_plain','event_plain')+tuple('target_'+v for v in k.TARGETS):args[1][key]=args[1][key].astype(object);args[1][key][:]=Poison()
            args[2]['training']['symbols']=[Poison() for _ in args[2]['training']['symbols']]
            with patch.object(k,'fit_ridge',side_effect=AssertionError('no fit')),patch.object(k,'_fit_tree',side_effect=AssertionError('no fit')):b=fit(args,n)
            self.assertEqual(b['audit']['counts']['actual_model_fits'],0);self.assertEqual(len(b['label_values']),0)
            for role in k.ROLES:self.assertEqual(b[role]['raw'].tobytes(),args[3]['raw_source_log'][:n].reshape(-1).tobytes())
    def test_future_training_numeric_and_query_metadata_not_consumed(self):
        args=clone(self.args);future=args[1]['ends']>np.datetime64('2015-12-31')
        for key in ('train_plain',)+tuple('target_'+v for v in k.TARGETS):args[1][key]=args[1][key].astype(object);args[1][key][future]=Poison()
        args[1]['event_plain']=args[1]['event_plain'].astype(object);args[1]['event_plain'][12:]=Poison()
        count=int(np.count_nonzero(future))//6
        for i in range(len(args[2]['training']['symbols'])-count,len(args[2]['training']['symbols'])):args[2]['training']['symbols'][i]=Poison()
        for key in ('dates','ends','symbols'):
            for i in (2,3):args[2]['monthly'][key][i]=Poison()
        b=fit(args,2)
        for role in k.ROLES:self.assertEqual(b[role]['raw'].tobytes(),self.full[role]['raw'][:12].tobytes())
    def test_all_logical_physical_cached_prefixes(self):
        for n in range(5):
            b=fit(self.args,n);small=fixtures.physical(self.args,n);c=fit(small)
            for role in k.ROLES:
                for key in ('raw','available','rank'):self.assertEqual(b[role][key].tobytes(),c[role][key].tobytes())
            x=decide(self.args,b,n);y=decide(small,self.full)
            self.assertEqual(x[0],y[0]);self.assertEqual(x[3],y[3])
            for arm in k.ARMS:self.assertEqual(x[1][arm].tobytes(),y[1][arm].tobytes());self.assertEqual(x[2][arm].tobytes(),y[2][arm].tobytes())
    def test_nonweak_years_fit_without_deployment(self):
        args=clone(self.args);args[3]['state_weak_narrow'][:]=False;b=fit(args);self.assertEqual(b['audit']['counts']['actual_model_fits'],6)
        o,f,p,a=decide(args,b)
        for arm in k.ARMS[1:]:self.assertEqual(p[arm].tobytes(),p[k.ARMS[0]].tobytes());self.assertEqual(o[arm],o[k.ARMS[0]])
    def test_no_query_support_is_not_cold(self):
        args=clone(self.args);args[3]['eligible'][1:]=False;refresh(args);b=fit(args);c=b['audit']['counts']
        self.assertEqual(c['supported_years'],2);self.assertEqual(c['no_prediction_years'],2);self.assertEqual(c['actual_model_fits'],0)
    def test_prefix_no_query_year_does_not_borrow_later_full_tree(self):
        args=clone(self.args);args[3]['eligible'][2]=False;refresh(args)
        full=fit(args);prefix=fit(args,3);physical=fit(fixtures.physical(args,3))
        self.assertEqual(full['audit']['counts']['actual_model_fits'],6)
        self.assertEqual(prefix['audit']['counts']['actual_model_fits'],3)
        self.assertEqual(prefix['audit']['counts']['no_prediction_years'],1)
        for role in k.ROLES:
            row=prefix[role]['audit']['annual'][1]
            self.assertEqual(row['status'],'no_prediction_rows');self.assertEqual(row['query_row_ids'],[])
            if role!='calibration_ridge':self.assertIsNone(row['tree']);self.assertIsNotNone(full[role]['audit']['annual'][1]['tree'])
            for key in ('raw','available','rank'):self.assertEqual(prefix[role][key].tobytes(),physical[role][key].tobytes())
        self.assertEqual(decide(args,prefix,3)[0],decide(args,full,3)[0])
    def test_initial_cached_selection_does_not_inspect_any_annual_tree(self):
        b=deepcopy(self.full)
        for role in k.ROLES:b[role]['audit']['annual']=Poison()
        for n in (0,1):
            got=decide(self.args,b,n);expected=decide(self.args,self.full,n)
            self.assertEqual(got[0],expected[0]);self.assertEqual(got[3],expected[3])
    def test_month18_gate_and_leaf50_not_source_gate(self):
        for months in (17,18):
            args=clone(self.args);t=args[1];t['common_good'][:]=False
            dates=t['dates'];allowed=np.unique(dates[(dates>np.datetime64('2011-01-01'))&(t['ends']<=np.datetime64('2015-12-31'))])[-months:]
            for d in allowed:t['common_good'][np.flatnonzero(dates==d)[:2]]=True
            refresh(args);b=fit(args,2);r=b['calibration_ridge']['audit']['annual'][0];self.assertEqual(r['training_months'],months);self.assertEqual(b['audit']['counts']['monotone_tree_fits'],int(months==18))
            if months==18:self.assertEqual(len(r['training_row_ids']),36)
    def test_all_four_targets_and_whole_month_cohort(self):
        args=clone(self.args);t=args[1];one=np.flatnonzero(t['dates']==np.datetime64('2014-02-28'));t['target_loss_utility'][one[1:]]=np.nan;two=np.flatnonzero(t['dates']==np.datetime64('2014-03-31'));t['target_vol_scaled'][two[2:]]=np.nan
        refresh(args);b=fit(args,2);ids=b['calibration_ridge']['audit']['annual'][0]['training_row_ids'];self.assertTrue(set(one).isdisjoint(ids));self.assertTrue(set(two[:2])<=set(ids));self.assertTrue(set(two[2:]).isdisjoint(ids))
    def test_exact_annual_cutoff_fiveyear_and12month_weights(self):
        for r in self.full['calibration_ridge']['audit']['annual']:
            ids=np.asarray(r['training_row_ids']);t=self.args[1];self.assertTrue(np.all(t['dates'][ids]>np.datetime64(r['lower_bound'])));self.assertTrue(np.all(t['ends'][ids]<=np.datetime64(r['cutoff'])))
            W=k.model.numeric.month_balanced_weights(t['dates'][ids],np.datetime64(r['cutoff']),12.);self.assertEqual(W.tobytes(),np.asarray(r['weights']).tobytes())
    def test_real_fit_calls_and_runtime_record_roles(self):
        with patch.object(k,'fit_ridge',wraps=k.fit_ridge) as rf,patch.object(k,'_fit_tree',wraps=k._fit_tree) as tf:b=fit(self.args)
        self.assertEqual((rf.call_count,tf.call_count),(2,4));self.assertEqual([x.args[-1] for x in tf.call_args_list],['unconstrained_control']*2+['monotone_candidate']*2)
        for row in b['audit']['execution_runtime']:
            self.assertIn(row['role'],k.TREE_ROLES)
            for key in ('fit_threadpools','predict_threadpools'):self.assertTrue(row[key]);self.assertTrue(all(x['num_threads']==1 for x in row[key]))
    def test_runtime_thread_limit_failure(self):
        with patch.object(k,'threadpool_info',return_value=[{'num_threads':2}]),patch.object(k,'_fit_tree') as tf,self.assertRaisesRegex(ValueError,'thread limit'):fit(self.args)
        self.assertEqual(tf.call_count,0)
    def test_solver_error_is_hard_failure(self):
        with patch.object(k,'fit_ridge',side_effect=np.linalg.LinAlgError('synthetic')),patch.object(k,'_fit_tree') as tf,self.assertRaises(np.linalg.LinAlgError):fit(self.args)
        self.assertEqual(tf.call_count,0)
    def test_source_and_frozen_parameters_unchanged(self):
        args=clone(self.args);before=k._fingerprint(args[1:]);params=deepcopy(k.legacy.TREE_PARAMETERS);fit(args);self.assertEqual(before,k._fingerprint(args[1:]));self.assertEqual(params,k.legacy.TREE_PARAMETERS)


class TreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.tree=cls.bundle['monotone_candidate']['audit']['annual'][0]['tree']
    def test_parameters_only_one_declared_difference(self):
        a=k.parameters('unconstrained_control');b=k.parameters('monotone_candidate');self.assertEqual([x for x in a if a[x]!=b[x]],['monotonic_cst']);self.assertEqual(b['monotonic_cst'],[1]+[0]*9);self.assertEqual(len(a),21)
    def test_native_constraints_and_all_parameter_types_enforced(self):
        for constraint in ([True]+[0]*9,[np.int64(1)]+[0]*9,[1.]+[0]*9,[1]+[0]*8,[0,1]+[0]*8,tuple([1]+[0]*9)):
            p=k.parameters('monotone_candidate');p['monotonic_cst']=constraint
            with self.assertRaises(ValueError):k._parameters_check(p,'monotone_candidate')
        p=k.parameters('unconstrained_control');p['max_depth']=2.
        with self.assertRaises(ValueError):k._parameters_check(p,'unconstrained_control')
    def test_serialized_role_and_exact_manual_prediction(self):
        for role in k.TREE_ROLES:
            for row in self.bundle[role]['audit']['annual']:
                t=row['tree'];self.assertEqual(tuple(t),k.TREE_KEYS);self.assertEqual(t['role'],role);q=self.args[1]['event_plain'][row['query_row_ids']]
                self.assertEqual(k.manual_predict(t,q,role).tobytes(),np.asarray(row['raw_predictions']).tobytes())
                with self.assertRaises(ValueError):k.validate_tree(t,k.TREE_ROLES[1-k.TREE_ROLES.index(role)])
    def test_supplied_certificate_must_equal_recomputed(self):
        for change in (lambda t:t['shape_certificate'].update(required=False),lambda t:t['shape_certificate'].update(feature_index=True),lambda t:t['shape_certificate']['constrained_nodes'].clear(),lambda t:t['shape_certificate'].update(passed=1)):
            t=tiny_tree(self.tree);change(t)
            with self.assertRaises(ValueError):k.validate_tree(t)
    def test_constraint_rejects_one_ulp_decrease_no_epsilon(self):
        t=tiny_tree(self.tree);t['trees'][0]['nodes'][1]['value']=np.nextafter(2.,3.).hex()
        with self.assertRaisesRegex(ValueError,'descendant-leaf'):k.validate_tree(t)
    def test_descendant_extrema_not_only_direct_children(self):
        t=tiny_tree(self.tree);ns=t['trees'][0]['nodes'];left=deepcopy(ns[0]);left.update(depth=1,feature_idx=1,left=3,right=4);ns[1]=left;lo=deepcopy(ns[2]);lo.update(depth=2,value=0.0.hex());hi=deepcopy(lo);hi['value']=3.0.hex();ns.extend([lo,hi])
        with self.assertRaisesRegex(ValueError,'descendant-leaf'):k.validate_tree(t)
        ns[4]['value']=2.0.hex();t['shape_certificate']=k._shape_certificate(t);k.validate_tree(t)
        c=t['shape_certificate']['constrained_nodes'][0];self.assertEqual(c['left_leaf_indices'],[3,4]);self.assertEqual(c['left_max'],2.0.hex())
    def test_other_coordinate_can_decrease_and_plateau_allowed(self):
        t=tiny_tree(self.tree);ns=t['trees'][0]['nodes'];ns[1]['value']=2.0.hex();t['shape_certificate']=k._shape_certificate(t);k.validate_tree(t)
        ns[0]['feature_idx']=1;ns[1]['value']=3.0.hex();t['shape_certificate']=k._shape_certificate(t);k.validate_tree(t);self.assertEqual(t['shape_certificate']['constrained_nodes'],[])
    def test_unconstrained_tree_can_decrease(self):
        t=tiny_tree(self.tree,'unconstrained_control');t['trees'][0]['nodes'][1]['value']=3.0.hex();t['shape_certificate']=k._shape_certificate(t);k.validate_tree(t);self.assertFalse(t['shape_certificate']['required'])
    def test_tiny_threshold_equality_and_nextafter(self):
        t=tiny_tree(self.tree);X=np.zeros((2,10));X[1,0]=np.nextafter(0.,1.);np.testing.assert_array_equal(k.manual_predict(t,X),[1.,2.])
    def test_malformed_tree_structure_and_float_values_rejected(self):
        mutations=[lambda t:t.update(n_iter=True),lambda t:t['trees'][0]['nodes'][0].update(left=0),lambda t:t['trees'][0]['nodes'][0].update(is_categorical=1),lambda t:t['trees'][1].update(iteration=0),lambda t:t['trees'][0]['nodes'][1].update(value='nan'),lambda t:t['baseline'].update(dtype='<f4'),lambda t:t['parameters'].update(max_depth=3)]
        for mutate in mutations:
            t=tiny_tree(self.tree);mutate(t)
            with self.assertRaises(ValueError):k.validate_tree(t)
    def test_direct_fits_negative_slope_control_vs_constrained(self):
        X=np.zeros((400,10));X[:,0]=np.linspace(-.5,.5,len(X));y=-X[:,0];W=np.ones(len(X));q=np.zeros((101,10));q[:,0]=np.linspace(-.5,.5,len(q))
        with threadpool_limits(limits=1):
            a=k._fit_tree(X,y,W,'unconstrained_control');b=k._fit_tree(X,y,W,'monotone_candidate')
            pa=a.predict(q);pb=b.predict(q)
        self.assertGreater(pa[0],pa[-1]);self.assertTrue(np.all(np.diff(pb)>=0.));t=k.serialize_tree(b,'monotone_candidate');self.assertEqual(pb.tobytes(),k.manual_predict(t,q).tobytes())
    def test_counterfactual_all_bins_hold_other_coordinates_fixed(self):
        t=self.tree;bounds=k._unpack(t['bin_thresholds'][0]);grid=np.unique(np.r_[-.5,bounds,np.nextafter(bounds,np.inf),.5]);q=np.repeat(np.array([[0.,.1,-.1,.2,-.2,.3,-.3,.4,-.4,0.]]),len(grid),axis=0);q[:,0]=grid
        self.assertTrue(np.all(np.diff(k.manual_predict(t,q))>=0.))
    def test_constant_target_valid_and_same_tie_convention(self):
        args=clone(self.args)
        for name in k.TARGETS:args[1]['target_'+name][:]=0.
        refresh(args);b=fit(args)
        for role in k.TREE_ROLES:
            np.testing.assert_array_equal(b[role]['rank'][1:],np.full((3,6),3.5/6-.5));self.assertTrue(b[role]['available'].all())
    def test_typed_float_preserves_signedzero(self):
        x=np.array([0.,-0.,np.nextafter(.1,1.)]);self.assertEqual(x.tobytes(),k._unpack(k._pack(x)).tobytes())


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.result=decide(cls.args,cls.bundle)
    def test_schema_and_each_arm_from_original_pre(self):
        ranks,deltas,ra=k.rank_and_delta(self.args[3],self.args[2]['monthly'],self.bundle);o,f,p,au=self.result
        self.assertEqual(tuple(au),k.AUDIT_KEYS);self.assertEqual(tuple(au['counts']),k.COUNT_FIELDS);self.assertEqual(tuple(ra),k.RANK_AUDIT_KEYS)
        for row in au['events']:self.assertEqual(tuple(row),k.EVENT_KEYS)
        for short,arm in zip(('control','candidate'),k.ARMS[1:]):
            for i,ids in enumerate(ra['active_indices']):np.testing.assert_array_equal(deltas[short][i,ids],.0375*(ranks[short][i,ids]-ranks['old_pool'][i,ids]))
            expected=p[k.ARMS[0]].copy();mask=deltas[short]!=0.;expected[mask]+=deltas[short][mask];self.assertEqual(expected.tobytes(),p[arm].tobytes())
    def test_partial_and_nonweak_exact(self):
        o,f,p,a=self.result
        for i in (0,2):
            for arm in k.ARMS[1:]:self.assertEqual(o[arm][i],o[k.ARMS[0]][i]);self.assertEqual(f[arm][i].tobytes(),f[k.ARMS[0]][i].tobytes());self.assertEqual(p[arm][i].tobytes(),p[k.ARMS[0]][i].tobytes())
    def test_zero_rank_delta_exact_original_bytes(self):
        b=deepcopy(self.bundle)
        for role in k.TREE_ROLES:b[role]=deepcopy(b['calibration_ridge'])
        o,f,p,a=decide(self.args,b)
        for arm in k.ARMS[1:]:self.assertEqual(o[arm],o[k.ARMS[0]]);self.assertEqual(f[arm].tobytes(),f[k.ARMS[0]].tobytes());self.assertEqual(p[arm].tobytes(),p[k.ARMS[0]].tobytes())
    def test_full_U_not_local_history(self):
        args=clone(self.args);args[3]['local_available'][:]=False;_,_,ra=k.rank_and_delta(args[3],args[2]['monthly'],self.bundle);self.assertEqual(ra['ranking_cells'],24)
    def test_selectors_count_separate_gates_and_final(self):
        for n in (0,4):
            study=self.args[0]
            with patch.object(k,'_episode',wraps=k._episode) as ep:
                k.calibration_decisions(study,self.args[3],self.args[2]['monthly'],self.bundle,n);self.assertEqual(ep.call_count,1)
                k.control_decisions(study,self.args[3],self.args[2]['monthly'],self.bundle,n);self.assertEqual(ep.call_count,2)
                result=decide(self.args,self.bundle,n);self.assertEqual(ep.call_count,5)
            self.assertEqual(result[3]['counts']['new_model_fits'],0)
            for key in ('incumbent_episode_calls','control_episode_calls','candidate_episode_calls'):self.assertEqual(result[3]['counts'][key],1)
    def test_incumbent_gate_fails_before_experimental_rank(self):
        with patch.object(k,'rank_and_delta') as rd,self.assertRaises(ValueError):decide(self.args,self.bundle,incumbent_validator=lambda *a:{'passed':False})
        self.assertEqual(rd.call_count,0)
    def test_original_mutation_in_incumbent_gate_rejected(self):
        args=clone(self.args)
        def gate(*a):args[3]['baseline'][0,0]+=.1;return {'passed':True}
        with self.assertRaisesRegex(ValueError,'changed by incumbent gate'):decide(args,self.bundle,incumbent_validator=gate)
    def test_consumed_raw_rank_availability_tampering_rejected(self):
        for role in k.ROLES:
            for key in ('raw','rank','available'):
                b=deepcopy(self.bundle)
                if key=='raw':b[role][key][6]=np.inf
                elif key=='rank':b[role][key][1,0]=np.nextafter(b[role][key][1,0],1.)
                else:b[role][key][6]=False
                with self.assertRaises((ValueError,AssertionError)):decide(self.args,b)
    def test_future_cached_values_not_consumed(self):
        b=deepcopy(self.bundle)
        for role in k.ROLES:b[role]['raw'][12:]=np.inf;b[role]['rank'][2:]=np.nan;b[role]['available'][12:]=False
        got=decide(self.args,b,2);ref=decide(self.args,self.bundle,2);self.assertEqual(got[0],ref[0]);self.assertEqual(got[3],ref[3])
    def test_empty_U_and_two_half_cash_conventions(self):
        for number in (0,1,2):
            args=clone(self.args);args[3]['eligible'][1:]=False;args[3]['eligible'][1:,:number]=True;refresh(args);b=fit(args);o,f,p,a=decide(args,b)
            for arm in k.ARMS:
                for row in o[arm][1:]:self.assertEqual(len(row['symbols']),number if number==2 else 0)


if __name__=='__main__':unittest.main(verbosity=2)
