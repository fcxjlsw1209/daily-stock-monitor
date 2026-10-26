"""Synthetic-only 261 source, fitting, serialization and execution tests."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import monthly_weak_tree261_kernel as k
import test_monthly_local_distance256_kernel as old


def fixture():return old.fixture()
def clone(args):return (args[0],)+deepcopy(args[1:])
def fit(args,n=None,**kw):return k.fit_models(args[1],args[2],args[3],n,**kw)
def decide(args,b,n=None,**kw):return k.decisions(args[0],args[3],args[2]['monthly'],b,n,**kw)
def physical(args,n):return old.physical(args,n)
def refresh(args):return old.refresh_original(args)

class Poison:
    def __float__(self):raise AssertionError('Future numeric consumed')
    def __deepcopy__(self,memo):raise AssertionError('Future body copied')
    def __len__(self):raise AssertionError('Future body inspected')
    def __iter__(self):raise AssertionError('Future body iterated')


def tiny_tree(reference):
    t=deepcopy(reference)
    def node(depth,leaf,value=0.):
        r={name:float(0).hex() if k.PREDICTOR_RECORD_DTYPE[name].kind=='f' else 0 for name in k.PREDICTOR_RECORD_DTYPE.names}
        r.update(depth=depth,is_leaf=leaf,count=100,value=float(value).hex());return r
    root=node(0,0);root.update(left=1,right=2,gain=float(1).hex())
    t['baseline']=k._pack(np.zeros((1,1)));t['bin_thresholds']=[k._pack(np.array([0.]))]+[k._pack(np.array([])) for _ in range(9)];t['n_bins_non_missing']=[2]+[1]*9
    for r in t['trees']:r['nodes']=[node(0,1)]
    t['trees'][0]['nodes']=[root,node(1,1,1.),node(1,1,2.)]
    k.validate_tree(t);return t


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.full=fit(cls.args)
    def test_complete_keys_counts_and_only_required_fits(self):
        b=self.full;c=b['audit']['counts'];self.assertEqual(tuple(b),k.BUNDLE_KEYS);self.assertEqual(tuple(c),k.MODEL_COUNT_FIELDS);self.assertEqual(tuple(b['audit']),k.MODEL_AUDIT_KEYS)
        self.assertEqual(c['matched_ridge_fits'],2);self.assertEqual(c['tree_fits'],2);self.assertEqual(c['actual_model_fits'],4);self.assertEqual(c['original_other_target_fits'],0);self.assertEqual(c['original_local_fits'],0)
        self.assertEqual(c['trees'],200);self.assertEqual(c['manual_tree_traversals'],100*c['prediction_rows'])
        for r in b['control']['audit']['annual']:self.assertEqual(tuple(r),k.RIDGE_ANNUAL_KEYS)
        for r in b['candidate']['audit']['annual']:self.assertEqual(tuple(r),k.TREE_ANNUAL_KEYS)
    def test_source_control_exact_and_all_labels_preserved(self):
        b=self.full;a=self.args[3];self.assertEqual(b['control']['raw'].tobytes(),a['raw_source_log'].reshape(-1).tobytes());self.assertEqual(b['control']['rank'].tobytes(),a['rank_log_excess'].tobytes())
        args,ids,_,_=k.model_inputs(self.args[1],self.args[2],a)
        self.assertEqual(b['label_values'].tobytes(),args[4]['log_excess'].tobytes());self.assertEqual(tuple(b['label_preservation']),k.LABEL_KEYS)
        for name in k.TARGETS:self.assertEqual(b['label_preservation']['target_sha256'][name],k._digest(args[4][name]))
    def test_all_ridge_years_before_any_tree_and_complete_gate(self):
        calls=[];ridge=k.fit_ridge;tree=k._fit_tree
        def rf(*args):calls.append('ridge');return ridge(*args)
        def tf(*args):self.assertEqual(calls[:3],['ridge','ridge','gate']);calls.append('tree');return tree(*args)
        def gate(view):
            self.assertIsNone(view['candidate']);self.assertEqual(view['audit']['counts']['tree_fits'],0);self.assertEqual(view['audit']['execution_runtime'],[])
            o,f,p=k.matched_decisions(self.args[0],self.args[3],self.args[2]['monthly'],view)
            orig=k.original_decisions(self.args[0],self.args[3],self.args[2]['monthly']);self.assertEqual(o,orig[0]);np.testing.assert_array_equal(f,orig[1]);np.testing.assert_array_equal(p,orig[2]);calls.append('gate');return {'passed':True,'test':'complete control'}
        with patch.object(k,'fit_ridge',side_effect=rf),patch.object(k,'_fit_tree',side_effect=tf):b=fit(self.args,control_validator=gate)
        self.assertEqual(calls,['ridge','ridge','gate','tree','tree']);self.assertEqual(b['control_gate']['test'],'complete control')
    def test_later_year_bad_cache_aborts_all_trees(self):
        args=clone(self.args);args[3]['raw_source_log'][3,0]=np.nextafter(args[3]['raw_source_log'][3,0],np.inf)
        with patch.object(k,'_fit_tree',side_effect=AssertionError('too early')) as tf:
            with self.assertRaises(AssertionError):fit(args)
        self.assertEqual(tf.call_count,0)
    def test_gate_failure_before_tree(self):
        with patch.object(k,'_fit_tree',side_effect=AssertionError('too early')) as tf:
            with self.assertRaisesRegex(ValueError,'gate failed'):fit(self.args,control_validator=lambda v:{'passed':False})
        self.assertEqual(tf.call_count,0)
    def test_gate_original_array_mutation_before_tree(self):
        args=clone(self.args)
        def gate(v):args[3]['baseline'][1,0]+=.01;return {'passed':True}
        with patch.object(k,'_fit_tree') as tf:
            with self.assertRaisesRegex(ValueError,'changed by control gate'):fit(args,control_validator=gate)
        self.assertEqual(tf.call_count,0)
    def test_gate_consumed_training_mutation_before_tree(self):
        for key in ('train_plain','target_log_excess','common_good'):
            args=clone(self.args)
            def gate(v):
                if key=='common_good':args[1][key][0]=False
                elif key=='train_plain':args[1][key][0,0]+=.01
                else:args[1][key][0]+=.01
                return {'passed':True}
            with patch.object(k,'_fit_tree') as tf:
                with self.assertRaisesRegex(ValueError,'source changed'):fit(args,control_validator=gate)
            self.assertEqual(tf.call_count,0)
    def test_n0_partial_no_training_numeric_and_no_fits(self):
        for n in (0,1):
            args=clone(self.args)
            for key in ('train_plain','event_plain')+tuple('target_'+v for v in k.TARGETS):
                args[1][key]=args[1][key].astype(object);args[1][key][:]=Poison()
            args[2]['training']['symbols']=[Poison() for _ in args[2]['training']['symbols']]
            with patch.object(k,'fit_ridge',side_effect=AssertionError('no fit')),patch.object(k,'_fit_tree',side_effect=AssertionError('no fit')):b=fit(args,n)
            self.assertEqual(b['audit']['counts']['actual_model_fits'],0);self.assertEqual(len(b['label_values']),0);self.assertEqual(b['candidate']['raw'].tobytes(),args[3]['raw_source_log'][:n].reshape(-1).tobytes())
    def test_future_numeric_and_identity_before_cast(self):
        args=clone(self.args);future=args[1]['ends']>np.datetime64('2015-12-31')
        for key in ('train_plain',)+tuple('target_'+v for v in k.TARGETS):args[1][key]=args[1][key].astype(object);args[1][key][future]=Poison()
        args[1]['event_plain']=args[1]['event_plain'].astype(object);args[1]['event_plain'][12:]=Poison()
        count=int(np.count_nonzero(future))//6
        for i in range(len(args[2]['training']['symbols'])-count,len(args[2]['training']['symbols'])):args[2]['training']['symbols'][i]=Poison()
        for key in ('dates','ends','symbols'):
            for i in (2,3):args[2]['monthly'][key][i]=Poison()
        b=fit(args,2);np.testing.assert_array_equal(b['candidate']['raw'],self.full['candidate']['raw'][:12])
    def test_all_prefixes_logical_physical_and_cached(self):
        for n in range(5):
            b=fit(self.args,n);small=physical(self.args,n);c=fit(small)
            for role in ('control','candidate'):
                for key in ('raw','available','rank'):np.testing.assert_array_equal(b[role][key],c[role][key])
            x=decide(self.args,b,n);y=decide(small,self.full)
            self.assertEqual(x[0],y[0]);self.assertEqual(x[3],y[3])
            for arm in k.ARMS:np.testing.assert_array_equal(x[1][arm],y[1][arm]);np.testing.assert_array_equal(x[2][arm],y[2][arm])
    def test_supported_nonweak_year_still_fits(self):
        args=clone(self.args);args[3]['state_weak_narrow'][:]=False;b=fit(args)
        self.assertEqual(b['audit']['counts']['tree_fits'],2);o,f,p,a=decide(args,b);self.assertEqual(a['counts']['changed_delta_cells'],0)
        for arm in k.ARMS[1:]:self.assertEqual(p[arm].tobytes(),p[k.ARMS[0]].tobytes())
    def test_original_availability_cannot_hide_required_row(self):
        args=clone(self.args);args[3]['forecast_available'][1,0]=False
        with patch.object(k,'_fit_tree') as tf:
            with self.assertRaises((AssertionError,ValueError)):fit(args)
        self.assertEqual(tf.call_count,0)
    def test_no_prediction_rows_not_cold(self):
        args=clone(self.args);args[3]['eligible'][1:]=False;refresh(args);b=fit(args)
        c=b['audit']['counts'];self.assertEqual(c['supported_years'],2);self.assertEqual(c['no_prediction_years'],2);self.assertEqual(c['actual_model_fits'],0)
    def test_min18_months_and_no_leaf50_admission_gate(self):
        for months in (17,18):
            args=clone(self.args);t=args[1];t['common_good'][:]=False
            dates=t['dates'];allowed=np.unique(dates[(dates>np.datetime64('2011-01-01'))&(t['ends']<=np.datetime64('2015-12-31'))])[-months:]
            for d in allowed:t['common_good'][np.flatnonzero(dates==d)[:2]]=True
            refresh(args);b=fit(args,2);r=b['control']['audit']['annual'][0];self.assertEqual(r['training_months'],months)
            self.assertEqual(b['audit']['counts']['tree_fits'],int(months==18))
            if months==18:self.assertEqual(len(r['training_row_ids']),36)
    def test_all_four_target_gate_and_whole_month_singleton(self):
        args=clone(self.args);t=args[1];month=np.flatnonzero(t['dates']==np.datetime64('2014-02-28'));t['target_loss_utility'][month[1:]]=np.nan
        second=np.flatnonzero(t['dates']==np.datetime64('2014-03-31'));t['target_vol_scaled'][second[2:]]=np.nan
        refresh(args);b=fit(args,2);used=b['control']['audit']['annual'][0]['training_row_ids'];self.assertTrue(set(month).isdisjoint(used));self.assertTrue(set(second[:2])<=set(used));self.assertTrue(set(second[2:]).isdisjoint(used))
    def test_strict_fiveyear_and_maturity_cohort(self):
        r=self.full['control']['audit']['annual'][0];t=self.args[1];ids=np.asarray(r['training_row_ids']);self.assertTrue(np.all(t['dates'][ids]>np.datetime64(r['lower_bound'])));self.assertTrue(np.all(t['ends'][ids]<=np.datetime64(r['cutoff'])))
        W=np.asarray(r['weights']);expected=k.model.numeric.month_balanced_weights(t['dates'][ids],np.datetime64(r['cutoff']),12.);self.assertEqual(W.tobytes(),expected.tobytes());self.assertAlmostEqual(W.sum(),len(W),11)
    def test_solve_failure_hard_no_lstsq(self):
        with patch.object(np.linalg,'solve',side_effect=np.linalg.LinAlgError('synthetic')),patch.object(np.linalg,'lstsq',side_effect=AssertionError('no rescue')),patch.object(k,'_fit_tree') as tf:
            with self.assertRaises(np.linalg.LinAlgError):fit(self.args)
        self.assertEqual(tf.call_count,0)
    def test_runtime_and_real_fit_calls(self):
        with patch.object(k,'fit_ridge',wraps=k.fit_ridge) as rf,patch.object(k,'_fit_tree',wraps=k._fit_tree) as tf,patch.object(k.model,'annual_forecasts',side_effect=AssertionError('no extra source model')),patch.object(k.model,'local_corrections',side_effect=AssertionError('no local fit')):b=fit(self.args)
        self.assertEqual(rf.call_count,2);self.assertEqual(tf.call_count,2)
        for row in b['audit']['execution_runtime']:
            for key in ('fit_threadpools','predict_threadpools'):
                self.assertTrue(row[key]);self.assertTrue(all(x['num_threads']==1 for x in row[key]))
    def test_runtime_wrong_effective_limit_rejected(self):
        with patch.object(k,'threadpool_info',return_value=[{'num_threads':2}]),patch.object(k,'_fit_tree') as tf:
            with self.assertRaisesRegex(ValueError,'thread limit'):fit(self.args)
        self.assertEqual(tf.call_count,0)
    def test_unchanged_inputs_and_labels(self):
        args=clone(self.args);t,m,a=args[1:];before=k.bridge._snapshot(a);tb={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in t.items()};mb=deepcopy(m);fit(args)
        self.assertEqual(before,k.bridge._snapshot(a));self.assertEqual(tb,{key:(v.dtype.str,v.shape,v.tobytes()) for key,v in t.items()});self.assertEqual(m,mb)


class TreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.tree=cls.bundle['candidate']['audit']['annual'][0]['tree']
    def test_complete_serialization_and_exact_manual(self):
        t=self.tree;self.assertEqual(tuple(t),k.TREE_KEYS);self.assertEqual(t['parameters'],k.TREE_PARAMETERS);self.assertEqual(len(k.TREE_PARAMETERS),21);self.assertEqual(len(t['trees']),100)
        row=self.bundle['candidate']['audit']['annual'][0];query=self.args[1]['event_plain'][row['query_row_ids']];self.assertEqual(k.manual_predict(t,query).tobytes(),np.asarray(row['raw_predictions']).tobytes())
    def test_threshold_equality_left_and_oneulp_right(self):
        t=tiny_tree(self.tree);X=np.zeros((2,10));X[1,0]=np.nextafter(0.,1.);np.testing.assert_array_equal(k.manual_predict(t,X),[1.,2.])
    def test_positive_inf_only_active_threshold(self):
        t=tiny_tree(self.tree);t['trees'][0]['nodes'][0]['num_threshold']='inf';np.testing.assert_array_equal(k.manual_predict(t,np.zeros((2,10))),[1.,1.])
        for key,value,index in [('num_threshold','-inf',0),('num_threshold','nan',0),('value','inf',1),('num_threshold','inf',1)]:
            bad=tiny_tree(self.tree);bad['trees'][0]['nodes'][index][key]=value
            with self.assertRaises(ValueError):k.validate_tree(bad)
    def test_topology_dtype_iteration_parameter_rejected(self):
        mutations=[lambda t:t.update(n_iter=99),lambda t:t['parameters'].update(max_depth=3),lambda t:t['node_dtype'][0].__setitem__(1,'<f4'),lambda t:t['trees'][0]['nodes'][0].update(left=0),lambda t:t['trees'][0]['nodes'][0].update(is_categorical=1),lambda t:t['trees'][1].update(iteration=0),lambda t:t['baseline'].update(dtype='<f4')]
        for change in mutations:
            t=tiny_tree(self.tree);change(t)
            with self.assertRaises(ValueError):k.validate_tree(t)
    def test_leaf_baseline_threshold_ulp_changes_are_not_rounded(self):
        X=np.zeros((1,10));t=tiny_tree(self.tree);base=k.manual_predict(t,X)
        for kind in ('leaf','baseline','threshold'):
            bad=deepcopy(t)
            if kind=='leaf':bad['trees'][0]['nodes'][1]['value']=np.nextafter(1.,2.).hex()
            elif kind=='baseline':bad['baseline']=k._pack(np.array([[np.finfo(float).eps]]))
            else:bad['trees'][0]['nodes'][0]['num_threshold']=np.nextafter(0.,-1.).hex()
            self.assertNotEqual(k.manual_predict(bad,X).tobytes(),base.tobytes())
    def test_iteration_order_and_no_second_learning_rate(self):
        t=tiny_tree(self.tree)
        for r in t['trees']:r['nodes']=deepcopy(t['trees'][1]['nodes'])
        for i,v in enumerate((1e16,-1e16,1.)):t['trees'][i]['nodes'][0]['value']=float(v).hex()
        self.assertEqual(k.manual_predict(t,np.zeros((1,10)))[0],1.)
        t['trees'][1]['nodes'],t['trees'][2]['nodes']=t['trees'][2]['nodes'],t['trees'][1]['nodes'];self.assertEqual(k.manual_predict(t,np.zeros((1,10)))[0],0.)
        t=tiny_tree(self.tree);self.assertEqual(k.manual_predict(t,np.zeros((1,10)))[0],1.);self.assertNotEqual(k.manual_predict(t,np.zeros((1,10)))[0],.05)
    def test_constant_response_valid_tree_exact_tie_rank(self):
        args=clone(self.args)
        for name in k.TARGETS:args[1]['target_'+name][:]=0.
        refresh(args);b=fit(args);self.assertEqual(b['audit']['counts']['tree_fits'],2);self.assertTrue(np.all(b['candidate']['available']))
        np.testing.assert_array_equal(b['candidate']['rank'][1:],np.full((3,6),3.5/6-.5));self.assertEqual(b['audit']['counts']['tree_nodes'],200)
    def test_typed_float_roundtrip_preserves_signed_zero(self):
        x=np.array([0.,-0.,np.nextafter(.1,1.)]);self.assertEqual(k._unpack(k._pack(x)).tobytes(),x.tobytes())


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.result=decide(cls.args,cls.bundle)
    def test_original_and_matched_exact_fullP_F_orders(self):
        s,t,m,a=self.args;c,post=s.choices(s.configs()[4],a,m['monthly'],4,policy=dict(s.POLICY_FIELDS));o,f,p,au=self.result
        for arm in k.ARMS[:2]:self.assertEqual(o[arm],s.rp.make_orders(c,post,m['monthly']));self.assertEqual(f[arm].tobytes(),post.tobytes());self.assertEqual(p[arm].tobytes(),s.modified_pre_score(s.configs()[4],a,4).tobytes())
    def test_exact_weak_delta_local_basis_unchanged(self):
        r,d,au=k.rank_and_delta(self.args[3],self.args[2]['monthly'],self.bundle);o,f,p,_=self.result
        for i,ids in enumerate(au['active_indices']):
            np.testing.assert_array_equal(d[i,ids],.0375*(r['tree'][i,ids]-r['old_pool'][i,ids]))
        expected=p[k.ARMS[0]].copy();changed=d!=0;expected[changed]+=d[changed];self.assertEqual(expected.tobytes(),p[k.ARMS[2]].tobytes())
    def test_calm_partial_and_zero_delta_exact_bytes(self):
        o,f,p,au=self.result
        for i in (0,2):self.assertEqual(p[k.ARMS[2]][i].tobytes(),p[k.ARMS[0]][i].tobytes());self.assertEqual(f[k.ARMS[2]][i].tobytes(),f[k.ARMS[0]][i].tobytes());self.assertEqual(o[k.ARMS[2]][i],o[k.ARMS[0]][i])
        b=deepcopy(self.bundle);b['candidate']=deepcopy(b['control']);o,f,p,au=decide(self.args,b);self.assertEqual(p[k.ARMS[2]].tobytes(),p[k.ARMS[0]].tobytes())
    def test_full_U_not_local_available(self):
        args=clone(self.args);args[3]['local_available'][:]=False;r,d,a=k.rank_and_delta(args[3],args[2]['monthly'],self.bundle);self.assertEqual(a['ranking_cells'],24)
    def test_one_selector_per_arm_including_empty(self):
        for n in (0,4):
            s=self.args[0]
            with patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:result=decide(self.args,self.bundle,n)
            self.assertEqual(ep.call_count,3);self.assertEqual(result[3]['counts']['new_model_fits'],0)
            for key in ('incumbent_episode_calls','matched_episode_calls','tree_episode_calls'):self.assertEqual(result[3]['counts'][key],1)
    def test_inc_gate_before_candidate_and_mutation_rejected(self):
        with patch.object(k,'rank_and_delta',side_effect=AssertionError('too early')) as rd:
            with self.assertRaises(ValueError):decide(self.args,self.bundle,incumbent_validator=lambda *x:{'passed':False})
        self.assertEqual(rd.call_count,0)
        args=clone(self.args)
        def gate(*x):args[3]['baseline'][0,0]+=.1;return {'passed':True}
        with patch.object(k,'rank_and_delta') as rd:
            with self.assertRaisesRegex(ValueError,'changed by incumbent gate'):decide(args,self.bundle,incumbent_validator=gate)
        self.assertEqual(rd.call_count,0)
    def test_consumed_tamper_rejected_future_ignored(self):
        for key in ('raw','rank','available'):
            b=deepcopy(self.bundle)
            if key=='raw':b['candidate'][key][6]=np.inf
            elif key=='rank':b['candidate'][key][1,0]=np.nextafter(b['candidate'][key][1,0],1.)
            else:b['candidate'][key][6]=False
            with self.assertRaises((ValueError,AssertionError)):decide(self.args,b,2)
        b=deepcopy(self.bundle);b['candidate']['raw'][12:]=np.inf;b['candidate']['rank'][2:]=np.inf
        x=decide(physical(self.args,2),b);y=decide(self.args,self.bundle,2);self.assertEqual(x[0],y[0]);self.assertEqual(x[3],y[3])
    def test_no_input_mutation_all_schemas(self):
        before=k.bridge._snapshot(self.args[3]);o,f,p,a=decide(self.args,self.bundle);self.assertEqual(before,k.bridge._snapshot(self.args[3]));self.assertEqual(tuple(a),k.AUDIT_KEYS);self.assertEqual(tuple(a['counts']),k.COUNT_FIELDS);self.assertEqual(tuple(a['rank_delta']),k.RANK_AUDIT_KEYS)
        for event in a['events']:self.assertEqual(tuple(event),k.EVENT_KEYS)

if __name__=='__main__':unittest.main(verbosity=2)
