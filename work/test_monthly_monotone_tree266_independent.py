"""Synthetic-only266 independent source, dual-tree, shape and decision tests."""
import copy
import hashlib
from functools import lru_cache
import unittest
from unittest.mock import patch
import numpy as np
import monthly_monotone_tree266_independent as audit
import test_monthly_asset_target255_independent as frozen

audit.require(hashlib.sha256((audit.ROOT/'work/test_monthly_asset_target255_independent.py').read_bytes()).hexdigest()==
 'b6aea9e714da159927d706ccd0e8ec4a83828a89b9c9df89e84c35cadf9febf8','Pinned synthetic fixture')

def fixture():return frozen.fixture()
@lru_cache(None)
def _bundle(stage='full'):return audit.model_oracle(*fixture(),stage=stage)[0]
def bundle(stage='full'):return copy.deepcopy(_bundle(stage))
def decide(a,m,b,n=None):
    F,choices,P,au=audit.selection_oracle(a,m,b,n);return audit.orders_from_choices(F,choices,m),F,P,au

def physical(t,m,a,n):
    mm=dict(m,monthly={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m['monthly'].items()})
    aa={k:(v[:n].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(m['monthly']['dates']) else copy.deepcopy(v)) for k,v in a.items()}
    tt=dict(t,event_dates=t['event_dates'][:n*a['eligible'].shape[1]],event_plain=t['event_plain'][:n*a['eligible'].shape[1]])
    return tt,mm,aa

class Poison:
    def __float__(self):raise AssertionError('Future numeric body')
    def __str__(self):raise AssertionError('Future identifier')
    def __deepcopy__(self,memo):raise AssertionError('Future copy')

class SourceTests(unittest.TestCase):
    def test_canonical_calibration_svd_before_any_tree(self):
        t,m,a=fixture();b=bundle('calibration')
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as svd,patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('tree')):
            r=audit.verify_calibration_models(t,m,a,b)
        self.assertEqual(svd.call_count,r['independent_svd_solves']);self.assertEqual(r['independent_tree_refits'],0)
        self.assertEqual(r['manual_tree_traversals'],0);self.assertFalse(r['candidate_models_computed'])

    def test_full_later_source_mismatch_stops_all_tree_fits(self):
        t,m,a=fixture();a['raw_source_log'][-1,0]=np.nextafter(a['raw_source_log'][-1,0],np.inf)
        with patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('early tree')),self.assertRaises(ValueError):audit.model_oracle(t,m,a)

    def test_n0_partial_retrieve_no_training_numeric(self):
        t,m,a=fixture();poison={k:Poison() for k in t}
        for n in (0,1):
            with patch.object(audit,'ridge_fit',side_effect=AssertionError('ridge')),patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('tree')):
                b,inp,_=audit.model_oracle(poison,m,a,n);r=audit.verify_models(poison,m,a,b,n)
            self.assertEqual(len(inp['dates']),0);self.assertEqual(r['counts']['actual_model_fits'],0)

    def test_future_numeric_and_symbol_bodies_are_not_consumed(self):
        t,m,a=fixture();expected=audit.model_oracle(t,m,a,2)[0];future=t['ends']>np.datetime64('2015-12-31')
        for key in ('train_plain',)+tuple('target_'+k for k in audit.TARGETS):t[key]=t[key].astype(object);t[key][future]=Poison()
        for j,end in enumerate(m['training']['ends']):
            if end>'2015-12-31':m['training']['symbols'][j]=Poison()
        t['event_plain']=t['event_plain'].astype(object);t['event_plain'][12:]=Poison()
        actual=audit.model_oracle(t,m,a,2)[0];audit.source.check_tree(actual,expected,'Future input poison')

    def test_all_four_target_common_rows_and_minimum_two_per_month(self):
        inp=audit.model_inputs(*fixture());idx=np.flatnonzero(inp['dates']==inp['dates'][0]);inp['good'][idx]=False;inp['good'][idx[:1]]=True
        self.assertFalse(audit.usable_rows(inp)[idx].any());inp['good'][idx[1]]=True;self.assertEqual(audit.usable_rows(inp)[idx].sum(),2)
        inp['targets']['loss_utility'][idx[1]]=np.nan;self.assertFalse(audit.usable_rows(inp)[idx].any())

    def test_12month_weights_and_duplicate_month_mass(self):
        dates=np.array(['2015-01-31']*2+['2015-02-28']*4,dtype='datetime64[D]');w=audit.month_weights(dates,np.datetime64('2015-12-31'))
        self.assertAlmostEqual(w.sum(),6);self.assertAlmostEqual(w[:2].sum()/w[2:].sum(),2**(-1/12))

    def test_five_year_cutoff_is_prior_december(self):
        t,m,a=fixture()
        for row in bundle()['calibration_ridge']['audit']['annual']:
            ids=row['training_row_ids'];self.assertTrue((t['dates'][ids]>np.datetime64(row['lower_bound'])).all());self.assertTrue((t['ends'][ids]<=np.datetime64(row['cutoff'])).all())

    def test_cold17_and_supported18_with_empty_prediction_population(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a,2);months=np.unique(inp['dates'].astype('datetime64[M]'))
        for count in (17,18):
            x=copy.deepcopy(inp);x['good']&=np.isin(x['dates'].astype('datetime64[M]'),months[-count:]);aa=copy.deepcopy(a)
            aa['eligible'][1]=False;aa['forecast_available'][1]=False;aa['model_available'][1]=False;aa.pop('rank_local_linear',None);x['query_good'][6:]=False
            r,_=audit.ridge_oracle(x,aa);self.assertEqual(r['audit']['annual'][0]['status'],'cold_start' if count==17 else 'no_prediction_rows')

    def test_no_prediction_prefix_does_not_borrow_later_same_year_tree(self):
        t,m,a=fixture();m['monthly']['dates'][2]='2016-09-30';m['monthly']['ends'][2]='2016-10-31';t['event_dates'][12:18]=np.datetime64('2016-09-30')
        a['eligible'][1]=False;a['forecast_available'][1]=False;a['model_available'][1]=False;a.pop('rank_local_linear',None)
        # Synchronize only fake pooled source after altering its fake query year.
        inp=audit.model_inputs(t,m,a);cal,_=audit.ridge_oracle(inp,a);a['raw_source_log'][:]=cal['raw'].reshape(4,6)
        for i in (2,3):a['rank_log_excess'][i]=audit.previous._rank(a['raw_source_log'][i])
        full,_,_=audit.model_oracle(t,m,a);prefix,_,_=audit.model_oracle(t,m,a,2)
        self.assertEqual(full['calibration_ridge']['audit']['annual'][0]['status'],'fit')
        self.assertEqual(prefix['calibration_ridge']['audit']['annual'][0]['status'],'no_prediction_rows')
        for role in audit.ROLES:self.assertIsNone(prefix[role]['audit']['annual'][0]['tree'])
        self.assertEqual(prefix['audit']['counts']['actual_model_fits'],0)

    def test_no_weak_gate_in_training_schedule(self):
        t,m,a=fixture();a['state_weak_narrow'][:]=False;b,_,_=audit.model_oracle(t,m,a)
        self.assertEqual(b['audit']['counts']['unconstrained_tree_fits'],3);self.assertEqual(b['audit']['counts']['monotone_tree_fits'],3)
        self.assertTrue(all(not d.any() for d in audit.rank_and_delta(a,m['monthly'],b)[1].values()))

    def test_ridge_failhard_and_plain_rank_bound(self):
        rng=np.random.default_rng(266);X=rng.uniform(-.4,.4,(80,10));y=X[:,0];Q=X[:3];w=np.ones(80)
        with patch.object(np.linalg,'solve',side_effect=np.linalg.LinAlgError('stop')),self.assertRaises(np.linalg.LinAlgError):audit.ridge_fit(X,y,Q,w)
        t,m,a=fixture();t['event_plain'][6,0]=np.nextafter(.5,np.inf)
        with self.assertRaises(ValueError):audit.model_oracle(t,m,a)

class ModelTests(unittest.TestCase):
    def test_full_actual_fit_svd_and_manual_work_counts(self):
        t,m,a=fixture();b=bundle()
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as svd,patch.object(audit.HistGradientBoostingRegressor,'fit',autospec=True,side_effect=audit.HistGradientBoostingRegressor.fit) as fit:
            r=audit.verify_models(t,m,a,b)
        self.assertEqual(svd.call_count,r['independent_svd_solves']);self.assertEqual(fit.call_count,r['independent_tree_refits'])
        self.assertEqual(r['independent_tree_refits'],r['unconstrained_tree_refits']+r['monotone_tree_refits'])
        self.assertEqual(r['manual_tree_traversals'],2*(r['counts']['unconstrained_manual_tree_traversals']+r['counts']['monotone_manual_tree_traversals']))
        self.assertEqual(r['shape_certificates_checked'],3*r['independent_tree_refits']);self.assertFalse(r['independent_tree_training_algorithm']);self.assertTrue(r['shared_tree_trainer'])

    def test_control_stage_no_monotone_fit(self):
        t,m,a=fixture();r=audit.verify_control_models(t,m,a,bundle('control'))
        self.assertEqual(r['monotone_tree_refits'],0);self.assertEqual(r['unconstrained_tree_refits'],3);self.assertFalse(r['candidate_models_computed'])

    def test_failed_gates_before_oracle_fits(self):
        for field in ('calibration_gate','control_gate'):
            b=bundle();b[field]={'passed':False}
            with patch.object(audit,'ridge_fit',side_effect=AssertionError('fit before gate')),self.assertRaises(ValueError):audit.verify_models(*fixture(),b)

    def test_exact_complete_receipts_and_tampered_work_rejected(self):
        t,m,a=fixture();cal=bundle('calibration');cg=audit.verify_calibration_models(t,m,a,cal)
        ctl=bundle('control');ctl['calibration_gate']=cg;ug=audit.verify_control_models(t,m,a,ctl)
        full=bundle();full['calibration_gate']=cg;full['control_gate']=ug;audit.verify_models(t,m,a,full)
        full['control_gate']['manual_tree_traversals']+=1
        with self.assertRaises(ValueError):audit.verify_models(t,m,a,full)

    def test_type_sensitive_constraint_vector_and_role(self):
        tree=copy.deepcopy(bundle()[audit.ROLES[1]]['audit']['annual'][0]['tree'])
        for mutate in (lambda t:t['parameters'].__setitem__('monotonic_cst',[True]+[0]*9),lambda t:t.__setitem__('role',audit.ROLES[0]),lambda t:t['parameters'].__setitem__('monotonic_cst',[1]*10)):
            bad=copy.deepcopy(tree);mutate(bad)
            with self.assertRaises(ValueError):audit.manual_predict(bad,np.zeros((1,10)),audit.ROLES[1])

    def test_serialized_node_and_manual_one_ulp_tamper(self):
        tree=copy.deepcopy(bundle()[audit.ROLES[1]]['audit']['annual'][0]['tree'])
        cert=tree['shape_certificate']
        if cert['constrained_nodes']:
            cert['constrained_nodes'][0]['left_max']=(np.nextafter(float.fromhex(cert['constrained_nodes'][0]['left_max']),np.inf)).hex()
            with self.assertRaises(ValueError):audit.manual_predict(tree,np.zeros((1,10)),audit.ROLES[1])
        b=bundle();b[audit.ROLES[0]]['audit']['annual'][0]['raw_predictions'][0]+=1e-12
        with self.assertRaises(ValueError):audit.verify_models(*fixture(),b)

    def test_full_labels_source_audit_counts_and_runtime_tamper(self):
        for field in ('labels','ids','counts','runtime'):
            b=bundle()
            if field=='labels':b['label_values'][0]+=1e-9
            elif field=='ids':b['training_row_ids'][0]+=1
            elif field=='counts':b['audit']['counts']['monotone_tree_nodes']+=1
            else:b['audit']['execution_runtime'][0]['role']='monotone_candidate'
            with self.subTest(field=field),self.assertRaises(ValueError):audit.verify_models(*fixture(),b)

    def test_legacy_control_projection_exact_261_fields(self):
        tr=bundle()[audit.ROLES[0]]['audit']['annual'][0]['tree'];projected=audit.legacy_unconstrained_tree(tr)
        self.assertEqual(projected['schema'],'monthly_weak_tree261_serialized_tree_v1');self.assertNotIn('shape_certificate',projected);self.assertNotIn('role',projected)
        expected=copy.deepcopy(tr);expected.pop('shape_certificate');expected.pop('role');expected['schema']=projected['schema'];audit.source.check_tree(projected,expected,'Explicit projection')

class ShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(2661);cls.X=rng.uniform(-.5,.5,(400,10));cls.y=-cls.X[:,0]+.01*cls.X[:,1]
        cls.trees={}
        with audit.threadpool_limits(limits=1):
            for role in audit.ROLES:
                fit=audit.HistGradientBoostingRegressor(**audit.parameters(role)).fit(cls.X,cls.y,sample_weight=np.ones(400));cls.trees[role]=audit.serialize_tree(fit,role)

    def test_monotone_conditional_query_lattice_not_cross_year_claim(self):
        Q=np.zeros((101,10));Q[:,0]=np.linspace(-.5,.5,101)
        unrestricted=audit.manual_predict(self.trees[audit.ROLES[0]],Q,audit.ROLES[0]);constrained=audit.manual_predict(self.trees[audit.ROLES[1]],Q,audit.ROLES[1])
        self.assertTrue((np.diff(constrained)>=0).all());self.assertTrue((np.diff(unrestricted)<0).any())

    def test_descendant_leaf_extrema_not_just_child_root_values(self):
        tree=copy.deepcopy(self.trees[audit.ROLES[0]]);tree['role']=audit.ROLES[1];tree['parameters']=audit.parameters(audit.ROLES[1])
        with self.assertRaisesRegex(ValueError,'descendant leaves'):audit.shape_certificate(tree,audit.ROLES[1])

    def test_threshold_equality_manual_iteration_and_no_second_shrink(self):
        tr=self.trees[audit.ROLES[0]];root=next(t['nodes'][0] for t in tr['trees'] if not t['nodes'][0]['is_leaf'])
        Q=np.zeros((2,10));Q[:,root['feature_idx']]=float.fromhex(root['num_threshold']);Q[1,root['feature_idx']]=np.nextafter(Q[1,root['feature_idx']],np.inf)
        expected=[]
        for x in Q:
            v=0.;v+=float.fromhex(tr['baseline']['values'][0][0])
            for t in tr['trees']:
                nodes=t['nodes'];j=0
                while not nodes[j]['is_leaf']:j=nodes[j]['left'] if x[nodes[j]['feature_idx']]<=float.fromhex(nodes[j]['num_threshold']) else nodes[j]['right']
                v+=float.fromhex(nodes[j]['value'])
            expected.append(v)
        actual=audit.manual_predict(tr,Q,audit.ROLES[0]);audit.exact_array(actual,np.asarray(expected),'Literal traversal')
        wrong=copy.deepcopy(tr)
        for t in wrong['trees']:
            for node in t['nodes']:
                if node['is_leaf']:node['value']=(.05*float.fromhex(node['value'])).hex()
        self.assertNotEqual(audit.manual_predict(wrong,Q,audit.ROLES[0]).tobytes(),actual.tobytes())

    def test_constant_leaf_and_no_constrained_split_are_valid(self):
        with audit.threadpool_limits(limits=1):fit=audit.HistGradientBoostingRegressor(**audit.parameters(audit.ROLES[1])).fit(self.X[:36],self.y[:36])
        tree=audit.serialize_tree(fit,audit.ROLES[1]);self.assertEqual(tree['shape_certificate']['constrained_nodes'],[])
        pred=audit.manual_predict(tree,np.zeros((5,10)),audit.ROLES[1]);self.assertTrue(np.all(pred==pred[0]))

    def test_invalid_structure_and_nonfinite_leaf_rejected(self):
        for field in ('cycle','leaf','categorical','depth','parameter'):
            t=copy.deepcopy(self.trees[audit.ROLES[0]])
            root=next(x['nodes'][0] for x in t['trees'] if not x['nodes'][0]['is_leaf'])
            if field=='cycle':root['left']=0
            elif field=='leaf':next(n for x in t['trees'] for n in x['nodes'] if n['is_leaf'])['num_threshold']='nan'
            elif field=='categorical':root['is_categorical']=1
            elif field=='depth':root['depth']=1
            else:t['parameters']['learning_rate']=.06
            with self.subTest(field=field),self.assertRaises(ValueError):audit.manual_predict(t,np.zeros((1,10)),audit.ROLES[0])

class DecisionTests(unittest.TestCase):
    def test_exact_full_three_arms_scores_choices_and_work(self):
        t,m,a=fixture();b=bundle();o,F,P,au=decide(a,m['monthly'],b)
        with patch.object(audit.accounts,'episode',wraps=audit.accounts.episode) as calls:r=audit.verify_selection(a,m['monthly'],b,o,scores=F,pre_scores=P,audit=au)
        self.assertEqual(calls.call_count,3);self.assertEqual(r['original_episode_calls'],3);self.assertEqual(r['ordered_choices_checked'],12)
        self.assertEqual(set(au['counts']),set(audit.COUNT_FIELDS));self.assertEqual(set(au['events'][0]),set(audit.EVENT_KEYS))

    def test_control_strategy_independent_without_candidate_body(self):
        t,m,a=fixture();b=bundle();o,F,P,_=decide(a,m['monthly'],b);b['monotone_candidate']=Poison()
        with patch.object(audit.accounts,'episode',wraps=audit.accounts.episode) as ep:r=audit.verify_control_selection(a,m['monthly'],b,o[audit.ARMS[1]],F[audit.ARMS[1]],P[audit.ARMS[1]])
        self.assertEqual(ep.call_count,1);self.assertTrue(r['candidate_selection_not_computed']);self.assertFalse(r['candidate_selection_computed'])

    def test_n0_and_physical_small_full_bundle_future_forecast_poison(self):
        t,m,a=fixture();full=bundle()
        for n in range(5):
            tt,mm,aa=physical(t,m,a,n);b=copy.deepcopy(full)
            for role in ('calibration_ridge',)+audit.ROLES:
                b[role]['raw'][n*6:]=np.inf;b[role]['rank'][n:]=np.inf;b[role]['audit']=Poison()
            o,F,P,au=decide(aa,mm['monthly'],b,n)
            r=audit.verify_selection(aa,mm['monthly'],b,o,scores=F,pre_scores=P,audit=au,n=n);self.assertEqual(r['checked_events'],n)

    def test_nonweak_and_partial_exact_copy(self):
        t,m,a=fixture();b=bundle();o,F,P,au=decide(a,m['monthly'],b);weak=a['state_weak_narrow'].copy();weak[0]=False
        for arm in audit.ARMS[1:]:
            audit.exact_array(P[arm][~weak],P[audit.ARMS[0]][~weak],'Nonweak P');audit.exact_array(F[arm][~weak],F[audit.ARMS[0]][~weak],'Nonweak F')
        base=audit.original_base(a,m['monthly'],4);_,d,_=audit.rank_and_delta(a,m['monthly'],b)
        for arm,key in zip(audit.ARMS[1:],('control','candidate')):
            expected=base['P'].copy();nz=d[key]!=0;expected[nz]+=d[key][nz];audit.exact_array(P[arm],expected,'Literal .0375 delta')

    def test_consumed_availability_raw_rank_and_audit_tamper(self):
        t,m,a=fixture()
        for field in ('calraw','av','rank','audit'):
            b=bundle();o,F,P,au=decide(a,m['monthly'],b)
            if field=='calraw':b['calibration_ridge']['raw'][6]+=1e-10
            elif field=='av':b[audit.ROLES[0]]['available'][6]=False
            elif field=='rank':b[audit.ROLES[1]]['rank'][1,0]+=1e-12
            else:au['events'][1]['candidate_delta'][0]+=1e-12
            with self.subTest(field=field),self.assertRaises(ValueError):audit.verify_selection(a,m['monthly'],b,o,scores=F,pre_scores=P,audit=au)

    def test_inc_gate_before_episode_and_no_new_fits(self):
        t,m,a=fixture();b=bundle();o,F,P,au=decide(a,m['monthly'],b);au['incumbent_gate']={'passed':False}
        with patch.object(audit.accounts,'episode',side_effect=AssertionError('episode')),self.assertRaises(ValueError):audit.verify_selection(a,m['monthly'],b,o,scores=F,pre_scores=P,audit=au)

class BilateralCross(unittest.TestCase):
    def test_producer_both_fixtures_gates_shapes_and_all_prefixes(self):
        import monthly_monotone_tree266_kernel as producer
        import test_monthly_monotone_tree266_kernel as producer_tests
        for name,sha in (('monthly_monotone_tree266_kernel.py','dd5e5afb31c836dfb25835b4f1215491049b9015ebf96d0f677b3ca0a27f3c26'),('test_monthly_monotone_tree266_kernel.py','e9d708e970dd2787dcb1b2cf4ba9475e3112a76cbc7763146e8df20cb8ca6338')):
            self.assertEqual(hashlib.sha256((audit.ROOT/'work'/name).read_bytes()).hexdigest(),sha)
        self.assertEqual(tuple(producer.BUNDLE_KEYS),audit.BUNDLE_KEYS);self.assertEqual(tuple(producer.MODEL_COUNT_FIELDS),audit.MODEL_COUNT_FIELDS)
        self.assertEqual(tuple(producer.COUNT_FIELDS),audit.COUNT_FIELDS);self.assertEqual(tuple(producer.EVENT_KEYS),audit.EVENT_KEYS)
        own=fixture()
        for study,t,m,a in ((frozen.study(),)+own,producer_tests.fixture()):
            # Synthetic reference uses frozen261; it is NOT a new actual cache.
            reference=producer.legacy.fit_models(t,m,a)
            old=producer.legacy.decisions(study,a,m['monthly'],reference)
            calls=[]
            def calibration(view):
                self.assertIsNone(view['unconstrained_control']);self.assertIsNone(view['monotone_candidate'])
                proof=audit.verify_calibration_models(t,m,a,view)
                o,F,P=producer.calibration_decisions(study,a,m['monthly'],view)
                audit.verify_incumbent_selection(a,m['monthly'],o,F,P);calls.append('calibration');return proof
            def control(view):
                self.assertIsNone(view['monotone_candidate']);proof=audit.verify_control_models(t,m,a,view)
                for fresh,prior in zip(view['unconstrained_control']['audit']['annual'],reference['candidate']['audit']['annual']):
                    for key in audit.TREE_ANNUAL_KEYS:
                        actual=audit.legacy_unconstrained_tree(fresh[key]) if key=='tree' and fresh[key] is not None else fresh[key]
                        audit.source.check_tree(actual,prior[key],'Old261 control annual/'+key)
                o,F,P=producer.control_decisions(study,a,m['monthly'],view)
                audit.verify_control_selection(a,m['monthly'],view,o,F,P)
                audit.record_check(o,old[0][producer.legacy.ARMS[2]],'Reference control ordered choices')
                audit.exact_array(F,old[1][producer.legacy.ARMS[2]],'Reference control F');audit.exact_array(P,old[2][producer.legacy.ARMS[2]],'Reference control P')
                calls.append('control');return proof
            actual=producer.fit_models(t,m,a,calibration_validator=calibration,control_validator=control)
            self.assertEqual(calls,['calibration','control']);audit.verify_models(t,m,a,actual)
            for n in range(len(m['monthly']['dates'])+1):
                tt,mm,aa=physical(t,m,a,n)
                def inc(o,F,P):return audit.verify_incumbent_selection(aa,mm['monthly'],o,F,P,n)
                out=producer.decisions(study,aa,mm['monthly'],actual,n,incumbent_validator=inc)
                audit.verify_selection(aa,mm['monthly'],actual,out[0],scores=out[1],pre_scores=out[2],audit=out[3],n=n)
                small=producer.fit_models(tt,mm,aa,n,calibration_validator=lambda v:audit.verify_calibration_models(tt,mm,aa,v,n),control_validator=lambda v:audit.verify_control_models(tt,mm,aa,v,n))
                audit.verify_models(tt,mm,aa,small,n)
                small_out=producer.decisions(study,aa,mm['monthly'],small,n,incumbent_validator=inc)
                for j in (0,3):audit.source.check_tree(out[j],small_out[j],'Full/prefix decisions')
                for j in (1,2):
                    for arm in audit.ARMS:audit.exact_array(out[j][arm],small_out[j][arm],'Physical/fullbundle P/F')
            for n in (0,2):
                tt,mm,aa=physical(t,m,a,n);poison=copy.deepcopy(actual)
                for role in ('calibration_ridge',)+audit.ROLES:
                    poison[role]['raw'][n*6:]=np.inf;poison[role]['rank'][n:]=np.inf;poison[role]['audit']=Poison()
                out=producer.decisions(study,aa,mm['monthly'],poison,n)
                audit.verify_selection(aa,mm['monthly'],poison,out[0],scores=out[1],pre_scores=out[2],audit=out[3],n=n)

if __name__=='__main__':unittest.main()
