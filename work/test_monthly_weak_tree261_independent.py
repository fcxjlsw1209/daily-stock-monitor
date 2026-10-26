"""Source-free261 independent mathematics, inference and integration tests."""
import copy
from functools import lru_cache
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import monthly_weak_tree261_independent as audit
import test_monthly_asset_target255_independent as frozen

audit.require(hashlib.sha256((audit.ROOT/'work/test_monthly_asset_target255_independent.py').read_bytes()).hexdigest()==
 'b6aea9e714da159927d706ccd0e8ec4a83828a89b9c9df89e84c35cadf9febf8','Pinned fake fixture')

def fixture():return frozen.fixture()
@lru_cache(None)
def cached_bundle():return audit.model_oracle(*fixture())[0]
def full_bundle():return copy.deepcopy(cached_bundle())
def decide(a,m,b,n=None):
    s,c,p,au=audit.selection_oracle(a,m,b,n);return audit.orders_from_choices(s,c,m),s,p,au

def physical(t,m,a,n):
    mm=dict(m,monthly={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m['monthly'].items()})
    aa={k:(v[:n].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(m['monthly']['dates']) else copy.deepcopy(v)) for k,v in a.items()}
    tt=dict(t,event_dates=t['event_dates'][:n*a['eligible'].shape[1]],event_plain=t['event_plain'][:n*a['eligible'].shape[1]])
    return tt,mm,aa

class Poison:
    def __float__(self):raise AssertionError('Unconsumed numeric body')
    def __str__(self):raise AssertionError('Unconsumed identity')
    def __deepcopy__(self,memo):raise AssertionError('Unconsumed deepcopy')

class CohortRidgeTests(unittest.TestCase):
    def test_control_canonical_svd_and_whole_gate_counts(self):
        t,m,a=fixture()
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as svd,patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('premature tree')):
            control,_,_=audit.model_oracle(t,m,a,candidate=False)
            r=audit.verify_control_models(t,m,a,control)
        self.assertEqual(svd.call_count,2*r['independent_svd_solves']);self.assertEqual(r['independent_tree_refits'],0)
        self.assertFalse(r['candidate_models_computed']);self.assertEqual(r['counts']['tree_fits'],0)
        audit.exact_array(control['control']['raw'],a['raw_source_log'].ravel(),'Frozen canonical')

    def test_later_bad_source_stops_all_tree_fits(self):
        t,m,a=fixture();a['raw_source_log'][-1,0]=np.nextafter(a['raw_source_log'][-1,0],np.inf)
        with patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('tree before all gates')),self.assertRaises(ValueError):audit.model_oracle(t,m,a)

    def test_n0_and_partial_do_not_retrieve_training_numeric(self):
        t,m,a=fixture()
        for n in (0,1):
            poisoned={k:Poison() for k in t}
            with patch.object(audit,'ridge_fit',side_effect=AssertionError('fit')),patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('tree')):
                b,inp,_=audit.model_oracle(poisoned,m,a,n)
            self.assertEqual(len(inp['dates']),0);self.assertEqual(b['audit']['counts']['actual_model_fits'],0)
            audit.verify_models(poisoned,m,a,b,n)

    def test_future_numeric_and_symbol_bodies_not_read(self):
        t,m,a=fixture();base=audit.model_oracle(t,m,a,2)[0];cutoff=np.datetime64('2015-12-31');future=t['ends']>cutoff
        for k in ('train_plain',)+tuple('target_'+k for k in audit.TARGETS):
            t[k]=t[k].astype(object);t[k][future]=Poison()
        for k,e in enumerate(m['training']['ends']):
            if e>'2015-12-31':m['training']['symbols'][k]=Poison()
        t['event_plain']=t['event_plain'].astype(object);t['event_plain'][12:]=Poison()
        actual=audit.model_oracle(t,m,a,2)[0];audit.source.check_tree(actual,base,'Future poison prefix')

    def test_all_four_targets_and_two_rows_month_filter(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);d=inp['dates'];first=np.flatnonzero(d==d[0]);inp['good'][first]=False;inp['good'][first[0]]=True
        use=audit.usable_rows(inp);self.assertFalse(use[first].any())
        inp['good'][first[1]]=True;self.assertEqual(audit.usable_rows(inp)[first].sum(),2)
        inp['targets']['loss_utility'][first[1]]=np.nan;self.assertFalse(audit.usable_rows(inp)[first].any())

    def test_weight_mass_half_life_and_row_multiplicity(self):
        d=np.array(['2015-01-31']*2+['2015-02-28']*4,dtype='datetime64[D]');w=audit.month_weights(d,np.datetime64('2015-12-31'))
        self.assertAlmostEqual(w.sum(),6.);self.assertAlmostEqual(w[:2].sum()/w[2:].sum(),2**(-1/12))

    def test_seventeen_eighteen_months_and_empty_U(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a,2)
        months=np.unique(inp['dates'].astype('datetime64[M]'))
        for count in (17,18):
            fake=copy.deepcopy(inp);fake['good']&=np.isin(fake['dates'].astype('datetime64[M]'),months[-count:]);empty=copy.deepcopy(a)
            empty['eligible'][1]=False;empty['forecast_available'][1]=False;empty['model_available'][1]=False;empty.pop('rank_local_linear',None);fake['query_good'][6:]=False
            result,_=audit.ridge_oracle(fake,empty)
            self.assertEqual(result['audit']['annual'][0]['status'],'cold_start' if count==17 else 'no_prediction_rows')

    def test_finite_query_and_mature_availability_cannot_be_falsified(self):
        t,m,a=fixture();a['forecast_available'][1,0]=False
        with self.assertRaises(ValueError):audit.model_oracle(t,m,a,2)

    def test_nonweak_year_still_has_one_fit(self):
        t,m,a=fixture();a['state_weak_narrow'][:]=False;b,_,_=audit.model_oracle(t,m,a)
        self.assertEqual(b['audit']['counts']['tree_fits'],3);self.assertEqual(audit.rank_and_delta(a,m['monthly'],b)[1].tobytes(),np.zeros((4,6)).tobytes())

    def test_ridge_constant_scale_and_no_solver_rescue(self):
        rng=np.random.default_rng(261);X=rng.normal(size=(70,10));X[:,0]=1.;Q=X[:4];y=X[:,1];w=np.ones(70)
        row,p=audit.ridge_fit(X,y,Q,w);self.assertEqual(row['scale'][0],1.)
        with patch.object(np.linalg,'solve',side_effect=np.linalg.LinAlgError('singular')),self.assertRaises(np.linalg.LinAlgError):audit.ridge_fit(X,y,Q,w)

    def test_five_year_and_prior_december_exact_endpoints(self):
        b=full_bundle();t,m,a=fixture()
        for row in b['control']['audit']['annual']:
            ids=row['training_row_ids'];self.assertTrue((t['dates'][ids]>np.datetime64(row['lower_bound'])).all());self.assertTrue((t['ends'][ids]<=np.datetime64(row['cutoff'])).all())

class TreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bundle=full_bundle();cls.tree=cls.bundle['candidate']['audit']['annual'][0]['tree']
    def test_full_refit_nodes_bins_predictions_and_actual_counts(self):
        t,m,a=fixture()
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as svd,patch.object(audit.HistGradientBoostingRegressor,'fit',autospec=True,side_effect=audit.HistGradientBoostingRegressor.fit) as fits:
            r=audit.verify_models(t,m,a,copy.deepcopy(self.bundle))
        self.assertEqual(svd.call_count,r['independent_svd_solves']);self.assertEqual(fits.call_count,r['independent_tree_refits'])
        self.assertEqual(r['manual_tree_traversals'],2*r['counts']['manual_tree_traversals']);self.assertFalse(r['independent_tree_training_algorithm']);self.assertTrue(r['shared_tree_trainer'])

    def test_exact_threshold_equality_and_single_shrinkage(self):
        tree=copy.deepcopy(self.tree);split=next(tr for tr in tree['trees'] if not tr['nodes'][0]['is_leaf']);root=split['nodes'][0]
        Q=np.zeros((2,10));Q[:,root['feature_idx']]=float.fromhex(root['num_threshold']);Q[1,root['feature_idx']]=np.nextafter(Q[1,root['feature_idx']],np.inf)
        actual=audit.manual_predict(tree,Q)
        # Scalar oracle using explicit equality-left and one already-shrunk leaf.
        expected=[]
        for x in Q:
            v=0.;v+=float.fromhex(tree['baseline']['values'][0][0])
            for tr in tree['trees']:
                nd=tr['nodes'];j=0
                while nd[j]['is_leaf']==0:j=nd[j]['left'] if x[nd[j]['feature_idx']]<=float.fromhex(nd[j]['num_threshold']) else nd[j]['right']
                v+=float.fromhex(nd[j]['value'])
            expected.append(v)
        audit.exact_array(actual,np.array(expected),'Equality route')
        wrong=copy.deepcopy(tree)
        for tr in wrong['trees']:
            for nd in tr['nodes']:
                if nd['is_leaf']:nd['value']=(.05*float.fromhex(nd['value'])).hex()
        self.assertNotEqual(audit.manual_predict(wrong,Q).tobytes(),actual.tobytes())

    def test_one_ulp_supplied_leaf_baseline_threshold_raw_rejected(self):
        t,m,a=fixture()
        for key in ('leaf','baseline','threshold','raw'):
            b=copy.deepcopy(self.bundle);row=b['candidate']['audit']['annual'][0];tr=row['tree']
            if key=='raw':row['raw_predictions'][0]=float(np.nextafter(row['raw_predictions'][0],np.inf))
            elif key=='baseline':tr['baseline']['values'][0][0]=float(np.nextafter(float.fromhex(tr['baseline']['values'][0][0]),np.inf)).hex()
            else:
                node=next(nd for tt in tr['trees'] for nd in tt['nodes'] if bool(nd['is_leaf'])==(key=='leaf'));field='value' if key=='leaf' else 'num_threshold';node[field]=float(np.nextafter(float.fromhex(node[field]),np.inf)).hex()
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_models(t,m,a,b)

    def test_bad_child_cycle_parameters_order_and_types(self):
        for key in ('child','cycle','params','order','dtype','categories'):
            tr=copy.deepcopy(self.tree)
            if key in ('child','cycle'):
                node=next(tt['nodes'][0] for tt in tr['trees'] if not tt['nodes'][0]['is_leaf']);node['left']=999 if key=='child' else 0
            if key=='params':tr['parameters']['learning_rate']=.06
            if key=='order':tr['trees'][0]['iteration']=1
            if key=='dtype':tr['node_dtype'][0][1]='<f4'
            if key=='categories':tr['is_categorical']=[False]*10
            with self.subTest(key=key),self.assertRaises(ValueError):audit.manual_predict(tr,np.zeros((1,10)))

    def test_constant_tree_is_available_original_tie_rank(self):
        rng=np.random.default_rng(2611);X=rng.uniform(-.4,.4,(36,10));y=np.linspace(-.2,.2,36);w=np.ones(36)
        with audit.threadpool_limits(limits=1):model=audit.HistGradientBoostingRegressor(**audit.TREE_PARAMETERS).fit(X,y,sample_weight=w)
        tr=audit.serialize_tree(model);p=audit.manual_predict(tr,X[:6]);self.assertTrue(np.all(p==p[0]));np.testing.assert_array_equal(audit.previous._rank(p),np.full(6,3.5/6-.5))
        self.assertTrue(all(len(t['nodes'])==1 for t in tr['trees']))

    def test_model_target_params_cohort_and_labels_tamper_rejected(self):
        t,m,a=fixture()
        for key in ('labels','ids','weights','nodes','cohort'):
            b=copy.deepcopy(self.bundle)
            if key=='labels':b['label_values'][0]+=1e-9
            if key=='ids':b['training_row_ids'][0]+=1
            if key=='weights':b['control']['audit']['annual'][0]['weights'][0]+=1e-9
            if key=='nodes':b['audit']['counts']['tree_nodes']+=1
            if key=='cohort':b['control']['audit']['annual'][0]['training_row_ids'].pop()
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_models(t,m,a,b)

class SelectionTests(unittest.TestCase):
    def test_exact_scores_orders_full_audit_and_original_budget(self):
        t,m,a=fixture();b=full_bundle();o,s,p,au=decide(a,m['monthly'],b)
        r=audit.verify_selection(a,m['monthly'],b,o,scores=s,pre_scores=p,audit=au)
        self.assertEqual(r['ordered_choices_checked'],12);self.assertEqual(r['original_episode_calls'],3)
        audit.exact_array(p[audit.ARMS[0]],p[audit.ARMS[1]],'Matched source');self.assertEqual(o[audit.ARMS[0]],o[audit.ARMS[1]])

    def test_local_quarter_calm_and_partial_exact_and_literal_delta(self):
        t,m,a=fixture();b=full_bundle();r,d,au=audit.rank_and_delta(a,m['monthly'],b);o,s,p,ev=decide(a,m['monthly'],b)
        for i in (0,1):audit.exact_array(p[audit.ARMS[2]][i],p[audit.ARMS[0]][i],'Bypass')
        np.testing.assert_array_equal(d[2:],.0375*(r['tree'][2:]-r['old_pool'][2:]));self.assertGreater(np.count_nonzero(d),0)

    def test_all_cached_and_physical_prefixes_no_fits_future_poison(self):
        t,m,a=fixture();b=full_bundle()
        for n in range(5):
            tt,mm,aa=physical(t,m,a,n);poison=copy.deepcopy(b)
            for f in ('control','candidate'):
                poison[f]['raw'][n*6:]=np.inf;poison[f]['rank'][n:]=np.inf
            with patch.object(audit,'ridge_fit',side_effect=AssertionError('cached fit')),patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('cached tree')):
                out=decide(aa,mm['monthly'],poison,n);o,s,p,au=out;audit.verify_selection(aa,mm['monthly'],poison,o,scores=s,pre_scores=p,audit=au,n=n)
            self.assertEqual(len(o[audit.ARMS[0]]),n)

    def test_physical_fresh_models_equal_mature_full_prefix(self):
        t,m,a=fixture();full=full_bundle()
        for n in (0,1,2,3,4):
            tt,mm,aa=physical(t,m,a,n);b,_,_=audit.model_oracle(tt,mm,aa,n)
            for role in ('control','candidate'):
                for key in ('raw','available'):audit.exact_array(b[role][key],full[role][key][:n*6],'Physical forecast')
                audit.exact_array(b[role]['rank'],full[role]['rank'][:n],'Physical rank')

    def test_bad_supplied_rank_score_order_audit_reject(self):
        t,m,a=fixture();b=full_bundle()
        for fault in ('rank','score','order','audit'):
            c=copy.deepcopy(b);o,s,p,au=decide(a,m['monthly'],c)
            if fault=='rank':c['candidate']['rank'][2,0]+=1e-8
            if fault=='score':s[audit.ARMS[2]][2,0]+=1e-8
            if fault=='order':o[audit.ARMS[2]][2]['symbols'].reverse()
            if fault=='audit':au['counts']['tree_episode_calls']=2
            with self.subTest(fault=fault),self.assertRaises(ValueError):audit.verify_selection(a,m['monthly'],c,o,scores=s,pre_scores=p,audit=au)

    def test_two_names_and_cash_do_not_change_original_E(self):
        t,m,a=fixture();b=full_bundle()
        for size in (0,1,2):
            aa=copy.deepcopy(a);aa['eligible'][:]=False;aa['eligible'][:,:size]=True
            # Recompute source ranks solely for this synthetic smaller original U.
            for i in range(4):
                aa['rank_log_excess'][i]=0.
                if size:aa['rank_log_excess'][i,:size]=audit.previous._rank(aa['raw_source_log'][i,:size])
            aa.pop('rank_local_linear',None);bb=copy.deepcopy(b)
            for f in ('control','candidate'):
                bb[f]['rank']=aa['rank_log_excess'].copy()
                for i in range(4):bb[f]['raw'][i*6+size:(i+1)*6]=aa['raw_source_log'][i,size:]
                for i in range(1,4):
                    if size:bb[f]['rank'][i,:size]=audit.previous._rank(bb[f]['raw'][i*6:i*6+size])
            out=decide(aa,m['monthly'],bb)
            for arm in audit.ARMS:self.assertTrue(all(len(r['symbols'])==(2 if size==2 else 0) for r in out[0][arm]))

class AdditionalGuardTests(unittest.TestCase):
    def test_control_raw_and_both_support_masks_exact_in_selection(self):
        t,m,a=fixture()
        for role,key in (('control','raw'),('control','available'),('candidate','available')):
            b=full_bundle()
            if key=='raw':b[role][key][6]=np.nextafter(b[role][key][6],np.inf)
            else:b[role][key][6]=not b[role][key][6]
            with self.subTest(role=role,key=key),self.assertRaises(ValueError):audit.rank_and_delta(a,m['monthly'],b)

    def test_only_active_positive_infinity_threshold_allowed(self):
        tr=full_bundle()['candidate']['audit']['annual'][0]['tree']
        node=next(t['nodes'][0] for t in tr['trees'] if not t['nodes'][0]['is_leaf']);node['num_threshold']='inf'
        audit.manual_predict(tr,np.zeros((1,10)))
        for bad in ('-inf','nan'):
            node['num_threshold']=bad
            with self.assertRaises(ValueError):audit.manual_predict(tr,np.zeros((1,10)))
        node['num_threshold']='inf'
        leaf=next(nd for tt in tr['trees'] for nd in tt['nodes'] if nd['is_leaf'])
        for bad in ('-inf','nan','inf'):
            leaf['num_threshold']=bad
            with self.assertRaises(ValueError):audit.manual_predict(tr,np.zeros((1,10)))

    def test_runtime_records_thread_limit_and_order_independence(self):
        t,m,a=fixture();b=full_bundle()
        for row in b['audit']['execution_runtime']:
            self.assertTrue(all(v['num_threads']==1 for v in row['fit_threadpools']))
            row['fit_threadpools'].reverse();row['predict_threadpools'].reverse()
        audit.verify_models(t,m,a,b)
        b['audit']['execution_runtime'][0]['fit_threadpools'][0]['num_threads']=2
        with self.assertRaises(ValueError):audit.verify_models(t,m,a,b)

class BilateralIntegrationTests(unittest.TestCase):
    def test_final_producer_host_all_gates_prefixes_and_poison(self):
        # Test-only imports exercise the seam, never provide oracle mathematics.
        import monthly_weak_tree261_kernel as producer
        import monthly_weak_tree261_host as host
        import test_monthly_weak_tree261_kernel as fixtures
        pins={'work/monthly_weak_tree261_kernel.py':'36595e322ca41138fd71721d0ff27507b42554602806b8b7ff7cf39748d3641f',
              'work/test_monthly_weak_tree261_kernel.py':'f59b75e551355a068a04873239d452ab5cbcf62729825cb7e44d1c02e452ab29',
              'work/monthly_weak_tree261_host.py':'e09558e19f73bbb9f149b7220358fffe003a0a577eab1ebb6b920dde3ef5311f',
              'work/test_monthly_weak_tree261_host.py':'3af94102006088a322eab25710626b66368e33caeab56cab9f4315c78e8cbde3'}
        for path,digest in pins.items():self.assertEqual(hashlib.sha256((audit.ROOT/path).read_bytes()).hexdigest(),digest)
        args=fixtures.fixture();study,train,metadata,arrays=args;n=len(metadata['monthly']['dates']);width=arrays['eligible'].shape[1]
        def original(aa,mm,count):
            base=audit.original_base(aa,mm,count);F,choices=audit.accounts.episode(base['P'],base)
            orders=audit.orders_from_choices({arm:F for arm in audit.ARMS},{arm:choices for arm in audit.ARMS},mm)[audit.ARMS[0]]
            return orders,F,base['P']
        expected=original(arrays,metadata['monthly'],n)
        bundle,got,proof=host.prepare_one(study,train,metadata,arrays,audit,n,expected[0],expected)
        self.assertEqual(set(proof),{'early_incumbent','matched_strategy','control','models','incumbent','selection','main_model_counts'})
        self.assertEqual(proof['models']['manual_tree_traversals'],2*bundle['audit']['counts']['manual_tree_traversals'])
        self.assertEqual(proof['control']['independent_tree_refits'],0)
        for prefix in range(n+1):
            tt,mm,aa,scope=host.physical_inputs(train,metadata,arrays,prefix);st=study;exp=original(aa,mm['monthly'],prefix)
            self.assertEqual(len(mm['monthly']['dates']),prefix);self.assertEqual(len(aa['eligible']),prefix)
            if prefix<=1:self.assertEqual(tt,{})
            else:
                self.assertEqual(tt['event_plain'].shape,(prefix*width,10))
                self.assertEqual(len(tt['dates']),scope['training_rows'])
                self.assertEqual(set('target_'+k for k in audit.TARGETS),{k for k in tt if k.startswith('target_')})
                self.assertTrue((tt['ends']<=np.datetime64(scope['cutoff'])).all())
            cb,receipts=host.incumbent_callback(audit,aa,mm['monthly'],prefix,exp[0])
            poisoned=copy.deepcopy(bundle)
            for role in ('control','candidate'):
                poisoned[role]['raw'][prefix*width:]=np.inf;poisoned[role]['rank'][prefix:]=np.inf
            with patch.object(audit,'ridge_fit',side_effect=AssertionError('cached Ridge')),patch.object(audit,'HistGradientBoostingRegressor',side_effect=AssertionError('cached tree')):
                oo,ss,pp,au=producer.decisions(st,aa,mm['monthly'],poisoned,n=prefix,incumbent_validator=cb,expected_incumbent_orders=exp[0])
                rr=audit.verify_selection(aa,mm['monthly'],poisoned,oo,scores=ss,pre_scores=pp,audit=au,n=prefix)
                host.receipt(rr,prefix,audit.SELECTION_FLAGS,'selection',width=width)
            self.assertEqual(oo[audit.ARMS[2]],got[0][audit.ARMS[2]][:prefix])
            fresh,out,receipt=host.prepare_one(st,tt,mm,aa,audit,prefix,exp[0],exp)
            host.bundle_parity(bundle,fresh,prefix);host.physical_scope_check(scope,prefix,fresh['audit']['counts'])
            for arm in audit.ARMS:
                self.assertEqual(out[0][arm],got[0][arm][:prefix]);audit.exact_array(out[1][arm],got[1][arm][:prefix],'Cross prefix score')
        # Strict shared schema identity, independent numeric implementation remains unimported from producer.
        self.assertEqual(producer.MODEL_COUNT_FIELDS,audit.MODEL_COUNT_FIELDS);self.assertEqual(producer.COUNT_FIELDS,audit.COUNT_FIELDS)

if __name__=='__main__':unittest.main(verbosity=2)
