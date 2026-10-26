"""Synthetic259 independent design/solver/selector and bilateral checks."""
import copy
from functools import lru_cache
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import monthly_joint_identity259_independent as audit
import test_monthly_asset_target255_independent as frozen

audit.require(hashlib.sha256((audit.ROOT/'work/test_monthly_asset_target255_independent.py').read_bytes()).hexdigest()==
 'b6aea9e714da159927d706ccd0e8ec4a83828a89b9c9df89e84c35cadf9febf8','Frozen synthetic fixture bytes')

def fixture():return frozen.fixture()
def study():return frozen.study()

def physical(args,n):
    t,m,a=copy.deepcopy(args);W=a['eligible'].shape[1]
    a={k:v[:n].copy() if isinstance(v,np.ndarray) and v.shape[:1]==(len(m['monthly']['dates']),) else v for k,v in a.items()}
    m['monthly']={k:v[:n] if k in ('dates','ends','symbols') else v for k,v in m['monthly'].items()}
    t['event_dates']=t['event_dates'][:n*W];t['event_plain']=t['event_plain'][:n*W]
    return t,m,a

def make_bundle(t,m,a,n=None,**kw):
    inp=audit.model_inputs(t,m,a,n);control=frozen.original_model(inp,inp['targets'])
    _,contexts=audit.control_oracle(inp,inp['targets'],**kw)
    diagnostic,candidate=audit.identity_forecasts(inp,control,contexts,min_own_months=kw.get('min_own_months',18))
    ma=audit.model_audit(inp,control,diagnostic,candidate)
    return dict(schema='monthly_joint_identity259_models_v1',n=inp['n'],width=inp['width'],training_row_ids=inp['training_row_ids'],
      control=control,diagnostic=diagnostic,candidate=candidate,label_values=inp['targets']['log_excess'].copy(),
      label_preservation=audit.label_preservation(inp),control_gate=copy.deepcopy(ma['original_cache_parity']),audit=ma)

@lru_cache(None)
def cached_bundle():return make_bundle(*fixture())
def full_bundle():return copy.deepcopy(cached_bundle())
def run(a,m,b,n=None):
    s,c,p,au=audit.selection_oracle(a,m,b,n)
    return audit.orders_from_choices(s,c,m),s,p,au
def verify(a,m,b,out,n=None):
    o,s,p,au=out
    return audit.verify_selection(a,m,b,o,scores=s,pre_scores=p,audit=au,n=n)

def pure_design(correlated=True):
    n=12;w=np.arange(1.,n+1);Z=np.zeros((n,10));H=np.zeros((n,2))
    H[:6,0]=1.;H[6:,1]=1.;H-=np.average(H,axis=0,weights=w)
    if correlated:Z[:,0]=np.linspace(-1,1,n);Z-=np.average(Z,axis=0,weights=w)
    y=.3*Z[:,0]+.8*H[:,0]+np.sin(np.arange(n))*.03;a=float(np.average(y,weights=w))
    beta,A,_=audit.normal_solve(Z,y-a,w,np.full(10,10.))
    return dict(status='fit',firms=['A','B'],w=w,Z=Z,Hc=H,y=y,a=a,beta0=beta,pooled_gram=A)

class IdentityMathTests(unittest.TestCase):
    def test_joint_feedback_and_matched_penalized_objective(self):
        d=pure_design();fit=audit.fit_identity(d)
        self.assertGreater(np.max(np.abs(fit['joint_beta']-fit['sequential_beta'])),1e-5)
        def objective(tag):
            beta,delta=fit[tag+'_beta'],fit[tag+'_delta']
            error=(d['y']-d['a'])-np.einsum('ij,j->i',d['Z'],beta)-np.einsum('ij,j->i',d['Hc'],delta)
            return np.sum(d['w']*error*error)+10*np.sum(beta*beta)+12*np.sum(delta*delta)
        self.assertLess(objective('joint'),objective('sequential'))

    def test_orthogonal_blocks_and_zero_response(self):
        d=pure_design(False);f=audit.fit_identity(d)
        np.testing.assert_allclose(f['joint_delta'],f['sequential_delta'],atol=2e-15,rtol=0)
        d['y'][:]=0.;d['a']=0.;d['beta0'][:]=0.;f=audit.fit_identity(d)
        np.testing.assert_array_equal(f['joint_delta'],np.zeros(2));np.testing.assert_array_equal(f['joint_beta'],np.zeros(10))

    def test_collinear_full_dummy_design_is_unique_without_reference_drop(self):
        d=pure_design();d['Hc'][:,1]=-d['Hc'][:,0]
        self.assertTrue(np.all(d['Hc'].sum(axis=1)==0.))
        f=audit.fit_identity(d);self.assertAlmostEqual(sum(f['joint_delta']),0.,places=14)
        self.assertTrue(np.all(np.linalg.eigvalsh(f['joint_gram'])>0))

    def test_zero_centered_design_does_not_solve_new_models(self):
        d=pure_design();d['Hc'][:]=0.;d['status']='zero_centered_design'
        with patch.object(audit,'normal_solve',side_effect=AssertionError('unexpected solve')):
            f=audit.fit_identity(d)
        np.testing.assert_array_equal(f['joint_beta'],d['beta0']);self.assertIsNone(f['joint_gram'])

    def test_training_only_columns_and_unqualified_rows_retained(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);_,ctx=audit.control_oracle(inp,inp['targets']);inp=copy.deepcopy(inp)
        # A rare identity has17 distinct months, stays in X/W but gets no column.
        ids=ctx[0]['ids'];own=ids[inp['symbols'][ids]=='ALPHA'][:17];inp['symbols'][own]='RARE'
        inp['query_symbols'][:6]=['NEW']*6
        d=audit.identity_design(inp,ctx[0]);self.assertIn('ALPHA',d['firms']);self.assertNotIn('RARE',d['firms'])
        self.assertEqual(d['firm_months']['RARE'],17);self.assertEqual(len(d['w']),len(ids))
        self.assertTrue(np.all(d['H'][np.isin(ids,own)]==0.));self.assertEqual(d['status'],'fit')
        # Query IDs cannot remove trained columns even when no query qualifies.
        inp['query_symbols'][ctx[0]['prediction_rows']]='NEW'
        d=audit.identity_design(inp,ctx[0]);self.assertEqual(d['status'],'no_qualified_query');self.assertGreater(len(d['firms']),0)

    def test_original_scale_floor_and_penalty_units(self):
        d=pure_design();f=audit.fit_identity(d);J=len(d['firms'])
        H=d['Hc'];S=d['w'].sum();p=np.average(np.r_[np.ones(6),np.zeros(6)],weights=d['w'])
        self.assertAlmostEqual(f['sequential_gram'][0,0],S*p*(1-p)+12.)
        self.assertEqual(f['joint_gram'].shape,(10+J,10+J))

    def test_17_to_18_and_unknown_ids_do_not_change_training_values(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);_,ctx=audit.control_oracle(inp,inp['targets']);inp=copy.deepcopy(inp)
        ids=ctx[0]['ids'];own=ids[inp['symbols'][ids]=='ALPHA'][:18];inp['symbols'][own]='RARE'
        before=inp['X'].copy();d=audit.identity_design(inp,ctx[0]);self.assertIn('RARE',d['firms'])
        np.testing.assert_array_equal(inp['X'],before)

class ModelTests(unittest.TestCase):
    def test_full_exact_design_models_svd_and_normal_solve_counts(self):
        t,m,a=fixture();b=full_bundle()
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as svd,patch.object(audit,'normal_solve',wraps=audit.normal_solve) as normal:
            proof=audit.verify_models(t,m,a,b)
        self.assertTrue(all(proof[k] for k in audit.MODEL_FLAGS))
        self.assertEqual(svd.call_count,proof['independent_solve_counts']['total_regression_solves'])
        self.assertEqual(normal.call_count,proof['independent_canonical_solve_counts']['total_regression_solves'])
        c=proof['counts'];self.assertEqual(c['actual_new_model_solves'],c['diagnostic_dummy_solves']+c['candidate_joint_solves'])
        self.assertEqual(c['diagnostic_pooled_fits']+c['candidate_pooled_fits'],0)

    def test_control_gate_does_not_build_identity_models(self):
        t,m,a=fixture();b=full_bundle()
        with patch.object(audit,'identity_design',side_effect=AssertionError('premature identity')):
            proof=audit.verify_control_models(t,m,a,b['control'])
        self.assertFalse(proof['candidate_models_computed']);self.assertEqual(proof['independent_design_builds'],0)
        self.assertEqual(proof['independent_canonical_solve_counts']['total_regression_solves'],0)

    def test_one_ulp_canonical_model_and_design_tampering_rejected(self):
        t,m,a=fixture();b=full_bundle()
        for key in ('source_beta','global_beta','dummy_delta','weights','dummy_mean','weighted_dummy_sum'):
            z=copy.deepcopy(b);row=z['candidate']['audit']['annual'][0];row[key][0]=float(np.nextafter(row[key][0],np.inf))
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_models(t,m,a,z)
        z=copy.deepcopy(b);z['diagnostic']['audit']['annual'][0]['centered_dummy_sha256']='0'*64
        with self.assertRaises(ValueError):audit.verify_models(t,m,a,z)

    def test_copied_targets_pool_constant_and_support_are_exact(self):
        t,m,a=fixture();b=full_bundle()
        for role in ('diagnostic','candidate'):
            for k in ('source_log','local_constant'):
                self.assertEqual(b[role]['raw'][k].tobytes(),b['control']['raw'][k].tobytes())
            for k in ('available','local_available'):np.testing.assert_array_equal(b[role][k],b['control'][k])
        for target in ('label_values','candidate'):
            z=copy.deepcopy(b)
            if target=='label_values':z[target][0]=np.nextafter(z[target][0],np.inf)
            else:z[target]['raw']['local_constant'][0]=np.nextafter(z[target]['raw']['local_constant'][0],np.inf)
            with self.assertRaises(ValueError):audit.verify_models(t,m,a,z)

    def test_no_qualified_columns_copy_pool_and_zero_new_solves(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);control=full_bundle()['control'];_,ctx=audit.control_oracle(inp,inp['targets'])
        diag,cand=audit.identity_forecasts(inp,control,ctx,min_own_months=1000)
        for model in (diag,cand):
            np.testing.assert_array_equal(model['raw']['local_linear'],control['raw']['source_log'])
            self.assertTrue(all(r['status']=='no_qualified_columns' for r in model['audit']['annual']))

    def test_n0_no_numeric_training_and_no_solves(self):
        t,m,a=physical(fixture(),0);t['train_plain'][:]=np.inf
        for k in audit.TARGETS:t['target_'+k][:]=np.inf
        b=make_bundle(t,m,a)
        with patch.object(audit.numeric,'least_squares',side_effect=AssertionError('SVD')),patch.object(audit,'normal_solve',side_effect=AssertionError('normal')):
            p=audit.verify_models(t,m,a,b)
        self.assertEqual(p['counts']['actual_regression_solves'],0);self.assertEqual(p['independent_design_builds'],0)

    def test_future_numeric_poison_and_all_physical_prefixes(self):
        full=fixture();b=full_bundle()
        for n in (0,1,2,4):
            t,m,a=physical(full,n);small=make_bundle(t,m,a)
            out=run(a,m['monthly'],b,n);verify(a,m['monthly'],b,out,n)
            expected=run(a,m['monthly'],small,n);audit.record_check(out[0],expected[0],'Physical full-bundle decisions')
        t,m,a=fixture();cutoff=np.datetime64('2015-12-31');future=t['ends']>cutoff
        t['train_plain'][future]=np.inf
        for key in audit.TARGETS:t['target_'+key][future]=np.inf
        small=make_bundle(t,m,a,1);audit.verify_models(t,m,a,small,1)

    def test_unknown_query_raw_copy_still_ranked_in_full_universe(self):
        b=full_bundle();model=b['candidate'];row=model['audit']['annual'][0]
        self.assertEqual(row['query_row_ids'],row['qualified_query_row_ids'])
        # Full-pool replacement verifies exact-copy shortcut but keeps all U.
        model['raw']['local_linear']=model['raw']['source_log'].copy()
        t,m,a=fixture();r,_,au=audit.rank_and_delta(a,m['monthly'],b)
        np.testing.assert_array_equal(r['new_local'],r['old_pool']);self.assertEqual(au['ranking_cells'],int(a['eligible'].sum()))

class SelectionTests(unittest.TestCase):
    def test_full_quarter_weak_and_calm_and_partial_copy(self):
        t,m,a=fixture();b=full_bundle();r,d,au=audit.rank_and_delta(a,m['monthly'],b)
        np.testing.assert_array_equal(d[0],np.zeros(6))
        np.testing.assert_array_equal(d[1],.0025*(r['new_local'][1]-r['old_local'][1]))
        np.testing.assert_array_equal(d[2],.0125*(r['new_local'][2]-r['old_local'][2]))
        p=verify(a,m['monthly'],b,run(a,m['monthly'],b));self.assertEqual(p['ordered_choices_checked'],12)

    def test_local_slot_not_double_counted_against_pool(self):
        t,m,a=fixture();b=full_bundle();r,d,_=audit.rank_and_delta(a,m['monthly'],b)
        expected=.0125*(r['new_local'][2]-r['old_local'][2]);np.testing.assert_array_equal(d[2],expected)
        self.assertTrue(np.any(r['old_local']!=r['old_pool']))

    def test_future_forecast_poison_consumed_prefix_only(self):
        t,m,a=physical(fixture(),1);b=full_bundle()
        for role in ('candidate','diagnostic'):
            for values in b[role]['raw'].values():values[6:]=np.inf
        verify(a,m['monthly'],b,run(a,m['monthly'],b,1),1)

    def test_order_score_audit_rejection(self):
        t,m,a=fixture();b=full_bundle();out=run(a,m['monthly'],b)
        for kind in ('score','pre','audit','order'):
            o,s,p,au=copy.deepcopy(out)
            if kind=='score':s[audit.ARMS[2]][1,0]+=1e-15
            if kind=='pre':p[audit.ARMS[1]][1,0]+=1e-15
            if kind=='audit':au['counts']['candidate_episode_calls']=2
            if kind=='order':o[audit.ARMS[2]][1]['symbols'].reverse()
            with self.subTest(kind=kind),self.assertRaises(ValueError):verify(a,m['monthly'],b,(o,s,p,au))

    def test_n0_three_calls_and_no_predictive_fits(self):
        t,m,a=physical(fixture(),0);b=full_bundle();out=run(a,m['monthly'],b,0);p=verify(a,m['monthly'],b,out,0)
        self.assertEqual(p['ordered_choices_checked'],0)
        for k in ('incumbent_episode_calls','diagnostic_episode_calls','candidate_episode_calls'):self.assertEqual(p['counts'][k],1)
        self.assertEqual(p['counts']['new_model_fits'],0)

    def test_inputs_not_mutated(self):
        t,m,a=fixture();b=full_bundle();before=copy.deepcopy((a,m,b));verify(a,m['monthly'],b,run(a,m['monthly'],b))
        for k,v in a.items():np.testing.assert_array_equal(v,before[0][k])
        self.assertEqual(m,before[1]);self.assertEqual(b['candidate']['raw']['local_linear'].tobytes(),before[2]['candidate']['raw']['local_linear'].tobytes())

class BilateralCross(unittest.TestCase):
    def test_final_producer_host_full_and_physical_prefixes(self):
        # Production imports are confined to this synthetic integration test;
        # the independent numerical implementation never imports them.
        import monthly_joint_identity259_kernel as kernel
        import test_monthly_joint_identity259_kernel as factory
        import monthly_joint_identity259_host as host
        for path,sha in (
          ('monthly_joint_identity259_kernel.py','c50ca6cd71673f27b65ae208089aad332eaaf5a7180e8008707f2d4019eb48bf'),
          ('test_monthly_joint_identity259_kernel.py','bb278f605a76bf4fcaf90f1e24b4121bd9458000b8e86999930836910cbf43fa'),
          ('monthly_joint_identity259_host.py','f19a29d199ba6e77e327d03dbbd974d30ea7b837f8a5ef46dc3422e8bce49124')):
            self.assertEqual(hashlib.sha256((audit.ROOT/'work'/path).read_bytes()).hexdigest(),sha)
        args=factory.fixture();st,t,m,a=args;N=len(m['monthly']['dates']);W=a['eligible'].shape[1]
        def models(arguments,n):
            _,train,meta,arrays=arguments
            callback,receipts=host.model_callbacks(audit,train,meta,arrays,n)
            bundle=kernel.fit_models(train,meta,arrays,n,control_validator=callback)
            proof=audit.verify_models(train,meta,arrays,bundle,n)
            host.receipt(proof,n,audit.MODEL_FLAGS,'models',width=W)
            host.bundle_preservation(bundle,n);host.fit_binding(receipts['control'],proof,bundle['audit'],n)
            return bundle
        def decisions(arguments,bundle,n):
            study,_,meta,arrays=arguments;monthly=meta['monthly'];base=audit.original_base(arrays,monthly,n)
            F,choices=audit.accounts.episode(base['P'],base)
            expected=audit.orders_from_choices({arm:F for arm in audit.ARMS},{arm:choices for arm in audit.ARMS},monthly)[audit.ARMS[0]]
            callback,receipts=host.incumbent_callback(audit,arrays,monthly,n,expected)
            output=kernel.decisions(study,arrays,monthly,bundle,n,incumbent_validator=callback,expected_incumbent_orders=expected)
            proof=verify(arrays,monthly,bundle,output,n)
            host.receipt(proof,n,audit.SELECTION_FLAGS,'selection',width=W)
            self.assertEqual(set(receipts),{'incumbent'})
            return output
        full=models(args,N);out=decisions(args,full,N)
        for n in (0,1,2,N):
            with self.subTest(n=n):
                physical=factory.physical(args,n);part=models(physical,n)
                host.bundle_parity(full,part,n)
                got=decisions(physical,full,n);fresh=decisions(physical,part,n)
                host.exact_prefix(out,got,n);host.exact_prefix(out,fresh,n)
                # Full logical arrays and physically cut inputs consume the same
                # saved forecast prefix without any model recomputation.
                logical=decisions(args,full,n);host.exact_prefix(out,logical,n)
        poisoned=copy.deepcopy(full)
        for role in ('candidate','diagnostic'):
            for values in poisoned[role]['raw'].values():values[W:]=np.inf
        decisions(factory.physical(args,1),poisoned,1)

if __name__=='__main__':unittest.main(verbosity=2)
