"""Only invented257 matrices/models/decisions; no actual caches or returns."""
import copy
from functools import lru_cache
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import monthly_correlation_distance257_independent as audit
import test_monthly_local_distance256_independent as frozen

audit.require(hashlib.sha256((audit.ROOT/'work/test_monthly_local_distance256_independent.py').read_bytes()).hexdigest()==
 '0d8b7d492450b2134ea525a5d1ae74d82fd1e170feef8c3e1aef460d8e62f8b6','Frozen synthetic fixtures')
fixture=frozen.fixture
study=frozen.study

def inputs():
    rng=np.random.default_rng(9257)
    X=rng.uniform(-.4,.4,(40,10));X[:,1]=.75*X[:,0]+.25*X[:,1]
    w=np.exp2(-np.arange(40)/12.);d=np.array('2010-01','datetime64[M]')+np.arange(40)
    return X,w,d.astype('datetime64[D]'),np.datetime64('2015-12-31'),X[:12],rng.uniform(-.4,.4,10)

def orthogonal(scales=None):
    # Exact binary orthogonality, equal weights and zero means.
    bits=((np.arange(1024)[:,None]>>np.arange(10))&1)*2-1
    return bits*(np.full(10,.25) if scales is None else scales),np.ones(1024)

class MatrixTests(unittest.TestCase):
    def test_original_weighted_moments_and_independent_quadratic_form(self):
        X,w,d,c,own,q=inputs();m=audit.moments_oracle(X,w);g=audit.geometry_oracle(d[:12],c,own,q,m)
        mu=np.average(X,axis=0,weights=w);C=np.array([[np.dot(w,(X[:,j]-mu[j])*(X[:,k]-mu[k]))/w.sum() for k in range(10)] for j in range(10)])
        np.testing.assert_allclose(m['covariance'],C,atol=1e-16,rtol=1e-14)
        D=np.asarray(m['metric']);np.testing.assert_allclose(g['distance'],[(x-q)@D@(x-q) for x in own],atol=2e-15,rtol=2e-15)
        self.assertAlmostEqual(np.trace(D),1.,places=14);self.assertFalse(m['identity_correlation'])

    def test_constant_and_tiny_scales_retained_as_unit_floor(self):
        X,w,*_=inputs();X[:,0]=0.;X[:,1]=np.arange(40)*1e-15
        m=audit.moments_oracle(X,w);R=np.asarray(m['correlation'])
        self.assertEqual(m['floored'][:2],[True,True]);self.assertEqual(m['source_scale'][:2],[1.,1.])
        np.testing.assert_array_equal(R[:2,:2],np.eye(2));np.testing.assert_array_equal(R[:2,2:],0.)
        self.assertEqual(np.asarray(m['metric']).shape,(10,10))

    def test_exact_identity_reuses_diagonal_with_unequal_scales(self):
        X,w=orthogonal(np.arange(1,11)/32.);m=audit.moments_oracle(X,w);self.assertTrue(m['identity_correlation'])
        _,_,d,c,_,q=inputs();g=audit.geometry_oracle(d[:12],c,X[:12],q,m)
        old=audit.previous.geometry_oracle(d[:12],c,X[:12],q,m['source_scale'])
        for k in ('weights','distance','unscaled_weights','original_weights'):np.testing.assert_array_equal(g[k],old[k])
        self.assertIsNotNone(g['diagonal_geometry'])

    def test_equal_scales_correlated_does_not_collapse(self):
        X,w=orthogonal();X[:,1]=X[:,0];m=audit.moments_oracle(X,w)
        self.assertTrue(np.all(np.asarray(m['source_scale'])==.25));self.assertFalse(m['identity_correlation'])
        _,_,d,c,_,q=inputs();g=audit.geometry_oracle(d[:12],c,X[:12],q,m)
        self.assertIsNone(g['diagonal_geometry']);self.assertFalse(g['weights_exact_original'])

    def test_exact_identity_not_allclose(self):
        X,w=orthogonal();X[0,0]=np.nextafter(X[0,0],np.inf)
        m=audit.moments_oracle(X,w)
        # Use an explicit tiny representable correlation, with a complete
        # compatible matrix construction, to avoid cancellation in the fixture.
        X[0,0]+=.000001;m=audit.moments_oracle(X,w)
        self.assertFalse(m['identity_correlation'])

    def test_no_mass_repair_or_effective_sample_constraint(self):
        X,w,d,c,own,q=inputs();g=audit.geometry_oracle(d[:12],c,own,q,audit.moments_oracle(X,w))
        np.testing.assert_array_equal(g['weights'],np.asarray(g['unscaled_weights'])*g['mass_multiplier'])
        self.assertLessEqual(g['mass_absolute_error'],audit.MASS_RTOL*g['original_mass'])
        self.assertNotEqual(g['effective_samples'],g['original_effective_samples'])

    def test_column_permutation_and_global_scale_metric(self):
        X,w,d,c,own,q=inputs();m=audit.moments_oracle(X,w);g=audit.geometry_oracle(d[:12],c,own,q,m)
        p=np.arange(9,-1,-1);mp=audit.moments_oracle(X[:,p],w);gp=audit.geometry_oracle(d[:12],c,own[:,p],q[p],mp)
        np.testing.assert_allclose(gp['distance'],g['distance'],atol=2e-15,rtol=2e-15)
        m2=audit.moments_oracle(X/2.,w);np.testing.assert_allclose(m2['metric'],m['metric'],atol=2e-15,rtol=2e-15)

    def test_invalid_moments_nonfinite_results_and_factorization_hardfail(self):
        X,w,*_=inputs()
        for fault in ('nan','weight','shape','rank'):
            xx=X.copy();ww=w.copy()
            if fault=='nan':xx[0,0]=np.nan
            if fault=='weight':ww[0]=0.
            if fault=='shape':xx=xx[:,:9]
            if fault=='rank':xx[0,0]=.6
            with self.subTest(fault=fault),self.assertRaises(ValueError):audit.moments_oracle(xx,ww)
        with patch.object(np.linalg,'cholesky',side_effect=np.linalg.LinAlgError('no SPD')),self.assertRaises(np.linalg.LinAlgError):audit.moments_oracle(X,w)
        with self.assertRaises(ValueError):audit._matmul(np.array([[np.inf]]),np.ones((1,1)))

    def test_saved_matrix_and_geometry_one_ulp_rejected(self):
        X,w,d,c,own,q=inputs();m=audit.moments_oracle(X,w)
        inp=dict(X=X,dates=d,query_X=q[None,:]);ctx=dict(ids=np.arange(40),prediction_rows=np.array([0]),pooled_weights=w,scale=np.asarray(m['source_scale']))
        a=dict(training_row_ids=list(range(40)),query_row_ids=[0],source_scale=m['source_scale'],moments=m)
        audit.verify_supplied_moments(inp,a,ctx)
        for key in ('covariance','correlation','shrinkage','transform','metric','source_scale','training_weights','trace','floored'):
            bad=copy.deepcopy(a);v=bad['moments'][key]
            if key=='floored':v[0]=not v[0]
            elif isinstance(v,list) and isinstance(v[0],list):v[0][0]=float(np.nextafter(v[0][0],np.inf))
            elif isinstance(v,list):v[0]=float(np.nextafter(v[0],np.inf))
            else:bad['moments'][key]=float(np.nextafter(v,np.inf))
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_supplied_moments(inp,bad,ctx)
        row=dict(own_row_ids=list(range(12)),event_row=0,cutoff=str(c),geometry=audit.geometry_oracle(d[:12],c,own,q,m))
        for key in ('weights','distance','unscaled_weights','matched_mass','effective_samples','weights_exact_original'):
            bad=copy.deepcopy(row);v=bad['geometry'][key]
            if type(v) is bool:bad['geometry'][key]=not v
            elif isinstance(v,list):v[0]=float(np.nextafter(v[0],np.inf))
            else:bad['geometry'][key]=float(np.nextafter(v,np.inf))
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_supplied_geometry(inp,bad,m)

def make_bundle(train,meta,arrays,n=None):
    inp=audit.model_inputs(train,meta,arrays,n);control=frozen.frozen_tests.original_model(inp,inp['targets'])
    _,contexts=audit.control_oracle(inp,inp['targets'])
    diagonal=audit.previous.candidate_oracle(inp,control,contexts);candidate=audit.candidate_oracle(inp,control,contexts)
    ma=audit.model_audit(inp,control,diagonal,candidate)
    return dict(schema='monthly_correlation_distance257_models_v1',n=inp['n'],width=inp['width'],training_row_ids=inp['training_row_ids'],
      control=control,diagonal=diagonal,candidate=candidate,label_values=inp['targets']['log_excess'].copy(),label_preservation=audit.label_preservation(inp),
      control_gate=copy.deepcopy(ma['original_cache_parity']),diagnostic_gate=dict(passed=True,events=inp['n'],frozen_diagonal_implementation=True,correlation_models_computed=False),audit=ma)

@lru_cache(None)
def cached_bundle():return make_bundle(*fixture())
def full_bundle():return copy.deepcopy(cached_bundle())
def run(a,m,b,n=None):
    s,c,p,au=audit.selection_oracle(a,m,b,n);return audit.orders_from_choices(s,c,m),s,p,au
def verify(a,m,b,out,n=None):
    o,s,p,au=out;return audit.verify_selection(a,m,b,o,scores=s,pre_scores=p,audit=au,n=n)

class ModelTests(unittest.TestCase):
    def test_full_original_moments_geometry_svd_counters(self):
        t,m,a=fixture();b=full_bundle()
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as solves,patch.object(audit,'moments_oracle',wraps=audit.moments_oracle) as matrices:
            r=audit.verify_models(t,m,a,b)
        self.assertTrue(all(r[k] for k in audit.MODEL_FLAGS));self.assertEqual(solves.call_count,r['independent_solve_counts']['total_regression_solves'])
        self.assertEqual(matrices.call_count,r['independent_matrix_builds'])
        for arm in ('diagonal','candidate'):self.assertEqual(r['independent_solve_counts'][arm]['pooled_log_solves'],0)
        self.assertEqual(r['counts']['actual_regression_solves']-solves.call_count,4*r['counts']['control_log_refits'])

    def test_diagnostic_model_gate_never_computes_correlation(self):
        t,m,a=fixture();b=full_bundle()
        with patch.object(audit,'moments_oracle',side_effect=AssertionError('early corr')),patch.object(audit,'candidate_oracle',side_effect=AssertionError('early corr')),\
             patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as solves:
            r=audit.verify_diagnostic_models(t,m,a,b['diagonal'])
        self.assertFalse(r['correlation_models_computed']);self.assertFalse(r['correlation_moments_computed']);self.assertEqual(r['independent_matrix_builds'],0)
        self.assertEqual(solves.call_count,r['independent_solve_counts']['total_regression_solves'])

    def test_unchanged_all_labels_pool_constant_availability(self):
        t,m,a=fixture();b=full_bundle();inp=audit.model_inputs(t,m,a)
        self.assertEqual(b['label_preservation'],audit.label_preservation(inp))
        for role in ('diagonal','candidate'):
            for k in ('source_log','local_constant'):audit.exact_array(b[role]['raw'][k],b['control']['raw'][k],k)
            for k in ('available','local_available'):audit.exact_array(b[role][k],b['control'][k],k)

    def test_no_own_history_keeps_pooled_fallback(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);inp['query_symbols'][:]='NEW'
        c,ctx=audit.control_oracle(inp,inp['targets']);b=audit.candidate_oracle(inp,c,ctx)
        self.assertEqual(audit._linear_solves(b),0);self.assertFalse(b['local_available'].any());self.assertTrue(all(r['geometry'] is None for r in b['audit']['local']))
        audit.exact_array(b['raw']['local_linear'],c['raw']['source_log'],'No own fallback')

    def test_zero_residual_still_checks_matrices_and_weights(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a)
        for v in inp['targets'].values():v[:]=0.
        c,ctx=audit.control_oracle(inp,inp['targets']);b=audit.candidate_oracle(inp,c,ctx)
        self.assertEqual(audit._linear_solves(b),0);self.assertTrue(all(r['status']=='zero_residual' and r['geometry'] is not None for r in b['audit']['local']))

    def test_future_training_and_query_body_poison(self):
        t,m,a=fixture();b=make_bundle(t,m,a,2);mask=t['ends']>np.datetime64('2015-12-31')
        t['train_plain']=t['train_plain'].astype(object);t['train_plain'][mask]='future'
        for k in audit.TARGETS:t['target_'+k]=t['target_'+k].astype(object);t['target_'+k][mask]='future'
        t['event_plain']=t['event_plain'].astype(object);t['event_plain'][12:]='future'
        audit.verify_models(t,m,a,b,2)

    def test_n0_no_matrices_or_fits(self):
        t,m,a=fixture();b=make_bundle(t,m,a,0)
        with patch.object(audit,'moments_oracle',side_effect=AssertionError('empty matrix')):r=audit.verify_models(t,m,a,b,0)
        self.assertTrue(all(v==0 for v in r['counts'].values()));self.assertEqual(r['independent_solve_counts']['total_regression_solves'],0)

    def test_copied_pool_constant_labels_counts_and_gates_tamper(self):
        t,m,a=fixture()
        for fault in ('pool','constant','label','hash','count','controlgate','diaggate','diagweights'):
            b=full_bundle()
            if fault in ('pool','constant'):
                k='source_log' if fault=='pool' else 'local_constant';b['candidate']['raw'][k][0]=np.nextafter(b['candidate']['raw'][k][0],np.inf)
            if fault=='label':b['label_values'][0]+=.01
            if fault=='hash':b['label_preservation']['common_good_sha256']='0'*64
            if fault=='count':b['audit']['counts']['candidate_pooled_fits']=1
            if fault=='controlgate':b['control_gate']['candidate_models_computed']=True
            if fault=='diaggate':b['diagnostic_gate']['correlation_models_computed']=True
            if fault=='diagweights':b['diagonal']['audit']['local'][0]['geometry']['weights'][0]+=.001
            with self.subTest(fault=fault),self.assertRaises(ValueError):audit.verify_models(t,m,a,b)

class SelectionTests(unittest.TestCase):
    def test_three_arms_same_pool_weak_delta_and_exact_partial(self):
        _,m,a=fixture();b=full_bundle();out=run(a,m['monthly'],b);r=verify(a,m['monthly'],b,out)
        self.assertEqual(r['ordered_choices_checked'],12)
        for arm in audit.ARMS[1:]:
            ranks,delta,*_=audit.rank_oracle(a,m['monthly'],b,arm=arm)
            audit.exact_array(ranks['new_pool'],ranks['old_pool'],'Old pool')
            audit.exact_array(delta[2:],.0125*(ranks['new_local'][2:]-ranks['old_local'][2:]),'Weak local delta')
            for i in (0,1):
                audit.exact_array(out[1][arm][i],out[1][audit.ARMS[0]][i],'Nonweak/partial')
                audit.exact_array(out[2][arm][i],out[2][audit.ARMS[0]][i],'Nonweak/partial pre')

    def test_diagnostic_gate_unread_candidate_body(self):
        _,m,a=fixture();b=full_bundle();out=run(a,m['monthly'],b)
        class Poison:
            def __getitem__(self,k):raise AssertionError('correlation touched')
        b['candidate']=Poison()
        r=audit.verify_diagnostic_selection(a,m['monthly'],b,out[0][audit.ARMS[1]],out[1][audit.ARMS[1]],out[2][audit.ARMS[1]])
        self.assertFalse(r['correlation_selection_computed']);self.assertEqual(r['episode_calls'],1)

    def test_all_logical_physical_prefix_and_future_forecast_poison(self):
        _,m,a=fixture();b=full_bundle();out=run(a,m['monthly'],b)
        for n in range(5):
            p=run(a,m['monthly'],b,n);verify(a,m['monthly'],b,p,n)
            mm={k:v[:n] if k in ('dates','ends','symbols') else v for k,v in m['monthly'].items()};aa={k:v[:n].copy() for k,v in a.items()}
            q=run(aa,mm,b,n);verify(aa,mm,b,q,n)
            for arm in audit.ARMS:
                self.assertEqual(p[0][arm],out[0][arm][:n]);audit.exact_array(p[1][arm],q[1][arm],'Physical prefix')
        for role in ('diagonal','candidate'):
            for v in b[role]['raw'].values():v[12:]=np.inf
        self.assertEqual(run(a,m['monthly'],b,2)[0],{k:v[:2] for k,v in out[0].items()})

    def test_complete_domain_missing_cannot_rerank(self):
        _,m,a=fixture();b=full_bundle();b['candidate']['local_available'][:]=False
        self.assertEqual(int(audit.rank_oracle(a,m['monthly'],b)[2].sum()),24)
        b['candidate']['raw']['local_linear'][12]=np.nan
        with self.assertRaises(ValueError):audit.rank_oracle(a,m['monthly'],b)

    def test_two_names_and_cash(self):
        _,m,a=fixture();b=full_bundle();a['eligible'][2]=False;a['eligible'][2,:2]=True;a['eligible'][3]=False;a['eligible'][3,0]=True
        for i in (2,3):
            use=a['eligible'][i];a['rank_log_excess'][i,use]=audit.previous._rank(a['raw_source_log'][i,use]);a['rank_local_linear'][i]=a['rank_log_excess'][i]
            a['rank_local_linear'][i,use]=audit.previous._rank(a['raw_local_linear'][i,use])
        out=run(a,m['monthly'],b)
        for arm in audit.ARMS:self.assertEqual(len(out[0][arm][2]['symbols']),2);self.assertEqual(out[0][arm][3]['symbols'],[])

    def test_full_unselected_score_pre_audit_and_choices_tamper(self):
        _,m,a=fixture();b=full_bundle()
        for fault in ('score','pre','order','count','diagdelta','corrdelta'):
            o,s,p,au=copy.deepcopy(run(a,m['monthly'],b))
            if fault=='score':s[audit.ARMS[2]][2,5]+=1e-9
            if fault=='pre':p[audit.ARMS[2]][2,5]+=1e-9
            if fault=='order':o[audit.ARMS[2]][2]['symbols'].reverse()
            if fault=='count':au['counts']['candidate_changed_delta_cells']+=1
            if fault=='diagdelta':au['events'][2]['diag_delta'][0]+=1e-9
            if fault=='corrdelta':au['events'][2]['delta'][0]+=1e-9
            with self.subTest(fault=fault),self.assertRaises(ValueError):verify(a,m['monthly'],b,(o,s,p,au))

class BilateralCross(unittest.TestCase):
    def test_both_fixtures_original_diag_corr_full_audits_and_host_prefix_gates(self):
        # New production code is confined to this synthetic seam, never oracle.
        import monthly_correlation_distance257_kernel as kernel
        import test_monthly_correlation_distance257_kernel as producer_tests
        import monthly_correlation_distance257_host as host
        import monthly_local_distance256_kernel as old_kernel
        pins=(('monthly_correlation_distance257_kernel.py','087a5db6d0190718ae828a2fcd7ddb075b07c3b4e938a394ff11b45f06a9a413'),
          ('test_monthly_correlation_distance257_kernel.py','0841ddb2d11df7fd98bc57cc3302f4871e91f48063233f925663c23410eea5db'),
          ('monthly_correlation_distance257_host.py','040ab134e8898d8a763935d0ad67aebaeceaed69b0b5328ee5447fcff709c02b'))
        for file,digest in pins:self.assertEqual(hashlib.sha256((audit.ROOT/'work'/file).read_bytes()).hexdigest(),digest)
        t,m,a=fixture()
        for args in ((study(),t,m,a),producer_tests.fixture()):
            s,t,m,a=args;n=len(m['monthly']['dates']);before=copy.deepcopy((t,m,a))
            old=old_kernel.fit_models(t,m,a,n);o,ss,pp,_=old_kernel.decisions(s,a,m['monthly'],old,n)
            forecasts={'candidate_'+key:value.copy() for key,value in old['candidate']['raw'].items()}
            for key in ('available','local_available'):forecasts['candidate_'+key]=old['candidate'][key].copy()
            saved=dict(forecasts=forecasts,orders=o,scores=ss,pres=pp,meta=m['monthly'],input_sha256={k:'0'*64 for k in host.DIAGNOSTIC_FILES})
            control,diagnostic,model_receipts=host.model_callbacks(audit,t,m,a,n,saved)
            b=kernel.fit_models(t,m,a,n,control_validator=control,diagnostic_validator=diagnostic)
            host.bundle_preservation(b,n)
            proof=audit.verify_models(t,m,a,b,n);host.receipt(proof,n,(),'models')
            self.assertEqual(model_receipts['control'],b['control_gate']);self.assertEqual(model_receipts['diagnostic_models'],b['diagnostic_gate'])
            for arm in audit.ARMS[1:]:
                r,d,ra=audit.rank_and_delta(a,m['monthly'],b,n,arm=arm);pr,pd,pa=kernel.rank_and_delta(a,m['monthly'],b,n,arm=arm)
                for key in r:audit.exact_array(pr[key],r[key],'Bilateral exact rank/'+arm+'/'+key)
                audit.exact_array(pd,d,'Bilateral exact delta');audit.record_check(pa,ra,'Bilateral rank audit')
            inc,diag,gate_receipts=host.decision_callbacks(audit,a,m['monthly'],b,n,saved)
            out=kernel.decisions(s,a,m['monthly'],b,n,incumbent_validator=inc,diagnostic_validator=diag,
              expected_incumbent_orders=o[audit.ARMS[0]],expected_diagnostic_orders=o[audit.ARMS[1]])
            receipt=verify(a,m['monthly'],b,out,n);host.receipt(receipt,n,(),'selection',width=a['baseline'].shape[1])
            self.assertEqual(out[3]['incumbent_gate'],gate_receipts['incumbent']);self.assertEqual(out[3]['diagnostic_gate'],gate_receipts['diagnostic_selection'])
            for count in range(n+1):
                inc,diag,_=host.decision_callbacks(audit,a,m['monthly'],b,count,saved)
                logical=kernel.decisions(s,a,m['monthly'],b,count,incumbent_validator=inc,diagnostic_validator=diag,
                  expected_incumbent_orders=o[audit.ARMS[0]][:count],expected_diagnostic_orders=o[audit.ARMS[1]][:count])
                verify(a,m['monthly'],b,logical,count);host.exact_prefix(out,logical,count)
                s2,t2,m2,a2=producer_tests.physical(args,count)
                control,diagnostic,_=host.model_callbacks(audit,t2,m2,a2,count,saved)
                small=kernel.fit_models(t2,m2,a2,count,control_validator=control,diagnostic_validator=diagnostic)
                p=audit.verify_models(t2,m2,a2,small,count);host.receipt(p,count,(),'models');host.bundle_preservation(small,count);host.bundle_parity(b,small,count)
                inc,diag,_=host.decision_callbacks(audit,a2,m2['monthly'],small,count,saved)
                physical=kernel.decisions(s2,a2,m2['monthly'],small,count,incumbent_validator=inc,diagnostic_validator=diag,
                  expected_incumbent_orders=o[audit.ARMS[0]][:count],expected_diagnostic_orders=o[audit.ARMS[1]][:count])
                verify(a2,m2['monthly'],small,physical,count);host.exact_prefix(out,physical,count)
                fullcache=kernel.decisions(s2,a2,m2['monthly'],b,count)
                verify(a2,m2['monthly'],b,fullcache,count);host.exact_prefix(out,fullcache,count)
            poisoned=copy.deepcopy(b)
            for role in ('diagonal','candidate'):
                for value in poisoned[role]['raw'].values():value[12:]=np.inf
            short=kernel.decisions(s,a,m['monthly'],poisoned,2);verify(a,m['monthly'],poisoned,short,2);host.exact_prefix(out,short,2)
            broken=copy.deepcopy(saved);broken['forecasts']['candidate_local_linear'][0]=np.nextafter(broken['forecasts']['candidate_local_linear'][0],np.inf)
            control,diagnostic,_=host.model_callbacks(audit,t,m,a,n,broken)
            with self.assertRaises((ValueError,AssertionError)):kernel.fit_models(t,m,a,n,control_validator=control,diagnostic_validator=diagnostic)
            broken=copy.deepcopy(saved);broken['scores'][audit.ARMS[1]][0,0]=np.nextafter(broken['scores'][audit.ARMS[1]][0,0],np.inf)
            inc,diag,_=host.decision_callbacks(audit,a,m['monthly'],b,n,broken)
            with self.assertRaises((ValueError,AssertionError)):kernel.decisions(s,a,m['monthly'],b,n,incumbent_validator=inc,diagnostic_validator=diag)
            for key,value in t.items():audit.exact_array(value,before[0][key],'No training mutation/'+key)
            self.assertEqual(m,before[1])
            for key,value in a.items():audit.exact_array(value,before[2][key],'No original mutation/'+key)

if __name__=='__main__':unittest.main()
