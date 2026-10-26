"""Pure synthetic256 geometry, model, selection and bilateral contracts."""
import copy
from functools import lru_cache
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import monthly_local_distance256_independent as audit
import test_monthly_asset_target255_independent as frozen_tests

audit.require(hashlib.sha256((audit.ROOT/'work/test_monthly_asset_target255_independent.py').read_bytes()).hexdigest()==
 'b6aea9e714da159927d706ccd0e8ec4a83828a89b9c9df89e84c35cadf9febf8','Frozen synthetic fixtures')

def fixture():return frozen_tests.fixture()
def study():return frozen_tests.study()

def geometry_fixture():
    rng=np.random.default_rng(9256)
    dates=np.array(['2014-%02d-28'%m for m in range(1,13)],dtype='datetime64[D]')
    return dates,np.datetime64('2015-12-31'),rng.uniform(-.48,.48,(12,10)),rng.uniform(-.48,.48,10),np.linspace(.08,.42,10)

class GeometryTests(unittest.TestCase):
    def test_hand_diagonal_coefficients_distance_and_mass(self):
        d,c,X,q,s=geometry_fixture();g=audit.geometry_oracle(d,c,X,q,s)
        a=1/(s*s);a/=a.sum();dist=((X-q)**2*a).sum(axis=1)
        np.testing.assert_array_equal(g['coefficients'],a);np.testing.assert_array_equal(g['distance'],dist)
        old=np.asarray(g['original_weights']);new=np.asarray(g['weights'])
        self.assertAlmostEqual(new.sum(),old.sum(),places=14);self.assertFalse(np.array_equal(new,old))
        self.assertNotEqual(g['effective_samples'],g['original_effective_samples'])

    def test_equal_scales_copy_exact_and_near_equal_does_not_branch(self):
        args=list(geometry_fixture());args[4][:]=.25
        g=audit.geometry_oracle(*args);self.assertTrue(g['equal_scales']);self.assertTrue(g['weights_exact_original'])
        self.assertEqual(np.asarray(g['weights']).tobytes(),np.asarray(g['original_weights']).tobytes())
        args[4][3]=np.nextafter(.25,np.inf);g=audit.geometry_oracle(*args);self.assertFalse(g['equal_scales'])

    def test_identical_rows_preserve_original_weight_bits(self):
        args=list(geometry_fixture());args[2]=np.tile(args[3],(12,1));g=audit.geometry_oracle(*args)
        self.assertTrue(g['weights_exact_original']);self.assertEqual(g['mass_multiplier'],1.)
        self.assertEqual(g['distance'],[0.]*12)

    def test_no_mass_repair_or_last_weight_patch(self):
        g=audit.geometry_oracle(*geometry_fixture())
        expected=np.asarray(g['unscaled_weights'])*g['mass_multiplier']
        np.testing.assert_array_equal(g['weights'],expected)
        self.assertEqual(g['mass_absolute_error'],abs(float(expected.sum())-g['original_mass']))

    def test_scale_rescale_and_column_permutation(self):
        args=list(geometry_fixture());g=audit.geometry_oracle(*args)
        args[4]=args[4]*2.;h=audit.geometry_oracle(*args)
        np.testing.assert_array_equal(h['coefficients'],g['coefficients']);np.testing.assert_array_equal(h['weights'],g['weights'])
        order=np.arange(9,-1,-1);args[2]=args[2][:,order];args[3]=args[3][order];args[4]=args[4][order]
        h=audit.geometry_oracle(*args);np.testing.assert_allclose(h['weights'],g['weights'],rtol=2e-15,atol=0)

    def test_invalid_scales_rank_clock_and_nonfinite_hard_fail(self):
        for fault in ('zero','negative','nan','inf','shape','rank','future'):
            args=list(geometry_fixture())
            if fault=='zero':args[4][1]=0.
            if fault=='negative':args[4][1]=-.1
            if fault in ('nan','inf'):args[4][1]=float(fault)
            if fault=='shape':args[4]=args[4][:9]
            if fault=='rank':args[2][0,0]=.51
            if fault=='future':args[0][0]=np.datetime64('2020-01-01')
            with self.subTest(fault=fault),self.assertRaises((ValueError,FloatingPointError)):audit.geometry_oracle(*args)

    def test_geometry_inputs_unchanged(self):
        args=geometry_fixture();before=copy.deepcopy(args);audit.geometry_oracle(*args)
        for a,b in zip(args,before):np.testing.assert_array_equal(a,b)

    def test_supplied_equal_branch_one_ulp_mutations_rejected(self):
        d,c,X,q,s=geometry_fixture();s[:]=.25
        inp=dict(dates=d,X=X,query_X=q[None,:])
        original=dict(own_row_ids=list(range(len(d))),event_row=0,cutoff=str(c),geometry=audit.geometry_oracle(d,c,X,q,s))
        audit.verify_supplied_geometry(inp,original)
        for key in ('weights','unscaled_weights','distance','coefficients','source_scale','mass_multiplier','matched_mass','weights_exact_original'):
            row=copy.deepcopy(original);g=row['geometry']
            if key=='weights_exact_original':g[key]=False
            elif isinstance(g[key],list):g[key][0]=float(np.nextafter(g[key][0],np.inf))
            else:g[key]=float(np.nextafter(g[key],np.inf))
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_supplied_geometry(inp,row)

def make_bundle(train,meta,arrays,n=None):
    inp=audit.model_inputs(train,meta,arrays,n)
    control=frozen_tests.original_model(inp,inp['targets'])
    _,contexts=audit.control_oracle(inp,inp['targets'])
    candidate=audit.candidate_oracle(inp,control,contexts)
    ma=audit.model_audit(inp,control,candidate)
    return dict(schema='monthly_local_distance256_models_v1',n=inp['n'],width=inp['width'],training_row_ids=inp['training_row_ids'],
      control=control,candidate=candidate,label_values=inp['targets']['log_excess'].copy(),label_preservation=audit.label_preservation(inp),
      control_gate=copy.deepcopy(ma['original_cache_parity']),audit=ma)

@lru_cache(None)
def cached_bundle():return make_bundle(*fixture())
def full_bundle():return copy.deepcopy(cached_bundle())
def run(a,m,b,n=None):
    s,c,p,au=audit.selection_oracle(a,m,b,n);return audit.orders_from_choices(s,c,m),s,p,au
def verify(a,m,b,out,n=None):
    o,s,p,au=out;return audit.verify_selection(a,m,b,o,scores=s,pre_scores=p,audit=au,n=n)

class ModelTests(unittest.TestCase):
    def test_full_model_weights_parameters_and_svd_counts(self):
        t,m,a=fixture();b=full_bundle()
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as spy:
            result=audit.verify_models(t,m,a,b)
        self.assertTrue(all(result[k] for k in audit.MODEL_FLAGS))
        solves=result['independent_solve_counts'];self.assertEqual(len(spy.call_args_list),solves['total_regression_solves'])
        self.assertEqual(solves['candidate']['pooled_log_solves'],0)
        dimensions=[call.args[0].shape[1] for call in spy.call_args_list]
        self.assertEqual(dimensions.count(10),solves['control']['pooled_log_solves'])
        self.assertEqual(dimensions.count(11),solves['control']['local_linear_solves']+solves['candidate']['local_linear_solves'])
        c=result['counts'];self.assertEqual(c['candidate_pooled_fits'],c['candidate_log_refits']);self.assertEqual(c['candidate_pooled_fits'],0)
        self.assertEqual(c['actual_regression_solves']-solves['total_regression_solves'],4*c['control_log_refits'])
        self.assertEqual(c['candidate_kernel_evaluations'],24)

    def test_original_labels_pool_constant_and_masks_exact(self):
        t,m,a=fixture();b=full_bundle();inp=audit.model_inputs(t,m,a)
        audit.exact_array(b['label_values'],inp['targets']['log_excess'],'Unchanged y')
        self.assertEqual(b['label_preservation'],audit.label_preservation(inp))
        for key in ('source_log','local_constant'):audit.exact_array(b['candidate']['raw'][key],b['control']['raw'][key],key)
        for key in ('available','local_available'):audit.exact_array(b['candidate'][key],b['control'][key],key)
        for old,new in zip(b['control']['audit']['local'],b['candidate']['audit']['local']):
            self.assertEqual(old['constant_correction'],new['constant_correction'])
            self.assertAlmostEqual(old['weight_mass'],new['weight_mass'],places=13)

    def test_control_gate_does_no_new_geometry_and_rejects_ulp_cache(self):
        t,m,a=fixture();b=full_bundle()
        with patch.object(audit,'candidate_oracle',side_effect=AssertionError('new candidate')),patch.object(audit,'geometry_oracle',side_effect=AssertionError('new geometry')):
            r=audit.verify_control(t,m,a,b['control'])
        self.assertFalse(r['candidate_models_computed']);self.assertEqual(r['episode_calls'],0)
        a['raw_source_log'][0,0]=np.nextafter(a['raw_source_log'][0,0],np.inf)
        with self.assertRaises(ValueError):audit.verify_control(t,m,a,b['control'])

    def test_no_own_history_retains_pooled_fallback_without_solve(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);inp['query_symbols'][:]='UNSEEN'
        control,contexts=audit.control_oracle(inp,inp['targets']);candidate=audit.candidate_oracle(inp,control,contexts)
        self.assertFalse(candidate['local_available'].any());audit.exact_array(candidate['raw']['local_linear'],control['raw']['source_log'],'No own sourcecopy')
        c=audit.model_counts(inp,control,candidate);self.assertEqual(c['candidate_local_evaluations'],24)
        self.assertEqual(c['candidate_local_solves'],0);self.assertEqual(c['candidate_kernel_evaluations'],0)

    def test_zero_residual_checks_geometry_without_local_solve(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a)
        for value in inp['targets'].values():value[:]=0.
        control,contexts=audit.control_oracle(inp,inp['targets']);candidate=audit.candidate_oracle(inp,control,contexts)
        c=audit.model_counts(inp,control,candidate);self.assertEqual(c['candidate_kernel_evaluations'],24);self.assertEqual(c['candidate_local_solves'],0)
        self.assertTrue(all(r['status']=='zero_residual' for r in candidate['audit']['local']))

    def test_effective_scale_floor_and_original_five_year_cohort(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);inp['X'][:,0]=0.;inp['query_X'][:,0]=0.
        control,contexts=audit.control_oracle(inp,inp['targets'])
        for ctx in contexts:
            self.assertEqual(ctx['scale'][0],1.)
            ids=ctx['ids'];self.assertTrue((inp['ends'][ids]<=ctx['cutoff']).all())
        candidate=audit.candidate_oracle(inp,control,contexts)
        self.assertTrue(all(r['geometry']['source_scale'][0]==1. for r in candidate['audit']['local']))

    def test_future_training_and_query_poison_not_consumed(self):
        t,m,a=fixture();b=make_bundle(t,m,a,2);future=t['ends']>np.datetime64('2015-12-31')
        t['train_plain']=t['train_plain'].astype(object);t['train_plain'][future]='future body'
        for k in audit.TARGETS:t['target_'+k]=t['target_'+k].astype(object);t['target_'+k][future]='future label'
        t['event_plain']=t['event_plain'].astype(object);t['event_plain'][12:]='future query'
        audit.verify_models(t,m,a,b,2)

    def test_n0_zero_rows_zero_models_zero_geometry(self):
        t,m,a=fixture();b=make_bundle(t,m,a,0)
        with patch.object(audit,'geometry_oracle',side_effect=AssertionError('empty geometry')):r=audit.verify_models(t,m,a,b,0)
        self.assertTrue(all(v==0 for v in r['counts'].values()));self.assertEqual(r['independent_solve_counts']['total_regression_solves'],0)

    def test_weights_scale_coefficients_labels_pool_audit_tamper_fails(self):
        t,m,a=fixture()
        for fault in ('weight','scale','coefficient','mass','label','labelhash','pool','constant','annualids','count','gate'):
            b=full_bundle();g=b['candidate']['audit']['local'][0]['geometry']
            if fault=='weight':g['weights'][0]+=.001
            if fault=='scale':g['source_scale'][0]+=.01
            if fault=='coefficient':g['coefficients'][0]+=.01
            if fault=='mass':g['matched_mass']+=.1
            if fault=='label':b['label_values'][0]+=.01
            if fault=='labelhash':b['label_preservation']['target_sha256']['simple_excess']='0'*64
            if fault=='pool':b['candidate']['raw']['source_log'][0]=np.nextafter(b['candidate']['raw']['source_log'][0],np.inf)
            if fault=='constant':b['candidate']['raw']['local_constant'][0]=np.nextafter(b['candidate']['raw']['local_constant'][0],np.inf)
            if fault=='annualids':b['candidate']['audit']['annual'][0]['training_row_ids'].reverse()
            if fault=='count':b['audit']['counts']['candidate_pooled_fits']=1
            if fault=='gate':b['control_gate']['candidate_models_computed']=True
            with self.subTest(fault=fault),self.assertRaises(ValueError):audit.verify_models(t,m,a,b)

    def test_supplied_general_geometry_ulp_is_not_continuous_tolerance(self):
        t,m,a=fixture()
        for key in ('weights','coefficients','distance','mass_multiplier','matched_mass'):
            b=full_bundle();g=b['candidate']['audit']['local'][0]['geometry']
            if isinstance(g[key],list):g[key][0]=float(np.nextafter(g[key][0],np.inf))
            else:g[key]=float(np.nextafter(g[key],np.inf))
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_models(t,m,a,b)

class SelectionTests(unittest.TestCase):
    def test_full_scores_orders_and_local_delta_only(self):
        _,m,a=fixture();b=full_bundle();out=run(a,m['monthly'],b);r=verify(a,m['monthly'],b,out)
        self.assertTrue(all(r[k] for k in audit.SELECTION_FLAGS));self.assertEqual(r['ordered_choices_checked'],8)
        ranks,d,_,_,_=audit.rank_oracle(a,m['monthly'],b)
        audit.exact_array(ranks['new_pool'],ranks['old_pool'],'Pool unchanged')
        for i in (2,3):audit.exact_array(d[i],.0125*(ranks['new_local'][i]-ranks['old_local'][i]),'Local delta')
        for i in (0,1):
            for item in (out[1],out[2]):self.assertEqual(item[audit.ARMS[0]][i].tobytes(),item[audit.ARMS[1]][i].tobytes())

    def test_equal_ranks_no_addition_and_negative_target_valid(self):
        _,m,a=fixture();b=full_bundle();b['candidate']=copy.deepcopy(b['control']);out=run(a,m['monthly'],b)
        self.assertEqual(out[0][audit.ARMS[0]],out[0][audit.ARMS[1]])
        audit.exact_array(out[2][audit.ARMS[0]],out[2][audit.ARMS[1]],'Zero delta bytes')
        b['candidate']['raw']['local_linear'][12:]=b['candidate']['raw']['local_linear'][12:][::-1]
        _,d,_,_,_=audit.rank_oracle(a,m['monthly'],b);self.assertTrue((d[2:]<0).any())

    def test_whole_U_no_local_history_rerank(self):
        _,m,a=fixture();b=full_bundle();b['candidate']['local_available'][:]=False
        _,_,U,_,_=audit.rank_oracle(a,m['monthly'],b);self.assertEqual(int(U.sum()),24)
        b['candidate']['raw']['local_linear'][12]=np.nan
        with self.assertRaises(ValueError):audit.rank_oracle(a,m['monthly'],b)

    def test_logical_physical_prefix_and_future_forecast_poison(self):
        _,m,a=fixture();b=full_bundle();full=run(a,m['monthly'],b)
        for n in range(5):
            small=run(a,m['monthly'],b,n);verify(a,m['monthly'],b,small,n)
            mm={k:v[:n] if k in ('dates','ends','symbols') else v for k,v in m['monthly'].items()};aa={k:v[:n].copy() for k,v in a.items()}
            physical=run(aa,mm,b,n)
            for arm in audit.ARMS:
                self.assertEqual(small[0][arm],full[0][arm][:n]);audit.exact_array(physical[1][arm],small[1][arm],'Physical prefix')
        for raw in b['candidate']['raw'].values():raw[12:]=np.inf
        self.assertEqual(run(a,m['monthly'],b,2)[0],{k:v[:2] for k,v in full[0].items()})

    def test_two_names_and_cash(self):
        _,m,a=fixture();b=full_bundle();a['eligible'][2]=False;a['eligible'][2,:2]=True;a['eligible'][3]=False;a['eligible'][3,0]=True
        for i in (2,3):
            use=a['eligible'][i];a['rank_log_excess'][i,use]=audit._rank(a['raw_source_log'][i,use]);a['rank_local_linear'][i]=a['rank_log_excess'][i]
            a['rank_local_linear'][i,use]=audit._rank(a['raw_local_linear'][i,use])
        out=run(a,m['monthly'],b);self.assertEqual(len(out[0][audit.ARMS[1]][2]['symbols']),2);self.assertEqual(out[0][audit.ARMS[1]][3]['symbols'],[])

    def test_full_score_pre_order_and_count_tamper_rejected(self):
        _,m,a=fixture();b=full_bundle()
        for fault in ('score','pre','order','count','audit'):
            o,s,p,au=copy.deepcopy(run(a,m['monthly'],b))
            if fault=='score':s[audit.ARMS[1]][2,5]+=1e-9
            if fault=='pre':p[audit.ARMS[1]][2,5]+=1e-9
            if fault=='order':o[audit.ARMS[1]][2]['symbols'].reverse()
            if fault=='count':au['counts']['changed_delta_cells']+=1
            if fault=='audit':au['events'][2]['delta'][0]+=1e-9
            with self.subTest(fault=fault),self.assertRaises(ValueError):verify(a,m['monthly'],b,(o,s,p,au))

class BilateralCross(unittest.TestCase):
    def test_both_fixtures_all_models_geometry_prefixes_and_host_gates(self):
        # New production imports are confined to this synthetic integration.
        import monthly_local_distance256_kernel as kernel
        import test_monthly_local_distance256_kernel as producer_tests
        import monthly_local_distance256_host as host
        for filename,digest in (('monthly_local_distance256_kernel.py','97ada679902942f31e4881354fcf95b1fccd2119f77c09760904b8c328364e30'),
          ('test_monthly_local_distance256_kernel.py','f4b9feef9c998ea22e28cbc684fc78ba9e3ed48e4473306ce07f7cccb826f621')):
            self.assertEqual(hashlib.sha256((audit.ROOT/'work'/filename).read_bytes()).hexdigest(),digest)
        t,m,a=fixture()
        for args in ((study(),t,m,a),producer_tests.fixture()):
            s,t,m,a=args;n=len(m['monthly']['dates']);before=copy.deepcopy((t,m,a))
            callback,control_receipts=host.model_callbacks(audit,t,m,a,n)
            b=kernel.fit_models(t,m,a,n,control_validator=callback)
            host.bundle_preservation(b,n)
            proof=audit.verify_models(t,m,a,b,n);host.receipt(proof,n,audit.MODEL_FLAGS,'models')
            self.assertEqual(control_receipts['control'],b['control_gate'])
            r,d,ra=audit.rank_and_delta(a,m['monthly'],b,n);pr,pd,pa=kernel.rank_and_delta(a,m['monthly'],b,n)
            for key in r:audit.exact_array(pr[key],r[key],'Bilateral exact rank/'+key)
            audit.exact_array(pd,d,'Bilateral exact delta');audit.record_check(pa,ra,'Bilateral full rank audit')
            expected=run(a,m['monthly'],b,n)
            callback,inc_receipts=host.incumbent_callback(audit,a,m['monthly'],n)
            out=kernel.decisions(s,a,m['monthly'],b,n,incumbent_validator=callback,expected_incumbent_orders=expected[0][audit.ARMS[0]])
            receipt=verify(a,m['monthly'],b,out,n);host.receipt(receipt,n,audit.SELECTION_FLAGS,'selection')
            self.assertEqual(out[3]['incumbent_gate'],inc_receipts['incumbent'])
            for count in range(n+1):
                inc,_=host.incumbent_callback(audit,a,m['monthly'],count)
                logical=kernel.decisions(s,a,m['monthly'],b,count,incumbent_validator=inc,expected_incumbent_orders=out[0][audit.ARMS[0]][:count])
                verify(a,m['monthly'],b,logical,count);host.exact_prefix(out,logical,count)
                ss,tt,mm,aa=producer_tests.physical(args,count)
                control,_=host.model_callbacks(audit,tt,mm,aa,count)
                small=kernel.fit_models(tt,mm,aa,count,control_validator=control)
                proof=audit.verify_models(tt,mm,aa,small,count);host.receipt(proof,count,audit.MODEL_FLAGS,'models');host.bundle_parity(b,small,count)
                inc,_=host.incumbent_callback(audit,aa,mm['monthly'],count)
                physical=kernel.decisions(ss,aa,mm['monthly'],small,count,incumbent_validator=inc,expected_incumbent_orders=out[0][audit.ARMS[0]][:count])
                verify(aa,mm['monthly'],small,physical,count);host.exact_prefix(out,physical,count)
                same=kernel.decisions(ss,aa,mm['monthly'],b,count)
                verify(aa,mm['monthly'],b,same,count);host.exact_prefix(out,same,count)
            poison=copy.deepcopy(b)
            for v in poison['candidate']['raw'].values():v[12:]=np.inf
            got=kernel.decisions(s,a,m['monthly'],poison,2);verify(a,m['monthly'],poison,got,2);host.exact_prefix(out,got,2)
            for key,value in t.items():audit.exact_array(value,before[0][key],'No train mutation/'+key)
            self.assertEqual(m,before[1])
            for key,value in a.items():audit.exact_array(value,before[2][key],'No cache mutation/'+key)

if __name__=='__main__':unittest.main()
