"""Pure synthetic independent135 tests. No historical arrays or outcomes read."""
import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import monthly_absolute_state135_prepare_review as r


def fixture(sd=.004):
    m=dict(available=True,status='available',reason=None,fit_attempted=True,
        scaler=dict(mean=.001,sd=sd),parameters=dict(means=[-.1,.2],variances=[1.,4.],
        initial=[.5,.5],transition=[[.9,.1],[.1,.9]]),filtered_posterior=[.7,.3],
        certificate=dict(converged=True,all_arithmetic_finite=True,low_variance_state_first=True))
    y=dict(year=2020,fit_attempted=True,status='available',reason=None,model=m,cutoff='2019-12-31')
    f=np.array([.3,.0,1.]);q=np.array([.8,1.,.0])
    rows=[dict(signal_index=i,signal_date=d,year=2020,available=True,reason=None,
        filtered_probability=float(f[i]),memoryless_probability=float(q[i]),first_invalid_query_date=None)
        for i,d in enumerate(('2020-01-31','2020-02-28','2020-03-31'))]
    return dict(years=[y],signals=rows,new_model_fits=1),f,q


def score_fixture():
    shape=(2,5);baseline=np.array([[.16,.12,.08,.07,-.10],[.15,.13,.10,.09,-.10]])
    core=np.array([[.10,.08,.05,.04,-.10],[.10,.08,.06,.05,-.10]])
    a=dict(baseline=baseline,core=core,eligible=np.ones(shape,bool),core_weight=np.array([.5,.52]),
       state_weak_narrow=np.array([False,True]),state_calm10=np.array([True,False]),pre_score=baseline.copy(),
       effects=np.zeros(shape),forecast_available=np.zeros(shape,bool),rank_log_excess=np.zeros(shape),
       old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),model_available=np.zeros(shape,bool),
       raw_local_constant=np.zeros(shape),raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
    raw=np.array([[.4,.3,.2,.1,-.5],[.4,.2,.3,.1,-.5]])
    risk=np.array([[.2,.4,.1,.3,-.5],[.2,.3,.1,.4,-.5]])
    return a,dict(symbols=[list('ABCDE'),list('ABCDE')]),raw,risk


class MappingTests(unittest.TestCase):
    def test_all_three_branches_preserve_each_stream(self):
        for sd,expected in ((.001,1.),(.02,0.),(.004,None)):
            a,f,q=fixture(sd);out=r.mapping_oracle(a,f,q)
            for key,original in (('filtered',f),('memoryless',q)):
                np.testing.assert_array_equal(out[key],original if expected is None else np.full(3,expected))
                if expected is None:self.assertEqual(out[key].tobytes(),original.tobytes())
    def test_exact_lower_threshold_is_nonquiet(self):
        a,_,_=fixture(.1/math.sqrt(252));sigma,mask=r.state_classification(a['years'][0]['model'])
        self.assertEqual(sigma[0],.1);self.assertEqual(mask,[False,False])
    def test_high_state_equal_threshold_straddles(self):
        a,_,_=fixture(.1/math.sqrt(252));a['years'][0]['model']['parameters']['variances']=[.25,1.]
        self.assertEqual(r.state_classification(a['years'][0]['model'])[1],[True,False])
    def test_nextafter_threshold_no_epsilon_band(self):
        sd=.1/math.sqrt(252)
        for value,expected in ((math.nextafter(sd,0),[True,False]),(math.nextafter(sd,math.inf),[False,False])):
            a,_,_=fixture(value);self.assertEqual(r.state_classification(a['years'][0]['model'])[1],expected)
    def test_exact_scalar_operation_order(self):
        a,_,_=fixture();sigma,_=r.state_classification(a['years'][0]['model'])
        self.assertEqual(sigma,[.004*math.sqrt(x)*math.sqrt(252) for x in [1.,4.]])
    def test_means_do_not_enter_volatility(self):
        a,_,_=fixture();m=a['years'][0]['model'];before=r.state_classification(m)
        m['scaler']['mean']=-9;m['parameters']['means']=[-3,4]
        self.assertEqual(r.state_classification(m),before)
    def test_nonmutation_and_zero_fits(self):
        a,f,q=fixture();before=copy.deepcopy(a);fb=f.tobytes();qb=q.tobytes()
        with patch.object(r.old,'fit_record',side_effect=AssertionError('No fitting')),patch.object(r.old,'annual_oracle',side_effect=AssertionError('No refiltering')):
            out=r.mapping_oracle(a,f,q)
        self.assertEqual(a,before);self.assertEqual(fb,f.tobytes());self.assertEqual(qb,q.tobytes())
        self.assertEqual(out['audit']['new_model_fits'],0);self.assertEqual(out['audit']['reused_model_fits'],1)
        self.assertEqual(out['audit']['years'][0]['model'],a['years'][0]['model'])
        self.assertIsNot(out['audit']['years'][0]['model'],a['years'][0]['model'])
    def test_missing_signal_retained_under_constant_mapping(self):
        a,f,q=fixture(.001);a['signals'][1].update(available=False,reason='invalid_query_history',filtered_probability=None,memoryless_probability=None)
        f[1]=q[1]=np.nan;out=r.mapping_oracle(a,f,q)
        self.assertTrue(math.isnan(out['filtered'][1]));self.assertIsNone(out['audit']['signals'][1]['filtered_probability'])
        self.assertEqual(out['filtered'][0],1.)
    def test_unavailable_year_before_fit(self):
        a,f,q=fixture();a['new_model_fits']=0;a['years'][0].update(fit_attempted=False,status='unavailable',reason='short',model=None)
        for row in a['signals']:row.update(available=False,reason='short',filtered_probability=None,memoryless_probability=None)
        f[:]=q[:]=np.nan;out=r.mapping_oracle(a,f,q)
        self.assertIsNone(out['audit']['years'][0]['quiet_state_mask']);self.assertIsNone(out['audit']['years'][0]['annualized_state_volatility'])
    def test_unavailable_postfit_is_not_classified(self):
        a,f,q=fixture();a['years'][0].update(status='unavailable',reason='unseparated',model=dict(available=False,status='unavailable',fit_attempted=True))
        for row in a['signals']:row.update(available=False,reason='unseparated',filtered_probability=None,memoryless_probability=None)
        f[:]=q[:]=np.nan;out=r.mapping_oracle(a,f,q)
        self.assertEqual(out['audit']['reused_model_fits'],1);self.assertIsNone(out['audit']['years'][0]['quiet_state_mask'])
    def test_unsorted_equal_negative_or_nonfinite_variances_rejected(self):
        for v in ([4.,1.],[1.,1.],[0.,1.],[-1.,1.],[1.,np.inf],[1.,np.nan]):
            a,_,_=fixture();a['years'][0]['model']['parameters']['variances']=v
            with self.subTest(v=v),self.assertRaises(ValueError):r.state_classification(a['years'][0]['model'])
    def test_malformed_scale_or_parameter_rejected(self):
        for sd in (0.,-1.,True,'0.004',np.inf,np.nan):
            a,_,_=fixture(sd)
            with self.subTest(sd=sd),self.assertRaises(ValueError):r.state_classification(a['years'][0]['model'])
        a,_,_=fixture();a['years'][0]['model']['parameters']['transition'][0]=[.4,.4]
        with self.assertRaises(ValueError):r.state_classification(a['years'][0]['model'])
    def test_derived_overflow_rejected(self):
        a,_,_=fixture(1e308)
        with self.assertRaises(ValueError):r.state_classification(a['years'][0]['model'])
    def test_model_certificate_required(self):
        for k in ('converged','all_arithmetic_finite','low_variance_state_first'):
            a,_,_=fixture();a['years'][0]['model']['certificate'][k]=False
            with self.subTest(k=k),self.assertRaises(ValueError):r.state_classification(a['years'][0]['model'])
    def test_probability_validated_before_constant_mapping(self):
        for sd in (.001,.02):
            for v in (-.1,1.1,np.inf,np.nan,True,'0.3',None):
                a,f,q=fixture(sd);a['signals'][0]['filtered_probability']=v
                with self.subTest(sd=sd,v=v),self.assertRaises(ValueError):r.mapping_oracle(a,f,q)
    def test_exact_array_mismatch_rejected(self):
        a,f,q=fixture();f[0]=math.nextafter(f[0],1.)
        with self.assertRaises(ValueError):r.mapping_oracle(a,f,q)
    def test_array_bool_string_shape_or_inf_rejected(self):
        a,f,q=fixture()
        for v in ([True,False,True],['.3','0','1'],f[:,None],f[:2],[.3,0,np.inf]):
            with self.subTest(v=str(v)),self.assertRaises(ValueError):r.mapping_oracle(a,v,q)
    def test_source_count_and_year_or_index_mismatch_rejected(self):
        changes=[lambda a:a.update(new_model_fits=2),lambda a:a['signals'][0].update(year=2019),
                 lambda a:a['signals'][0].update(signal_index=True),lambda a:a['years'].append(copy.deepcopy(a['years'][0]))]
        for change in changes:
            a,f,q=fixture();change(a)
            with self.assertRaises(ValueError):r.mapping_oracle(a,f,q)
    def test_audit_retains_every_nonmapping_field_exactly(self):
        a,f,q=fixture();a['extra_note']={'arbitrary':['kept']};a['years'][0]['model']['log_likelihood_history']=[-50.,-45.]
        b=r.mapping_oracle(a,f,q)['audit'];self.assertEqual(b['extra_note'],a['extra_note'])
        for old,new in zip(a['years'],b['years']):
            self.assertEqual(old,{k:v for k,v in new.items() if k not in ('annualized_state_volatility','quiet_state_mask')})
    def test_later_year_changes_do_not_affect_prefix(self):
        a,f,q=fixture();y=copy.deepcopy(a['years'][0]);y['year']=2021;y['model']['scaler']['sd']=.001
        a['years'].append(y);a['new_model_fits']=2
        row=copy.deepcopy(a['signals'][0]);row.update(signal_index=3,signal_date='2021-01-29',year=2021)
        a['signals'].append(row);f=np.r_[f,.3];q=np.r_[q,.8];before=r.mapping_oracle(a,f,q)
        a['years'][1]['model']['scaler']['sd']=.02;after=r.mapping_oracle(a,f,q)
        self.assertEqual(before['audit']['signals'][:3],after['audit']['signals'][:3]);self.assertNotEqual(before['filtered'][3],after['filtered'][3])
    def test_producer_full_synthetic_parity(self):
        import monthly_absolute_state135_model as producer
        for sd in (.001,.004,.02):
            a,f,q=fixture(sd);expected=r.mapping_oracle(a,f,q);got=producer.transform(a,f,q)
            self.assertEqual(got['audit'],expected['audit']);self.assertEqual(got['filtered'].tobytes(),expected['filtered'].tobytes())
            self.assertEqual(got['memoryless'].tobytes(),expected['memoryless'].tobytes())


class ScoreTests(unittest.TestCase):
    def test_zero_means_risk_not_missing(self):
        a,m,raw,risk=score_fixture();zero=np.zeros(2);scores,orders,delta=r.old.score_oracle(a,m,raw,risk,zero,zero)
        np.testing.assert_allclose(delta[r.NAMES[2]],a['core_weight'][:,None]*risk-a['core'],rtol=0,atol=0)
        self.assertFalse(np.array_equal(scores[r.NAMES[2]],scores[r.NAMES[0]]))
    def test_one_means_raw(self):
        a,m,raw,risk=score_fixture();one=np.ones(2);_,_,delta=r.old.score_oracle(a,m,raw,risk,one,one)
        np.testing.assert_allclose(delta[r.NAMES[1]],a['core_weight'][:,None]*raw-a['core'],rtol=0,atol=0)
    def test_missing_exact_baseline(self):
        a,m,raw,risk=score_fixture();na=np.full(2,np.nan);scores,orders,_=r.old.score_oracle(a,m,raw,risk,na,na)
        for name in r.NAMES[1:]:self.assertEqual(scores[name].tobytes(),scores[r.NAMES[0]].tobytes());self.assertEqual(orders[name],orders[r.NAMES[0]])
    def test_mixed_probability_keeps_original_core_weight(self):
        a,m,raw,risk=score_fixture();q=np.array([.25,.6]);_,_,delta=r.old.score_oracle(a,m,raw,risk,q,q)
        np.testing.assert_allclose(delta[r.NAMES[2]],a['core_weight'][:,None]*(q[:,None]*raw+(1-q[:,None])*risk)-a['core'],atol=0,rtol=0)
    def test_prefix_scores_and_order_exact(self):
        a,m,raw,risk=score_fixture();q=np.array([.3,.7]);scores,orders,_=r.old.score_oracle(a,m,raw,risk,q,q)
        ss,cc,_=r.old.score_oracle(a,m,raw,risk,q[:1],q[:1],1)
        for name in r.NAMES:self.assertEqual(ss[name].tobytes(),scores[name][:1].tobytes());self.assertEqual(cc[name],orders[name][:1])
    def test_score_oracle_uses_single_episode_per_arm(self):
        a,m,raw,risk=score_fixture();helper=r.old.original_helper();original=helper.episode
        with patch.object(helper,'episode',wraps=original) as episode,patch.object(r.old,'original_helper',return_value=helper):
            r.old.score_oracle(a,m,raw,risk,np.ones(2),np.ones(2));self.assertEqual(episode.call_count,3)


class ClosureTests(unittest.TestCase):
    def test_bound_reuse_dependency_accepts_and_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            p=(Path(d)/'evidence').resolve();p.write_bytes(b'proof');sha=hashlib.sha256(b'proof').hexdigest();reader=r.old.Reader(d)
            self.assertEqual(r.require_bound(reader,{str(p):sha},{str(p):sha}),{str(p):sha})
            with self.assertRaises(ValueError):r.require_bound(reader,{str(p):sha},{})
    def test_wrong_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=(Path(d)/'evidence').resolve();p.write_bytes(b'changed');reader=r.old.Reader(d)
            with self.assertRaises(ValueError):r.require_bound(reader,{str(p):'0'*64},{str(p):'0'*64})
    def test_wrong_reused_path_rejected_before_reads(self):
        reader=r.old.Reader('/tmp');inputs={k:Path('/tmp/wrong') for k in r.INPUTS if k.startswith('reused_')}
        with self.assertRaises(ValueError):r.reuse_certificate(reader,{},inputs)
    def test_no_direct_producer_import_in_reviewer(self):
        import ast
        tree=ast.parse(Path(r.__file__).read_text());modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules += [v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom):modules.append(node.module)
        self.assertFalse(any(x and ('absolute_state135_model' in x or 'absolute_state135_runner' in x) for x in modules))
    def test_no_fit_or_annual_call_in_actual_audit(self):
        import ast,inspect
        tree=ast.parse(inspect.getsource(r.audit));names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(names)&{'fit_oracle','fit_record','annual_oracle','load_panel'})

if __name__=='__main__':unittest.main()
