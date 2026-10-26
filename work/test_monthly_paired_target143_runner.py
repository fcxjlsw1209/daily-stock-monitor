"""Artificial-data-only scorer, selection and lifecycle integrity checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_paired_target143_runner as r
import register_monthly_paired_target143 as reg

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = r.module('paired143_test_adapter', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
POLICY = dict(budget=.05, ceiling=5, gap=.02)


def arrays(n=4, width=5):
    scores = np.tile(np.linspace(.2, .1, width), (n, 1))
    a = dict(baseline=scores.copy(), core=.5*scores, core_weight=np.full(n,.5),
        eligible=np.ones((n,width),bool), forecast_available=np.ones((n,width),bool),
        effects=np.zeros((n,width)), state_calm10=np.ones(n,bool),
        state_weak_narrow=np.zeros(n,bool), extension_available=np.ones((n,width),bool),
        rank_log_excess=np.tile(np.linspace(-.3,.5,width),(n,1)),
        paired_raw_close=np.tile(np.arange(width,dtype=float),(n,1)),
        paired_raw_open=np.tile(np.arange(width,dtype=float)[::-1],(n,1)),
        paired_available=np.ones((n,width),bool))
    m = dict(dates=['2020-%02d-28'%(i+1) for i in range(n)],
             symbols=[['S'+str(j) for j in range(width)] for i in range(n)])
    return a,m


class FakeBackend:
    adapter = ADAPTER
    def __init__(self): self.episodes=0
    def scope_pre(self,a,n):
        out=np.zeros(a['baseline'].shape);out[:n]=a['baseline'][:n];return out
    def selector(self,scores,good,symbols,effects,original,policy):
        self.episodes += 1
        choices=[sorted(np.flatnonzero(ok).tolist(),key=lambda j:(-sc[j],names[j]))[:3]
            for sc,ok,names in zip(scores,good,symbols)]
        return choices,scores.copy()
    def baseline(self,a,m,n,policy):
        # Deliberately different from scope_pre, proving missing fallback must
        # retain local67, not silently become scope67.
        out=self.scope_pre(a,n);out[:n,3] += .2
        return ADAPTER.score_and_choose('incumbent66',0.,np.full(a['baseline'].shape,np.nan),a,m,n,
            pre_score=out,effects=a['effects'],episode_policy=policy,boundary_selector=self.selector)


def trials():
    return [dict(name=name,training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),
        early=dict(cagr=.2),recent=dict(cagr=.2)) for name in r.NAMES]


class Scoring(unittest.TestCase):
    def setUp(self): self.a,self.m=arrays();self.b=FakeBackend()
    def test_two_paired_targets_different_rank(self):
        ranks,ready,use=r.ranked_pair(self.a,4)
        self.assertTrue(ready.all());self.assertTrue(use.all())
        self.assertFalse(np.array_equal(ranks['close'],ranks['open']))
    def test_rank_population_includes_nonboundary(self):
        ranks,_,_=r.ranked_pair(self.a,1)
        self.assertEqual(ranks['close'][0,0],-.3)
        self.assertEqual(ranks['close'][0,-1],.5)
    def test_tie_average_rank(self):
        self.a['paired_raw_close'][0]=1
        ranks,_,_=r.ranked_pair(self.a,1)
        np.testing.assert_allclose(ranks['close'][0],.1)
    def test_unavailable_single_member_entire_month_fallback(self):
        self.a['paired_available'][0,0]=False
        choices,scores,audit=r.choices_all(self.a,self.m,1,self.b,POLICY)
        for name in r.NAMES[1:]:
            self.assertEqual(choices[name],choices[r.NAMES[0]])
            self.assertEqual(scores[name].tobytes(),scores[r.NAMES[0]].tobytes())
        self.assertTrue(audit[0]['fallback_to_local_incumbent'])
        self.assertGreater(scores[r.NAMES[0]][0,3],self.a['baseline'][0,3])
    def test_one_arm_nan_causes_both_fallback(self):
        self.a['paired_raw_open'][0,0]=np.nan
        c,s,_=r.choices_all(self.a,self.m,1,self.b,POLICY)
        for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes())
    def test_missing_outside_original_population_does_not_shrink_rank(self):
        self.a['forecast_available'][0,0]=False
        self.a['paired_raw_close'][0,0]=np.nan
        rank,ready,use=r.ranked_pair(self.a,1)
        self.assertTrue(ready[0]);self.assertFalse(use[0,0])
        self.assertEqual(rank['close'][0,0],self.a['rank_log_excess'][0,0])
    def test_empty_population_falls_back(self):
        self.a['eligible'][0]=False
        _,ready,_=r.ranked_pair(self.a,1);self.assertFalse(ready[0])
    def test_infinite_prediction_rejected(self):
        self.a['paired_raw_open'][0,0]=np.inf
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)
    def test_string_and_bool_prediction_rejected(self):
        for bad in ('x',True):
            self.a['paired_raw_open']=self.a['paired_raw_open'].astype(object);self.a['paired_raw_open'][0,0]=bad
            with self.assertRaises(ValueError):r.ranked_pair(self.a,1)
    def test_future_poison_ignored(self):
        before=r.ranked_pair(self.a,1)
        self.a['paired_raw_open']=self.a['paired_raw_open'].astype(object);self.a['paired_raw_open'][1:]='future'
        after=r.ranked_pair(self.a,1)
        np.testing.assert_array_equal(before[0]['open'],after[0]['open'])
    def test_invalid_prefix(self):
        for n in (-1,5,True,1.5):
            with self.assertRaises(ValueError):r.ranked_pair(self.a,n)
    def test_missing_availability_boolean_invalid(self):
        self.a['paired_available']=self.a['paired_available'].astype(int)
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)
    def test_inputs_unchanged_and_all_prefixes_equal(self):
        snap={k:v.tobytes() for k,v in self.a.items()}
        c,s,_=r.choices_all(self.a,self.m,4,self.b,POLICY)
        for n in range(5):
            d,t,_=r.choices_all(self.a,self.m,n,self.b,POLICY)
            for name in r.NAMES:
                self.assertEqual(c[name][:n],d[name]);self.assertEqual(s[name][:n].tobytes(),t[name].tobytes())
        self.assertEqual(snap,{k:v.tobytes() for k,v in self.a.items()})
    def test_once_episode_per_arm(self):
        r.choices_all(self.a,self.m,2,self.b,POLICY);self.assertEqual(self.b.episodes,3)
    def test_calm_narrow_and_weak_coefficient(self):
        for weak,expected in ((False,.0025),(True,.0125)):
            self.a['state_weak_narrow'][:]=weak
            _,scores,_=r.choices_all(self.a,self.m,1,self.b,POLICY)
            delta=scores[r.NAMES[2]][0]-self.a['baseline'][0]
            rank,_,_=r.ranked_pair(self.a,1)
            np.testing.assert_allclose(delta,expected*(rank['open'][0]-self.a['rank_log_excess'][0]),atol=1e-16)
    def test_no_extension_calm_zero_slot(self):
        self.a['extension_available'][:]=False
        _,s,_=r.choices_all(self.a,self.m,1,self.b,POLICY)
        np.testing.assert_array_equal(s[r.NAMES[1]],self.a['baseline'][:1])


class Gates(unittest.TestCase):
    def test_tie_never_admits(self):self.assertFalse(r.select(trials(),trials())['admitted'])
    def test_both_controls_and_open_confirmation(self):
        rows,op=trials(),trials();rows[2]['training']['cagr']=.21;op[2]['training']['cagr']=.21
        self.assertTrue(r.select(rows,op)['admitted'])
    def test_open_failure_blocks_close_win(self):
        rows=trials();rows[2]['training']['cagr']=.3
        self.assertFalse(r.select(rows,trials())['admitted'])
    def test_open_must_not_harm_incumbent(self):
        rows,op=trials(),trials();rows[2]['training']['cagr']=.3
        op[1]['training']['cagr']=.1;op[2]['training']['cagr']=.15
        self.assertFalse(r.select(rows,op)['admitted'])
    def test_each_stability_tolerance_blocks(self):
        for group,field,val in [('training','sharpe',.949),('training','max_drawdown',-.231),('early','cagr',.174),('recent','cagr',.174)]:
            rows,op=trials(),trials();rows[2]['training']['cagr']=.3;op[2]['training']['cagr']=.3
            op[2][group][field]=val
            self.assertFalse(r.select(rows,op)['admitted'])
    def test_bad_order_rejected(self):
        with self.assertRaises(ValueError):r.select(trials()[::-1],trials())
    def test_nonfinite_or_boolean_fails(self):
        for value in (np.nan,np.inf,True,'0.3'):
            rows=trials();rows[2]['training']['cagr']=value
            self.assertFalse(r.select(rows,trials())['admitted'])


class ForecastValidation(unittest.TestCase):
    def forecast(self):
        x=np.array([.1,np.nan]);return dict(raw={k:x.copy() for k in r.ARMS},pooled={k:x.copy() for k in r.ARMS},
            available=np.array([True,False]),local_available=np.array([False,False]))
    def test_valid(self):r.validate_forecasts(self.forecast(),2)
    def test_cold_local_must_equal_pooled(self):
        f=self.forecast();f['raw']['open'][0]=.2
        with self.assertRaises(ValueError):r.validate_forecasts(f,2)
    def test_invalid_availability_or_output(self):
        for key in ('available','local_available'):
            f=self.forecast();f[key]=f[key].astype(int)
            with self.assertRaises(ValueError):r.validate_forecasts(f,2)
        f=self.forecast();f['raw']['open'][1]=1
        with self.assertRaises(ValueError):r.validate_forecasts(f,2)


class ReviewBinding(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'out';self.out.mkdir()
        self.runner=object.__new__(r.Runner);self.runner.root=self.root;self.runner.out=self.out
        self.code=self.root/'review.py';self.code.write_text('# synthetic reviewer\n')
        self.runner.pins={str(self.code):r.sha(self.code)}
        self.runner.plan={'independent_prepare_source_hashes':dict(self.runner.pins)}
        self.cache={}
        for name in (*r.CACHE_FILES,'PLAN.json','CACHE.json'):
            (self.out/name).write_text('{}')
            self.cache[str(self.out/name)]=r.sha(self.out/name)
        self.review=dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
            source_hashes=dict(self.runner.pins),artifact_hashes=dict(self.cache),
            plan_sha256=self.cache[str(self.out/'PLAN.json')],cache_sha256=self.cache[str(self.out/'CACHE.json')],
            orders_sha256=self.cache[str(self.out/'PRETRAIN_ORDERS.json')],**{k:True for k in r.CHECKS})
    def check(self):
        path=self.out/'INDEPENDENT_PREPARE.json';path.write_text(json.dumps(self.review))
        with patch.object(self.runner,'cache',return_value={k:v for k,v in self.cache.items() if Path(k).name in r.CACHE_FILES}):self.runner.require_prepare_review()
    def test_complete_bound_receipt_passes(self):self.check()
    def test_passed_bool_without_artifacts_fails(self):
        self.review['artifact_hashes']={}
        with self.assertRaises(ValueError):self.check()
    def test_wrong_source_fails(self):
        self.code.write_text('# changed')
        with self.assertRaises(ValueError):self.check()
    def test_missing_check_fails(self):
        self.review[r.CHECKS[-1]]=False
        with self.assertRaises(ValueError):self.check()
    def test_wrong_plan_digest_fails(self):
        self.review['plan_sha256']='f'*64
        with self.assertRaises(ValueError):self.check()
    def test_unbound_extra_artifact_fails(self):
        extra=self.root/'extra';extra.write_text('extra');self.review['artifact_hashes'][str(extra)]=r.sha(extra)
        with self.assertRaises(ValueError):self.check()


class Registration(unittest.TestCase):
    def test_predecessor_exact(self):
        state={'completed':[{'trials':0} for _ in range(141)]+[{'trials':926,'directory':'monthly_formation142'}]}
        reg.require_predecessor(state)
        for bad in (141,143):
            wrong=deepcopy(state);wrong['completed']=wrong['completed'][:bad] if bad<142 else wrong['completed']+[{'trials':0}]
            with self.assertRaises(ValueError):reg.require_predecessor(wrong)
    def test_proposal_binding(self):
        self.assertEqual(r.sha(ROOT/'work/monthly_paired_target143_proposal.json'),r.PROPOSAL_SHA)
        self.assertEqual(reg.PROPOSAL_SHA,r.PROPOSAL_SHA)
    def test_test_receipt_bool_not_sufficient(self):
        with self.assertRaises(ValueError):reg.validate_tests({'passed':True,'suites':[],'test_count':0},lambda *a:Path(a[0]))
    def test_train_phase_requires_review_before_loading(self):
        runner=object.__new__(r.Runner);runner.out=Path('/unlikely/missing/path')
        with patch.object(runner,'refresh'),patch.object(runner,'require_prepare_review',side_effect=ValueError('gate')), \
             patch.object(runner,'loaded') as load:
            with self.assertRaises(ValueError):runner.select_first88()
            load.assert_not_called()


class QueueIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'outputs'/reg.EXPERIMENT;self.out.mkdir(parents=True)
        self.statepath=self.root/reg.STATE_RELATIVE;self.statepath.parent.mkdir(parents=True)
        self.state=dict(completed=[dict(trials=0) for _ in range(141)]+[dict(trials=926,directory='monthly_formation142')],registered_queue=[])
        self.statepath.write_text(json.dumps(self.state))
        dep=self.root/'input';dep.write_text('immutable')
        self.planpath=self.out/'PLAN.json'
        self.plan=dict(round=143,status='REGISTERED',arms=list(r.NAMES),primary_challenger=r.NAMES[2],output_directory=str(self.out),
            registration_precondition=dict(path=reg.STATE_RELATIVE,sha256=r.sha(self.statepath)),hashes={'input':r.sha(dep)},registered_utc='synthetic')
        self.planpath.write_text(json.dumps(self.plan));self.digest=r.sha(self.planpath)
    def test_enqueue_and_exact_resume(self):
        self.assertTrue(reg.enqueue_registered_plan(self.root,self.planpath,self.digest))
        self.assertFalse(reg.enqueue_registered_plan(self.root,self.planpath,self.digest))
        state=json.loads(self.statepath.read_text());self.assertEqual(state['registered_queue'][0]['round'],143)
        self.assertEqual(state['current'],reg.EXPERIMENT)
    def test_changed_input_prevents_queue(self):
        (self.root/'input').write_text('changed')
        with self.assertRaises(ValueError):reg.enqueue_registered_plan(self.root,self.planpath,self.digest)
        self.assertEqual(json.loads(self.statepath.read_text())['registered_queue'],[])
    def test_changed_state_prevents_overwrite(self):
        self.state['note']='concurrent mutation';self.statepath.write_text(json.dumps(self.state))
        with self.assertRaises(ValueError):reg.enqueue_registered_plan(self.root,self.planpath,self.digest)
        self.assertEqual(json.loads(self.statepath.read_text())['note'],'concurrent mutation')
    def test_conflicting_queue_prevents_overwrite(self):
        self.state['registered_queue']=[{'directory':'unrelated'}];self.statepath.write_text(json.dumps(self.state))
        with self.assertRaises(ValueError):reg.enqueue_registered_plan(self.root,self.planpath,self.digest)
        self.assertEqual(json.loads(self.statepath.read_text())['registered_queue'],[{'directory':'unrelated'}])
    def test_changed_plan_prevents_queue(self):
        self.plan['round']=144;self.planpath.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError):reg.enqueue_registered_plan(self.root,self.planpath,self.digest)
    def test_wrong_directory_prevents_queue(self):
        wrong=self.root/'PLAN.json';wrong.write_text(self.planpath.read_text())
        with self.assertRaises(ValueError):reg.enqueue_registered_plan(self.root,wrong,self.digest)
    def test_synthetic_validation_requires_actual_log_match(self):
        suites=[]
        for name,source in reg.SUITE_SOURCES.items():
            file=self.root/source;file.write_text('# artificial source')
            log=self.root/(name+'.log');log.write_text('Ran 1 test in 0.001s\n\nOK\n')
            suites.append(dict(name=name,passed=True,count=1,source=dict(path=str(file),sha256=r.sha(file)),log=dict(path=str(log),sha256=r.sha(log))))
        def add(path,digest):
            self.assertEqual(r.sha(path),digest);return Path(path)
        receipt=dict(passed=True,suites=suites,test_count=len(suites))
        self.assertEqual(len(reg.validate_tests(receipt,add)),len(suites))
        receipt['test_count']+=1
        with self.assertRaises(ValueError):reg.validate_tests(receipt,add)
        receipt['test_count']-=1
        suites[0]['count']=2
        with self.assertRaises(ValueError):reg.validate_tests(receipt,add)

class PhaseIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.runner=object.__new__(r.Runner);self.runner.out=Path(self.tmp.name).resolve()
    def test_partial_prepare_must_not_restart(self):
        (self.runner.out/'LABELS.npz').write_text('partial')
        self.runner.backend_factory=lambda *args:self.fail('backend should not load')
        with patch.object(self.runner,'refresh'):
            with self.assertRaises(ValueError):self.runner.prepare()
    def test_partial_training_must_not_restart(self):
        (self.runner.out/'TRAIN_OPEN_TRIALS.json').write_text('partial')
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_prepare_review') as review:
            with self.assertRaises(ValueError):self.runner.select_first88()
            review.assert_not_called()
    def test_partial_diagnostics_must_not_restart(self):
        (self.runner.out/'DIAGNOSE_STARTED.json').write_text('partial')
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_prepare_review') as review:
            with self.assertRaises(ValueError):self.runner.diagnose_later34()
            review.assert_not_called()
    def test_cache_requires_exact_file_schema(self):
        (self.runner.out/'CACHE.json').write_text('{}')
        with self.assertRaises(ValueError):self.runner.cache()

if __name__ == '__main__':unittest.main()
