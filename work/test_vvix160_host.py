"""Synthetic lifecycle tests; never load real strategy arrays or market values."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import vvix160_host as h

def plan():
    return dict(round=160,status='REGISTERED',arms=list(h.r.NAMES),runtime=h.runtime(),
                measurement=h.r.measurement_contract(),selection_contract=h.r.selection_contract(),
                output_directory=str(h.OUT),no_observer_writes=True,inputs={'protected_manifest':{'sha256':h.PROTECTED_SHA}},
                actual_features_before_registration=0,actual_orders_before_registration=0,
                actual_returns_before_registration=0)

class PlanTests(unittest.TestCase):
    def test_fixed_contract(self): h.validate_plan(plan())
    def test_mutated_contracts_fail(self):
        for key,value in [('round',159),('status','DRAFT'),('arms',[]),('runtime',{}),
                          ('measurement',{}),('selection_contract',{}),('output_directory','/tmp'),
                          ('no_observer_writes',False),('inputs',{})]:
            p=plan();p[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): h.validate_plan(p)
    def test_prior_numerical_activity_or_bool_is_rejected(self):
        for key in ('actual_features_before_registration','actual_orders_before_registration','actual_returns_before_registration'):
            for value in (1,True,False,None):
                p=plan();p[key]=value
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):h.validate_plan(p)
    def test_named_paths_do_not_reach_observer_or_cloud(self):
        paths=h.paths()
        self.assertEqual(len(paths),14)
        self.assertTrue(all(p.is_relative_to(h.ROOT) for p in paths.values()))
        self.assertFalse(any('stock-observer-runtime' in str(p) or 'CloudStorage' in str(p) for p in paths.values()))

class SourceTests(unittest.TestCase):
    def fixture(self):
        val=dict(passed=True,signals_checked=122,noninitial_signals=121,
                 first_required_date='2015-09-01',last_required_date='2026-08-28',
                 distinct_required_sessions=2764,input_hashes={'fake':'hash'})
        review=dict(passed=True,strict_pit_vintage=False,strategy_admitted=False,
                    conditional_research_sources_ready=True,validation_sha256='validation',
                    summary={k:val[k] for k in ('signals_checked','noninitial_signals',
                    'first_required_date','last_required_date','distinct_required_sessions')},input_hashes={'other':'hash'})
        p={k:Path(k) for k in ('source_validation','source_review','vol_feature','vol_meta','meta','protected_manifest')}
        hashes={'protected_manifest':h.PROTECTED_SHA,'source_validation':'validation','vol_feature':'1d44b505ddcfd2979e30ba4e73ff528cd1d72efc5ca7e745c3f1c51401fbf5b5',
                'vol_meta':'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
                'meta':'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'}
        return val,review,p,hashes
    def run_gate(self,val,review,p,hashes):
        with patch.object(h,'read',side_effect=lambda p: {'source_validation':val,'source_review':review}[str(p)]),\
             patch.object(h,'sha',side_effect=lambda p:hashes[str(p)]),patch.object(h,'verify') as verify:
            result=h.source_gate(p)
            self.assertEqual(verify.call_count,2)
            return result
    def test_valid_conditional_coverage_only(self):
        self.run_gate(*self.fixture())
    def test_incomplete_or_wrong_fixed_count(self):
        for k,v in [('passed',False),('signals_checked',121),('noninitial_signals',120)]:
            val,review,p,hashes=self.fixture();val[k]=v
            with self.subTest(key=k),self.assertRaises(ValueError):self.run_gate(val,review,p,hashes)
    def test_review_flags_and_binding_required(self):
        for k,v in [('passed',False),('strict_pit_vintage',True),('strategy_admitted',True),
                    ('conditional_research_sources_ready',False),('validation_sha256','other')]:
            val,review,p,hashes=self.fixture();review[k]=v
            with self.subTest(key=k),self.assertRaises(ValueError):self.run_gate(val,review,p,hashes)
    def test_mismatched_source_totals(self):
        val,review,p,hashes=self.fixture();review['summary']['distinct_required_sessions']=2763
        with self.assertRaises(ValueError):self.run_gate(val,review,p,hashes)
    def test_volatility_axis_and_cache_pins_required(self):
        for k in ('vol_feature','vol_meta','meta','protected_manifest'):
            val,review,p,hashes=self.fixture();hashes[k]='mutated'
            with self.subTest(key=k),self.assertRaises(ValueError):self.run_gate(val,review,p,hashes)

class PublicationTests(unittest.TestCase):
    def test_original_history_and_incumbent_must_stay_identical(self):
        before={'completed':[{'trials':6} for _ in range(158)]+[{'trials':44}],
                'best_training_selected_candidate':{'id':4}}
        h.validate_prior_state(deepcopy(before),before)
        for key in ('completed','best_training_selected_candidate'):
            altered=deepcopy(before)
            if key=='completed':altered[key][0]['extra']='changed'
            else:altered[key]['id']=5
            with self.subTest(key=key),self.assertRaises(ValueError):h.validate_prior_state(altered,before)
    def test_exact_prior_backup_and_atomic_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json';original=b'{"count": 159}\n';path.write_bytes(original)
            old=h.sha(path)
            with patch.object(h,'STATE',path):h.state_commit(old,{'count':160})
            self.assertEqual(json.loads(path.read_text()),{'count':160})
            self.assertEqual((path.parent/('PRE160_'+old+'.json')).read_bytes(),original)
            self.assertEqual(sorted(x.name for x in path.parent.iterdir()),sorted(['state.json','PRE160_'+old+'.json']))
    def test_stale_state_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json';path.write_bytes(b'original')
            with patch.object(h,'STATE',path),self.assertRaises(ValueError):h.state_commit('bad',{})
            self.assertEqual(path.read_bytes(),b'original')
            self.assertEqual(len(list(path.parent.iterdir())),1)
    def test_conflicting_backup_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json';path.write_bytes(b'original');old=h.sha(path)
            (path.parent/('PRE160_'+old+'.json')).write_bytes(b'wrong')
            with patch.object(h,'STATE',path),self.assertRaises(ValueError):h.state_commit(old,{})
            self.assertEqual(path.read_bytes(),b'original')
    def test_artifact_write_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'artifact.json';h.write(p,{'x':1})
            with self.assertRaises(FileExistsError):h.write(p,{'x':2})
            self.assertEqual(h.read(p),{'x':1})
    def test_prepare_and_select_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'PREPARE_STARTED.json').write_text('{}');(out/'SELECT_STARTED.json').write_text('{}')
            host=object.__new__(h.Host)
            with patch.object(h,'OUT',out):
                with self.assertRaises(ValueError):host.prepare()
                with self.assertRaises(ValueError):host.select()
    def test_changed_dependency_is_rejected(self):
        host=object.__new__(h.Host);host.pins={}
        with tempfile.TemporaryDirectory() as temp:
            p=(Path(temp)/'source').resolve();p.write_text('one')
            with self.assertRaises(ValueError):host.bound_path(p)
            host.pins[str(p)]=h.sha(p);self.assertEqual(host.bound_path(p),p)
            p.write_text('two')
            with self.assertRaises(ValueError):host.bound_path(p)

if __name__=='__main__':unittest.main(verbosity=2)
