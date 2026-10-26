"""Synthetic lifecycle tests; never load real strategy arrays or market values."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run163_host as h

def plan():
    return dict(round=163,status='REGISTERED',arms=list(h.r.NAMES),runtime=h.runtime(),
                measurement=h.r.measurement_contract(),selection_contract=h.r.selection_contract(),
                output_directory=str(h.OUT),no_observer_writes=True,inputs={'protected_manifest':{'sha256':h.PROTECTED_SHA}},
                actual_features_before_registration=0,actual_orders_before_registration=0,
                actual_returns_before_registration=0)

class PlanTests(unittest.TestCase):
    def test_fixed_contract(self): h.validate_plan(plan())
    def test_mutated_contracts_fail(self):
        for key,value in [('round',162),('status','DRAFT'),('arms',[]),('runtime',{}),
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
        self.assertEqual(len(paths),10)
        self.assertTrue(all(p.is_relative_to(h.ROOT) for p in paths.values()))
        self.assertFalse(any('stock-observer-runtime' in str(p) or 'CloudStorage' in str(p) for p in paths.values()))

class SourceTests(unittest.TestCase):
    def fixture(self):
        p={k:Path('/private/tmp/run163-'+k) for k in ('source_review','proposal','training_meta','calendar','prices','meta','protected_manifest')}
        hashes={str(value):key+'-hash' for key,value in p.items()}
        hashes[str(p['protected_manifest'])]=h.PROTECTED_SHA
        hashes[str(p['meta'])]='3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'
        pins={str(p[k]):hashes[str(p[k])] for k in ('proposal','training_meta','calendar','prices','meta')}
        proposal=dict(round=163,arms=list(h.r.NAMES),actual_features_computed=0,actual_orders_computed=0,actual_returns_computed=0)
        review=dict(passed=True,strict_pit_vintage=False,strict_pit_membership=False,strategy_admitted=False,
                    conditional_research_sources_ready=True,proposal_sha256=hashes[str(p['proposal'])],
                    actual_features_computed=0,actual_orders_computed=0,actual_returns_computed=0,
                    input_hashes=pins,source_hashes=pins,summary=dict(all_overlap_sets_equal=True,
                    noninitial_dates_verified_monthend=True,event_months=122,historical_names_absent_from_price_panel=[],market_proxy_present=True))
        return proposal,review,p,hashes
    def run_gate(self,proposal,review,p,hashes):
        with patch.object(h,'read',side_effect=lambda path: {p['proposal']:proposal,p['source_review']:review}[path]),\
             patch.object(h,'sha',side_effect=lambda path:hashes[str(path)]),patch.object(h,'verify') as verify:
            result=h.source_gate(p)
            verify.assert_called_once()
            return result
    def test_valid_conditional_design_only(self):
        proposal,review,p,hashes=self.fixture()
        self.assertEqual(self.run_gate(proposal,review,p,hashes),review)
    def test_review_flags_and_binding_required(self):
        for k,v in [('passed',False),('strict_pit_vintage',True),('strict_pit_membership',True),('strategy_admitted',True),
                    ('conditional_research_sources_ready',False),('proposal_sha256','other')]:
            proposal,review,p,hashes=self.fixture();review[k]=v
            with self.subTest(key=k),self.assertRaises(ValueError):self.run_gate(proposal,review,p,hashes)
    def test_no_prior_numerical_activity_in_review_or_proposal(self):
        for target in ('proposal','review'):
            for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed'):
                proposal,review,p,hashes=self.fixture()
                (proposal if target=='proposal' else review)[key]=True
                with self.subTest(target=target,key=key),self.assertRaises(ValueError):self.run_gate(proposal,review,p,hashes)
    def test_incomplete_source_pins_and_metadata_evidence_rejected(self):
        for key in ('meta','training_meta','prices','calendar','proposal'):
            proposal,review,p,hashes=self.fixture();review['input_hashes'].pop(str(p[key]))
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_gate(proposal,review,p,hashes)
        for key,value in [('all_overlap_sets_equal',False),('noninitial_dates_verified_monthend',False),
                          ('event_months',121),('historical_names_absent_from_price_panel',['missing']),('market_proxy_present',False)]:
            proposal,review,p,hashes=self.fixture();review['summary'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_gate(proposal,review,p,hashes)
    def test_original_axis_and_protected_pins_required(self):
        for key in ('meta','protected_manifest'):
            proposal,review,p,hashes=self.fixture();hashes[str(p[key])]='mutated'
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_gate(proposal,review,p,hashes)

class ReceiptTests(unittest.TestCase):
    def fixture(self):
        implementation=Path('/private/tmp/run163-implementation')
        validation=Path('/private/tmp/run163-validation')
        pins={str(implementation):'implementation-sha'}
        review=dict(passed=True,host_sha256='host-sha',registration_tests_sha256='validation-sha',
                    implementation_hashes=pins.copy())
        hashes={h.SELF:'host-sha',validation:'validation-sha',implementation:'implementation-sha'}
        return review,validation,implementation,pins,hashes
    def verify_review(self,review,validation,implementation,pins,hashes):
        with patch.object(h,'validation_paths',return_value=[implementation]),\
             patch.object(h,'sha',side_effect=lambda p:hashes[p]),\
             patch.object(h,'read',return_value={'implementation_hashes':pins}),patch.object(h,'verify'):
            h.validate_registration_review(review,validation)
    def test_current_review_binds_test_receipt_and_all_implementations(self):
        self.verify_review(*self.fixture())
    def test_stale_or_incomplete_review_binding_is_rejected(self):
        for change in ('receipt','host','implementation','missing','different_test'):
            review,validation,implementation,pins,hashes=self.fixture()
            if change=='receipt':review['registration_tests_sha256']='old'
            elif change=='host':review['host_sha256']='old'
            elif change=='implementation':review['implementation_hashes'][str(implementation)]='old'
            elif change=='missing':review['implementation_hashes']={}
            else:pins[str(implementation)]='old'
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.verify_review(review,validation,implementation,pins,hashes)
    def test_full_preparation_counts_required(self):
        h.validate_prepare_counts(dict(h.PREPARE_COUNTS))
        for key,value in h.PREPARE_COUNTS.items():
            for changed in (None,True,value-1):
                review=dict(h.PREPARE_COUNTS);review[key]=changed
                with self.subTest(key=key,value=changed),self.assertRaises(ValueError):h.validate_prepare_counts(review)

class CalendarTests(unittest.TestCase):
    def fixture(self):
        return dict(provenance=dict(source_hashes={'/private/tmp/manifest':'mh','/private/tmp/evidence':'eh'},
                    provenance=dict(manifest='/private/tmp/manifest',manifest_sha256='mh',
                    closures=[dict(source_file='/private/tmp/evidence',source_sha256='eh')])))
    def test_calendar_manifest_and_evidence_are_bound(self):
        calendar=self.fixture()
        with patch.object(h,'read',return_value=calendar),patch.object(h,'verify') as verify:
            self.assertEqual(h.calendar_pins({'calendar':'unused'}),calendar['provenance']['source_hashes'])
            verify.assert_called_once()
    def test_incomplete_or_wrong_calendar_closure_is_rejected(self):
        for change in ('missing','changed'):
            calendar=self.fixture()
            if change=='missing':del calendar['provenance']['source_hashes']['/private/tmp/evidence']
            else:calendar['provenance']['provenance']['manifest_sha256']='wrong'
            with patch.object(h,'read',return_value=calendar),patch.object(h,'verify'),self.assertRaises(ValueError):
                h.calendar_pins({'calendar':'unused'})

class PublicationTests(unittest.TestCase):
    def test_best_metadata_requires_training_admission_and_full_gain(self):
        old={'cagr':.67,'spec':{'id':4}}
        results={'selection':{'primary':'run_positive','admitted':False},'arms':{'run_positive':{
            'same_close':{'metrics':{'cagr':.71,'sharpe':1.8,'max_drawdown':-.3}},
            'next_open_cost':{'metrics':{'cagr':.68}}}}}
        self.assertEqual(h.best_after({'best_training_selected_candidate':old},results),old)
        results['selection']['admitted']=True
        new=h.best_after({'best_training_selected_candidate':old},results)
        self.assertEqual(new['cagr'],.71)
        self.assertFalse(new['observer_promotion'])
        before={'completed':[{'trials':6} for _ in range(161)]+[{'trials':35}],
                'best_training_selected_candidate':old}
        after=deepcopy(before);after['completed'].append({'trials':3});after['best_training_selected_candidate']=new
        h.validate_completed_state(after,before,new)
        with self.assertRaises(ValueError):h.validate_completed_state(after,before)
        results['arms']['run_positive']['same_close']['metrics']['cagr']=.66
        self.assertEqual(h.best_after(before,results),old)
    def test_original_history_and_incumbent_must_stay_identical(self):
        before={'completed':[{'trials':6} for _ in range(161)]+[{'trials':35}],
                'best_training_selected_candidate':{'id':4}}
        h.validate_prior_state(deepcopy(before),before)
        for key in ('completed','best_training_selected_candidate'):
            altered=deepcopy(before)
            if key=='completed':altered[key][0]['extra']='changed'
            else:altered[key]['id']=5
            with self.subTest(key=key),self.assertRaises(ValueError):h.validate_prior_state(altered,before)
    def test_completed_state_preserves_all_prior_history_and_best(self):
        before={'completed':[{'trials':6} for _ in range(161)]+[{'trials':35}],
                'best_training_selected_candidate':{'id':4}}
        after=deepcopy(before);after['completed'].append({'trials':3})
        h.validate_completed_state(after,before)
        for change in ('history','best','trials'):
            bad=deepcopy(after)
            if change=='history':bad['completed'][1]['changed']=True
            elif change=='best':bad['best_training_selected_candidate']['id']=5
            else:bad['completed'][-1]['trials']=4
            with self.subTest(change=change),self.assertRaises(ValueError):h.validate_completed_state(bad,before)
    def test_exact_prior_backup_and_atomic_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json';original=b'{"count": 162}\n';path.write_bytes(original)
            old=h.sha(path)
            with patch.object(h,'STATE',path):h.state_commit(old,{'count':163})
            self.assertEqual(json.loads(path.read_text()),{'count':163})
            self.assertEqual((path.parent/('PRE163_'+old+'.json')).read_bytes(),original)
            self.assertEqual(sorted(x.name for x in path.parent.iterdir()),sorted(['state.json','PRE163_'+old+'.json']))
    def test_stale_state_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json';path.write_bytes(b'original')
            with patch.object(h,'STATE',path),self.assertRaises(ValueError):h.state_commit('bad',{})
            self.assertEqual(path.read_bytes(),b'original')
            self.assertEqual(len(list(path.parent.iterdir())),1)
    def test_conflicting_backup_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json';path.write_bytes(b'original');old=h.sha(path)
            (path.parent/('PRE163_'+old+'.json')).write_bytes(b'wrong')
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
