"""Fabricated state/phase/source-integrity tests; no market data or actual grid."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import monthly_revenue_persistence156_lifecycle as c


def state():
    return dict(completed=[dict(directory='old'+str(i), trials=6) for i in range(154)]+[
        dict(directory='monthly_endpoint155', trials=54, completion=dict(sha256=c.PREVIOUS_DONE_SHA))],
        registered_queue=[], best_training_selected_candidate=dict(cagr=.67), unrelated={'preserve': [1, 2]})


def rows():
    return [dict(name=n, training=dict(cagr=.5, sharpe=1.5, max_drawdown=-.3),
                 early=dict(cagr=.3), recent=dict(cagr=.6)) for n in c.NAMES]


class SelectionTests(unittest.TestCase):
    def test_equal_controls_keep_incumbent(self):
        self.assertFalse(c.selection(rows(), rows())['admitted'])

    def test_primary_must_beat_every_close_control(self):
        close, opening = rows(), rows()
        close[3]['training']['cagr'] = opening[3]['training']['cagr'] = .6
        self.assertTrue(c.selection(close, opening)['admitted'])
        for i in range(3):
            candidate = deepcopy(close); candidate[i]['training']['cagr'] = .6
            self.assertFalse(c.selection(candidate, opening)['admitted'])

    def test_open_incumbent_tie_allowed_but_control_tie_not(self):
        close, opening = rows(), rows(); close[3]['training']['cagr'] = .6
        opening[1]['training']['cagr'] = opening[2]['training']['cagr'] = .4
        self.assertTrue(c.selection(close, opening)['admitted'])
        opening[2]['training']['cagr'] = .5
        self.assertFalse(c.selection(close, opening)['admitted'])

    def test_subperiod_and_risk_guards(self):
        for section, key, value in [('training', 'sharpe', 1.44), ('training', 'max_drawdown', -.331),
                                    ('early', 'cagr', .274), ('recent', 'cagr', .574)]:
            close, opening = rows(), rows(); close[3]['training']['cagr'] = opening[3]['training']['cagr'] = .6
            close[3][section][key] = value
            self.assertFalse(c.selection(close, opening)['admitted'])

    def test_nonfinite_or_boolean_rejected(self):
        for bad in [float('nan'), float('inf'), True, None]:
            close = rows(); close[3]['training']['cagr'] = bad
            self.assertFalse(c.selection(close, rows())['admitted'])

    def test_changed_arm_order_fails(self):
        with self.assertRaises(ValueError):
            c.selection(list(reversed(rows())), rows())


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic-revenue156-', dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.out = self.root/'outputs'/c.STUDY; self.out.mkdir(parents=True)
        self.put(c.STATE, state())

    def put(self, name, value):
        p = Path(name); p = p if p.is_absolute() else self.root/p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value if isinstance(value, bytes) else c.old.encoded(value))
        return p

    def plan(self):
        source = self.put('work/pin.py', b'synthetic source')
        return dict(round=156, status='REGISTERED', arms=list(c.NAMES), primary_challenger=c.NAMES[3], dose=.01,
            output_directory=str(self.out), registered_utc='2020-01-01T00:00:00+00:00',
            registration_precondition=dict(path=c.STATE, sha256=c.sha(self.root/c.STATE)),
            hashes={str(source): c.sha(source)})


class StateTests(Files):
    def test_exact_predecessor_required(self):
        c.predecessor(state())
        for action in [lambda x: x['completed'].pop(), lambda x: x['completed'][0].update(trials=True),
                       lambda x: x['completed'][0].update(trials=7),
                       lambda x: x['completed'][-1]['completion'].update(sha256='a'*64)]:
            doc = state(); action(doc)
            with self.assertRaises(ValueError): c.predecessor(doc)

    def test_state_update_preserves_exact_original(self):
        p = self.root/c.STATE; before = p.read_bytes(); h = c.sha(p)
        changed = state(); changed['new'] = True
        c.update_state(p, h, changed)
        self.assertEqual((p.parent/('PRE156_'+h+'.json')).read_bytes(), before)
        self.assertEqual(c.old.decode(p.read_bytes()), changed)

    def test_stale_state_does_not_overwrite(self):
        p = self.root/c.STATE; h = c.sha(p); self.put(p, {'concurrent': True}); before = p.read_bytes()
        with self.assertRaises(ValueError): c.update_state(p, h, state())
        self.assertEqual(p.read_bytes(), before)

    def test_conflicting_backup_fails(self):
        p = self.root/c.STATE; h = c.sha(p); self.put(p.parent/('PRE156_'+h+'.json'), b'wrong')
        with self.assertRaises(ValueError): c.update_state(p, h, state())

    def test_partial_serialization_publishes_nothing(self):
        p = self.root/'bad.json'
        with self.assertRaises(ValueError): c.write(p, {'bad': float('nan')})
        self.assertFalse(p.exists())
        self.assertEqual(list(self.root.glob('.revenue156-*')), [])

    def test_exclusive_publication_preserves_prior(self):
        p = self.put('existing', {'original': True}); before = p.read_bytes()
        with self.assertRaises(FileExistsError): c.write(p, {'new': True})
        self.assertEqual(p.read_bytes(), before)

    def test_enqueue_idempotent(self):
        p = self.put(self.out/'PLAN.json', self.plan()); h = c.sha(p)
        self.assertFalse(c.enqueue(self.root, h)['already_enqueued'])
        before = (self.root/c.STATE).read_bytes()
        self.assertTrue(c.enqueue(self.root, h)['already_enqueued'])
        self.assertEqual((self.root/c.STATE).read_bytes(), before)

    def test_enqueue_changed_pin_fails(self):
        p = self.put(self.out/'PLAN.json', self.plan()); h = c.sha(p)
        self.put('work/pin.py', b'changed')
        with self.assertRaises(ValueError): c.enqueue(self.root, h)

    def test_prepare_artifacts_block_registration(self):
        p = self.put(self.out/'DATA.npz', b'already prepared')
        with patch.object(c, 'runner', return_value=SimpleNamespace(CACHE_FILES=['DATA.npz'])):
            with self.assertRaisesRegex(ValueError, 'precede actual preparation'):
                c.build_plan(self.root, 'unused', 'a'*64)
        self.assertEqual(p.read_bytes(), b'already prepared')

    def test_conflicting_queue_fails_without_state_change(self):
        p = self.put(self.out/'PLAN.json', self.plan()); h = c.sha(p)
        changed = state(); changed['registered_queue'] = [{'other': True}]; self.put(c.STATE, changed)
        before = (self.root/c.STATE).read_bytes()
        with self.assertRaises(ValueError): c.enqueue(self.root, h)
        self.assertEqual((self.root/c.STATE).read_bytes(), before)


class SourceGateTests(Files):
    def source_fixture(self):
        source = self.put('work/source', b'source bytes')
        self.put(c.SOURCE_PLAN, {'synthetic': 'plan'}); self.put(c.SOURCE_CALENDAR, {'synthetic': 'calendar'})
        values = {'ROWS.json': [], 'REPORTS.json': {}, 'EDGES.json': {}, 'SUMMARY.json': {},
                  'COVERAGE_GATE.json': dict(source_coverage_floor_passed=True, rows_verified=5978,
                                            source_reviews_verified_by_preparation_driver=True)}
        outputs, hashes = {}, {str(source): c.sha(source)}
        for name, value in values.items():
            p = self.put(c.SOURCE_ROOT+'/'+name, value)
            outputs[name] = dict(sha256=c.sha(p), bytes=p.stat().st_size); hashes[str(p)] = c.sha(p)
        manifest = self.put(c.SOURCE_MANIFEST, dict(input_hashes={str(source): c.sha(source)}, outputs=outputs,
                                                  source_only=True, strategy_registration=False))
        hashes[str(manifest)] = c.sha(manifest)
        review = self.put(c.SOURCE_REVIEW, dict(passed=True, actual_grid_verified=True, input_hashes=hashes))
        return dict(path=c.SOURCE_REVIEW, sha256=c.sha(review))

    def run_gate(self, ref):
        with patch.object(c, 'SOURCE_PLAN_SHA', c.sha(self.root/c.SOURCE_PLAN)), \
             patch.object(c, 'SOURCE_CALENDAR_SHA', c.sha(self.root/c.SOURCE_CALENDAR)):
            return c.source_gate(c.old.Reader(self.root), ref)

    def test_complete_independent_source_closure(self):
        self.run_gate(self.source_fixture())

    def test_review_boolean_without_actual_grid_evidence_fails(self):
        ref = self.source_fixture(); p = self.root/c.SOURCE_REVIEW
        doc = c.old.decode(p.read_bytes()); doc['actual_grid_verified'] = False; self.put(p, doc)
        ref['sha256'] = c.sha(p)
        with self.assertRaisesRegex(ValueError, 'actual source-grid review'):
            self.run_gate(ref)

    def test_changed_source_output_fails(self):
        ref = self.source_fixture(); self.put(c.SOURCE_ROOT+'/ROWS.json', [{'changed': True}])
        with self.assertRaises(ValueError): self.run_gate(ref)

    def test_omitted_review_input_fails(self):
        ref = self.source_fixture(); p = self.root/c.SOURCE_REVIEW
        doc = c.old.decode(p.read_bytes()); del doc['input_hashes'][str(self.root/'work/source')]
        self.put(p, doc); ref['sha256'] = c.sha(p)
        with self.assertRaisesRegex(ValueError, 'omitted exact input/output'):
            self.run_gate(ref)


class RegistrationTests(Files):
    def fixture(self):
        review_ref = SourceGateTests.source_fixture(self)
        core = {}
        for key in ('meta', 'policy', 'original_orders', 'prices', 'protected_manifest'):
            path = self.put('core/'+key, b'synthetic only')
            core[key] = dict(path=str(path), sha256=c.sha(path))
        source = self.put('work/core.py', b'synthetic core')
        raw = {str(self.put('raw/'+str(i), b'original raw bytes')): None for i in range(64)}
        raw = {p: c.sha(p) for p in raw}
        runtime = dict(python=sys.version, numpy='synthetic', pandas='synthetic')
        self.put(c.PREVIOUS_DONE, {'synthetic': 'completion'})
        self.put(c.PREVIOUS_PLAN, dict(hashes={str(source): c.sha(source)}, inputs=core,
                                       raw_source_hashes=raw, runtime=runtime))
        self.put(c.DESIGN, {'synthetic': 'design'})
        self.put(c.SESSIONS, {'synthetic': 'sessions'})
        source_inputs = {key: dict(path=path, sha256=c.sha(self.root/path)) for key, path in (
            ('source_manifest', c.SOURCE_MANIFEST), ('source_grid_review', c.SOURCE_REVIEW),
            ('source_plan', c.SOURCE_PLAN), ('source_calendar', c.SOURCE_CALENDAR))}
        self.put(c.PROPOSAL, dict(design={'sha256': c.sha(self.root/c.DESIGN)}, source_inputs=source_inputs))
        implementation = {}
        for name in ('after155_revenue_pilot_factor.py', 'monthly_revenue_persistence156_runner.py',
                     'monthly_revenue_persistence156_lifecycle.py', 'prepare_revenue_persistence156_proposal.py'):
            p = self.put('work/'+name, b'synthetic implementation')
            implementation[str(p)] = c.sha(p)
        factor_test = self.put('work/test_after155_revenue_pilot_factor.py', b'factor test')
        factor_hashes = {str(self.root/'work/after155_revenue_pilot_factor.py'):
                         c.sha(self.root/'work/after155_revenue_pilot_factor.py'),
                         str(factor_test): c.sha(factor_test), str(self.root/c.DESIGN): c.sha(self.root/c.DESIGN)}
        activity = dict.fromkeys(('actual_factors_computed', 'actual_financial_values_read', 'actual_orders_computed',
                                 'actual_returns_computed', 'frozen_files_changed', 'observer_changes', 'strategy_registrations'), 0)
        self.put(c.FACTOR_REVIEW, dict(passed=True, synthetic_only=True, status='PURE_REVENUE_FACTOR_REVIEW_PASSED',
                                     activity_counts=activity, input_hashes=factor_hashes))
        suites = []
        for name, filename in c.SUITES.items():
            p = self.root/'work'/filename
            if not p.exists(): self.put(p, b'synthetic test')
            log = self.put('logs/'+name, b'Ran 1 test in 0.001s\n\nOK\n')
            suites.append(dict(name=name, count=1, passed=True, source=dict(path=str(p), sha256=c.sha(p)),
                               log=dict(path=str(log), sha256=c.sha(log))))
        tests = self.put(self.out/'TESTS.json', dict(passed=True, synthetic_only=True, test_count=6, suites=suites))
        proposal_sha = c.sha(self.root/c.PROPOSAL)
        for name, path in c.PREREVIEWS.items():
            reviewer = self.put('work/'+name+'_independent.py', b'independent synthetic reviewer')
            self.put(path, dict(passed=True, round=156, stage='preregistration_'+name+'_review', synthetic_only=True,
                proposal_sha256=proposal_sha, actual_factors_calculated=0, actual_returns_calculated=0,
                actual_orders_calculated=0, actual_model_fits=0, source_hashes={str(reviewer): c.sha(reviewer)},
                implementation_hashes=implementation,
                input_hashes={str(self.root/p): c.sha(self.root/p) for p in (c.PROPOSAL, c.DESIGN, c.SOURCE_PLAN, c.SOURCE_MANIFEST)}))
        fake = SimpleNamespace(CACHE_FILES=['DATA.npz'], np=SimpleNamespace(__version__='synthetic'),
            pd=SimpleNamespace(__version__='synthetic'), validate_proposal=lambda _: None,
            source_files=lambda _: {str(source): c.sha(source)}, TOLERANCES={'scores': 1e-12},
            INPUTS=('meta', 'policy', 'original_orders', 'prices', 'protected_manifest', 'sessions', 'proposal',
                    'source_manifest', 'source_grid_review', 'source_plan', 'source_calendar'))
        overrides = {key: c.sha(self.root/path) for key, path in (
            ('PREVIOUS_DONE_SHA', c.PREVIOUS_DONE), ('PREVIOUS_PLAN_SHA', c.PREVIOUS_PLAN),
            ('SOURCE_PLAN_SHA', c.SOURCE_PLAN), ('SOURCE_CALENDAR_SHA', c.SOURCE_CALENDAR),
            ('DESIGN_SHA', c.DESIGN), ('SESSIONS_SHA', c.SESSIONS))}
        s = state(); s['completed'][-1]['completion']['sha256'] = overrides['PREVIOUS_DONE_SHA']; self.put(c.STATE, s)
        return tests, proposal_sha, fake, overrides

    def test_full_synthetic_build_and_registration_bind_source_and_predecessor(self):
        tests, h, fake, overrides = self.fixture()
        with patch.multiple(c, **overrides), patch.object(c, 'runner', return_value=fake):
            plan = c.build_plan(self.root, tests, h)
            self.assertEqual(plan['round'], 156)
            self.assertEqual(set(plan['inputs']), set(fake.INPUTS))
            self.assertIn(str(self.root/c.SOURCE_REVIEW), plan['hashes'])
            self.assertNotIn(str(self.root/c.STATE), plan['hashes'])
            self.assertFalse((self.out/'PLAN.json').exists())
            result = c.register(self.root, tests, h)
            self.assertTrue(result['registered'])
            registered = c.old.decode((self.root/c.STATE).read_bytes())
            self.assertEqual(len(registered['registered_queue']), 1)
            self.assertEqual(registered['unrelated'], {'preserve': [1, 2]})

    def test_prereview_must_cover_final_proposal_and_actual_activity(self):
        tests, h, fake, overrides = self.fixture()
        path = self.root/c.PREREVIEWS['prepare']
        doc = c.old.decode(path.read_bytes()); doc['actual_returns_calculated'] = 1; self.put(path, doc)
        with patch.multiple(c, **overrides), patch.object(c, 'runner', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'actual_returns_calculated'):
                c.build_plan(self.root, tests, h)
        self.assertFalse((self.out/'PLAN.json').exists())


if __name__ == '__main__':
    unittest.main()
