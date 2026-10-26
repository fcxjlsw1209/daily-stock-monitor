"""Fabricated-only runner checks; never load the actual source grid or market cache."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_revenue_persistence156_runner as r


def source_fixture():
    dates = ['%04d-%02d-28' % (2024 + i // 12, i % 12 + 1) for i in range(122)]
    names = ['MSFT', 'AAPL', 'ABBV'] + ['STOCK%d' % i for i in range(29)] + ['ETF%d' % i for i in range(17)]
    meta = dict(dates=dates, symbols=[list(names) for _ in dates], etfs=names[-17:])
    cutoffs = {day: int(pd.Timestamp(day + 'T21:00:00Z').value) for day in dates}
    rows = [dict(signal_index=i, candidate_index=j, decision_date=day, symbol=symbol,
                 decision_cutoff=day + 'T21:00:00Z', decision_cutoff_ns=cutoffs[day],
                 status='missing_required_source', reasons=['synthetic_hold'],
                 report_ids=[], edge_ids=[], latest_report_id=None)
            for i, day in enumerate(dates) for j, symbol in enumerate(names)]
    reports, edges = {issuer: [] for issuer in r.FIXED}, {issuer: [] for issuer in r.FIXED}
    for i, year in enumerate(range(2020, 2024)):
        raw = dict(path='synthetic_%d.html' % year, sha256='a' * 64, bytes=10)
        record = dict(issuer='MSFT', fiscal_year=year, report_id='MSFT:%d' % year,
            source_checked=True, currency='USD', measure='consolidated_gaap_total_revenue',
            known_ns=int(pd.Timestamp('%d-07-20T20:00:00Z' % year).value),
            period_start='%d-07-01' % (year - 1), period_end='%d-06-30' % year,
            current=str(100 * 2 ** i), scale='1000000', source=dict(raw=raw))
        if i:
            previous = reports['MSFT'][-1]
            record.update(comparative=previous['current'], comparative_scale=previous['scale'],
                comparative_currency='USD', comparative_measure=record['measure'],
                comparative_period_start=previous['period_start'], comparative_period_end=previous['period_end'],
                comparative_source_report_id=record['report_id'])
            edges['MSFT'].append(dict(edge_id='edge:%d' % year, earlier_fiscal_year=year - 1,
                later_fiscal_year=year, earlier_report_id=previous['report_id'],
                later_report_id=record['report_id'], known_ns=record['known_ns'],
                evidence_known_ns=[previous['known_ns'], record['known_ns']],
                status='checked_reported_total_edge'))
        reports['MSFT'].append(record)
    for i in (0, 1):
        row = rows[i * 49]
        row.update(r.evaluate_msft(reports['MSFT'], edges['MSFT'], cutoffs[dates[i]], dates[i]))
        row['source_refs'] = [record['source']['raw'] for record in reports['MSFT']]
    plan = dict(fixed_issuer_scope=list(r.FIXED), gate_before_strategy_registration={
        'minimum_issuers': 2, 'minimum_usable_candidate_rows_per_qualifying_issuer_in_first88': 12,
        'minimum_distinct_latest_original_annual_endpoints_per_qualifying_issuer_in_first88': 2,
        'minimum_first88_cutoffs_with_at_least_two_usable_issuers': 12})
    return meta, rows, reports, edges, cutoffs, plan


def measured(name, cagr=.4, sharpe=1.5, dd=-.25, early=.3, recent=.5):
    return dict(name=name, training=dict(cagr=cagr, sharpe=sharpe, max_drawdown=dd),
                early=dict(cagr=early), recent=dict(cagr=recent))


class Adapter:
    def __init__(self):
        self.calls = 0

    def score_and_choose(self, *args, **kwargs):
        self.calls += 1
        assert args[:2] == ('incumbent66', 0.)
        assert kwargs['boundary_selector'] == 'original'
        scores = kwargs['pre_score'][:args[5]].copy()
        choices = [sorted(range(49), key=lambda j: (-row[j], j))[:3] for row in scores]
        return choices, scores


def scoring_fixture():
    meta, _, _, _, _, _ = source_fixture()
    arrays = dict(baseline=np.zeros((122, 49)), eligible=np.ones((122, 49), bool),
                  effects=np.zeros((122, 49)), common_available=np.zeros((122, 49), bool))
    for key in r.HANDS:
        arrays[key] = np.full((122, 49), np.nan)
        arrays[key][:, 0] = .5
    arrays['common_available'][:, 0] = True
    backend = SimpleNamespace(adapter=Adapter(), selector='original')
    backend.pre = lambda arrays, n: np.zeros((n, 49))
    backend.baseline = lambda arrays, meta, n, policy: ([[0, 1, 2] for _ in range(n)], np.zeros((n, 49)))
    backend.make_orders = lambda choices, scores, meta: [dict(symbols=[meta['symbols'][i][j] for j in row])
        for i, row in enumerate(choices)]
    return arrays, meta, backend


class SourceFeatureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = source_fixture()

    def test_complete_grid_and_identical_three_feature_masks(self):
        result = r.build_features(*self.fixture)
        self.assertEqual(result['audit']['rows'], 5978)
        self.assertEqual(result['audit']['usable_rows'], 2)
        common = result['arrays']['common_available']
        self.assertEqual(int(common.sum()), 2)
        for key in r.HANDS:
            np.testing.assert_array_equal(np.isfinite(result['arrays'][key]), common)
            self.assertEqual(list(result['arrays'][key][common]), [.5, .5])
        self.assertFalse(common[:, 1:].any())

    def test_no_input_mutation(self):
        before = deepcopy(self.fixture)
        r.build_features(*self.fixture)
        self.assertEqual(self.fixture, before)

    def test_drop_or_duplicate_candidate_rejected_before_factor_call(self):
        for mode in ('drop', 'duplicate'):
            sample = deepcopy(self.fixture)
            sample[1][-1:] = [] if mode == 'drop' else [deepcopy(sample[1][0])]
            with patch.object(r.factor, 'measure_levels') as calculate:
                with self.assertRaises(ValueError):
                    r.build_features(*sample)
                calculate.assert_not_called()

    def test_all_cutoffs_verified_before_first_factor(self):
        self.fixture[1][-1]['decision_cutoff_ns'] += 1
        with patch.object(r.factor, 'measure_levels') as calculate:
            with self.assertRaises(ValueError):
                r.build_features(*self.fixture)
            calculate.assert_not_called()

    def test_naive_or_wrong_day_timestamp_rejected(self):
        for value in ('2024-01-28T21:00:00', '2024-01-27T21:00:00Z'):
            sample = deepcopy(self.fixture)
            sample[1][0]['decision_cutoff'] = value
            with self.assertRaises(ValueError):
                r.build_features(*sample)

    def test_wrong_endpoint_or_order_is_not_silently_repaired(self):
        for mode in ('latest', 'reverse', 'duplicate'):
            sample = deepcopy(self.fixture)
            if mode == 'latest':
                sample[1][0]['latest_report_id'] = 'older'
            elif mode == 'reverse':
                sample[1][0]['report_ids'].reverse()
            else:
                sample[1][0]['report_ids'][0] = sample[1][0]['report_ids'][1]
            with self.assertRaises(ValueError):
                r.build_features(*sample)

    def test_unchecked_latest_never_falls_back(self):
        self.fixture[2]['MSFT'][-1]['source_checked'] = False
        with self.assertRaises(ValueError):
            r.build_features(*self.fixture)

    def test_future_edge_rejected(self):
        self.fixture[3]['MSFT'][-1]['known_ns'] = self.fixture[4][self.fixture[0]['dates'][0]] + 1
        with self.assertRaises(ValueError):
            r.build_features(*self.fixture)

    def test_mixed_original_scale_rejected_even_when_witness_matches(self):
        self.fixture[2]['MSFT'][0]['scale'] = '1000'
        self.fixture[2]['MSFT'][1]['comparative_scale'] = '1000'
        with self.assertRaises(ValueError):
            r.build_features(*self.fixture)

    def test_source_ref_relabel_rejected(self):
        self.fixture[1][0]['source_refs'] = []
        with self.assertRaises(ValueError):
            r.build_features(*self.fixture)

    def test_unusable_row_stays_nan_even_when_catalog_could_supply_it(self):
        self.fixture[1][0].update(status='unresolved_source_or_comparability', reasons=['retained_hold'])
        result = r.build_features(*self.fixture)
        self.assertTrue(all(np.isnan(result['arrays'][key][0, 0]) for key in r.HANDS))
        self.assertEqual(result['audit']['cells'][0]['reasons'], ['retained_hold'])

    def test_nonpilot_or_etf_source_cannot_be_admitted(self):
        for j in (3, 48):
            sample = deepcopy(self.fixture)
            sample[1][j].update(status=r.USABLE, latest_report_id='invalid')
            with self.assertRaises(ValueError):
                r.build_features(*sample)


class ScoringSelectionTests(unittest.TestCase):
    def test_reuses_frozen_pre_score_helper_and_backend_prefix_fix(self):
        self.assertIs(r.corrected, r.prior.corrected)
        self.assertIs(r.Backend.pre, r.prior.Backend.pre)
        self.assertIs(r.Backend.baseline, r.prior.Backend.baseline)
        self.assertEqual(r.selection_contract(), r.prior.selection_contract())

    def test_exact_dose_eligibility_and_one_original_selector(self):
        arrays, meta, backend = scoring_fixture()
        arrays['eligible'][0, 0] = False
        choices, scores = r.choose(r.NAMES[1], arrays, meta, 2, backend, {})
        self.assertEqual(scores[0, 0], 0.)
        self.assertEqual(scores[1, 0], .005)
        self.assertEqual(backend.adapter.calls, 1)
        self.assertEqual(len(choices), 2)

    def test_full_panels_prefixes_null_fallback_and_no_mutation(self):
        arrays, meta, backend = scoring_fixture()
        before = {key: value.tobytes() for key, value in arrays.items()}
        orders, scores = r.panels(arrays, meta, backend, {})
        self.assertEqual(list(orders), list(r.NAMES))
        self.assertTrue(all(len(value) == 122 for value in orders.values()))
        self.assertEqual(backend.adapter.calls, 3 * 124 + 6)
        self.assertEqual(before, {key: value.tobytes() for key, value in arrays.items()})

    def test_future_bad_features_cannot_affect_earlier_prefix(self):
        arrays, meta, backend = scoring_fixture()
        arrays[r.HANDS[0]][121, 0] = np.inf
        r.choose(r.NAMES[1], arrays, meta, 88, backend, {})
        with self.assertRaises(ValueError):
            r.choose(r.NAMES[1], arrays, meta, 122, backend, {})

    def test_mismatched_mask_rejected_before_scoring(self):
        arrays, meta, backend = scoring_fixture()
        arrays[r.HANDS[0]][0, 0] = np.nan
        with self.assertRaises(ValueError):
            r.panels(arrays, meta, backend, {})
        self.assertEqual(backend.adapter.calls, 0)

    def test_all_three_close_controls_and_two_open_controls_must_be_beaten(self):
        rows = [measured(name, cagr=.4 + i * .01) for i, name in enumerate(r.NAMES)]
        self.assertTrue(r.select(rows, rows)['admitted'])
        for i in range(3):
            failed = deepcopy(rows)
            failed[i]['training']['cagr'] = .9
            self.assertFalse(r.select(failed, rows)['admitted'])
        for i in (1, 2):
            failed = deepcopy(rows)
            failed[i]['training']['cagr'] = .9
            self.assertFalse(r.select(rows, failed)['admitted'])

    def test_open_incumbent_noninferiority_and_risk_guards(self):
        rows = [measured(name, cagr=.4 + i * .01) for i, name in enumerate(r.NAMES)]
        opened = deepcopy(rows)
        opened[0]['training']['cagr'] = opened[3]['training']['cagr']
        self.assertTrue(r.select(rows, opened)['admitted'])
        for key, bad in (('sharpe', 1.44), ('dd', -.281), ('early', .274), ('recent', .474)):
            failed = deepcopy(rows)
            failed[3] = measured(r.NAMES[3], cagr=.9, **{key: bad})
            self.assertFalse(r.select(failed, opened)['admitted'])

    def test_equal_nonfinite_or_wrong_order_selection_cannot_pass(self):
        rows = [measured(name) for name in r.NAMES]
        self.assertFalse(r.select(rows, rows)['admitted'])
        rows[-1]['training']['cagr'] = np.nan
        self.assertFalse(r.select(rows, rows)['admitted'])
        with self.assertRaises(ValueError):
            r.select(list(reversed(rows)), rows)

    def test_training_never_consumes_later34_orders(self):
        backend = object.__new__(r.Backend)
        meta = dict(symbols=[['S'] for _ in range(122)], dates=['d'] * 122, ends=['e'] * 122)
        orders = {name: [dict(symbols=['S']) for _ in range(88)] + [None] * 34 for name in r.NAMES}
        calls = []
        rp = SimpleNamespace(paths=lambda px, meta, n: (None, None, list(range(123)), None),
            fast_returns=lambda selected, blocks, valid: (calls.append(len(selected)) or np.zeros(88)),
            stats_rr=lambda values, start, end: {'count': len(values)})
        backend.local = SimpleNamespace(rp=rp, old=SimpleNamespace(session=SimpleNamespace(
            measure=lambda *args: {key: value for key, value in measured('ignored').items() if key != 'name'})))
        opens = []
        backend.open_training = lambda orders, meta, px, op, n, names: (
            opens.append(n) or [measured(name) for name in names])
        _, _, walk = backend.training(orders, meta, None, None)
        self.assertEqual(calls, [88] * 4)
        self.assertEqual(opens, [88, 36, 60, 76])
        self.assertEqual([(fold['n'], fold['end']) for fold in walk['folds']], [(36, 60), (60, 76), (76, 88)])

    def test_diagnostics_uses_exactly_four_arms_and_three_execution_paths(self):
        backend = object.__new__(r.Backend)
        calls, written = [], []
        def replay(px, op, orders, **kwargs):
            calls.append((len(orders), kwargs['execution'], kwargs['cost']))
            return ({'annual_returns': [{'year': 2016}, {'year': 2026}]},
                    pd.Series([1., 1.01], index=pd.to_datetime(['2024-01-01', '2024-01-02'])), [], [])
        previous = SimpleNamespace(cuts=lambda eq, meta: {})
        backend.local = SimpleNamespace(
            ex=SimpleNamespace(e=SimpleNamespace(engine=SimpleNamespace(replay=replay))),
            arch=SimpleNamespace(old=SimpleNamespace(prior=SimpleNamespace(old=SimpleNamespace(previous=previous)))))
        orders = {name: [{} for _ in range(122)] for name in r.NAMES}
        result = backend.diagnostics(orders, {'dates': ['2024-01-01'], 'ends': ['2024-01-02']}, None, None,
                                     lambda name, value: written.append(name))
        self.assertEqual(list(result), list(r.NAMES))
        self.assertEqual(calls, [(122, 'close', 0.), (122, 'open', 0.), (122, 'open', .0005)] * 4)
        self.assertEqual(len(written), len(set(written)))
        self.assertEqual(len(written), 36)


class RegistrationPhaseTests(unittest.TestCase):
    def test_proposal_exact_contract_and_zero_preregistration_activity(self):
        proposal = dict(round=156, arms=list(r.NAMES), primary_challenger=r.NAMES[3], dose=.01,
            selection_contract=r.selection_contract(), measurement=r.measurement_contract(),
            status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
            actual_new_factors=0, actual_new_orders=0, actual_new_returns=0)
        r.validate_proposal(proposal)
        for key, value in (('dose', .02), ('primary_challenger', r.NAMES[1]),
                           ('actual_new_factors', True), ('actual_new_returns', 1)):
            with self.assertRaises(ValueError):
                r.validate_proposal(dict(proposal, **{key: value}))

    def test_unregistered_plan_rejected_before_source_or_backend_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / 'PLAN.json'
            plan.write_text('{"round":156,"status":"PROPOSED"}')
            with patch.object(r.Runner, 'require_source_grid') as source, patch.object(r, 'Backend') as backend:
                with self.assertRaises(ValueError):
                    r.Runner(root, plan, r.sha(plan))
                source.assert_not_called()
                backend.assert_not_called()

    def test_selection_and_diagnostics_require_independent_preparation(self):
        runner = object.__new__(r.Runner)
        with patch.object(runner, 'refresh'), patch.object(runner, 'require_prepare_review',
                side_effect=ValueError('independent review missing')), patch.object(runner, 'loaded') as loaded:
            for action in (runner.select_first88, runner.diagnose_later34):
                with self.assertRaisesRegex(ValueError, 'independent review missing'):
                    action()
            loaded.assert_not_called()

    def test_prepare_receipt_requires_exact_stage_and_full_audit_counts(self):
        runner = object.__new__(r.Runner)
        runner.out = Path('/private/tmp/synthetic_revenue_review')
        runner.plan_sha = 'a' * 64
        runner.plan = {'independent_prepare_source_hashes': {'review.py': 'b' * 64}}
        report = dict(passed=True, stage='prepare', producer_modules_imported=False,
            returns_or_nav_computed=False, orders_checked=488, candidate_cells_checked=5978,
            months_checked=122, prefixes_checked=123, revenue_feature_matrices_verified=3,
            new_model_fits=0, plan_sha256='a' * 64, cache_sha256='a' * 64, orders_sha256='a' * 64,
            source_hashes={'review.py': 'b' * 64},
            artifact_hashes={str(runner.out / name): 'a' * 64
                             for name in ('PLAN.json', 'CACHE.json', *r.CACHE_FILES)})
        report.update({key: True for key in r.CHECKS})
        with patch.object(runner, 'cache'), patch.object(runner, 'verify', side_effect=lambda value: value), \
                patch.object(r, 'sha', return_value='a' * 64), patch.object(r, 'read', return_value=report):
            runner.require_prepare_review()
            for key, bad in [('stage', 'synthetic'), ('orders_checked', 487),
                             ('candidate_cells_checked', 5977), ('months_checked', 121),
                             ('prefixes_checked', 122), ('revenue_feature_matrices_verified', 2),
                             ('new_model_fits', True)]:
                old = report[key]
                report[key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.require_prepare_review()
                report[key] = old

    def test_diagnostic_start_marker_binds_frozen_selection_before_backend_load(self):
        runner = object.__new__(r.Runner)
        runner.out = Path('/private/tmp/synthetic_revenue_marker')
        runner.plan_sha = 'a' * 64
        rows = [measured(name, cagr=.4 + i * .01) for i, name in enumerate(r.NAMES)]
        frozen = dict(selection=r.select(rows, rows), hashes={name: 'a' * 64 for name in r.FROZEN_FILES})
        written = {}
        def read(path):
            return frozen if Path(path).name == 'FROZEN.json' else rows
        with patch.object(runner, 'refresh'), patch.object(runner, 'require_prepare_review'), \
                patch.object(runner, 'verify'), patch.object(r, 'read', side_effect=read), \
                patch.object(r, 'sha', return_value='f' * 64), \
                patch.object(runner, 'write', side_effect=lambda name, value: written.update({name: value})), \
                patch.object(runner, 'loaded', side_effect=ValueError('stop_after_marker')):
            with self.assertRaisesRegex(ValueError, 'stop_after_marker'):
                runner.diagnose_later34()
        marker = written['DIAGNOSE_STARTED.json']
        self.assertEqual(marker['plan_sha256'], 'a' * 64)
        self.assertEqual(marker['frozen_sha256'], 'f' * 64)
        self.assertTrue(marker['started_utc'].endswith('+00:00'))

    def test_nonlocal_data_paths_rejected(self):
        runner = object.__new__(r.Runner)
        runner.root = Path('/private/tmp/revenue_test')
        with self.assertRaises(ValueError):
            runner.path('/private/tmp/outside.json')

    def test_missing_actual_review_or_failed_coverage_blocks_amount_parsing(self):
        for reviewed in (False, True):
            with self.subTest(actual_grid_verified=reviewed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                out = root / r.SOURCE_OUT
                out.mkdir(parents=True)
                meta, rows, _, _, _, plan = source_fixture()
                for row in rows:
                    row['status'] = 'missing_required_source'
                payload = {'ROWS.json': rows, 'COVERAGE_GATE.json': r.coverage_gate(meta, rows, plan),
                           'SUMMARY.json': {}}
                for name in r.SOURCE_FILES:
                    # Valid hashed bytes but intentionally not JSON: a gate failure must precede parsing.
                    (out / name).write_text(json.dumps(payload[name]) if name in payload else 'INVALID_AMOUNT_JSON')
                (root / 'stub.json').write_text('{}')
                manifest = dict(source_only=True, strategy_registration=False,
                    input_hashes={'stub.json': r.sha(root / 'stub.json')},
                    outputs={name: dict(sha256=r.sha(out / name), bytes=(out / name).stat().st_size)
                             for name in r.SOURCE_FILES})
                (out / 'MANIFEST.json').write_text(json.dumps(manifest))
                for name, value in (('meta', meta), ('source_plan', plan)):
                    (root / (name + '.json')).write_text(json.dumps(value))
                inputs = dict(source_manifest=out / 'MANIFEST.json', source_grid_review=root / 'review.json',
                              meta=root / 'meta.json', source_plan=root / 'source_plan.json')
                closure = {str(root / 'stub.json'): r.sha(root / 'stub.json'),
                           str(out / 'MANIFEST.json'): r.sha(out / 'MANIFEST.json')}
                closure.update({str(out / name): r.sha(out / name) for name in r.SOURCE_FILES})
                review = dict(passed=True, actual_grid_verified=reviewed, input_hashes=closure)
                inputs['source_grid_review'].write_text(json.dumps(review))
                runner = object.__new__(r.Runner)
                runner.root, runner.inputs = root, inputs
                runner.pins = dict(closure)
                runner.pins.update({str(path): r.sha(path) for path in inputs.values()})
                expected = 'coverage floor' if reviewed else 'actual source-grid review'
                with self.assertRaisesRegex(ValueError, expected):
                    runner.require_source_grid()


if __name__ == '__main__':
    unittest.main()
