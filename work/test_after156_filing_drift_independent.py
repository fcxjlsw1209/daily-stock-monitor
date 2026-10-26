"""Independent synthetic oracles for the filing-drift numerical kernel only."""
import ast
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import unittest

import after156_filing_drift_kernel as kernel

ROOT = Path(__file__).resolve().parents[1]
DESIGN = 'work/after156_filing_drift_design.json'
IMPLEMENTATION = 'work/after156_filing_drift_kernel.py'
PRODUCER_TEST = 'work/test_after156_filing_drift_kernel.py'
THIS = 'work/test_after156_filing_drift_independent.py'
OUT = 'work/after156_filing_drift_kernel_review.json'
EXPECTED = {DESIGN: '4a902f27fd542a0c00f75661d231346c63f99c61e1c8e95bbf45cb94fd73ee75',
            IMPLEMENTATION: '2e60fa089951935868c79e299a2bf305e8f564701fe7cd7d40f2b0bf80c753cc',
            PRODUCER_TEST: 'b5e975d0a82b14fd079b33944bda6218b1398ba8d94c6d874311ba450e73e3d3'}
ARMS = ('whole_formation_beta_residual_momentum', 'filing_window_density',
        'filing_vs_nonfiling_beta_residual_contrast')
T = 800


def admission(t=T):
    return kernel.Admission(True, True, True, t - 271, t - 42)


def events(t=T):
    return [kernel.Event('a', t - 240, '10-Q'), kernel.Event('b', t - 170, '10-K'),
            kernel.Event('c', t - 100, '10-Q')]


def partition_oracle(t, filings):
    """Brute-force set ownership, independent of the producer's moving cursor."""
    unique = {x.accession: x for x in filings}
    relevant = [x for x in unique.values() if t - 271 <= x.anchor <= t - 42]
    by_day = {}
    for day in range(t - 251, t - 41):
        active = [x for x in relevant if 0 <= day - x.anchor <= 20]
        if not active:
            by_day[day] = ('N', None, [])
            continue
        latest = max(x.anchor for x in active)
        owners = [x for x in active if x.anchor == latest]
        blackout = day - latest < 2 or any(x.form.endswith('/A') for x in owners)
        by_day[day] = ('B' if blackout else 'E', latest, sorted(x.accession for x in owners))
    full = []
    for anchor in sorted({x.anchor for x in relevant}):
        # All nineteen prescribed dates must exist and belong to this anchor.
        if all(by_day.get(day, (None, None))[:2] == ('E', anchor)
               for day in range(anchor + 2, anchor + 21)):
            full.append(anchor)
    counts = {label: sum(row[0] == label for row in by_day.values()) for label in 'EBN'}
    return by_day, full, counts


def fixture(t=T, filings=None, beta=1.7, alpha=.0013, residual=None):
    filings = events(t) if filings is None else filings
    labels, _, _ = partition_oracle(t, filings)
    stock, market = [83.], [117.]
    for day in range(1, t + 31):
        m = .0007 + .0011 * ((day % 11) - 5)
        if t - 503 <= day <= t - 252:
            y = alpha + beta * m
        elif day in labels:
            z = {'E': .003, 'N': -.0004, 'B': .008}[labels[day][0]] if residual is None else residual(day, labels[day][0])
            y = beta * m + z
        else:
            y = -.0002 + .3 * m
        stock.append(stock[-1] * math.exp(y))
        market.append(market[-1] * math.exp(m))
    return stock, market


def numeric_oracle(t, stock, market, filings):
    """Decimal log differences + raw-moment OLS, not producer centered sums."""
    labels, _, counts = partition_oracle(t, filings)
    with localcontext() as context:
        context.prec = 60
        n = Decimal(252)
        ys = [Decimal(stock[d]).ln() - Decimal(stock[d - 1]).ln() for d in range(t - 503, t - 251)]
        xs = [Decimal(market[d]).ln() - Decimal(market[d - 1]).ln() for d in range(t - 503, t - 251)]
        slope = (sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys) / n) / (sum(x * x for x in xs) - sum(xs) ** 2 / n)
        residuals = {}
        for day in labels:
            y = Decimal(stock[day]).ln() - Decimal(stock[day - 1]).ln()
            x = Decimal(market[day]).ln() - Decimal(market[day - 1]).ln()
            residuals[day] = y - slope * x
        e = [v for d, v in residuals.items() if labels[d][0] == 'E']
        other = [v for d, v in residuals.items() if labels[d][0] == 'N']
        raw = [Decimal(21) * sum(residuals.values()) / 210,
               Decimal(counts['E']) / (counts['E'] + counts['N']),
               Decimal(21) * (sum(e) / len(e) - sum(other) / len(other))]
        hands = [x if i == 1 else (x / Decimal('.05')) / (1 + abs(x / Decimal('.05'))) for i, x in enumerate(raw)]
        return float(slope), list(map(float, raw)), list(map(float, hands))


class IndependentKernelReview(unittest.TestCase):
    def compare_partition(self, t, filings):
        expected, full, counts = partition_oracle(t, filings)
        actual = kernel.partition(t, filings, admission(t))
        self.assertEqual({d['session']: (d['label'], d['owner_anchor'], d['accessions']) for d in actual['days']}, expected)
        self.assertEqual(actual['full_original_anchors'], full)
        self.assertEqual(actual['counts'], counts)
        self.assertEqual(actual['minimum_passed'], len(full) >= 2 and counts['E'] >= 38 and counts['N'] >= 80)
        return actual

    def test_01_decimal_oracle_nonzero_means_and_no_residual_intercept(self):
        stock, market = fixture()
        actual = kernel.evaluate_cell(T, stock, market, events(), admission())
        beta, raw, hands = numeric_oracle(T, stock, market, events())
        self.assertAlmostEqual(actual['beta'], beta, places=11)
        self.assertAlmostEqual(beta, 1.7, places=11)
        for index, name in enumerate(ARMS):
            self.assertAlmostEqual(actual['raw'][name], raw[index], places=11)
            self.assertAlmostEqual(actual['hands'][name], hands[index], places=11)
        self.assertNotAlmostEqual(actual['raw'][ARMS[0]], raw[0] - 21 * .0013, places=8)

    def test_02_randomized_bruteforce_partition_oracle(self):
        rng = random.Random(918275)
        for trial in range(240):
            t = rng.randrange(600, 1300)
            filings = [kernel.Event(str(i), rng.randrange(t - 280, t + 5), rng.choice(('10-Q', '10-K', '10-Q/A', '10-K/A')))
                       for i in range(rng.randrange(1, 44))]
            filings += filings[:3]
            rng.shuffle(filings)
            with self.subTest(trial=trial):
                self.compare_partition(t, filings)

    def test_03_exact_nonevent_minimum_eighty_vs_seventy_nine(self):
        left = T - 251
        full = [kernel.Event(str(i), left + i, '10-Q') for i in (10, 40, 70, 100, 130, 160)]
        at = self.compare_partition(T, full + [kernel.Event('clipped', left - 17, '10-Q')])
        below = self.compare_partition(T, full + [kernel.Event('clipped', left - 16, '10-Q')])
        self.assertEqual(at['counts']['N'], 80)
        self.assertTrue(at['minimum_passed'])
        self.assertEqual(below['counts']['N'], 79)
        self.assertFalse(below['minimum_passed'])

    def test_04_many_event_days_do_not_substitute_for_two_full_bundles(self):
        filings = [kernel.Event('leftpartial', T - 254, '10-Q'), kernel.Event('full', T - 150, '10-K'),
                   kernel.Event('rightpartial', T - 44, '10-Q')]
        part = self.compare_partition(T, filings)
        self.assertEqual(part['counts']['E'], 38)
        self.assertEqual(len(part['full_original_anchors']), 1)
        self.assertFalse(part['minimum_passed'])

    def test_05_latest_amendment_and_original_overlapping_windows(self):
        filings = events() + [kernel.Event('amend', T - 234, '10-Q/A'),
                             kernel.Event('replacement', T - 230, '10-K'),
                             kernel.Event('same-day-amend', T - 170, '10-K/A')]
        part = self.compare_partition(T, filings)
        self.assertNotIn(T - 240, part['full_original_anchors'])
        self.assertNotIn(T - 170, part['full_original_anchors'])
        self.assertIn(T - 230, part['full_original_anchors'])

    def test_06_identical_bundle_duplicates_and_conflicts(self):
        baseline = self.compare_partition(T, events())
        self.assertEqual(baseline, self.compare_partition(T, events() + events()))
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            kernel.partition(T, events() + [replace(events()[0], form='10-Q/A')], admission())

    def test_07_left_right_boundaries_and_future_events(self):
        filings = events() + [kernel.Event('earliest', T - 271, '10-Q'),
                             kernel.Event('last', T - 42, '10-Q/A')]
        before = self.compare_partition(T, filings)
        after = self.compare_partition(T, filings + [kernel.Event('tooearly', T - 272, '10-Q'), kernel.Event('future', T - 41, '10-Q/A')])
        self.assertEqual(before, after)
        self.assertEqual(before['days'][0]['label'], 'E')
        self.assertEqual(before['days'][-1]['label'], 'B')

    def test_08_only_consumed_prices_are_ever_accessed_and_shifted_axes_match(self):
        stock, market = fixture()
        expected = kernel.evaluate_cell(T, stock, market, events(), admission())
        class Guard:
            def __init__(self, values): self.values = values
            def __len__(self): return len(self.values)
            def __getitem__(self, key):
                if not isinstance(key, slice) or key.start < T - 504 or key.stop > T - 41:
                    raise AssertionError('Price access outside exact consumed interval')
                return self.values[key]
        actual = kernel.evaluate_cell(T, Guard(stock), Guard(market), events(), admission())
        self.assertEqual(expected, actual)
        first, stop = T - 504, T - 41
        self.assertEqual(expected, kernel.evaluate_cell(T, stock[first:stop], market[first:stop], events(), admission(), price_start_index=first))

    def test_09_preformation_boundary_separation(self):
        stock, market = fixture()
        original = kernel.evaluate_cell(T, stock, market, events(), admission())
        formation = stock[:]
        for d in range(T - 251, len(formation)):
            formation[d] *= math.exp(.4)
        changed = kernel.evaluate_cell(T, formation, market, events(), admission())
        self.assertEqual(original['beta'], changed['beta'])
        self.assertNotEqual(original['raw'][ARMS[0]], changed['raw'][ARMS[0]])
        training = stock[:]
        for d in range(T - 252, len(training)):
            training[d] *= math.exp(.4)
        earlier = kernel.evaluate_cell(T, training, market, events(), admission())
        self.assertNotAlmostEqual(original['beta'], earlier['beta'], places=8)

    def test_10_source_failure_precedes_any_price_access(self):
        class Untouchable:
            def __len__(self): raise AssertionError('Unadmitted prices accessed')
        for name in ('clock_admitted', 'chronology_complete', 'identity_admitted'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                kernel.evaluate_cell(T, Untouchable(), Untouchable(), events(), replace(admission(), **{name: False}))

    def test_11_all_missing_causes_share_exact_three_arm_mask(self):
        stock, market = fixture()
        for location in (T - 504, T - 252, T - 251, T - 42):
            for value in (None, 0., -1., float('nan'), float('inf')):
                corrupted = stock[:]
                corrupted[location] = value
                result = kernel.evaluate_cell(T, corrupted, market, events(), admission())
                self.assertFalse(result['available'])
                self.assertEqual(tuple(result['hands']), ARMS)
                self.assertTrue(all(math.isnan(x) for x in result['hands'].values()))
                self.assertEqual(result['nuisance_beta_estimates_completed'], 0)

    def test_12_no_model_count_false_zero_after_beta(self):
        stock, market = fixture()
        fitted = kernel.evaluate_cell(T, stock, market, events(), admission())
        constant = kernel.evaluate_cell(T, stock, [100.] * len(market), events(), admission())
        self.assertEqual((fitted['nuisance_beta_estimates_attempted'], fitted['nuisance_beta_estimates_completed']), (1, 1))
        self.assertEqual((constant['nuisance_beta_estimates_attempted'], constant['nuisance_beta_estimates_completed']), (1, 0))
        self.assertEqual(fitted['new_predictive_model_fits'], 0)

    def test_13_constant_residual_zero_contrast_density_differs(self):
        stock, market = fixture(residual=lambda day, label: .002)
        first = kernel.evaluate_cell(T, stock, market, events(), admission())
        second = kernel.evaluate_cell(T, stock, market, events()[:2], admission())
        for result in (first, second):
            self.assertAlmostEqual(result['raw'][ARMS[0]], .042, places=11)
            self.assertAlmostEqual(result['raw'][ARMS[2]], 0., places=11)
        self.assertNotEqual(first['hands'][ARMS[1]], second['hands'][ARMS[1]])

    def test_14_scale_invariance_and_no_input_mutation(self):
        stock, market = fixture()
        filings = events()
        saved = deepcopy((stock, market, filings))
        baseline = kernel.evaluate_cell(T, stock, market, filings, admission())
        scaled = kernel.evaluate_cell(T, [x * 1e4 for x in stock], [x * 1e-3 for x in market], filings, admission())
        self.assertEqual(saved, (stock, market, filings))
        self.assertAlmostEqual(baseline['beta'], scaled['beta'], places=11)
        for name in ARMS:
            self.assertAlmostEqual(baseline['hands'][name], scaled['hands'][name], places=11)

    def test_15_kernel_import_surface_has_no_data_or_strategy_dependency(self):
        tree = ast.parse((ROOT / IMPLEMENTATION).read_text())
        imports = [x.module for x in ast.walk(tree) if isinstance(x, ast.ImportFrom)]
        imports += [a.name for x in ast.walk(tree) if isinstance(x, ast.Import) for a in x.names]
        self.assertEqual(set(imports), {'dataclasses', 'math', 'numbers'})


def main():
    write = '--write-review' in sys.argv
    if write:
        sys.argv.remove('--write-review')
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
              for p in (DESIGN, IMPLEMENTATION, PRODUCER_TEST, THIS, 'run_local.sh', 'local_run.py', 'local.sandbox.sb')}
    for path, expected in EXPECTED.items():
        if hashes[path] != expected:
            raise ValueError('Reviewed input changed: ' + path)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentKernelReview)
    names = [x.id() for x in suite]
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    for path, expected in hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Input changed during tests: ' + path)
    print(buffer.getvalue())
    if write:
        receipt = dict(schema_version=1, stage='INDEPENDENT_SYNTHETIC_FILING_DRIFT_KERNEL_REVIEW',
            passed=result.wasSuccessful(), tests_run=result.testsRun, test_names=names,
            randomized_partition_cases=240, input_hashes=hashes,
            oracle='Independent per-day brute-force ownership plus 60-digit Decimal log differences and raw-moment OLS; no producer helper used to construct expected numeric values.',
            findings=[] if result.wasSuccessful() else [str(x) for x in result.failures + result.errors],
            scope='Pure admitted-integer-anchor and supplied-array kernel only. Synthetic values exclusively.',
            caller_obligations=['Authentic clock/identity/chronology receipts must be enforced outside this kernel; True booleans do not prove provenance.',
                'Source adapter must resolve duplicate original report keys and orphan amendments before forming Event records; Event does not contain reportDate or proof objects.',
                'Full XNYS timestamp-to-anchor conversion, whole122x49 grid/mask, source breadth gates, original27-array/score/selector parity, lifecycle and outcome verification are outside this kernel review.'],
            actual_activity=dict(price_values_read=0, factor_calculations=0, returns=0, ranks=0, strategy_registrations=0, network_requests=0, frozen_files_modified=0),
            synthetic_activity=dict(beta_estimation=True, fabricated_log_returns=True),
            admissions=dict(real_source_gate=False, exact_clock=False, actual_factor_build=False, strategy_registered=False))
        with (ROOT / OUT).open('x') as output:
            json.dump(receipt, output, sort_keys=True, indent=2)
            output.write('\n')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
