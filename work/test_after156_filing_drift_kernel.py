"""Fabricated paths only: no source, price cache, producer or strategy imports."""
from copy import deepcopy
from dataclasses import replace
import math
import random
import unittest

import after156_filing_drift_kernel as k


T = 504
SOURCE = k.Admission(True, True, True, T - 271, T - 42)
EVENTS = [k.Event('q1', 260, '10-Q'), k.Event('q2', 330, '10-Q'), k.Event('annual', 400, '10-K')]


def prices(returns, start=100.):
    result = [start]
    for value in returns:
        result.append(result[-1] * math.exp(value))
    return result


def fixture(residual=None, beta=2., training_intercept=.001):
    residual = residual or {}
    market = [.004 * ((day % 7) - 3) for day in range(1, T + 1)]
    stock = [beta * value + (training_intercept if day <= T - 252 else residual.get(day, 0.))
             for day, value in enumerate(market, 1)]
    return prices(stock), prices(market)


def evaluate(stock=None, market=None, events=EVENTS, admission=SOURCE, **kwargs):
    if stock is None:
        stock, market = fixture()
    return k.evaluate_cell(T, stock, market, events, admission, **kwargs)


class FilingPartitionTest(unittest.TestCase):
    def test_exact_day_zero_one_two_twenty_twentyone(self):
        result = k.partition(T, EVENTS, SOURCE)
        days = {row['session']: row for row in result['days']}
        self.assertEqual([days[d]['label'] for d in (260, 261, 262, 280, 281)], ['B', 'B', 'E', 'E', 'N'])
        self.assertEqual(result['counts'], {'E': 57, 'B': 6, 'N': 147})
        self.assertEqual(result['full_original_anchors'], [260, 330, 400])
        self.assertTrue(result['minimum_passed'])
        self.assertEqual(len(days), 210)

    def test_same_anchor_bundles_do_not_duplicate_days(self):
        result = k.partition(T, EVENTS + [k.Event('other-period', 260, '10-K')], SOURCE)
        self.assertEqual(result['counts']['E'], 57)
        self.assertEqual(result['full_original_anchors'], [260, 330, 400])
        self.assertEqual(result['bundles'][0]['accessions'], ['other-period', 'q1'])

    def test_amendment_overrides_same_session_original(self):
        result = k.partition(T, EVENTS + [k.Event('amended', 260, '10-Q/A')], SOURCE)
        self.assertEqual(result['counts'], {'E': 38, 'B': 25, 'N': 147})
        self.assertEqual(result['full_original_anchors'], [330, 400])
        self.assertTrue(result['minimum_passed'])

    def test_latest_anchor_owns_overlap_and_amendment_blackout(self):
        events = EVENTS + [k.Event('amended', 270, '10-Q/A'), k.Event('new-period', 280, '10-Q')]
        result = k.partition(T, events, SOURCE)
        days = {row['session']: row for row in result['days']}
        for d in range(262, 270):
            self.assertEqual((days[d]['label'], days[d]['owner_anchor']), ('E', 260))
        for d in range(270, 280):
            self.assertEqual((days[d]['label'], days[d]['owner_anchor']), ('B', 270))
        self.assertEqual((days[280]['label'], days[280]['owner_anchor']), ('B', 280))
        self.assertEqual((days[282]['label'], days[282]['owner_anchor']), ('E', 280))
        self.assertEqual(result['full_original_anchors'], [280, 330, 400])

    def test_exact_left_edge_and_immature_right_edge(self):
        events = EVENTS + [k.Event('left', 233, '10-Q'), k.Event('late', 459, '10-Q'),
                           k.Event('last', 462, '10-K')]
        result = k.partition(T, events, SOURCE)
        days = {row['session']: row for row in result['days']}
        self.assertEqual((days[253]['label'], days[253]['owner_anchor']), ('E', 233))
        self.assertEqual(days[254]['label'], 'N')
        self.assertEqual(days[461]['label'], 'E')
        self.assertEqual((days[462]['label'], days[462]['owner_anchor']), ('B', 462))
        self.assertEqual(result['full_original_anchors'], [260, 330, 400])
        self.assertTrue(result['bundles'][0]['treatment_left_clipped'])
        self.assertTrue(result['bundles'][-1]['treatment_right_clipped'])

    def test_future_and_too_old_events_do_not_change_partition(self):
        outside = [k.Event('old', 232, '10-Q'), k.Event('future', 463, '10-Q/A')]
        self.assertEqual(k.partition(T, EVENTS, SOURCE), k.partition(T, outside + EVENTS, SOURCE))

    def test_exact_duplicates_coalesce_conflicts_fail(self):
        self.assertEqual(k.partition(T, EVENTS, SOURCE), k.partition(T, EVENTS + [EVENTS[0]], SOURCE))
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            k.partition(T, EVENTS + [replace(EVENTS[0], anchor=261)], SOURCE)

    def test_explicit_admission_and_full_boundary_required(self):
        for field in ('clock_admitted', 'chronology_complete', 'identity_admitted'):
            for value in (False, 1, None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'not admitted'):
                    k.partition(T, EVENTS, replace(SOURCE, **{field: value}))
        for source in (replace(SOURCE, first_anchor=234), replace(SOURCE, last_anchor=461)):
            with self.assertRaisesRegex(ValueError, 'every relevant anchor'):
                k.partition(T, EVENTS, source)

    def test_unadmitted_clock_rejected_before_price_access(self):
        class NoPriceAccess:
            def __len__(self):
                raise AssertionError('Prices accessed before clock gate')
        with self.assertRaisesRegex(ValueError, 'clock_admitted'):
            evaluate(NoPriceAccess(), NoPriceAccess(), admission=replace(SOURCE, clock_admitted=False))

    def test_typed_anchors_and_forms_fail_closed(self):
        for event in (k.Event('bad', True, '10-Q'), k.Event('bad', 260.0, '10-Q'),
                      k.Event('bad', 260, '8-K'), k.Event('', 260, '10-K')):
            with self.subTest(event=event), self.assertRaises(ValueError):
                k.partition(T, [event], SOURCE)

    def test_partition_invariants_for_seeded_adversarial_calendars(self):
        rng = random.Random(1701)
        for trial in range(40):
            events = [k.Event(str(i), rng.randrange(225, 480), rng.choice(sorted(k.ORIGINAL | k.AMENDMENT)))
                      for i in range(20)]
            result = k.partition(T, events, SOURCE)
            self.assertEqual(sum(result['counts'].values()), 210)
            self.assertEqual([d['session'] for d in result['days']], list(range(253, 463)))
            for row in result['days']:
                if row['label'] == 'E':
                    self.assertIn(row['session'] - row['owner_anchor'], range(2, 21))
                    owner = [event for event in events if event.anchor == row['owner_anchor']]
                    self.assertTrue(all(event.form in k.ORIGINAL for event in owner))
                if row['label'] == 'N':
                    self.assertIsNone(row['owner_anchor'])


class FilingReturnsTest(unittest.TestCase):
    def assert_unavailable(self, result, reason):
        self.assertFalse(result['available'])
        self.assertEqual(result['reason'], reason)
        self.assertTrue(all(math.isnan(value) for value in result['hands'].values()))
        self.assertEqual(result['nuisance_beta_estimates_completed'], 0)

    def test_known_beta_intercept_and_residual_means(self):
        labels = {d['session']: d['label'] for d in k.partition(T, EVENTS, SOURCE)['days']}
        residual = {d: {'E': .003, 'N': .001, 'B': .02}[label] for d, label in labels.items()}
        result = evaluate(*fixture(residual))
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['beta'], 2., places=12)
        self.assertAlmostEqual(result['raw'][k.NAMES[0]], 21 * (57 * .003 + 147 * .001 + 6 * .02) / 210, places=12)
        self.assertAlmostEqual(result['raw'][k.NAMES[2]], 21 * .002, places=12)
        self.assertEqual(result['hands'][k.NAMES[1]], 57 / 204)
        self.assertEqual(result['nuisance_beta_estimates_attempted'], 1)
        self.assertEqual(result['nuisance_beta_estimates_completed'], 1)
        self.assertEqual(result['new_predictive_model_fits'], 0)

    def test_genuine_zero_remains_exact_zero(self):
        _, market = fixture()
        result = evaluate(market[:], market)
        self.assertEqual(result['beta'], 1.)
        self.assertEqual(result['hands'][k.NAMES[0]], 0.)
        self.assertEqual(result['hands'][k.NAMES[2]], 0.)
        self.assertGreater(result['hands'][k.NAMES[1]], 0.)

    def test_preformation_beta_is_not_refit_on_formation_shock(self):
        before = evaluate(*fixture())
        shock = {d: 4. * (.004 * ((d % 7) - 3)) for d in range(253, 463)}
        after = evaluate(*fixture(shock))
        self.assertEqual(before['beta'], after['beta'])
        self.assertNotEqual(before['raw'][k.NAMES[0]], after['raw'][k.NAMES[0]])

    def test_beta_window_last_return_included_first_formation_excluded(self):
        original = evaluate(*fixture())
        changed_formation = evaluate(*fixture({253: .2}))
        self.assertEqual(original['beta'], changed_formation['beta'])
        stock, market = fixture()
        for i in range(252, len(stock)):
            stock[i] *= math.exp(.2)
        changed_beta = evaluate(stock, market)
        self.assertNotAlmostEqual(original['beta'], changed_beta['beta'], places=6)

    def test_shock_location_distinguishes_contrast_from_whole_momentum(self):
        event = evaluate(*fixture({262: .19}))
        nonevent = evaluate(*fixture({300: .19}))
        excluded = evaluate(*fixture({260: .19}))
        for result in (nonevent, excluded):
            self.assertAlmostEqual(event['raw'][k.NAMES[0]], result['raw'][k.NAMES[0]], places=12)
            self.assertEqual(event['hands'][k.NAMES[1]], result['hands'][k.NAMES[1]])
        self.assertAlmostEqual(event['raw'][k.NAMES[2]], 21 * .19 / 57, places=12)
        self.assertAlmostEqual(nonevent['raw'][k.NAMES[2]], -21 * .19 / 147, places=12)
        self.assertAlmostEqual(excluded['raw'][k.NAMES[2]], 0., places=12)

    def test_constant_residual_contrast_neutral_with_nonzero_momentum(self):
        result = evaluate(*fixture({d: .002 for d in range(253, 463)}))
        self.assertAlmostEqual(result['raw'][k.NAMES[2]], 0., places=12)
        self.assertAlmostEqual(result['raw'][k.NAMES[0]], .042, places=12)

    def test_density_changes_without_manufacturing_drift(self):
        _, market = fixture()
        three = evaluate(market[:], market)
        two = evaluate(market[:], market, events=EVENTS[:2])
        self.assertNotEqual(three['hands'][k.NAMES[1]], two['hands'][k.NAMES[1]])
        for result in (three, two):
            self.assertEqual(result['hands'][k.NAMES[0]], 0.)
            self.assertEqual(result['hands'][k.NAMES[2]], 0.)

    def test_future_price_poison_and_array_truncation_invariance(self):
        stock, market = fixture({262: .1})
        expected = evaluate(stock, market)
        stock[463:] = [float('nan')] * len(stock[463:])
        market[463:] = [-1.] * len(market[463:])
        self.assertEqual(expected, evaluate(stock, market))
        self.assertEqual(expected, evaluate(stock[:463], market[:463]))
        self.assertEqual(expected, evaluate(stock, market, events=EVENTS + [k.Event('future', 470, '10-Q/A')]))
        self.assert_unavailable(evaluate(stock[:462], market[:462]), 'incomplete_formation_price_window')

    def test_price_unit_scale_invariance(self):
        stock, market = fixture({262: .01, 300: -.03})
        expected = evaluate(stock, market)
        scaled = evaluate([v * 1000 for v in stock], [v * .01 for v in market])
        self.assertAlmostEqual(expected['beta'], scaled['beta'], places=12)
        for name in k.NAMES:
            self.assertAlmostEqual(expected['hands'][name], scaled['hands'][name], places=12)

    def test_missing_warmup_and_prices_share_all_hand_mask(self):
        stock, market = fixture()
        self.assert_unavailable(evaluate(stock[1:], market[1:], price_start_index=1), 'insufficient_preformation_warmup')
        for index, suffix in ((0, 'beta'), (252, 'beta'), (253, 'formation'), (462, 'formation')):
            for value in (None, 0., -1., float('nan'), float('inf')):
                a = stock[:];a[index] = value
                with self.subTest(index=index, value=value):
                    result = evaluate(a, market)
                    self.assert_unavailable(result, 'invalid_required_' + suffix + '_price')
                    self.assertEqual(result['nuisance_beta_estimates_attempted'], 0)

    def test_constant_market_reports_failed_nuisance_attempt(self):
        stock, _ = fixture()
        result = evaluate(stock, [100.] * len(stock))
        self.assert_unavailable(result, 'zero_preformation_market_variance')
        self.assertEqual(result['nuisance_beta_estimates_attempted'], 1)

    def test_too_few_full_events_and_too_few_ordinary_days_are_unavailable(self):
        for events in (EVENTS[:1], [k.Event(str(a), a, '10-Q') for a in range(253, 454, 22)]):
            result = evaluate(events=events)
            self.assert_unavailable(result, 'insufficient_event_or_nonevent_days')
            self.assertEqual(result['nuisance_beta_estimates_attempted'], 0)

    def test_axis_and_price_type_errors_are_not_silent_neutrality(self):
        stock, market = fixture()
        with self.assertRaisesRegex(ValueError, 'axes differ'):
            evaluate(stock[:-1], market)
        for value in (True, '100'):
            modified = stock[:];modified[200] = value
            with self.assertRaisesRegex(ValueError, 'real scalar'):
                evaluate(modified, market)

    def test_inputs_unchanged_and_bounded_outputs(self):
        stock, market = fixture({262: 1., 300: -1.})
        events = list(EVENTS)
        before = deepcopy((stock, market, events))
        result = evaluate(stock, market, events)
        self.assertEqual(before, (stock, market, events))
        self.assertTrue(all(abs(value) <= 1 for value in result['hands'].values()))
        self.assertEqual(k.DOSE, .01)


if __name__ == '__main__':
    unittest.main()
