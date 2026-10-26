"""Synthetic-only checks for fixed monthly conditional-payoff measurement."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import monthly_transition175_kernel as kernel


def prices_from_factors(factors):
    if len(factors) != 60:
        raise ValueError('Fixture needs 60 factors')
    prices = np.empty(1261)
    prices[0] = 100.
    for i, factor in enumerate(factors):
        prices[i*21+1:(i+1)*21] = prices[i*21]
        prices[(i+1)*21] = prices[i*21]*factor
    return prices


def fixture():
    return prices_from_factors([2., 4., .5, .25]*15)


def panel_fixture(length=1380):
    # Synthetic weekdays only; the host supplies its pinned official calendar.
    days = pd.bdate_range('2000-01-03', periods=length)
    frame = pd.DataFrame(100., index=days, columns=['A', 'B'])
    frame.loc[days[:1261], 'A'] = fixture()
    calendar = days.strftime('%Y-%m-%d').tolist()
    return {'adj': frame}, [(calendar[1302], ['A', 'B'])], calendar


class MeasureTests(unittest.TestCase):
    def test_exact_dimensions(self):
        self.assertEqual(kernel.ADJUSTED_COUNT, 1261)
        self.assertEqual(kernel.ADJUSTED_START_OFFSET, 1302)
        self.assertEqual((kernel.BLOCK, kernel.BLOCK_COUNT, kernel.TRANSITION_COUNT), (21, 60, 59))
        self.assertEqual((kernel.SKIP, kernel.PRIOR_MASS, kernel.MIN_MATCHES), (42, 20, 10))

    def test_conditional_mean_with_matched_unconditional_prior(self):
        out = kernel.measure(fixture())
        self.assertTrue(out['available'])
        expected = [math.log(v) for v in [2., 4., .5, .25]*15]
        labels = expected[1:]
        states = [1, 1, -1, -1]*15
        matches = [j for j in range(1, 60) if states[j-1] == -1]
        matched = [labels[j-1] for j in matches]
        mean = math.fsum(labels)/59
        self.assertEqual(out['states'], states)
        self.assertEqual(out['current_state'], -1)
        self.assertEqual(out['matching_transition_indices'], matches)
        self.assertEqual(out['matching_count'], 29)
        np.testing.assert_allclose(out['log_returns'], expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(out['training_labels'], labels, atol=1e-14, rtol=0)
        self.assertAlmostEqual(out['control'], mean)
        self.assertAlmostEqual(out['candidate'], (sum(matched)+20*mean)/(29+20))
        self.assertEqual(out['candidate'], out['shrunken_payoff'])
        self.assertAlmostEqual(out['matching_mean'], sum(matched)/29)
        self.assertNotAlmostEqual(out['candidate'], out['matching_mean']-mean)

    def test_q1_is_not_a_training_label(self):
        prices = fixture()
        before = kernel.measure(prices)
        prices[0] *= .75
        after = kernel.measure(prices)
        self.assertNotEqual(after['log_returns'][0], before['log_returns'][0])
        for key in ('candidate', 'control', 'training_labels', 'states', 'matching_transition_indices'):
            self.assertEqual(after[key], before[key])

    def test_q60_is_last_label_but_not_training_predictor(self):
        prices = fixture()
        out = kernel.measure(prices)
        self.assertEqual(len(out['training_labels']), 59)
        self.assertEqual(out['training_labels'][-1], out['log_returns'][-1])
        self.assertEqual(out['matching_transition_indices'][-1], 59)
        self.assertNotIn(60, out['matching_transition_indices'])
        prices[-1] *= .8  # Retains negative query sign, changes the matured q60 label.
        changed = kernel.measure(prices)
        self.assertEqual(changed['matching_transition_indices'], out['matching_transition_indices'])
        self.assertNotEqual(changed['candidate'], out['candidate'])

    def test_zero_state_is_exactly_separate_from_positive_and_negative(self):
        prices = prices_from_factors([1., 2., .5]*20)
        prices[-1] = prices[-22]
        out = kernel.measure(prices)
        self.assertTrue(out['available'])
        self.assertEqual(out['current_state'], 0)
        self.assertEqual(out['matching_count'], 20)
        self.assertTrue(all(out['states'][j-1] == 0 for j in out['matching_transition_indices']))

    def test_constant_prices_valid_and_exactly_zero(self):
        out = kernel.measure(np.full(1261, 100.))
        self.assertTrue(out['available'])
        self.assertEqual(out['matching_count'], 59)
        self.assertEqual(out['matching_transition_indices'], list(range(1, 60)))
        self.assertEqual(out['states'], [0]*60)
        self.assertEqual(out['candidate'], 0.)
        self.assertEqual(out['control'], 0.)

    def test_minimum_ten_is_inclusive_and_both_arms_share_support(self):
        for n in (9, 10):
            out = kernel.measure(prices_from_factors([2.]*n+[.5]*(59-n)+[2.]))
            self.assertEqual(out['matching_count'], n)
            self.assertEqual(out['available'], n == 10)
            if n == 9:
                self.assertEqual(out['reason'], 'insufficient_matches')
                self.assertTrue(math.isnan(out['candidate']) and math.isnan(out['control']))
                self.assertIsNone(out['shrunken_payoff'])
                self.assertIsNotNone(out['unconditional_mean'])

    def test_no_matches_preserves_audit_without_dividing_by_zero(self):
        out = kernel.measure(prices_from_factors([2.]*59+[.5]))
        self.assertEqual(out['matching_count'], 0)
        self.assertEqual(out['matching_sum'], 0.)
        self.assertIsNone(out['matching_mean'])
        self.assertFalse(out['available'])

    def test_interior_prices_are_required_even_when_not_endpoints(self):
        for value in (np.nan, np.inf, -np.inf, 0., -1.):
            prices = fixture()
            prices[17] = value
            out = kernel.measure(prices)
            self.assertFalse(out['available'])
            self.assertEqual(out['reason'], 'invalid_adjusted_price')
            self.assertEqual(out['endpoint_indices'], [])

    def test_valid_interior_path_does_not_change_endpoint_statistic(self):
        prices = fixture()
        before = kernel.measure(prices)
        prices[17] *= 10.
        self.assertEqual(kernel.measure(prices), before)

    def test_nonnumeric_is_unavailable(self):
        prices = fixture().astype(object)
        prices[3] = 'not-a-price'
        self.assertEqual(kernel.measure(prices)['reason'], 'nonnumeric_adjusted_price')

    def test_wrong_shape_is_an_explicit_programming_error(self):
        for prices in ([1.]*1260, [1.]*1262, np.ones((1261, 1))):
            with self.assertRaises(ValueError):
                kernel.measure(prices)

    def test_endpoint_sign_uses_exact_comparison_for_one_ulp(self):
        prices = np.full(1261, 1.)
        for k in range(1, 61):
            prices[k*21] = np.nextafter(prices[(k-1)*21], np.inf)
        out = kernel.measure(prices)
        self.assertEqual(out['states'], [1]*60)
        self.assertTrue(all(r > 0 for r in out['log_returns']))

    def test_extreme_positive_prices_do_not_overflow_ratios(self):
        prices = np.ones(1261)
        prices[::21] = np.array([1e-300, 1e300]*30+[1e-300])
        out = kernel.measure(prices)
        self.assertTrue(out['available'])
        self.assertTrue(math.isfinite(out['candidate']))
        self.assertTrue(all(math.isfinite(r) for r in out['log_returns']))
        self.assertAlmostEqual(out['log_returns'][0], 600*math.log(10), places=10)

    def test_scale_invariance(self):
        before, after = kernel.measure(fixture()), kernel.measure(fixture()*8.)
        self.assertEqual(before, after)

    def test_measure_does_not_mutate_sources(self):
        prices = fixture()
        original = prices.copy()
        prices.flags.writeable = False
        kernel.measure(prices)
        np.testing.assert_array_equal(prices, original)


class BuildTests(unittest.TestCase):
    def test_window_offbyone_and_dated_audit(self):
        panel, events, calendar = panel_fixture()
        values, audit = kernel.build_features(panel, events, calendar)
        self.assertEqual(values['candidate'].shape, (1, 2))
        self.assertTrue(values['available'].all())
        row = audit['event_history'][0]
        self.assertEqual((row['adjusted_start'], row['adjusted_end']), (calendar[0], calendar[1260]))
        cell = row['cells'][0]
        self.assertEqual(cell['endpoint_indices'], list(range(0, 1261, 21)))
        self.assertEqual(cell['endpoint_dates'], calendar[:1261:21])
        self.assertEqual(cell['matching_predictor_end_dates'][-1], calendar[1239])
        self.assertEqual(cell['matching_label_end_dates'][-1], calendar[1260])
        self.assertEqual(audit['available_cells'], 2)
        self.assertEqual(audit['predictive_model_fits'], 0)
        self.assertFalse(audit['trading_eligibility_modified'])

    def test_skipped_42_and_future_poison_never_converted(self):
        panel, events, calendar = panel_fixture()
        expected = kernel.build_features(panel, events, calendar)
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].iloc[1261:, :] = 'FUTURE-MUST-NOT-BE-CONVERTED'
        actual = kernel.build_features(panel, events, calendar)
        for key in expected[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        self.assertEqual(actual[1], expected[1])

    def test_truncating_after_signal_preserves_measurement(self):
        panel, events, calendar = panel_fixture()
        expected = kernel.build_features(panel, events, calendar)
        short = {'adj': panel['adj'].iloc[:1303]}
        actual = kernel.build_features(short, events, calendar[:1303])
        self.assertEqual(actual[1], expected[1])
        np.testing.assert_array_equal(actual[0]['candidate'], expected[0]['candidate'])

    def test_prefix_preserves_prior_rows(self):
        panel, events, calendar = panel_fixture()
        events += [(calendar[1323], ['B', 'A']), (calendar[1344], ['A', 'B'])]
        full, audit = kernel.build_features(panel, events, calendar)
        prefix, prior = kernel.build_features(panel, events, calendar, limit=1)
        for key in full:
            np.testing.assert_array_equal(prefix[key], full[key][:1])
        self.assertEqual(prior['event_history'], audit['event_history'][:1])

    def test_limit_zero_and_empty_events(self):
        panel, events, calendar = panel_fixture()
        values, audit = kernel.build_features(panel, events, calendar, limit=0)
        self.assertEqual(values['candidate'].shape, (0, 2))
        self.assertEqual(audit['event_history'], [])
        values, audit = kernel.build_features(panel, [], calendar)
        self.assertEqual(values['candidate'].shape, (0, 0))

    def test_limit_must_be_integer_prefix(self):
        panel, events, calendar = panel_fixture()
        for limit in (True, -1, 2, .5):
            with self.assertRaises(ValueError):
                kernel.build_features(panel, events, calendar, limit=limit)

    def test_missing_official_session_does_not_compress_or_fill(self):
        panel, events, calendar = panel_fixture()
        panel['adj'] = panel['adj'].drop(pd.Timestamp(calendar[17]))
        values, audit = kernel.build_features(panel, events, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['reason_counts'], {'missing_official_session': 2})
        self.assertEqual(audit['event_history'][0]['missing_official_sessions'], [calendar[17]])

    def test_missing_symbol_only_invalidates_that_security(self):
        panel, events, calendar = panel_fixture()
        events[0] = (events[0][0], ['A', 'MISSING'])
        values, audit = kernel.build_features(panel, events, calendar)
        np.testing.assert_array_equal(values['available'], [[True, False]])
        self.assertIsNone(audit['event_history'][0]['cells'][1]['candidate'])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'], 'missing_symbol')

    def test_missing_cell_is_not_filled_from_another_security(self):
        panel, events, calendar = panel_fixture()
        panel['adj'].iloc[17, 0] = np.nan
        values, audit = kernel.build_features(panel, events, calendar)
        np.testing.assert_array_equal(values['available'], [[False, True]])
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'invalid_adjusted_price')

    def test_history_must_be_full_window(self):
        panel, events, calendar = panel_fixture()
        events[0] = (calendar[1301], ['A', 'B'])
        values, audit = kernel.build_features(panel, events, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['reason_counts'], {'insufficient_history': 2})
        self.assertIsNone(audit['event_history'][0]['adjusted_start'])

    def test_extra_nonofficial_panel_row_cannot_shift_offsets(self):
        panel, events, calendar = panel_fixture()
        before = kernel.build_features(panel, events, calendar)
        panel['adj'].loc[pd.Timestamp('2000-01-08')] = [999., 888.]
        panel['adj'] = panel['adj'].sort_index()
        after = kernel.build_features(panel, events, calendar)
        self.assertEqual(before[1], after[1])

    def test_reordered_symbols_follow_event_identity(self):
        panel, events, calendar = panel_fixture()
        normal, _ = kernel.build_features(panel, events, calendar)
        reverse, audit = kernel.build_features(panel, [(events[0][0], ['B', 'A'])], calendar)
        np.testing.assert_array_equal(normal['candidate'][:, ::-1], reverse['candidate'])
        self.assertEqual([c['symbol'] for c in audit['event_history'][0]['cells']], ['B', 'A'])

    def test_calendar_and_event_contracts(self):
        panel, events, calendar = panel_fixture()
        for bad in (calendar[::-1], calendar+[calendar[-1]], []):
            with self.assertRaises(ValueError):
                kernel.build_features(panel, events, bad)
        for bad in ([events[0], events[0]], [(events[0][0], ['A', 'A'])],
                    [(events[0][0], ['A', 'B']), (calendar[1323], ['A'])],
                    [('2000-01-08', ['A', 'B'])], [(events[0][0], 'AB')]):
            with self.assertRaises((TypeError, ValueError)):
                kernel.build_features(panel, bad, calendar)

    def test_panel_contracts(self):
        panel, events, calendar = panel_fixture()
        for bad in ({}, dict(panel, volume=panel['adj']), {'adj': panel['adj'].iloc[::-1]},
                    {'adj': panel['adj'].set_axis(['A', 'A'], axis=1)}):
            with self.assertRaises(ValueError):
                kernel.build_features(bad, events, calendar)

    def test_sources_not_mutated(self):
        panel, events, calendar = panel_fixture()
        original = copy.deepcopy((panel, events, calendar))
        kernel.build_features(panel, events, calendar)
        pd.testing.assert_frame_equal(panel['adj'], original[0]['adj'])
        self.assertEqual(events, original[1])
        self.assertEqual(calendar, original[2])


class IndependentCrossChecks(unittest.TestCase):
    def test_production_fixture_against_decimal_oracle(self):
        import monthly_transition175_independent as independent
        panel, events, calendar = panel_fixture()
        values, audit = kernel.build_features(panel, events, calendar)
        self.assertTrue(independent.verify_features(panel, events, calendar, values, audit)['passed'])

    def test_independently_authored_fixture_against_decimal_oracle(self):
        import monthly_transition175_independent as independent
        from test_monthly_transition175_independent import fixture as other_fixture
        panel, events, calendar = other_fixture()
        values, audit = kernel.build_features(panel, events, calendar)
        self.assertTrue(independent.verify_features(panel, events, calendar, values, audit)['passed'])

    def test_support_threshold_against_decimal_oracle(self):
        import monthly_transition175_independent as independent
        for n in (0, 9, 10, 59):
            panel, events, calendar = panel_fixture()
            panel['adj'].iloc[:1261, 0] = prices_from_factors([2.]*n+[.5]*(59-n)+[2.])
            values, audit = kernel.build_features(panel, events, calendar)
            self.assertTrue(independent.verify_features(panel, events, calendar, values, audit)['passed'])

    def test_source_failures_against_independent_oracle(self):
        import monthly_transition175_independent as independent
        for kind in ('history', 'symbol', 'session', 'invalid', 'nonnumeric'):
            panel, events, calendar = panel_fixture()
            if kind == 'history':
                events[0] = (calendar[1301], ['A', 'B'])
            elif kind == 'symbol':
                events[0] = (events[0][0], ['A', 'MISSING'])
            elif kind == 'session':
                panel['adj'] = panel['adj'].drop(pd.Timestamp(calendar[17]))
            elif kind == 'invalid':
                panel['adj'].iloc[17, 0] = np.nan
            else:
                panel['adj'] = panel['adj'].astype(object)
                panel['adj'].iloc[17, 0] = 'not-a-price'
            values, audit = kernel.build_features(panel, events, calendar)
            self.assertTrue(independent.verify_features(panel, events, calendar, values, audit)['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
