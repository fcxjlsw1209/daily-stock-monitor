"""Synthetic-only exact arithmetic, availability and causal-prefix tests."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, localcontext
import json
import unittest

import numpy as np
import operating_margin172_kernel as kernel


def record(symbol='MSFT', known='2016-08-01', report_id='synthetic-1', **overrides):
    result = dict(symbol=symbol, fiscal_year=2016, report_id=report_id,
                  release_date=(date.fromisoformat(known) - timedelta(days=1)).isoformat(),
                  known_session=known,
                  current=dict(revenue='120', operating_income='24', days=366),
                  comparative=dict(revenue='100', operating_income='10', days=365),
                  usable=True, hold_reasons=[])
    result.update(overrides)
    return result


def one(source=None, day='2016-08-17', symbol='MSFT'):
    return kernel.build_features([record() if source is None else source], [(day, [symbol])])


class FutureGuard(dict):
    """Fail on reading any not-yet-known facts or qualification metadata."""
    guarded = {'current', 'comparative', 'usable', 'hold_reasons'}

    def __getitem__(self, key):
        if key in self.guarded:
            raise AssertionError('Future field read: ' + key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.guarded:
            raise AssertionError('Future field read: ' + key)
        return super().get(key, default)


class KernelTests(unittest.TestCase):
    def assert_signs(self, source, candidate, control):
        values, audit = one(source)
        self.assertTrue(values['available'][0, 0])
        self.assertEqual(values['candidate'][0, 0], candidate)
        self.assertEqual(values['control'][0, 0], control)
        self.assertEqual(audit['reason_counts'], {'available': 1})
        return values, audit

    def assert_unavailable(self, source, reason):
        values, audit = one(source)
        self.assertFalse(values['available'][0, 0])
        self.assertTrue(np.isnan(values['candidate'][0, 0]))
        self.assertTrue(np.isnan(values['control'][0, 0]))
        self.assertIn(reason, audit['event_history'][0]['cells'][0]['failure_reasons'])
        return audit

    def test_both_formulas_use_same_report_annual_pair(self):
        self.assert_signs(record(), 1, 1)
        source = record()
        source['current'].update(revenue='50', operating_income='-8', days=365)
        source['comparative'].update(revenue='100', operating_income='-10', days=365)
        self.assert_signs(source, -1, 1)

    def test_signed_and_zero_income_are_valid(self):
        for current, prior, expected in (('-5', '-10', 1), ('-10', '-5', -1),
                                         ('0', '-1', 1), ('-1', '0', -1), ('0', '0', 0)):
            source = record()
            for side, amount in (('current', current), ('comparative', prior)):
                source[side].update(revenue='100', operating_income=amount, days=365)
            with self.subTest(current=current, prior=prior):
                self.assert_signs(source, expected, expected)

    def test_each_zero_is_exact_without_forcing_other_arm_to_zero(self):
        source = record()
        source['current'].update(revenue='200', operating_income='20', days=365)
        source['comparative'].update(revenue='100', operating_income='10', days=365)
        self.assert_signs(source, 0, 1)
        source['current'].update(revenue='1000', operating_income='364', days=364)
        source['comparative'].update(revenue='1000', operating_income='365', days=365)
        self.assert_signs(source, -1, 0)

    def test_tiny_difference_survives_low_decimal_precision(self):
        source = record()
        source['current'].update(revenue='1e100', operating_income=str(10 ** 100 + 1), days=365)
        source['comparative'].update(revenue='1e100', operating_income='1e100', days=365)
        with localcontext() as context:
            context.prec = 2
            self.assert_signs(source, 1, 1)

    def test_common_positive_monetary_unit_scaling_is_invariant(self):
        source = record()
        source['current'].update(revenue='50.005', operating_income='-8.004')
        source['comparative'].update(revenue='100.0001', operating_income='-10.002')
        expected, _ = one(source)
        for factor in ('0.001', '1000', '1e300', '1e-300'):
            changed = deepcopy(source)
            with localcontext() as context:
                context.prec = 1000
                for side in ('current', 'comparative'):
                    for field in ('revenue', 'operating_income'):
                        changed[side][field] = str(Decimal(changed[side][field]) * Decimal(factor))
            got, _ = one(changed)
            for key in expected:
                np.testing.assert_array_equal(got[key], expected[key])

    def test_missing_or_invalid_days_jointly_disable_both_features(self):
        for side in ('current', 'comparative'):
            for days in (None, True, False, 365.0, '365', 0, -1, 329, 401):
                source = record()
                source[side]['days'] = days
                reason = ('missing_' if days is None else 'invalid_') + side + '_days'
                with self.subTest(side=side, days=days):
                    self.assert_unavailable(source, reason)
        for days in (330, 400):
            source = record()
            source['current']['days'] = source['comparative']['days'] = days
            self.assert_signs(source, 1, 1)

    def test_invalid_amounts_are_unavailable_not_zero(self):
        for side in ('current', 'comparative'):
            for field in ('revenue', 'operating_income'):
                for value in (None, True, 1.25, 'NaN', 'Infinity', '-Infinity', ' 2', '2 ', '1,000', '—'):
                    source = record()
                    source[side][field] = value
                    with self.subTest(side=side, field=field, value=value):
                        self.assert_unavailable(source, 'invalid_' + side + '_' + field)
            for value in ('0', '-1', '-0'):
                source = record()
                source[side]['revenue'] = value
                self.assert_unavailable(source, 'invalid_' + side + '_revenue')

    def test_first_report_missing_comparative_duration_has_no_special_fallback(self):
        source = record(symbol='AAPL')
        source['comparative']['days'] = None
        values, audit = one(source, symbol='AAPL')
        self.assertFalse(values['available'][0, 0])
        self.assertEqual(audit['reason_counts'], {'missing_comparative_days': 1})

    def test_initial_event_is_not_special_cased_by_kernel(self):
        values, audit = one()
        self.assertTrue(values['available'][0, 0])
        self.assertEqual(audit['event_history'][0]['date'], '2016-08-17')
        self.assertFalse(audit['initial_allocation_neutralized'])

    def test_latest_known_report_selected_before_qualification_no_old_fallback(self):
        old = record(known='2016-07-01', report_id='old')
        new = record(known='2016-08-01', report_id='new', usable=False,
                     hold_reasons=['missing_comparative_start'], current=object(), comparative=object())
        values, audit = kernel.build_features([new, old], [('2016-08-17', ['MSFT'])])
        self.assertFalse(values['available'][0, 0])
        cell = audit['event_history'][0]['cells'][0]
        self.assertEqual(cell['selected_report_id'], 'new')
        self.assertEqual(cell['reason'], 'latest_report_unusable')
        self.assertEqual(cell['hold_reasons'], ['missing_comparative_start'])
        self.assertFalse(cell['input_period']['current']['days_inspected'])

    def test_inherited_holds_do_not_override_explicit_usable_flag(self):
        _, audit = self.assert_signs(record(hold_reasons=['retrospective_retrieval', 'unknown_iso']), 1, 1)
        self.assertEqual(audit['inherited_hold_reason_counts'], {'retrospective_retrieval': 1, 'unknown_iso': 1})

    def test_never_select_or_read_future_facts(self):
        early = record()
        future = FutureGuard(record(known='2016-10-01', report_id='future'))
        expected, _ = one(early)
        got, _ = kernel.build_features([future, early], [('2016-08-17', ['MSFT'])])
        for key in expected:
            np.testing.assert_array_equal(got[key], expected[key])
        values, audit = kernel.build_features([future], [('2016-08-17', ['MSFT'])])
        self.assertFalse(values['available'][0, 0])
        self.assertEqual(audit['reason_counts'], {'no_known_report': 1})

    def test_latest_means_information_clock_not_fiscal_year_or_record_order(self):
        old = record(known='2016-07-01', report_id='old', fiscal_year=2016)
        correction = record(known='2016-08-01', report_id='correction', fiscal_year=2015)
        correction['current']['operating_income'] = '-100'
        for records in ([old, correction], [correction, old]):
            values, audit = kernel.build_features(records, [('2016-08-17', ['MSFT'])])
            self.assertEqual(values['candidate'][0, 0], -1)
            self.assertEqual(audit['event_history'][0]['cells'][0]['selected_report_id'], 'correction')

    def test_age_400_inclusive_401_stale_uses_known_session(self):
        source = record(known='2016-08-01')
        source['release_date'] = '2016-07-01'
        known = date.fromisoformat(source['known_session'])
        events = [((known + timedelta(days=days)).isoformat(), ['MSFT']) for days in (0, 400, 401)]
        values, audit = kernel.build_features([source], events)
        np.testing.assert_array_equal(values['available'][:, 0], [True, True, False])
        self.assertTrue(np.isnan(values['candidate'][2, 0]))
        self.assertEqual([h['cells'][0]['age_days'] for h in audit['event_history']], [0, 400, 401])
        self.assertEqual(audit['event_history'][2]['reasons'], ['latest_report_stale'])

    def test_stale_values_and_qualification_are_not_read(self):
        source = FutureGuard(record(known='2015-01-01'))
        values, audit = one(source)
        self.assertFalse(values['available'][0, 0])
        self.assertEqual(audit['reason_counts'], {'latest_report_stale': 1})

    def test_events_preserve_member_order_scope_and_matrix_shape(self):
        records = [record(symbol=symbol, report_id=symbol) for symbol in ('MSFT', 'ABBV', 'AAPL')]
        events = [('2016-08-17', ['MSFT', 'ZZZ', 'ABBV', 'AAPL']),
                  ('2016-08-18', ['AAPL', 'ABBV', 'ZZZ', 'MSFT'])]
        values, audit = kernel.build_features(records, events)
        self.assertEqual(values['candidate'].shape, (2, 4))
        self.assertEqual(values['available'].dtype, np.dtype(bool))
        np.testing.assert_array_equal(values['available'], [[True, False, True, True], [True, True, False, True]])
        self.assertEqual(audit['reason_counts'], {'available': 6, 'missing_symbol': 2})

    def test_prefix_and_future_mutation_invariance(self):
        records = [record(), record(known='2016-10-01', report_id='later')]
        records[1]['current']['operating_income'] = '-20'
        events = [(day, ['MSFT', 'ABBV']) for day in ('2016-08-17', '2016-09-01', '2016-10-03')]
        full, audit = kernel.build_features(records, events)
        for limit in range(4):
            partial, info = kernel.build_features(records, events, limit=limit)
            for key in full:
                np.testing.assert_array_equal(partial[key], full[key][:limit])
            self.assertEqual(info['event_history'], audit['event_history'][:limit])
        poisoned = [records[0], FutureGuard(records[1])]
        partial, info = kernel.build_features(poisoned, events, limit=2)
        for key in full:
            np.testing.assert_array_equal(partial[key], full[key][:2])
        self.assertEqual(info['event_history'], audit['event_history'][:2])

    def test_audit_has_exhaustive_counts_provenance_and_period_inputs(self):
        source = record()
        source['current'].update(revenue='0', days=None)
        source['comparative']['operating_income'] = 'NaN'
        values, audit = kernel.build_features([source], [('2016-08-17', ['MSFT', 'ABBV', 'QQQ'])])
        self.assertEqual(sum(audit['reason_counts'].values()), values['available'].size)
        self.assertEqual(audit['available_cells'] + audit['unavailable_cells'], 3)
        self.assertEqual(audit['failure_reason_counts'], {
            'missing_current_days': 1, 'invalid_current_revenue': 1,
            'invalid_comparative_operating_income': 1, 'no_known_report': 1, 'missing_symbol': 1})
        cell = audit['event_history'][0]['cells'][0]
        self.assertEqual(cell['selected_report_id'], source['report_id'])
        self.assertEqual(cell['known_session'], source['known_session'])
        self.assertEqual(cell['input_period']['comparative'], dict(fiscal_year=2015, days=365, days_inspected=True))
        json.dumps(audit, allow_nan=False)

    def test_input_records_and_events_unchanged(self):
        records, events = [record()], [('2016-08-17', ['MSFT'])]
        before = deepcopy((records, events))
        kernel.build_features(records, events)
        self.assertEqual((records, events), before)

    def test_empty_events_and_zero_prefix_have_explicit_shapes(self):
        values, audit = kernel.build_features([], [])
        self.assertEqual(values['candidate'].shape, (0, 0))
        self.assertEqual(audit['reason_counts'], {})
        values, _ = kernel.build_features([FutureGuard(record())], [('2016-08-17', ['MSFT', 'ABBV'])], limit=0)
        self.assertEqual(values['candidate'].shape, (0, 2))

    def test_rejects_noninteger_bool_and_out_of_range_prefixes(self):
        for limit in (True, False, -1, 2, 1.0, '1', np.int64(1)):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'prefix'):
                kernel.build_features([record()], [('2016-08-17', ['MSFT'])], limit=limit)

    def test_rejects_event_order_duplicates_dates_members_and_width(self):
        malformed = [
            [('2016-08-18', ['MSFT']), ('2016-08-17', ['MSFT'])],
            [('2016-08-17', ['MSFT']), ('2016-08-17', ['MSFT'])],
            [('2016-08-17', ['MSFT', 'MSFT'])], [('2016-08-17', [])],
            [('2016-08-17', ['MSFT']), ('2016-08-18', ['MSFT', 'ABBV'])],
            [('2016-8-17', ['MSFT'])], [('2016-08-17T00:00:00', ['MSFT'])],
            [('2016-08-17', 'MSFT')], [('2016-08-17', [''])],
            [('2016-08-17', [' MSFT'])], [('2016-08-17', [None])], [('2016-08-17',)],
        ]
        for events in malformed:
            with self.subTest(events=events), self.assertRaises(ValueError):
                kernel.build_features([record()], events)

    def test_duplicate_identity_and_information_clock_are_ambiguous(self):
        original = record()
        for other in (deepcopy(original), record(report_id='same-clock'),
                      record(known='2016-09-01', report_id=original['report_id'])):
            with self.subTest(other=other['report_id']), self.assertRaises(ValueError):
                kernel.build_features([original, other], [('2016-08-17', ['MSFT'])])

    def test_rejects_invalid_selection_metadata_and_qualification_schema(self):
        for key, value in (('symbol', 'ZZZ'), ('fiscal_year', True), ('fiscal_year', '2016'),
                           ('report_id', ''), ('known_session', 'invalid'),
                           ('release_date', '2016-08-02'), ('usable', 1), ('hold_reasons', 'hold')):
            source = record()
            source[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                one(source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
