"""Synthetic-only checks for the unregistered pure peer earnings factor."""
from copy import deepcopy
from datetime import date, timedelta
import json
import math
import unittest

import numpy as np

import monthly_peer_earnings139_factor as f


def fixture(months=2, width=6):
    sessions = [(date(2000, 1, 1) + timedelta(days=i)).isoformat() for i in range(200)]
    symbols = ['A', 'B', 'C', 'D', 'X', 'ETF'][:width]
    event = np.tile(np.array([.02, .04, -.03, .01, .08, .9])[:width], (months, 1))
    return dict(meta=dict(dates=[sessions[100 + i * 20] for i in range(months)],
                         symbols=[symbols[:] for _ in range(months)], etfs=['ETF']),
        event10=event, matched_recent10=event * 2., valid10=np.ones((months, width), dtype=bool),
        anchor_sessions=np.array([[sessions[80 + i * 20]] * width for i in range(months)], dtype=object),
        event_ids=np.array([['event-' + s for s in symbols] for _ in range(months)], dtype=object),
        official_sessions=sessions,
        sector_benchmarks=np.tile(np.array(['S', 'S', 'S', 'S', 'T', None], dtype=object)[:width], (months, 1)),
        issuer_keys=np.tile(np.array(['issuer-' + s for s in symbols], dtype=object), (months, 1)))


def result(data):
    return f.build_peer_features(**data)


class PeerFeatureTests(unittest.TestCase):
    def test_synthetic_production_shape(self):
        sessions = [(date(2000, 1, 1) + timedelta(days=i)).isoformat() for i in range(500)]
        names = ['S' + str(i) for i in range(32)] + ['E' + str(i) for i in range(17)]
        shape = (122, 49)
        d = dict(meta=dict(dates=[sessions[100 + i * 2] for i in range(122)],
                          symbols=[names[:] for _ in range(122)], etfs=names[32:]),
            event10=np.full(shape, .02), matched_recent10=np.full(shape, -.01),
            valid10=np.ones(shape, dtype=bool), official_sessions=sessions,
            anchor_sessions=np.array([[sessions[80 + i * 2]] * 49 for i in range(122)], dtype=object),
            event_ids=np.tile(np.array(['event-' + s for s in names], dtype=object), (122, 1)),
            issuer_keys=np.tile(np.array(['issuer-' + s for s in names], dtype=object), (122, 1)),
            sector_benchmarks=np.full(shape, 'SECTOR', dtype=object))
        got = result(d)
        self.assertEqual(got['arrays']['common_available'].shape, shape)
        self.assertTrue(got['arrays']['common_available'][:, :32].all())
        self.assertFalse(got['arrays']['common_available'][:, 32:].any())
        self.assertTrue((got['arrays']['peer_count'][:, :32] == 31).all())
        self.assertEqual(len(got['audit']['cells']), 122 * 49)

    def test_formulas_common_mask_self_exclusion_and_no_rank(self):
        d = fixture(); got = result(d); a = got['arrays']
        self.assertEqual(set(a), set(f.EXTRA))
        np.testing.assert_array_equal(a['common_available'][0], [True, True, True, True, False, False])
        self.assertEqual(a['peer_count'][0, 0], 3)
        self.assertAlmostEqual(a['raw_peer_event10'][0, 0], (.04 - .03 + .01) / 3)
        self.assertAlmostEqual(a['raw_peer_recent10'][0, 0], (.08 - .06 + .02) / 3)
        self.assertAlmostEqual(a['raw_negative_own_event10'][0, 0], -.02 / 3)
        for raw, hand in zip(f.RAW_NAMES, f.HAND_NAMES):
            self.assertAlmostEqual(a[hand][0, 0], .5 * math.tanh(a[raw][0, 0] / .05))
            np.testing.assert_array_equal(np.isfinite(a[hand]), a['common_available'])
        cell = got['audit']['cells'][0]
        self.assertEqual(cell['peer_symbols'], ['B', 'C', 'D'])
        self.assertNotIn(cell['issuer_key'], cell['peer_issuer_keys'])
        self.assertEqual(cell['peer_event_ids'], ['event-B', 'event-C', 'event-D'])
        json.dumps(got['audit'], allow_nan=False)

    def test_minimum_two_other_issuers(self):
        d = fixture(); d['sector_benchmarks'][:, 2:4] = 'T'
        got = result(d)
        self.assertEqual(got['arrays']['peer_count'][0, 0], 1)
        self.assertFalse(got['arrays']['common_available'][0, 0])
        self.assertEqual(got['audit']['cells'][0]['peer_symbols'], ['B'])
        self.assertEqual(got['audit']['cells'][0]['reason'], 'insufficient_peer_issuers')
        for key in f.RAW_NAMES + f.HAND_NAMES:
            self.assertTrue(np.isnan(got['arrays'][key][0, 0]))

    def test_target_own_missing_gates_every_arm(self):
        d = fixture(); d['valid10'][:, 0] = False
        d['event10'][:, 0] = np.nan; d['matched_recent10'][:, 0] = np.nan
        d['anchor_sessions'][:, 0] = None; d['event_ids'][:, 0] = None
        got = result(d)
        self.assertFalse(got['arrays']['source_available'][:, 0].any())
        self.assertTrue(got['arrays']['common_available'][:, 1:4].all())
        self.assertEqual(got['arrays']['peer_count'][0, 1], 2)
        self.assertEqual(got['audit']['cells'][0]['reason'], 'missing_event')

    def test_unknown_mappings_and_etf_fallback(self):
        d = fixture(); d['issuer_keys'][:, 0] = None; d['sector_benchmarks'][:, 1] = None
        got = result(d)
        self.assertEqual([c['reason'] for c in got['audit']['cells'][:2]], ['unknown_issuer', 'unknown_sector'])
        self.assertEqual(got['audit']['cells'][5]['reason'], 'etf')
        self.assertFalse(got['arrays']['source_available'][0, [0, 1, 5]].any())
        self.assertTrue(np.isnan(got['arrays']['hand_peer_event10'][0, [0, 1, 5]]).all())

    def test_duplicate_issuer_fails_even_if_second_is_missing(self):
        d = fixture(); d['issuer_keys'][:, 1] = d['issuer_keys'][:, 0]
        with self.assertRaisesRegex(ValueError, 'duplicate issuer'):
            result(d)
        d['valid10'][:, 1] = False; d['event10'][:, 1] = np.nan; d['matched_recent10'][:, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'duplicate issuer'):
            result(d)

    def test_official_session_age_boundaries(self):
        for age, expected in ((9, False), (10, True), (63, True), (64, False)):
            d = fixture(months=1); d['anchor_sessions'][0, 0] = d['official_sessions'][100 - age]
            got = result(d)
            self.assertEqual(got['arrays']['common_available'][0, 0], expected)
            self.assertEqual(got['audit']['cells'][0]['age_sessions'], age)
        d = fixture(months=1)
        d['anchor_sessions'][0, 0] = d['official_sessions'][101]
        with self.assertRaisesRegex(ValueError, 'Future'):
            result(d)

    def test_calendar_gaps_use_session_positions(self):
        d = fixture(months=1)
        d['official_sessions'] = [(date(2000, 1, 1) + timedelta(days=i * 3)).isoformat() for i in range(200)]
        d['meta']['dates'] = [d['official_sessions'][100]]
        d['anchor_sessions'][:] = d['official_sessions'][90]
        got = result(d)
        self.assertEqual(got['audit']['cells'][0]['age_sessions'], 10)
        self.assertTrue(got['arrays']['common_available'][0, 0])

    def test_exact_zero_is_not_missing_and_positive_zero(self):
        d = fixture(); d['event10'][:] = 0.; d['matched_recent10'][:] = -0.
        got = result(d)
        for key in f.RAW_NAMES + f.HAND_NAMES:
            values = got['arrays'][key][got['arrays']['common_available']]
            self.assertTrue((values == 0.).all()); self.assertFalse(np.signbit(values).any())
        self.assertTrue(got['arrays']['common_available'][:, :4].all())

    def test_prefix_future_data_do_not_change_earlier_features(self):
        full = fixture(); original = result(full)
        prefix = deepcopy(full)
        for key in ('event10', 'matched_recent10', 'valid10', 'anchor_sessions', 'event_ids', 'sector_benchmarks', 'issuer_keys'):
            prefix[key] = prefix[key][:1]
        prefix['meta']['dates'] = prefix['meta']['dates'][:1]
        prefix['meta']['symbols'] = prefix['meta']['symbols'][:1]
        prefix['official_sessions'] = prefix['official_sessions'][:101]
        got = result(prefix)
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], original['arrays'][key][:1])
        full['event10'][1] *= -7; full['matched_recent10'][1] *= 13
        poisoned = result(full)
        for key in f.EXTRA:
            np.testing.assert_array_equal(poisoned['arrays'][key][:1], original['arrays'][key][:1])
        self.assertEqual(got['audit']['cells'], original['audit']['cells'][:6])

    def test_no_input_mutation_or_trading_eligibility_filter(self):
        d = fixture(); d['meta']['eligible'] = [[False] * 6] * 2
        old = deepcopy(d); got = result(d)
        self.assertTrue(got['arrays']['common_available'][:, :4].all())
        for key in d:
            if isinstance(d[key], np.ndarray):
                np.testing.assert_array_equal(d[key], old[key])
                self.assertEqual(d[key].tobytes(), old[key].tobytes())
            else:
                self.assertEqual(d[key], old[key])

    def test_candidate_permutation_invariance(self):
        d = fixture(); original = result(d); order = [3, 1, 5, 0, 4, 2]
        p = deepcopy(d)
        for key in ('event10', 'matched_recent10', 'valid10', 'anchor_sessions', 'event_ids', 'sector_benchmarks', 'issuer_keys'):
            p[key] = p[key][:, order]
        p['meta']['symbols'] = [[row[k] for k in order] for row in p['meta']['symbols']]
        got = result(p)
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], original['arrays'][key][:, order])

    def test_invalid_source_contract_fails(self):
        for key, value in (('event10', True), ('event10', np.inf), ('matched_recent10', np.nan),
                           ('anchor_sessions', '2000-02-30'), ('anchor_sessions', None),
                           ('event_ids', None), ('issuer_keys', ''), ('sector_benchmarks', ' S')):
            d = fixture()
            if value is True:
                d[key] = np.full((2, 6), True)
            else:
                d[key][0, 0] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                result(d)
        d = fixture(); d['valid10'] = d['valid10'].astype(int)
        with self.assertRaises(ValueError): result(d)
        d = fixture(); d['official_sessions'][1] = d['official_sessions'][0]
        with self.assertRaises(ValueError): result(d)
        d = fixture(); d['meta']['dates'][0] = '1999-01-01'
        with self.assertRaises(ValueError): result(d)
        d = fixture(); d['meta']['symbols'][0][1] = 'A'
        with self.assertRaises(ValueError): result(d)


if __name__ == '__main__':
    unittest.main()
