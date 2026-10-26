"""Synthetic-only independent preparation tests; no producer or market I/O."""
from copy import deepcopy
from collections import Counter
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
import unittest

import numpy as np
import annual_lag158_independent_prepare as review


def fixture():
    start = date(2011, 1, 1)
    sessions = [(start+timedelta(days=i)).isoformat() for i in range(5500)
                if (start+timedelta(days=i)).weekday() < 5]
    closes = {s: review.ns(s+'T20:00:00Z') for s in sessions}
    signals = [sessions[1800+10*i] for i in range(122)]
    names = ['KO', 'WMT'] + ['STOCK'+str(i) for i in range(30)] + ['ETF'+str(i) for i in range(17)]
    meta = dict(dates=signals, ends=signals[:], symbols=[names[:] for _ in signals], etfs=names[-17:])
    rows, events = [], {}
    source = np.zeros((122, 49), bool)
    source[:115, :2] = True
    eligible = np.ones((122, 49), bool)
    for i, signal in enumerate(signals):
        cut = sessions[1800+10*i-42]
        for j, symbol in enumerate(names):
            ids = []
            if j < 2:
                for previous in (False, True):
                    year = int(cut[:4])-1-int(previous)
                    report = date(year, 1, 1)
                    lag = ((50 if previous else 30) if j == 0 else (70 if previous else 40))
                    key = '%d:%d:%d' % (i, j, previous)
                    events[key] = dict(event_id=key, cik=review.ISSUERS[symbol], form='10-K',
                        accessionNumber='%010d-%02d-%06d' % (j+1, year % 100, i*2+int(previous)),
                        primaryDocument='annual.htm', reportDate=report.isoformat(),
                        filingDate=(report+timedelta(days=lag)).isoformat())
                    ids.append(key)
            admitted = bool(source[i, j])
            row = dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=signal,
                cik=review.ISSUERS.get(symbol), is_etf=j >= 32, tier2=admitted, tier3=False,
                normal_period_admission=False, selected_event_ids=ids, known_amendment_event_ids=[],
                information_cutoff_date=cut, information_cutoff=cut+'T20:00:00Z', information_cutoff_ns=closes[cut],
                original_decision_cutoff=signal+'T20:00:00Z', reasons=[review.MISSING] if admitted else ['NO_SOURCE'])
            rows.append(dict(original_row=row, body_pair_context_verified=admitted,
                body_context_source=symbol if admitted else None, old_admission_flags_changed=False,
                conditional_annual_pair_admitted=admitted, remaining_row_reasons=[] if admitted else ['NO_SOURCE']))
    return dict(meta=meta, rows=rows, event_metadata=events, source_mask=source,
                eligible=eligible, calendar=dict(sessions=sessions, closes=closes))


def score_fixture():
    baseline = np.array([[.16, .12, .08, .07, -.1], [.15, .13, .10, .09, -.1]])
    shape = baseline.shape
    arrays = dict(baseline=baseline, core=np.array([[.10, .08, .05, .04, -.1], [.10, .08, .06, .05, -.1]]),
        eligible=np.ones(shape, bool), core_weight=np.array([.5, .52]), state_weak_narrow=np.array([False, True]),
        state_calm10=np.array([True, False]), pre_score=baseline.copy(), effects=np.zeros(shape),
        forecast_available=np.zeros(shape, bool), rank_log_excess=np.zeros(shape), old_ml4=np.zeros(shape),
        extension_available=np.zeros(shape, bool), model_available=np.zeros(shape, bool),
        raw_local_constant=np.zeros(shape), raw_local_linear=np.zeros(shape), raw_source_log=np.zeros(shape))
    meta = dict(symbols=[list('ABCDE'), list('ABCDE')], etfs=['E'],
        dates=['2020-03-02', '2020-04-01'], ends=['2020-03-31', '2020-04-30'])
    ranks = np.array([[.5, -.5, 0, 0, 0], [-.5, .5, 0, 0, 0]])
    return arrays, meta, dict(current_lag_rank=ranks, lag_change_rank=-ranks)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture()

    def setUp(self):
        self.data = deepcopy(self.base)

    def build(self, n=1):
        return review.features_oracle(**self.data, n=n)

    def event(self, candidate=0, previous=False):
        key = self.data['rows'][candidate]['original_row']['selected_event_ids'][int(previous)]
        return self.data['event_metadata'][key]

    def test_independent_known_lags_and_opposed_rank_directions(self):
        a = self.build()['arrays']
        np.testing.assert_array_equal(a[review.RAW[0]][0, :2], [-30., -40.])
        np.testing.assert_array_equal(a[review.RAW[1]][0, :2], [20., 30.])
        np.testing.assert_array_equal(a[review.RANKS[0]][0, :2], [.5, -.5])
        np.testing.assert_array_equal(a[review.RANKS[1]][0, :2], [-.5, .5])
        self.assertTrue(np.isnan(a[review.RAW[0]][0, 2:]).all())
        self.assertTrue((a[review.RANKS[0]][0, 2:] == 0).all())

    def test_full_denominator_and_repeated_pair_counts(self):
        got = self.build(None)
        self.assertEqual(got['arrays']['common_available'].shape, (122, 49))
        self.assertEqual(got['arrays']['common_available'].sum(), 230)
        counts = review.coverage_oracle(self.data['rows'], got)
        self.assertEqual(counts['all122']['candidate_cells'], 5978)
        self.assertEqual(counts['all122']['unique_source_pairs'], 230)
        self.assertEqual(counts['all122']['months_with_two_usable'], 115)
        # Reusing the same IDs is one pair, not two independent annual events.
        rows = deepcopy(self.data['rows'])
        rows[49]['original_row']['selected_event_ids'] = rows[0]['original_row']['selected_event_ids'][:]
        self.assertEqual(review.coverage_oracle(rows, got)['all122']['unique_source_pairs'], 229)

    def test_current_is_highest_report_date_not_id_or_filing_order(self):
        self.data['rows'][0]['original_row']['selected_event_ids'].reverse()
        self.assertEqual(self.build()['arrays'][review.RAW[0]][0, 0], -30.)
        previous = self.event(previous=False)
        previous['filingDate'] = self.event(previous=True)['filingDate']
        self.assertEqual(self.build()['arrays'][review.RAW[0]][0, 0], -30.)

    def test_calendar_days_count_leap_day_and_genuine_zero(self):
        self.event().update(reportDate='2016-02-28', filingDate='2016-03-01')
        self.assertEqual(self.build()['arrays'][review.RAW[0]][0, 0], -2.)
        self.event()['filingDate'] = self.event()['reportDate']
        got = self.build()['arrays']
        self.assertEqual(got[review.RAW[0]][0, 0], 0.)
        self.assertTrue(got['common_available'][0, 0])

    def test_latest_invalid_never_falls_back(self):
        self.data['event_metadata']['older-clean'] = deepcopy(self.event(previous=True))
        self.event()['reportDate'] = 'bad'
        with self.assertRaises(ValueError): self.build()

    def test_admitted_invalid_is_hard_failure_even_if_ineligible(self):
        self.data['eligible'][0, 0] = False
        self.event()['form'] = '10-K/A'
        with self.assertRaisesRegex(ValueError, 'issuer/form'): self.build()

    def test_unadmitted_invalid_events_are_not_consumed(self):
        row = self.data['rows'][0]
        row['conditional_annual_pair_admitted'] = False
        self.data['source_mask'][0, 0] = False
        self.data['event_metadata'][row['original_row']['selected_event_ids'][0]] = object()
        got = self.build()['arrays']
        self.assertFalse(got['common_available'][0, 0])
        self.assertEqual(got['current_lag_rank'][0, 1], 0.)

    def test_ineligible_is_missing_not_numeric_zero_and_singleton_neutral(self):
        self.data['eligible'][0, 0] = False
        got = self.build()['arrays']
        self.assertTrue(np.isnan(got['negative_current_lag'][0, 0]))
        self.assertTrue(got['common_available'][0, 1])
        self.assertTrue((got['current_lag_rank'] == 0).all())

    def test_shared_mask_and_ties(self):
        self.event(candidate=1)['filingDate'] = self.event()['filingDate']
        got = self.build()['arrays']
        self.assertTrue((got['current_lag_rank'] == 0).all())
        self.assertFalse((got['lag_change_rank'] == 0).all())
        for key in review.RAW:
            np.testing.assert_array_equal(np.isfinite(got[key]), got['common_available'])

    def test_second_strict_session_inclusive_cutoff(self):
        row = self.data['rows'][0]['original_row']
        sessions = self.data['calendar']['sessions']; cut = sessions.index(row['information_cutoff_date'])
        self.event()['filingDate'] = sessions[cut-2]
        self.assertTrue(self.build()['arrays']['common_available'][0, 0])
        self.event()['filingDate'] = sessions[cut-1]
        with self.assertRaisesRegex(ValueError, 'later than t-42'): self.build()

    def test_weekend_second_strict_session(self):
        sessions = self.data['calendar']['sessions']
        friday = next(d for d in sessions if d > self.event()['filingDate'] and date.fromisoformat(d).weekday() == 4)
        saturday = (date.fromisoformat(friday)+timedelta(days=1)).isoformat()
        self.event()['filingDate'] = saturday
        detail = self.build()['cells'][0]
        self.assertEqual(date.fromisoformat(detail['current_available_date']).weekday(), 1)

    def test_all_cutoff_references_and_early_close(self):
        row = self.data['rows'][0]['original_row']; cut = row['information_cutoff_date']
        self.data['calendar']['closes'][cut] = review.ns(cut+'T17:00:00Z')
        for item in self.data['rows'][:49]:
            item['original_row'].update(information_cutoff_ns=review.ns(cut+'T17:00:00Z'), information_cutoff=cut+'T17:00:00Z')
        self.build()
        row['information_cutoff_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'cutoff mismatch'): self.build()

    def test_future_filing_and_negative_lag_rejected(self):
        self.event()['filingDate'] = self.data['meta']['dates'][0]
        with self.assertRaises(ValueError): self.build()
        self.data = deepcopy(self.base)
        self.event()['filingDate'] = '2011-01-03'
        with self.assertRaisesRegex(ValueError, 'Negative'): self.build()

    def test_duplicate_period_accession_and_ids_rejected(self):
        for mutation in ('period', 'accession', 'id'):
            self.data = deepcopy(self.base)
            if mutation == 'period': self.event(previous=True)['reportDate'] = self.event()['reportDate']
            elif mutation == 'accession': self.event(previous=True)['accessionNumber'] = self.event()['accessionNumber']
            else:
                ids = self.data['rows'][0]['original_row']['selected_event_ids']; ids[1] = ids[0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.build()

    def test_retained_source_blockers_rejected(self):
        for field, value in [('remaining_row_reasons', ['BAD']), ('body_context_source', 'WMT'),
                             ('old_admission_flags_changed', True), ('body_pair_context_verified', False)]:
            self.data = deepcopy(self.base); self.data['rows'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.build()
        self.data = deepcopy(self.base); self.data['rows'][0]['original_row']['known_amendment_event_ids'] = ['amendment']
        with self.assertRaises(ValueError): self.build()

    def test_initial_partial_missing_with_zero_correction(self):
        day = review.INITIAL; sessions = self.data['calendar']['sessions']
        cut = sessions[sessions.index(day)-42]; self.data['meta']['dates'][0] = day
        for item in self.data['rows'][:49]:
            item['original_row'].update(signal_date=day, original_decision_cutoff=day+'T20:00:00Z',
                information_cutoff_date=cut, information_cutoff=cut+'T20:00:00Z', information_cutoff_ns=review.ns(cut+'T20:00:00Z'))
        got = self.build()['arrays']
        self.assertFalse(got['common_available'].any())
        self.assertTrue((got['lag_change_rank'] == 0).all())

    def test_axis_mask_and_boolean_guards(self):
        self.data['source_mask'][0, 0] = False
        with self.assertRaises(ValueError): self.build()
        self.data = deepcopy(self.base); self.data['rows'][0]['original_row']['candidate_index'] = True
        with self.assertRaises(ValueError): self.build()
        self.data = deepcopy(self.base); self.data['eligible'] = self.data['eligible'].astype(int)
        with self.assertRaises(ValueError): self.build()

    def test_future_rows_events_dates_and_masks_not_consumed(self):
        expected = self.build()
        self.data['rows'][49:] = [object()]*(5978-49)
        self.data['meta']['dates'][1:] = [object()]*121
        self.data['meta']['symbols'][1:] = [object()]*121
        self.data['eligible'][1:] = False; self.data['source_mask'][1:] = False
        for key in list(self.data['event_metadata']):
            if not key.startswith('0:'): self.data['event_metadata'][key] = object()
        actual = self.build()
        for key in review.EXTRA: np.testing.assert_array_equal(actual['arrays'][key], expected['arrays'][key])
        self.assertEqual(actual['cells'], expected['cells'])

    def test_all123_prefixes(self):
        expected = self.build(None)['arrays']
        for n in range(123):
            actual = self.build(n)['arrays']
            for key in review.EXTRA: np.testing.assert_array_equal(actual[key], expected[key][:n])

    def test_inputs_unchanged(self):
        before = deepcopy(self.data); self.build(None)
        for key in ('source_mask', 'eligible'): np.testing.assert_array_equal(self.data[key], before[key])
        for key in ('meta', 'rows', 'event_metadata', 'calendar'): self.assertEqual(self.data[key], before[key])


class RankAndScoreTests(unittest.TestCase):
    def test_average_ties_fraction_oracle(self):
        expected = [float(Fraction(-1, 3)), .5, float(Fraction(-1, 3)), float(Fraction(1, 6))]
        np.testing.assert_array_equal(review.rank_oracle([2, 8, 2, 5]), expected)
        for values in ([], [9], [5, 5]):
            np.testing.assert_array_equal(review.rank_oracle(values), np.zeros(len(values)))

    def test_rank_invalid_and_monotone_scale_invariance(self):
        for bad in ([True], ['1'], [np.nan], [np.inf], [[1]]):
            with self.assertRaises(ValueError): review.rank_oracle(bad)
        np.testing.assert_array_equal(review.rank_oracle([2, 4, 8]), review.rank_oracle([12, 14, 18]))

    def test_exact_small_addition_and_unchanged_core(self):
        arrays, meta, ranks = score_fixture(); before = {k: v.tobytes() for k, v in arrays.items()}
        _, _, deltas = review.score_oracle(arrays, meta, ranks)
        np.testing.assert_array_equal(deltas[review.NAMES[1]], .01*ranks['current_lag_rank'])
        self.assertEqual(max(abs(deltas[review.NAMES[2]]).flat), .005)
        self.assertEqual(before, {k: v.tobytes() for k, v in arrays.items()})

    def test_score_neutral_and_prefix_exact(self):
        arrays, meta, ranks = score_fixture()
        scores, choices, _ = review.score_oracle(arrays, meta, {k: np.zeros((2, 5)) for k in review.RANKS})
        for name in review.NAMES[1:]:
            self.assertEqual(choices[name], choices[review.NAMES[0]])
            self.assertEqual(scores[name].tobytes(), scores[review.NAMES[0]].tobytes())
        full, choices, _ = review.score_oracle(arrays, meta, ranks)
        for n in range(3):
            part, cc, _ = review.score_oracle(arrays, meta, ranks, n)
            for name in review.NAMES:
                self.assertEqual(part[name].tobytes(), full[name][:n].tobytes()); self.assertEqual(cc[name], choices[name][:n])

    def test_score_bounds_and_etf_guard(self):
        arrays, meta, ranks = score_fixture()
        for value in (.500001, np.nan, np.inf):
            bad = deepcopy(ranks); bad['current_lag_rank'][0, 0] = value
            with self.assertRaises(ValueError): review.score_oracle(arrays, meta, bad)
        ranks['current_lag_rank'][0, 4] = .5
        with self.assertRaisesRegex(ValueError, 'ETF'): review.score_oracle(arrays, meta, ranks)

    def test_original27_byte_and_dtype_preservation(self):
        old = {'a'+str(i): np.array([i], dtype=float) for i in range(27)}
        prepared = dict(old, **{k: np.zeros((122, 49)) for k in review.EXTRA})
        review.verify_original_arrays(old, prepared)
        prepared['a3'] = np.array([3], dtype=np.int64)
        with self.assertRaises(ValueError): review.verify_original_arrays(old, prepared)

    def test_no_producer_imports(self):
        review.independent_imports()

    def test_real_audit_rejects_missing_registration_before_inputs(self):
        for digest in (None, 'x', 'a'*63):
            with self.assertRaisesRegex(ValueError, 'Explicit PLAN SHA'):
                review.audit('/synthetic', 'PLAN.json', digest)


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.expected = review.features_oracle(**cls.data)

    def receipt(self):
        e = self.expected
        return dict(source_overlay_sha256='a'*64, source_review_sha256='b'*64, source_prior_sha256='c'*64,
            new_predictive_model_fits=0, nuisance_estimates=0, source_admission_changed=False,
            exact_acceptance_certified=False, historical_price_identity_certified=False,
            strategy_returns_or_nav_computed=False, numerical_features_computed=True, source_contradictions_hard_failed=True,
            rows=5978, months=122, dose=.01, max_absolute_score_correction=.005, fixed_issuers=dict(review.ISSUERS),
            usable_rows=230, lag_pairs_evaluated=230, source_conditionally_admitted_rows=230,
            cells=deepcopy(e['cells']), monthly=deepcopy(e['monthly']),
            reason_counts=dict(Counter(r for c in e['cells'] for r in c['reasons'])), informative_rank_months=115,
            coverage=review.coverage_oracle(self.data['rows'], e), feature_prefixes_checked=123,
            negative_current_lag_definition='-(filingDate_current-reportDate_current).days',
            negative_lag_change_definition='previous_lag_days-current_lag_days',
            rank_definition='(average_rank1-1)/(n-1)-0.5; n<=1 gives0; rank only common original-eligible mask')

    def verify(self, receipt):
        review.verify_feature_audit(receipt, self.expected, self.data['rows'],
            dict(source_overlay='a'*64, source_review='b'*64, source_prior='c'*64))

    def test_exact_receipt_and_bad_lag_rejected(self):
        receipt = self.receipt(); self.verify(receipt)
        receipt['cells'][0]['previous_lag_days'] += 1
        with self.assertRaises(ValueError): self.verify(receipt)

    def test_receipt_mask_coverage_and_boolean_tamper_rejected(self):
        for mutation in ('mask', 'coverage', 'boolean', 'prefixes'):
            receipt = self.receipt()
            if mutation == 'mask': receipt['cells'][0]['available'] = False
            elif mutation == 'coverage': receipt['coverage']['all122']['unique_usable_pairs'] += 1
            elif mutation == 'boolean': receipt['source_admission_changed'] = 0
            else: receipt['feature_prefixes_checked'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.verify(receipt)

    def test_no_incremental_evidence_distinguishes_scores_and_orders(self):
        choices = {name: [[0, 1, 2] for _ in range(122)] for name in review.NAMES}
        nonzero = dict.fromkeys(review.NAMES[1:], 115)
        receipt = dict(nonzero_correction_months=nonzero,
            ordered_holdings_changed_months=dict.fromkeys(review.NAMES[1:], 0), primary_no_incremental_evidence=True)
        review.verify_change_counts(receipt, self.expected, choices)
        receipt['primary_no_incremental_evidence'] = False
        with self.assertRaises(ValueError): review.verify_change_counts(receipt, self.expected, choices)

    def test_only_exact_registered_historical_state_alias(self):
        class Fake:
            root = Path('/synthetic')
            def path(self, path):
                p = Path(path); return p if p.is_absolute() else self.root/p
            def sha(self, path):
                self.last = self.path(path); return review.HISTORICAL_SHA
        reader = Fake()
        archive = str(reader.path(review.HISTORICAL_ARCHIVE))
        got = review.historical_pins(reader, {review.HISTORICAL_STATE: review.HISTORICAL_SHA}, {archive: review.HISTORICAL_SHA})
        self.assertEqual(got, {archive: review.HISTORICAL_SHA})
        self.assertEqual(reader.last, Path(archive))
        with self.assertRaises(ValueError): review.historical_pins(reader, {review.HISTORICAL_STATE: 'a'*64}, {})
        with self.assertRaises(ValueError): review.historical_pins(reader, {review.HISTORICAL_STATE: review.HISTORICAL_SHA}, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
