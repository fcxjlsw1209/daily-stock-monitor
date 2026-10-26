"""Fabricated checks of the independent source-grid oracle only."""
from copy import deepcopy
from datetime import date, timedelta
import unittest

import after155_revenue_pilot_output_independent as review


def originals(first=2013, last=2025):
    reports = [dict(year=y, id='r%s' % y, known=review.utc_ns('%s-02-03T21:00:00+00:00' % (y+1)),
        end='%s-12-31' % y, checked=True, amount=str(y), witness=str(y-1)) for y in range(first, last+1)]
    edges = [dict(id='e%s' % b['year'], years=(a['year'], b['year']), reports=(a['id'], b['id']),
        known=b['known'], evidence=[a['known'], b['known']],
        status='pending_identified_reclassification_note' if b['year'] == 2018 else 'checked_reported_total_edge')
        for a, b in zip(reports, reports[1:])]
    return reports, edges


class IndependentOracleTests(unittest.TestCase):
    def test_release_close_cutoff_is_inclusive(self):
        reports, edges = originals()
        instant = reports[3]['known']
        before = review.own_oracle(reports, edges, instant-1, '2017-02-03')
        after = review.own_oracle(reports, edges, instant, '2017-02-03')
        self.assertEqual(before['latest_report_id'], 'r2015')
        self.assertEqual(before['status'], 'missing_required_source')
        self.assertEqual(after['status'], review.USABLE)
        self.assertEqual(after['report_ids'], ['r2013', 'r2014', 'r2015', 'r2016'])

    def test_no_known_original_stays_missing(self):
        reports, edges = originals()
        result = review.own_oracle(reports, edges, reports[0]['known']-1, '2014-02-03')
        self.assertEqual(result['reasons'], ['no_known_original_annual_report'])
        self.assertEqual(result['report_ids'], [])

    def test_pending_edge_blocks_every_containing_four_year_window(self):
        reports, edges = originals()
        for year in (2018, 2019, 2020):
            r = next(r for r in reports if r['year'] == year)
            result = review.own_oracle(reports, edges, r['known'], '%s-02-03' % (year+1))
            self.assertEqual(result['status'], 'unresolved_source_or_comparability')
            self.assertEqual(result['reasons'], ['unreviewed_edge_2018'])
        r = next(r for r in reports if r['year'] == 2021)
        result = review.own_oracle(reports, edges, r['known'], '2022-02-03')
        self.assertEqual(result['status'], review.USABLE)
        self.assertNotIn('e2018', result['edge_ids'])

    def test_later_comparative_does_not_supply_a_missing_original(self):
        reports, edges = originals(last=2016)
        reports = [r for r in reports if r['year'] != 2014]
        result = review.own_oracle(reports, edges, reports[-1]['known'], '2017-02-03')
        self.assertEqual(result['reasons'], ['missing_own_year_2014'])

    def test_invalid_latest_does_not_fall_back(self):
        reports, edges = originals(last=2017)
        reports[-1]['checked'] = False
        result = review.own_oracle(reports, edges, reports[-1]['known'], '2018-02-03')
        self.assertEqual(result['latest_report_id'], 'r2017')
        self.assertEqual(result['reasons'], ['latest_source_not_checked'])

    def test_later_edge_evidence_is_unavailable(self):
        reports, edges = originals(last=2016)
        edges[-1]['known'] += 1
        result = review.own_oracle(reports, edges, reports[-1]['known'], '2017-02-03')
        self.assertEqual(result['reasons'], ['missing_or_conflicting_edge_2016'])

    def test_550_day_staleness_boundary(self):
        reports, edges = originals(last=2016)
        for days, expected in [(550, review.USABLE), (551, 'stale_latest_annual')]:
            day = (date(2016, 12, 31) + timedelta(days=days)).isoformat()
            result = review.own_oracle(reports, edges, review.utc_ns(day+'T21:00:00+00:00'), day)
            self.assertEqual(result['status'], expected)

    def test_unequal_comparative_blocks(self):
        reports, edges = originals(last=2016)
        reports[-1]['witness'] = '999'
        result = review.own_oracle(reports, edges, reports[-1]['known'], '2017-02-03')
        self.assertEqual(result['status'], 'blocked_known_basis_break')
        self.assertEqual(result['reasons'], ['comparative_mismatch_2016'])

    def test_conflicting_latest_original_is_not_selected(self):
        reports, edges = originals(last=2016)
        duplicate = deepcopy(reports[-1])
        duplicate['id'] = 'conflict'
        reports.append(duplicate)
        result = review.own_oracle(reports, edges, reports[-1]['known'], '2017-02-03')
        self.assertEqual(result['reasons'], ['conflicting_latest_release'])

    def test_recent_coverage_can_pass_without_early_coverage(self):
        dates = ['date%s' % i for i in range(122)]
        meta = dict(dates=dates, etfs=[])
        plan = dict(fixed_issuer_scope=['MSFT', 'AAPL', 'ABBV'], gate_before_strategy_registration=dict(
            minimum_issuers=2, minimum_usable_candidate_rows_per_qualifying_issuer_in_first88=12,
            minimum_distinct_latest_original_annual_endpoints_per_qualifying_issuer_in_first88=2,
            minimum_first88_cutoffs_with_at_least_two_usable_issuers=12))
        rows = [dict(symbol=s, signal_index=i, status=review.USABLE if s != 'AAPL' and 53 <= i < 74 else 'pending',
                     latest_report_id='%s_%s' % (s, i//12)) for i in range(122) for s in plan['fixed_issuer_scope']]
        result = review.independent_gate(meta, rows, plan)
        self.assertIs(result['source_coverage_floor_passed'], True)
        self.assertEqual(len(result['qualifying_joint_cutoffs']), 21)
        self.assertEqual(result['segments']['early53']['cutoffs_with_minimum_issuers'], [])
        self.assertEqual(result['segments']['early53']['issuers']['ABBV']['usable_rows'], 0)


if __name__ == '__main__':
    unittest.main()
