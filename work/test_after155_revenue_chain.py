"""Source timing, original-vintage and eligibility tests; no market data."""
import copy
import unittest
from after155_revenue_chain import evaluate


class SourceChainTest(unittest.TestCase):
    def setUp(self):
        self.reports=[]; self.edges=[]
        for year in range(2015,2022):
            self.reports.append(dict(report_id=str(year),known_ns=year, fiscal_year=year,
                source_checked=True, issuer='MSFT', currency='USD',
                measure='consolidated_gaap_total_revenue',period_start=f'{year-1}-07-01',
                period_end=f'{year}-06-30',current='100',scale='1000000',
                comparative='100',comparative_scale='1000000',comparative_currency='USD',
                comparative_measure='consolidated_gaap_total_revenue',
                comparative_source_report_id=str(year),
                comparative_period_start=f'{year-2}-07-01',
                comparative_period_end=f'{year-1}-06-30'))
            if year>2015:
                self.edges.append(dict(edge_id=str(year-1)+'_'+str(year),known_ns=year,
                    earlier_fiscal_year=year-1,later_fiscal_year=year,
                    earlier_report_id=str(year-1),later_report_id=str(year),
                    evidence_known_ns=[year-1,year],
                    status='checked_reported_total_edge'))

    def run_chain(self, cutoff=2021, day='2021-08-01'):
        return evaluate(self.reports,self.edges,cutoff,day)

    def test_four_originals(self):
        result=self.run_chain()
        self.assertEqual(result['report_ids'],['2018','2019','2020','2021'])
        self.assertEqual(result['status'],'checked_reported_total_chain_with_declared_source_scope')

    def test_future_report_not_member(self):
        result=self.run_chain(2020,'2020-08-01')
        self.assertEqual(result['latest_report_id'],'2020')
        self.assertNotIn('2021',result['report_ids'])

    def test_invalid_latest_blocks_older(self):
        self.reports[-1]['source_checked']=False
        self.assertEqual(self.run_chain()['reasons'],['latest_source_not_checked'])

    def test_missing_original_not_replaced_by_comparative(self):
        self.reports=[r for r in self.reports if r['fiscal_year']!=2019]
        self.assertEqual(self.run_chain()['reasons'],['missing_own_year_2019'])

    def test_later_restatement_not_backfilled(self):
        self.reports[-1]['comparative']='105'
        old=copy.deepcopy(self.reports)
        self.assertEqual(self.run_chain()['status'],'blocked_known_basis_break')
        self.assertEqual(self.reports,old)
        self.assertEqual(self.run_chain(2020,'2020-08-01')['status'],
                         'checked_reported_total_chain_with_declared_source_scope')

    def test_known_basis_break(self):
        self.edges[-2]['status']='blocked_known_basis_break'
        self.assertEqual(self.run_chain()['status'],'blocked_known_basis_break')

    def test_future_witness_cannot_validate(self):
        self.edges[-1]['known_ns']=2022
        self.assertEqual(self.run_chain()['reasons'],['missing_or_conflicting_edge_2021'])

    def test_stale_no_fallback(self):
        self.assertEqual(self.run_chain(day='2023-08-01')['status'],'stale_latest_annual')

    def test_quarter_cannot_substitute_year(self):
        self.reports[-2]['period_start']='2020-04-01'
        self.assertEqual(self.run_chain()['reasons'],['invalid_period_2020'])

    def test_segment_cannot_substitute_total(self):
        self.reports[-2]['measure']='segment_revenue'
        self.assertEqual(self.run_chain()['reasons'],['invalid_member_2020'])

    def test_currency_mismatch(self):
        self.reports[-1]['comparative_currency']='EUR'
        self.assertEqual(self.run_chain()['reasons'],['incompatible_witness_2021'])

    def test_different_scale_not_silently_normalized(self):
        self.reports[-1]['comparative']='100000'
        self.reports[-1]['comparative_scale']='1000'
        self.assertEqual(self.run_chain()['reasons'],['incompatible_witness_scale_2021'])

    def test_comparative_annual_start_required(self):
        self.reports[-1]['comparative_period_start']='2020-04-01'
        self.assertEqual(self.run_chain()['reasons'],['incompatible_witness_2021'])

    def test_comparative_total_measure_required(self):
        self.reports[-1]['comparative_measure']='segment_revenue'
        self.assertEqual(self.run_chain()['reasons'],['incompatible_witness_2021'])

    def test_comparative_must_belong_to_later_report(self):
        self.reports[-1]['comparative_source_report_id']='different_2021_document'
        self.assertEqual(self.run_chain()['reasons'],['incompatible_witness_2021'])

    def test_edge_must_bind_selected_originals(self):
        self.edges[-1]['earlier_report_id']='restated_2020'
        self.assertEqual(self.run_chain()['reasons'],['unbound_or_mistimed_edge_2021'])

    def test_edge_cannot_precede_report(self):
        self.edges[-1]['known_ns']=2020
        self.assertEqual(self.run_chain()['reasons'],['unbound_or_mistimed_edge_2021'])

    def test_later_note_not_backdated_to_edge(self):
        self.edges[-1]['evidence_known_ns'].append(2022)
        self.assertEqual(self.run_chain()['reasons'],['unbound_or_mistimed_edge_2021'])

    def test_edge_without_evidence_times_fails(self):
        self.edges[-1]['evidence_known_ns']=[]
        self.assertEqual(self.run_chain()['reasons'],['unbound_or_mistimed_edge_2021'])

    def test_future_period_rejected(self):
        self.assertEqual(self.run_chain(day='2021-05-01')['reasons'],['future_latest_period'])

    def test_freshness_boundary_is_inclusive(self):
        from datetime import date, timedelta
        end=date(2021,6,30)
        self.assertEqual(self.run_chain(day=str(end+timedelta(days=550)))['status'],
                         'checked_reported_total_chain_with_declared_source_scope')
        self.assertEqual(self.run_chain(day=str(end+timedelta(days=551)))['status'],
                         'stale_latest_annual')

    def test_conflicting_latest(self):
        self.reports.append(copy.deepcopy(self.reports[-1]))
        self.assertEqual(self.run_chain()['reasons'],['conflicting_latest_release'])

    def test_reject_nan(self):
        self.reports[-2]['current']='NaN'
        self.assertEqual(self.run_chain()['reasons'],['invalid_value_2020'])

    def test_long_literal_difference_is_not_rounded_away(self):
        self.reports[-2]['current']='1234567890123456789012345678901'
        self.reports[-2]['comparative']='100'
        self.reports[-1]['comparative']='1234567890123456789012345678902'
        self.assertEqual(self.run_chain()['reasons'],['comparative_mismatch_2021'])

    def test_missing_field_returns_explicit_hold(self):
        del self.reports[-1]['period_end']
        self.assertEqual(self.run_chain()['reasons'],['malformed_required_source_record'])

    def test_unknown_edge_holds(self):
        self.edges[-1]['status']='unresolved'
        self.assertEqual(self.run_chain()['status'],'unresolved_source_or_comparability')


if __name__=='__main__':
    unittest.main()
