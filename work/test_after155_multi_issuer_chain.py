"""Fabricated source-only tests; no original amounts or performance read."""
from copy import deepcopy
from datetime import date, timedelta
import unittest
from after155_multi_issuer_chain import evaluate, expected_period, MEASURE


def fixture(issuer='AAPL'):
    reports, edges = [], []
    for y in range(2020, 2024):
        start, end = expected_period(issuer, y)
        r = dict(issuer=issuer, fiscal_year=y, report_id=f'{issuer}_{y}',
                 known_ns=y, source_checked=True, currency_basis_checked=True,
                 source_sha256='a'*64, source_review_sha256='b'*64,
                 period_start=start, period_end=end, current=str(100+y),
                 scale='1000000', measure=MEASURE, currency='USD',
                 currency_basis='iso_evidenced',currency_iso='USD',currency_iso_certified=True,
                 currency_policy_sha256='d'*64)
        ps, pe = expected_period(issuer, y-1)
        r.update(comparative=str(99+y), comparative_scale='1000000',
                 comparative_period_start=ps, comparative_period_end=pe,
                 comparative_currency='USD', comparative_measure=MEASURE,
                 comparative_currency_basis='iso_evidenced',comparative_currency_iso='USD',
                 comparative_currency_iso_certified=True,
                 comparative_source_report_id=r['report_id'])
        reports.append(r)
        if y>2020:
            edges.append(dict(issuer=issuer, earlier_fiscal_year=y-1,
                              later_fiscal_year=y, earlier_report_id=f'{issuer}_{y-1}',
                              later_report_id=r['report_id'], edge_id=f'edge_{y}',
                              known_ns=y, evidence_known_ns=[y-1,y],
                              source_review_sha256='c'*64,
                              status='checked_reported_total_edge'))
    return reports, edges


class EligibilityTest(unittest.TestCase):
    def setUp(self):
        self.reports, self.edges = fixture()

    def run_case(self, reason=None, status=None, **kw):
        result=evaluate('AAPL',self.reports,self.edges,kw.get('cutoff',2024),kw.get('day','2024-01-31'))
        if reason is not None: self.assertEqual(result['reasons'],[reason])
        if status is not None: self.assertEqual(result['status'],status)
        self.assertFalse(result['factor_computed'])
        return result

    def test_all_profiles(self):
        for issuer in ['MSFT','AAPL','ABBV','ABT']:
            r,e=fixture(issuer)
            self.assertEqual(evaluate(issuer,r,e,2024,'2024-01-31')['status'],
                             'checked_reported_total_chain_with_declared_source_scope')

    def test_apple_53week_period(self):
        a,b=expected_period('AAPL',2023)
        self.assertEqual((a,b),('2022-09-25','2023-09-30'))
        self.assertEqual((date.fromisoformat(b)-date.fromisoformat(a)).days+1,371)

    def test_apple_52week_period(self):
        a,b=expected_period('AAPL',2022)
        self.assertEqual((date.fromisoformat(b)-date.fromisoformat(a)).days+1,364)

    def test_missing_own_year_not_comparative(self):
        self.reports.pop(0)
        self.run_case('missing_own_year_2020','missing_required_source')

    def test_future_report_not_known(self):
        self.run_case('missing_own_year_2019','missing_required_source',cutoff=2022)

    def test_latest_invalid_no_fallback(self):
        self.reports[-1]['source_checked']=False
        self.run_case('latest_source_not_checked')

    def test_no_known(self):
        self.run_case('no_known_original_annual_report','missing_required_source',cutoff=1)

    def test_550_days_inclusive(self):
        end=date.fromisoformat(self.reports[-1]['period_end'])
        self.run_case(status='checked_reported_total_chain_with_declared_source_scope',
                      day=(end+timedelta(days=550)).isoformat())

    def test_551_days_stale(self):
        end=date.fromisoformat(self.reports[-1]['period_end'])
        self.run_case('latest_annual_period_older_than_550_days','stale_latest_annual',
                      day=(end+timedelta(days=551)).isoformat())

    def test_future_period(self):
        self.run_case('future_latest_period',day='2023-01-31')

    def test_wrong_issuer(self):
        self.reports[0]['issuer']='MSFT'
        self.run_case('unbound_issuer_or_report_clock')

    def test_wrong_currency(self):
        self.reports[0]['currency']='CAD'
        self.run_case('malformed_required_source_record')

    def test_bare_currency_not_evidence(self):
        self.reports[0]['currency_basis_checked']=False
        self.run_case('invalid_or_unreviewed_member_2020')

    def dollar_basis(self):
        for r in self.reports:
            for prefix in ('','comparative_'):
                r[prefix+'currency']='reported_dollar'
                r[prefix+'currency_basis']='declared_within_issuer_reported_dollar_v1'
                r[prefix+'currency_iso']=None
                r[prefix+'currency_iso_certified']=False

    def test_explicit_reported_dollar_assumption(self):
        self.dollar_basis()
        self.run_case(status='checked_reported_total_chain_with_declared_source_scope')

    def test_reported_dollar_never_certifies_iso(self):
        self.dollar_basis()
        self.reports[0]['currency_iso_certified']=True
        self.run_case('malformed_required_source_record')

    def test_mixed_current_basis_not_silently_coerced(self):
        original=deepcopy(self.reports[0])
        self.dollar_basis()
        self.reports[0]=original
        self.run_case('mixed_currency_basis_2021')

    def test_mixed_comparative_basis_not_silently_coerced(self):
        self.reports[1]['comparative_currency_basis']='declared_within_issuer_reported_dollar_v1'
        self.run_case('malformed_required_source_record')

    def test_missing_currency_policy(self):
        self.reports[0].pop('currency_policy_sha256')
        self.run_case('unbound_currency_policy_2020')

    def test_mixed_currency_policy(self):
        self.reports[1]['currency_policy_sha256']='e'*64
        self.run_case('mixed_currency_basis_2021')

    def test_missing_source_hash(self):
        self.reports[0].pop('source_sha256')
        self.run_case('invalid_or_unreviewed_member_2020')

    def test_missing_review_hash(self):
        self.edges[0].pop('source_review_sha256')
        self.run_case('unbound_or_mistimed_edge_2021')

    def test_duplicate_latest(self):
        self.reports.append(deepcopy(self.reports[-1]))
        self.run_case('conflicting_latest_release')

    def test_duplicate_own_year(self):
        self.reports.append(deepcopy(self.reports[0]))
        self.run_case('conflicting_own_year_2020')

    def test_later_older_year_correction(self):
        self.reports[0]['known_ns']=2024
        self.run_case('later_release_for_older_year_requires_review')

    def test_float_amount_rejected(self):
        self.reports[0]['current']=2120.0
        self.run_case('malformed_required_source_record')

    def test_invalid_amounts(self):
        for v in ['NaN','Infinity','0','-1']:
            self.reports[0]['current']=v
            self.run_case('malformed_required_source_record')

    def test_scale_mismatch_no_rescale(self):
        self.reports[1]['comparative_scale']='1000'
        self.run_case('incompatible_witness_scale_2021')

    def test_common_original_scale_required(self):
        self.reports[1]['scale']='1000'
        self.reports[2]['comparative_scale']='1000'
        self.run_case('mixed_original_scale_2021')

    def test_witness_mismatch(self):
        self.reports[1]['comparative']='2000'
        self.run_case('comparative_mismatch_2021','blocked_known_basis_break')

    def test_witness_belongs_to_current_release(self):
        self.reports[1]['comparative_source_report_id']='AAPL_2020'
        self.run_case('incompatible_witness_2021')

    def test_witness_wrong_start(self):
        self.reports[1]['comparative_period_start']='2019-09-30'
        self.run_case('incompatible_witness_2021')

    def test_invalid_annual_period(self):
        self.reports[0]['period_start']='2020-01-01'
        self.run_case('invalid_period_2020')

    def test_duplicate_edge(self):
        self.edges.append(deepcopy(self.edges[0]))
        self.run_case('missing_or_conflicting_edge_2021')

    def test_future_edge(self):
        self.edges[0]['known_ns']=2025
        self.run_case('missing_or_conflicting_edge_2021')

    def test_edge_predates_later_release(self):
        self.edges[0]['known_ns']=2020
        self.run_case('unbound_or_mistimed_edge_2021')

    def test_future_note(self):
        self.edges[0]['evidence_known_ns']=[2025]
        self.run_case('unbound_or_mistimed_edge_2021')

    def test_wrong_edge_report(self):
        self.edges[0]['later_report_id']='wrong'
        self.run_case('unbound_or_mistimed_edge_2021')

    def test_unreviewed_edge(self):
        self.edges[0]['status']='pending'
        self.run_case('unreviewed_edge_2021')

    def test_basis_break(self):
        self.edges[0]['status']='blocked_known_basis_break'
        self.run_case('edge_2021','blocked_known_basis_break')

    def test_bad_clock_never_silently_filtered(self):
        self.reports[-1]['known_ns']=True
        self.run_case('unbound_issuer_or_report_clock')

    def test_bad_decision_clock(self):
        self.run_case('invalid_decision_timestamp',cutoff=True)

    def test_invalid_catalog_member(self):
        self.reports.append(None)
        self.run_case('invalid_catalog_member')

    def test_empty_edge_id(self):
        self.edges[0]['edge_id']=''
        self.run_case('invalid_or_duplicate_edge_identity')

    def test_duplicate_edge_id(self):
        self.edges[1]['edge_id']=self.edges[0]['edge_id']
        self.run_case('invalid_or_duplicate_edge_identity')


if __name__ == '__main__':
    unittest.main(verbosity=2)
