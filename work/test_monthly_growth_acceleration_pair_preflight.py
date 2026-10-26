"""Synthetic-only tests: no real event, financial amount, or price reads."""
from copy import deepcopy
import hashlib
import unittest

import monthly_growth_acceleration_pair_preflight as p


def period(year, q):
    return dict(fiscal_year=year, fiscal_quarter=q, quarter_identity_verified=True)


def clock(day, at=None, version=None):
    return dict(visible_date=day, published_at=at, modified_at=None, known_version_not_before=version,
                time_kind='aware_release' if at else 'date_only', publication_date_verified=True,
                version_evidence_status='known_version_bound' if version else 'checked_no_known_late_version',
                modification_history_complete=False)


def event(eid, day, year, q, at=None):
    nq = (year+1, 1) if q==4 else (year,q+1)
    s = dict(path='synthetic/'+eid, sha256=hashlib.sha256(eid.encode()).hexdigest(),
             identity_verified=True, receipt_verified=True, body_complete=True)
    semantics = dict(currency='USD', metric='reported_total_revenue', scope='consolidated_reported_revenue',
                     accounting_basis='reported', scope_verified=True, currency_verified=True, amount_verified=True)
    e = dict(event_id=eid, symbol='SYN', issuer_id='synthetic_issuer', event_kind='earnings_release',
             earnings_event_verified=True, source=s, clock=clock(day,at), reported_period=period(year,q),
             forecast=dict(semantics,status='numeric',target_relation_verified=True,comparison_basis='as_reported',
                           required_prior_basis_id=None,target_period=period(*nq)),actuals={})
    for k,y,qq in [('target_prior_year',nq[0]-1,nq[1]),('current_quarter',year,q),('current_prior_year',year-1,q)]:
        e['actuals'][k] = dict(semantics, issuer_id='synthetic_issuer', quarter_only_verified=True,basis_id='as_reported',
                              period=period(y,qq),source=deepcopy(s),clock=deepcopy(e['clock']))
    return e


class PairTests(unittest.TestCase):
    def setUp(self):
        self.old = event('old','2023-10-20',2023,3)
        self.cur = event('cur','2024-01-20',2023,4)
        self.es = [self.old,self.cur]
        self.reviewed = {'old','cur'}

    def run_pair(self, es=None): return p.pair_one(self.es if es is None else es,'cur',self.reviewed)

    def test_quarter_year_rollover(self):
        r=self.run_pair(); self.assertTrue(r['pair_available']); self.assertEqual(r['previous_event_id'],'old')
        self.assertEqual(r['current_target_period'],period(2024,1))

    def test_intervening_unreviewed_no_skip(self):
        mid=event('mid','2023-12-01',2023,3)
        r=self.run_pair([self.old,mid,self.cur]); self.assertEqual(r['previous_event_id'],'mid')
        self.assertIn('previous_not_independently_reviewed',r['reason_codes']); self.assertFalse(r['pair_available'])

    def test_intervening_unknown_forecast_no_skip(self):
        mid=event('mid','2023-12-01',2023,3); mid['forecast']['status']='unresolved'; self.reviewed.add('mid')
        r=self.run_pair([self.old,mid,self.cur]); self.assertEqual(r['previous_event_id'],'mid')
        self.assertIn('previous_forecast_not_numeric',r['reason_codes'])

    def test_intervening_unresolved_version_stays_barrier(self):
        mid=event('mid','2023-12-01',2023,3); mid['clock']['version_evidence_status']='unresolved'
        r=self.run_pair([self.old,mid,self.cur]); self.assertEqual(r['previous_event_id'],'mid')
        self.assertIn('previous_guide_reference_clock_version_evidence_unresolved',r['reason_codes'])

    def test_prior_guide_version_after_current_not_month(self):
        self.old['clock']=clock('2023-10-20',version='2024-01-21T12:00:00-05:00')
        r=self.run_pair(); self.assertIn('previous_guide_known_version_after_current',r['reason_codes'])

    def test_prior_denominator_version_after_current(self):
        self.old['actuals']['target_prior_year']['clock']=clock('2022-10-20',version='2024-01-21T12:00:00-05:00')
        r=self.run_pair(); self.assertIn('previous_target_prior_year_known_version_after_current',r['reason_codes'])

    def test_prior_denominator_version_before_current(self):
        self.old['actuals']['target_prior_year']['clock']=clock('2022-10-20',version='2024-01-19T12:00:00-05:00')
        self.assertTrue(self.run_pair()['pair_available'])

    def test_same_day_version_requires_current_instant(self):
        self.old['clock']=clock('2023-10-20',version='2024-01-20T12:00:00-05:00')
        self.assertIn('previous_guide_same_day_version_order_unknown',self.run_pair()['reason_codes'])
        self.cur['clock']=clock('2024-01-20','2024-01-20T13:00:00-05:00')
        self.assertTrue(self.run_pair()['pair_available'])
        self.cur['clock']=clock('2024-01-20','2024-01-20T11:00:00-05:00')
        self.assertIn('previous_guide_known_version_after_current',self.run_pair()['reason_codes'])

    def test_all_version_fields_offset_order(self):
        self.cur['clock']=clock('2024-01-20','2024-01-20T12:00:00-05:00')
        self.old['clock']=clock('2023-10-20',version='2024-01-20T18:30:00+02:00')
        self.old['clock']['modified_at']='2024-01-20T12:30:00-05:00'
        self.assertIn('previous_guide_known_version_after_current',self.run_pair()['reason_codes'])

    def test_same_day_publication_order_not_guessed(self):
        self.old['clock']=clock('2024-01-20')
        self.assertIn('ambiguous_current_day_order',self.run_pair()['reason_codes'])
        self.old['clock']=clock('2024-01-20','2024-01-20T10:00:00-05:00')
        self.cur['clock']=clock('2024-01-20','2024-01-20T12:00:00-05:00')
        self.assertTrue(self.run_pair()['pair_available'])

    def test_equal_publication_instants_ambiguous(self):
        for e in self.es: e['clock']=clock('2024-01-20','2024-01-20T12:00:00-05:00')
        self.assertIn('ambiguous_current_day_order',self.run_pair()['reason_codes'])

    def test_latest_prior_same_day_requires_all_instants(self):
        other=deepcopy(self.old); other['event_id']='other'; other['source']['sha256']='b'*64
        other['clock']=clock('2023-10-20','2023-10-20T13:00:00-04:00'); self.reviewed.add('other')
        self.assertIn('ambiguous_prior_day_order',self.run_pair([*self.es,other])['reason_codes'])
        self.old['clock']=clock('2023-10-20','2023-10-20T12:00:00-04:00')
        self.assertEqual(self.run_pair([*self.es,other])['previous_event_id'],'other')

    def test_same_raw_representations_preserved(self):
        other=deepcopy(self.old); other['event_id']='other'; self.reviewed.add('other')
        self.old['clock']=clock('2023-10-20','2023-10-20T12:00:00-04:00')
        other['clock']=clock('2023-10-20','2023-10-20T13:00:00-04:00')
        self.assertIn('ambiguous_same_raw_representations',self.run_pair([*self.es,other])['reason_codes'])

    def test_quarter_gap_no_skip(self):
        old=event('old','2023-07-20',2023,2)
        r=self.run_pair([old,self.cur]); self.assertIn('target_quarters_not_adjacent',r['reason_codes'])

    def test_currency_mismatch(self):
        self.old['forecast']['currency']='EUR'
        self.assertIn('between_event_currency_mismatch',self.run_pair()['reason_codes'])

    def test_missing_reference_and_wrong_issuer(self):
        self.old['actuals']['target_prior_year']=None
        self.assertIn('previous_target_prior_year_missing',self.run_pair()['reason_codes'])
        self.old['issuer_id']='other'
        self.assertIn('no_prior_located_observed_event',self.run_pair()['reason_codes'])

    def test_undated_disclosed_not_invented(self):
        unknown=event('unlocated','2023-12-01',2023,3); unknown['clock']['visible_date']=None
        r=self.run_pair([*self.es,unknown]); self.assertTrue(r['pair_available'])
        self.assertEqual(r['unlocated_event_ids'],['unlocated']); self.assertFalse(r['complete_event_stream'])

    def test_future_prefix_and_order_invariance(self):
        future=event('future','2024-04-20',2024,1); es=[future,*self.es]
        r=self.run_pair(es); self.assertEqual(r,self.run_pair(list(reversed(es))))
        self.assertEqual(r,self.run_pair(p.prefix(es,self.cur)))
        later=event('later','2024-01-20',2023,4,'2024-01-20T14:00:00-05:00')
        self.cur['clock']=clock('2024-01-20','2024-01-20T12:00:00-05:00')
        self.assertEqual(self.run_pair([*self.es,later]),self.run_pair())

    def test_project_never_reads_amounts_or_quotes(self):
        class Poison(dict):
            def get(self,k,*args):
                if k in ('amount','quote'): raise AssertionError('numerical or text field read')
                return super().get(k,*args)
        e=deepcopy(self.cur); e['forecast']=Poison(e['forecast'])
        e['actuals']={k:Poison(v) for k,v in e['actuals'].items()}
        r=p.project_event(e); self.assertEqual(r,self.cur)

    def test_duplicate_event_ids_error(self):
        with self.assertRaises(ValueError): self.run_pair([*self.es,self.old])

    def test_known_versions_never_prefix_pruned(self):
        self.old['clock']=clock('2023-10-20',version='2026-01-20T12:00:00-05:00')
        pr=p.prefix(self.es,self.cur)
        self.assertEqual(self.run_pair(pr),self.run_pair()); self.assertFalse(self.run_pair(pr)['pair_available'])

if __name__=='__main__': unittest.main(verbosity=2)
