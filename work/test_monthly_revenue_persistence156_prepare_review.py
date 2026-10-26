"""Fabricated data only; no actual source values or performance are read."""
from copy import deepcopy
import unittest
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_revenue_persistence156_prepare_review as r
from test_monthly_cash_quality146_prepare_review import score_fixture


def fixture():
    names=['MSFT','ABBV','AAPL']+['S'+str(i) for i in range(29)]+['E'+str(i) for i in range(17)]
    meta=dict(dates=['2023-03-31','2023-04-28'],symbols=[names[:],names[:]],etfs=names[32:])
    reports={key:[dict(report_id=key+str(year),fiscal_year=year,known_ns=year,
        current=str(value),scale='1000000') for year,value in zip(range(2019,2023),[100,200,100,400])]
        for key in ('MSFT','ABBV')}
    reports['AAPL']=[]; rows=[]
    for i,date in enumerate(meta['dates']):
        clock=pd.Timestamp(date+'T20:00:00Z')
        for j,symbol in enumerate(names):
            row=dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=date,
                decision_cutoff=str(clock),decision_cutoff_ns=int(clock.value),status='unavailable',reasons=['held'])
            if symbol in ('MSFT','ABBV'):
                ids=[x['report_id'] for x in reports[symbol]]
                row.update(status=r.USABLE,reasons=[],report_ids=ids,latest_report_id=ids[-1],edge_ids=['a','b','c'])
            rows.append(row)
    return meta,rows,reports


class FeatureTests(unittest.TestCase):
    def test_common_mask_and_independent_closed_form(self):
        m,rows,reports=fixture(); x=r.features_oracle(m,rows,reports)
        self.assertEqual(int(x['arrays']['common_available'].sum()),4)
        self.assertAlmostEqual(x['arrays'][r.HANDS[0]][0,0],.75)
        self.assertAlmostEqual(x['arrays'][r.HANDS[2]][0,0],-1/3)
        for key in r.HANDS:self.assertTrue(np.array_equal(np.isfinite(x['arrays'][key]),x['arrays']['common_available']))

    def test_source_prefixes(self):
        m,rows,reports=fixture(); full=r.features_oracle(m,rows,reports)
        for n in (0,1,2):
            got=r.features_oracle(m,rows,reports,n)
            for key in r.EXTRA:np.testing.assert_array_equal(got['arrays'][key],full['arrays'][key][:n])

    def test_hold_is_not_converted_to_zero_or_removed(self):
        m,rows,reports=fixture();rows[0]['status']='blocked_known_basis_break'
        got=r.features_oracle(m,rows,reports)
        self.assertEqual(len(got['cells']),98);self.assertFalse(got['cells'][0]['available'])
        self.assertTrue(np.isnan(got['arrays'][r.HANDS[0]][0,0]))

    def test_future_original_rejected(self):
        m,rows,reports=fixture();reports['MSFT'][-1]['known_ns']=rows[0]['decision_cutoff_ns']+1
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_latest_known_blocks_old_chain(self):
        m,rows,reports=fixture();reports['MSFT'].append(dict(report_id='new',fiscal_year=2023,known_ns=2024,current='bad',scale='1000000'))
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_future_catalog_addition_cannot_change_prefix(self):
        m,rows,reports=fixture();before=r.features_oracle(m,rows,reports)
        reports['MSFT'].append(dict(report_id='future',fiscal_year=2023,known_ns=rows[-1]['decision_cutoff_ns']+1,current='bad',scale='1'))
        after=r.features_oracle(m,rows,reports)
        for key in r.EXTRA:np.testing.assert_array_equal(before['arrays'][key],after['arrays'][key])

    def test_axis_change_rejected(self):
        m,rows,reports=fixture();rows[1]['candidate_index']=2
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_missing_slot_rejected(self):
        m,rows,reports=fixture()
        with self.assertRaises(ValueError):r.features_oracle(m,rows[:-1],reports)

    def test_scale_conflict_rejected(self):
        m,rows,reports=fixture();reports['MSFT'][0]['scale']='1000'
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_common_nonpositive_or_nonfinite_scale_rejected(self):
        for scale in ('-1','0','NaN','Infinity'):
            m,rows,reports=fixture()
            for report in reports['MSFT']:report['scale']=scale
            with self.subTest(scale=scale), self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_tied_latest_timestamp_rejected(self):
        m,rows,reports=fixture(); extra=deepcopy(reports['MSFT'][-1]);extra['report_id']='conflicting_latest'
        reports['MSFT'].append(extra)
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_duplicate_original_fiscal_year_rejected(self):
        m,rows,reports=fixture();extra=deepcopy(reports['MSFT'][0]);extra.update(report_id='duplicate_year',known_ns=2018)
        reports['MSFT'].append(extra)
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_clock_for_another_day_rejected_even_if_ns_matches(self):
        m,rows,reports=fixture();clock=pd.Timestamp('2023-03-30T20:00:00Z')
        rows[0].update(decision_cutoff=str(clock),decision_cutoff_ns=int(clock.value))
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_year_gap_rejected(self):
        m,rows,reports=fixture();reports['MSFT'][1]['fiscal_year']=2018
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_aapl_unadmitted_not_silently_included(self):
        m,rows,reports=fixture();rows[2]['status']=r.USABLE
        with self.assertRaises(ValueError):r.features_oracle(m,rows,reports)

    def test_input_unchanged(self):
        m,rows,reports=fixture();before=deepcopy((m,rows,reports));r.features_oracle(m,rows,reports)
        self.assertEqual((m,rows,reports),before)


class ScoreTests(unittest.TestCase):
    def test_exact_point01_adjustment_and_one_episode_per_arm(self):
        a,m,hand=score_fixture();a['eligible'][0,0]=False
        helper=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=helper), \
                patch.object(helper,'episode',wraps=helper.episode) as episode:
            _,_,deltas=r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand))
        self.assertEqual(episode.call_count,4)
        expected=np.where(a['eligible'] & np.isfinite(hand),.01*np.nan_to_num(hand),0.)
        np.testing.assert_array_equal(deltas[r.NAMES[0]],np.zeros(hand.shape))
        for name in r.NAMES[1:]:np.testing.assert_array_equal(deltas[name],expected)

    def test_four_arms_have_shared_exact_incumbent_fallback(self):
        a,m,_=score_fixture();null=np.full(a['core'].shape,np.nan)
        scores,choices,_=r.score_oracle(a,m,dict.fromkeys(r.HANDS,null))
        for name in r.NAMES[1:]:
            self.assertEqual(choices[name],choices[r.NAMES[0]])
            self.assertEqual(scores[name].tobytes(),scores[r.NAMES[0]].tobytes())

    def test_score_prefix_isolation(self):
        a,m,hand=score_fixture();hands=dict.fromkeys(r.HANDS,hand)
        scores,choices,_=r.score_oracle(a,m,hands)
        for n in (0,1,2):
            ps,pc,_=r.score_oracle(a,m,hands,n)
            for name in r.NAMES:
                self.assertEqual(pc[name],choices[name][:n]);self.assertEqual(ps[name].tobytes(),scores[name][:n].tobytes())

    def test_all123_prefixes_of_a_full122_row_score_fixture(self):
        original,m,hand=score_fixture()
        a={key:np.concatenate([value]*61,axis=0) for key,value in original.items()}
        hand=np.concatenate([hand]*61,axis=0)
        dates=pd.date_range('2010-01-01',periods=122,freq='MS')
        m=dict(m,symbols=m['symbols']*61,dates=[str(day.date()) for day in dates],
               ends=[str((day+pd.offsets.MonthEnd()).date()) for day in dates])
        ss,cc,_=r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand))
        for n in range(123):
            ps,pc,_=r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand),n)
            for name in r.NAMES:
                self.assertEqual(pc[name],cc[name][:n]);self.assertEqual(ps[name].tobytes(),ss[name][:n].tobytes())

    def test_common_mask_mismatch_rejected(self):
        a,m,hand=score_fixture();hands={key:hand.copy() for key in r.HANDS};hands[r.HANDS[2]][0,0]=np.nan
        with self.assertRaises(ValueError):r.score_oracle(a,m,hands)

    def test_etf_data_rejected(self):
        a,m,hand=score_fixture();hand[:,-1]=.5
        with self.assertRaises(ValueError):r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand))

    def test_ordered_record_dates_and_fixed_two_or_three(self):
        a,m,hand=score_fixture();ss,cc,_=r.score_oracle(a,m,dict.fromkeys(r.HANDS,hand));out=r.order_oracle(m,ss,cc)
        self.assertEqual(set(out),set(r.NAMES))
        for name in r.NAMES:
            self.assertEqual(out[name][1]['signal_date'],m['dates'][1])
            self.assertEqual(len(out[name][1]['symbols']),3)

    def test_plan_hash_required_before_io(self):
        with patch.object(r.old.Reader,'sha',side_effect=AssertionError('Unexpected read')):
            for bad in (None,'z'*64,'a'*63):
                with self.subTest(hash=bad),self.assertRaises(ValueError):r.audit('/synthetic','plan.json',bad)

    def test_no_new_producer_module_imported(self):
        for name in ('monthly_revenue_persistence156_runner','after155_revenue_pilot_factor',
                     'after155_revenue_pilot_prepare','after155_multi_issuer_chain'):
            self.assertNotIn(name,sys.modules)


class SourceReviewContractTests(unittest.TestCase):
    def test_explicit_full_source_calendar_and_evidence_checks_required(self):
        review=dict(passed=True,actual_grid_verified=True,checks=dict.fromkeys(r.SOURCE_REVIEW_CHECKS,True))
        review['checks'].update(candidate_rows=5978,decision_dates=122,candidates_per_date=49)
        r.validate_source_review(review)
        for key in r.SOURCE_REVIEW_CHECKS:
            review['checks'][key]=False
            with self.subTest(check=key),self.assertRaises(ValueError):r.validate_source_review(review)
            review['checks'][key]=True
        review['checks']['candidate_rows']=5977
        with self.assertRaises(ValueError):r.validate_source_review(review)


if __name__=='__main__':unittest.main()
