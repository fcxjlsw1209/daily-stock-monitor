"""Synthetic accounting, source-mask, chronology and tamper tests only."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_filing_association157_outcome_review as m
from test_monthly_filing_association157_prepare_review import fixture


def order(day, end, symbols):
    return dict(date=str(day.date()), signal_date=str(day.date()), end=str(end.date()),
                symbols=list(symbols), scores=[.2]*len(symbols))


def synthetic_outputs(high_control=False):
    dates = pd.bdate_range('2016-01-04', periods=245); x = np.arange(245)
    close = pd.DataFrame({s: price*np.exp(rate*x+.007*np.sin(x/period)) for s, price, rate, period in
                         [('A', 10, .001, 4), ('B', 20, .0007, 7), ('C', 30, .03 if high_control else .0013, 6)]}, index=dates)
    opening = close*np.array([.999, 1.001, 1.0002])
    meta = dict(dates=[str(d.date()) for d in dates[:-1:2]], ends=[str(d.date()) for d in dates[2::2]],
                symbols=[['A', 'B', 'C']+[f'S{i}' for i in range(3, 49)] for _ in range(122)], etfs=[])
    orders = {n: [order(d, e, ['A', 'C'] if k and i % (k+2) == 0 else ['A', 'B'])
                  for i, (d, e) in enumerate(zip(dates[:-1:2], dates[2::2]))] for k, n in enumerate(m.NAMES)}
    data, results, navs, opens = {}, {}, {}, {}
    bounds = dict(early2016_2020=(meta['dates'][0], meta['ends'][52]), original2021_2026=(meta['dates'][53], meta['ends'][-1]),
                  training2016_2023=(meta['dates'][0], meta['ends'][87]), later34=(meta['dates'][88], meta['ends'][-1]))
    for name in m.NAMES:
        results[name] = {}
        for execution, cost, key in m.PATHS:
            met, nav, trades, holdings = m.replay(close, opening, orders[name], dates, execution, cost)
            prefix = name+'_'+key
            results[name][key] = dict(metrics=met, slices={k: m.h.slice_stat(dates, nav, *span) for k, span in bounds.items()})
            data[prefix+'_DAILY.json'] = [dict(date=str(d.date()), equity=float(v)) for d, v in zip(dates, nav)]
            data[prefix+'_TRADES.json'] = trades; data[prefix+'_HOLDINGS.json'] = holdings
            if key == 'same_close': navs[name] = nav
            if key == 'next_open_cost': opens[name] = nav
    rows = [dict(name=n, **m.h.measures(dates, navs[n], meta)) for n in m.NAMES]
    opened = [dict(name=n, **m.h.measures(dates, opens[n], meta)) for n in m.NAMES]
    choice = m.select(rows, opened)
    data.update({'RESULTS.json': results, 'TRAIN_TRIALS.json': rows, 'TRAIN_OPEN_TRIALS.json': opened,
        'FROZEN.json': dict(selection=choice), 'WALKFORWARD.json': m.walkforward(dates, navs, opens, meta),
        'VALIDATION.json': dict(selection=choice, pristine_oos=False, independent_outcome_review_pending=True,
                                no_observer_writes=True, no_intramonth_trades=True,
                                nuisance_beta_estimates_attempted=1600,nuisance_beta_estimates_completed=1598,new_predictive_model_fits=0),
        'FEATURE_AUDIT.json': dict(nuisance_beta_estimates_attempted=1600,nuisance_beta_estimates_completed=1598,new_predictive_model_fits=0)})
    return close, opening, orders, meta, data


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.original = synthetic_outputs()

    def setUp(self): self.close, self.opening, self.orders, self.meta, self.data = deepcopy(self.original)
    def check(self): return m.check_outcomes(self.close, self.opening, self.orders, self.meta, self.data.__getitem__)

    def test_twelve_paths_and_488_orders(self):
        report = self.check()
        self.assertTrue(report['passed']); self.assertEqual(report['execution_paths_verified'], 12)
        self.assertEqual(report['total_monthly_orders_verified'], 488)
        self.assertEqual(len(report['independent_monthly_comparison']), 122)

    def test_tampered_nav_rejected(self):
        self.data[m.NAMES[3]+'_next_open_cost_DAILY.json'][5]['equity'] += 10
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_trade_cost_rejected(self):
        self.data[m.NAMES[0]+'_next_open_cost_TRADES.json'][0]['cost'] += 10
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_holdings_rejected(self):
        self.data[m.NAMES[1]+'_same_close_HOLDINGS.json'][0]['symbols'] = []
        with self.assertRaises((AssertionError, ValueError)): self.check()

    def test_wrong_next_open_execution_day_rejected(self):
        self.data[m.NAMES[0]+'_next_open_zero_TRADES.json'][0]['date'] = self.meta['dates'][0]
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_frozen_choice_rejected(self):
        self.data['FROZEN.json']['selection']['primary'] = 'forged'
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_training_measure_rejected(self):
        self.data['TRAIN_TRIALS.json'][2]['training']['sharpe'] += .1
        with self.assertRaises(ValueError): self.check()

    def test_tampered_walkforward_rejected(self):
        self.data['WALKFORWARD.json']['folds'][1]['selection']['admitted'] = not self.data['WALKFORWARD.json']['folds'][1]['selection']['admitted']
        with self.assertRaises(ValueError): self.check()

    def test_single_holding_not_original_rule(self):
        self.orders[m.NAMES[0]][0].update(symbols=['A'], scores=[.1])
        with self.assertRaises(ValueError): m.validate_orders(self.close, self.opening, self.orders, self.meta)

    def test_later34_cannot_promote_challenger(self):
        rows = [dict(name=n, training=dict(cagr=.5, sharpe=1., max_drawdown=-.2), early={'cagr': .4}, recent={'cagr': .5}) for n in m.NAMES]
        rows[3]['later34'] = {'cagr': 1000.}
        self.assertFalse(m.select(rows, deepcopy(rows))['admitted'])

    def test_no_new_producer_imports(self):
        m.independent_imports()
        self.assertFalse(any(key in sys.modules for key in m.FORBIDDEN))

    def test_nuisance_estimates_are_reported_separately(self):
        report=self.check()
        self.assertEqual([report[k] for k in m.COUNTERS],[1600,1598,0])
        self.assertTrue(report['nuisance_estimate_counts_verified'])
        self.assertNotIn('models_fit',report)
        self.assertNotIn('zero_model_fit_count_verified',report)

    def test_validation_counter_tamper_rejected(self):
        self.data['VALIDATION.json'][m.COUNTERS[1]]-=1
        with self.assertRaises(ValueError):self.check()

    def test_unqualified_zero_fit_claim_rejected(self):
        self.data['FEATURE_AUDIT.json']['new_model_fits']=0
        with self.assertRaises(ValueError):self.check()

    def test_all_changed_months_are_reported(self):
        report=self.check()
        for i,row in enumerate(report['independent_monthly_comparison']):
            self.assertEqual(row['changed_arms'],[name for name in m.NAMES[1:]
                if self.orders[name][i]['symbols']!=self.orders[m.NAMES[0]][i]['symbols']])

    def test_target_uses_selected_arm_only(self):
        report=self.check()
        chosen=self.data['FROZEN.json']['selection']['primary']
        metric=self.data['RESULTS.json'][chosen]['same_close']['metrics']['cagr']
        self.assertEqual(report['selected_name'],chosen)
        self.assertAlmostEqual(report['primary_cagr'],metric)
        self.assertEqual(report['primary_target_reached'],metric>=.7)

    def test_high_return_unselected_control_cannot_claim_target(self):
        close,opening,orders,meta,data=synthetic_outputs(high_control=True)
        report=m.check_outcomes(close,opening,orders,meta,data.__getitem__)
        self.assertGreater(data['RESULTS.json'][m.NAMES[1]]['same_close']['metrics']['cagr'],.7)
        self.assertEqual(report['selected_name'],m.NAMES[0])
        self.assertFalse(report['primary_target_reached'])

    def test_selection_contract_preserves_controls_without_zero_nuisance_claim(self):
        contract=m.proposal_contract()
        self.assertEqual(contract['new_predictive_model_fits'],0)
        self.assertNotIn('new_model_fits',contract)
        self.assertIn('counted_separately',contract['nuisance_estimates'])

    def test_only_designated_primary_can_be_selected(self):
        rows=[dict(name=n,training=dict(cagr=.3,sharpe=1.,max_drawdown=-.2),
            early={'cagr':.3},recent={'cagr':.3}) for n in m.NAMES]
        rows[1]['training']['cagr']=9.
        self.assertEqual(m.select(rows,deepcopy(rows))['primary'],m.NAMES[0])


class FixedUnitTests(unittest.TestCase):
    def fixture(self):
        days = pd.bdate_range('2016-01-04', periods=4)
        close = pd.DataFrame(dict(A=[10., 20., 30., 40.], B=[20., 20., 20., 20.], C=[10., 10., 10., 10.]), index=days)
        opening = close.copy(); opening.iloc[1] = [10., 10., 10.]
        return days, close, opening

    def test_fixed_units_not_daily_rebalance(self):
        days, close, opening = self.fixture()
        _, nav, _, _ = m.replay(close, opening, [order(days[0], days[-1], ['A', 'B'])], days)
        np.testing.assert_allclose(nav, [100000, 150000, 200000, 250000])

    def test_entry_and_exit_fees(self):
        days, close, opening = self.fixture()
        met, nav, _, _ = m.replay(close, opening, [order(days[0], days[2], ['B', 'C']), order(days[2], days[-1], [])], days, 'close', .0005)
        self.assertAlmostEqual(nav[-1], 100000/1.0005*(1-.0005))
        self.assertAlmostEqual(met['fees_paid'], 100000-nav[-1])

    def test_next_open_lag(self):
        days, close, opening = self.fixture()
        _, nav, trades, _ = m.replay(close, opening, [order(days[0], days[-1], ['A', 'B'])], days, 'open', 0.)
        self.assertEqual(nav[0], 100000); self.assertEqual(nav[1], 200000)
        self.assertEqual(trades[0]['date'], str(days[1].date()))


class FeatureTests(unittest.TestCase):
    """Exercise the full independent matrix comparator with constructed oracle output."""
    @classmethod
    def setUpClass(cls):
        names=['S'+str(i) for i in range(32)]+['E'+str(i) for i in range(17)]
        days=[str(d.date()) for d in pd.bdate_range('2020-01-01',periods=122)]
        cls.meta=dict(dates=days,symbols=[names[:] for _ in days],etfs=names[32:])
        cls.rows=[];cells=[]
        arrays={key:np.full((122,49),np.nan) for key in m.HANDS}
        arrays['common_available']=np.zeros((122,49),bool)
        for i,day in enumerate(days):
            for j,name in enumerate(names):
                active=j<16
                row=dict(signal_index=i,candidate_index=j,symbol=name,decision_date=day,
                    decision_cutoff=day+'T20:00:00Z',decision_cutoff_ns=int(pd.Timestamp(day+'T20:00:00Z').value),
                    research_eligible=active,reasons=[] if active else ['held'],issuer_id=name if j<32 else None)
                cls.rows.append(row)
                cells.append(dict(signal_index=i,candidate_index=j,symbol=name,issuer_id=row['issuer_id'],
                    decision_date=day,source_research_eligible=active,available=active,reason=None if active else 'held',
                    beta=1. if active else None,raw={key:0. if active else None for key in m.HANDS},
                    price_window=['old','formation_end'] if active else None,
                    nuisance_beta_estimates_attempted=int(active),nuisance_beta_estimates_completed=int(active),
                    new_predictive_model_fits=0))
                if active:
                    for key in m.HANDS:arrays[key][i,j]=0.
                    arrays['common_available'][i,j]=True
        cls.expected=dict(arrays=arrays,cells=cells,nuisance_beta_estimates_attempted=1952,
            nuisance_beta_estimates_completed=1952,new_predictive_model_fits=0)
        cls.saved=dict(rows=5978,usable_rows=1952,cells=cells,**{k:cls.expected[k] for k in m.COUNTERS},
            clock_policy_id=m.CLOCK,exact_acceptance_certified=False,strategy_returns_or_nav_computed=False,
            original_eligibility_unchanged=True,filing_source_grid_verified=True,missing_reason_counts={'held':4026},
            common_coverage=m.preparation.coverage_oracle(cls.meta,cls.rows,arrays['common_available']))

    def setUp(self):
        self.arrays=deepcopy(self.expected['arrays']);self.feature=deepcopy(self.saved)

    def check(self):
        with patch.object(m.preparation,'features_oracle',return_value=self.expected):
            return m.verify_features(self.arrays,self.feature,self.meta,self.rows,[],None,{})

    def test_genuine_zero_preserved_on_available_rows(self):
        self.assertEqual(self.check(),1952)
        self.assertEqual(int(self.arrays['common_available'].sum()),1952)

    def test_held_row_cannot_be_zero_imputed(self):
        self.arrays[m.HANDS[0]][0,16]=0.
        with self.assertRaises(AssertionError):self.check()

    def test_common_mask_tamper_rejected(self):
        self.arrays['common_available'][0,16]=True
        with self.assertRaises(AssertionError):self.check()

    def test_numeric_oracle_mismatch_rejected(self):
        self.arrays[m.HANDS[2]][0,0]=.001
        with self.assertRaises(AssertionError):self.check()

    def test_beta_and_cutoff_identity_tamper_rejected(self):
        for field,value in [('beta',1.2),('decision_date','2099-01-01')]:
            self.feature=deepcopy(self.saved);self.feature['cells'][0][field]=value
            with self.assertRaises(ValueError):self.check()

    def test_false_exact_clock_claim_rejected(self):
        self.feature['exact_acceptance_certified']=True
        with self.assertRaises(ValueError):self.check()

    def test_nuisance_totals_and_missing_reason_tamper_rejected(self):
        self.feature[m.COUNTERS[0]]=1953
        with self.assertRaises(ValueError):self.check()
        self.feature=deepcopy(self.saved);self.feature['missing_reason_counts']={}
        with self.assertRaises(ValueError):self.check()

    def test_post_price_floor_cannot_use_pre_price_counts(self):
        # Reject a source-only gate masquerading as the post-price common mask.
        self.feature['common_coverage']['eras']['first88']['usable_slots']+=1
        with self.assertRaises(AssertionError):self.check()

    def test_real_independent_oracle_respects_old_window_and_counts(self):
        args=list(fixture(1));first=m.preparation.features_oracle(*args)
        self.assertEqual([first[k] for k in m.COUNTERS],[2,2,0])
        end=args[2][args[1][0]['signal_session_index']-42]
        args[3].loc[args[3].index>pd.Timestamp(end),'AAPL']=999999.
        after=m.preparation.features_oracle(*args)
        for key in (*m.HANDS,'common_available'):
            np.testing.assert_array_equal(first['arrays'][key],after['arrays'][key])
        args[3]['SPY']=1.
        failed=m.preparation.features_oracle(*args)
        self.assertEqual([failed[k] for k in m.COUNTERS],[2,0,0])


class ChronologyTests(unittest.TestCase):
    def test_no_numeric_reads_before_bindings_succeed(self):
        with patch.object(m,'bindings',side_effect=ValueError('source or phase not ready')),patch.object(m.Reader,'npz') as numeric:
            with self.assertRaisesRegex(ValueError,'source or phase'):
                m.audit('/private/tmp','a'*64)
            numeric.assert_not_called()

    def test_diagnostic_binds_frozen_and_precedes_audit(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as root:
            root = Path(root); (root/'FROZEN.json').write_text('{}')
            sha = hashlib_sha(root/'FROZEN.json')
            marker = dict(plan_sha256='a'*64, frozen_sha256=sha, started_utc='2020-01-02T00:00:00+00:00')
            (root/'DIAGNOSE_STARTED.json').write_text(json.dumps(marker))
            plan = dict(registered_utc='2020-01-01T00:00:00+00:00')
            m.verify_diagnostic_start(m.Reader(root), root, plan, 'a'*64, datetime(2020, 1, 3, tzinfo=timezone.utc))
            marker['frozen_sha256'] = 'b'*64
            (root/'DIAGNOSE_STARTED.json').write_text(json.dumps(marker))
            with self.assertRaises(ValueError):
                m.verify_diagnostic_start(m.Reader(root), root, plan, 'a'*64, datetime(2020, 1, 3, tzinfo=timezone.utc))

    def test_diagnostics_must_follow_registration_and_precede_audit(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as root:
            root=Path(root);(root/'FROZEN.json').write_text('{}')
            for timestamp in ('2019-12-31T00:00:00Z','2020-01-03T00:00:00Z','2020-01-02T00:00:00'):
                marker=dict(plan_sha256='a'*64,frozen_sha256=hashlib_sha(root/'FROZEN.json'),started_utc=timestamp)
                (root/'DIAGNOSE_STARTED.json').write_text(json.dumps(marker))
                with self.subTest(timestamp=timestamp),self.assertRaises(ValueError):
                    m.verify_diagnostic_start(m.Reader(root),root,dict(registered_utc='2020-01-01T00:00:00Z'),
                        'a'*64,datetime(2020,1,3,tzinfo=timezone.utc))


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.feature=dict(nuisance_beta_estimates_attempted=1952,nuisance_beta_estimates_completed=1950,new_predictive_model_fits=0)
        self.prep=dict(self.feature,passed=True,stage='prepare',orders_checked=488,candidate_cells_checked=5978,
            months_checked=122,prefixes_checked=123,filing_feature_matrices_verified=3,
            producer_modules_imported=False,strategy_returns_or_nav_computed=False,exact_acceptance_certified=False,
            **dict.fromkeys(m.PREPARE_CHECKS,True))

    def test_exact_preparation_scope_passes(self):
        m.check_prepare_scope(self.prep,self.feature)

    def test_missing_or_false_mandatory_check_rejected(self):
        for key in m.PREPARE_CHECKS:
            bad=deepcopy(self.prep);bad[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):m.check_prepare_scope(bad,self.feature)

    def test_wrong_stage_or_existing_outcomes_rejected(self):
        for key,value in [('stage','evaluate'),('producer_modules_imported',True),
                          ('strategy_returns_or_nav_computed',True),('exact_acceptance_certified',True)]:
            bad=deepcopy(self.prep);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.check_prepare_scope(bad,self.feature)

    def test_count_types_scope_and_nuisance_mismatch_rejected(self):
        for key in ('orders_checked','candidate_cells_checked','months_checked','prefixes_checked',
                    'filing_feature_matrices_verified',*m.COUNTERS):
            for value in (True,None,-1,self.prep[key]+1):
                bad=deepcopy(self.prep);bad[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.check_prepare_scope(bad,self.feature)

    def test_nuisance_bounds_and_ambiguous_fit_labels_rejected(self):
        for change in (dict(nuisance_beta_estimates_completed=1953),dict(nuisance_beta_estimates_attempted=3905),
                       dict(new_model_fits=0)):
            with self.assertRaises(ValueError):m.counters(dict(self.feature,**change))


def hashlib_sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    unittest.main()

