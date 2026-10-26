"""Generated-input accounting, selection and evidence-tamper tests only."""
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import annual_lag158_independent_outcome as m


def order(day, end, names):
    return dict(date=str(day.date()), signal_date=str(day.date()), end=str(end.date()),
                symbols=list(names), scores=[.2]*len(names))


def small_prices():
    days = pd.bdate_range('2016-01-04', periods=4)
    close = pd.DataFrame({'A':[10.,20.,30.,40.], 'B':[20.,20.,20.,20.], 'C':[10.,10.,10.,10.]}, index=days)
    opening = pd.DataFrame({'A':[5.,10.,20.,30.], 'B':[10.,10.,20.,20.], 'C':[10.,10.,10.,10.]}, index=days)
    return days, close, opening


def synthetic_outputs():
    days = pd.bdate_range('2016-01-05', periods=245); x = np.arange(245.)
    close = pd.DataFrame({s:b*np.exp(r*x+.007*np.sin(x/f)) for s,b,r,f in
        [('A',10,.001,4), ('B',20,.0007,7), ('C',30,.0013,6), ('D',15,.0009,5)]}, index=days)
    opening = close*np.array([.999,1.001,1.0002,.9998])
    pool = ['A','B','C','D']+['S'+str(i) for i in range(4,32)]+['ETF'+str(i) for i in range(17)]
    meta = dict(dates=[str(d.date()) for d in days[:-1:2]], ends=[str(d.date()) for d in days[2::2]],
        symbols=[pool[:] for _ in range(122)], etfs=pool[-17:])
    orders = {name:[] for name in m.NAMES}
    for i,(day,end) in enumerate(zip(meta['dates'],meta['ends'])):
        for name in m.NAMES:
            changed = (name == m.NAMES[2] and i%3 == 0) or (name == m.NAMES[1] and i%4 == 0)
            orders[name].append(order(pd.Timestamp(day),pd.Timestamp(end), ['A','B','D'] if changed else ['A','B','C']))
    data, results, navs, opens, metrics = {}, {}, {}, {}, {}
    bounds = dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),
        original2021_2026=(meta['dates'][53],meta['ends'][-1]),
        training2016_2023=(meta['dates'][0],meta['ends'][87]), later34=(meta['dates'][88],meta['ends'][-1]))
    for name in m.NAMES:
        results[name] = {}
        for execution,cost,key in m.PATHS:
            met,nav,tr,hold = m.replay(close,opening,orders[name],days,execution,cost)
            results[name][key] = dict(metrics=met, slices={k:m.h.slice_stat(days,nav,*span) for k,span in bounds.items()})
            prefix = name+'_'+key
            data[prefix+'_DAILY.json'] = [dict(date=str(d.date()),equity=float(v)) for d,v in zip(days,nav)]
            data[prefix+'_TRADES.json'],data[prefix+'_HOLDINGS.json'] = tr,hold
            if key == 'same_close':navs[name],metrics[name] = nav,met
            if key == 'next_open_cost':opens[name] = nav
    rows = [dict(name=name,**m.h.measures(days,navs[name],meta)) for name in m.NAMES]
    opened = [dict(name=name,**m.h.measures(days,opens[name],meta)) for name in m.NAMES]
    selected = m.select(rows,opened)
    months = [dict(signal_index=i,date=d,shared_query_population=2,informative_pair=True,
        initial_partial_fallback=False,unchanged_current_correction=False,unchanged_change_correction=False)
        for i,d in enumerate(meta['dates'])]
    data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'TRAIN_OPEN_TRIALS.json':opened,
        'FROZEN.json':dict(selection=selected,selection_count=1),
        'RANK_AUDIT.json':dict(months=months,prefixes_checked=123),
        'FEATURE_AUDIT.json':dict(new_predictive_model_fits=0,nuisance_estimates=0)})
    data['MONTH_COMPARISON.json'] = m.monthly(orders,close,meta,months)
    data['ATTRIBUTION.json'] = m.attribution(data['MONTH_COMPARISON.json'],metrics)
    data['VALIDATION.json'] = dict(primary=selected['primary'],selection=selected,selection_count=1,
        original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,
        independent_outcome_review_pending=True,pristine_oos=False,new_predictive_model_fits=0,nuisance_estimates=0)
    incremental=m.incremental_evidence(orders,months)
    data['FEATURE_AUDIT.json'].update(incremental)
    data['VALIDATION.json']['primary_no_incremental_evidence']=incremental['primary_no_incremental_evidence']
    return close,opening,orders,meta,data


class AccountingTests(unittest.TestCase):
    def test_fixed_units_not_daily_rebalance(self):
        d,p,o = small_prices(); _,nav,tr,_ = m.replay(p,o,[order(d[0],d[-1],['A','B'])],d)
        np.testing.assert_array_equal(nav,[100000.,150000.,200000.,250000.]);self.assertEqual(len(tr),2)

    def test_three_equal_start_weights(self):
        d,p,o = small_prices(); _,nav,tr,_ = m.replay(p,o,[order(d[0],d[-1],['A','B','C'])],d)
        np.testing.assert_allclose(nav,[100000.,100000.*4/3,100000.*5/3,200000.],rtol=1e-14)
        self.assertEqual(len(tr),3)

    def test_next_session_open(self):
        d,p,o = small_prices(); _,nav,tr,_ = m.replay(p,o,[order(d[0],d[-1],['A','B'])],d,'open',0.)
        self.assertEqual(nav[0],100000.);self.assertEqual(nav[1],200000.)
        self.assertEqual(tr[0]['date'],str(d[1].date()));self.assertEqual({t['adjusted_price'] for t in tr},{10.})

    def test_fee_equation_purchase_and_liquidation(self):
        d,p,o = small_prices(); rows=[order(d[0],d[2],['B','C']),order(d[2],d[-1],[])]
        met,nav,tr,_ = m.replay(p,o,rows,d,'close',.0005)
        final=100000/1.0005*(1-.0005)
        self.assertAlmostEqual(nav[-1],final,places=7);self.assertAlmostEqual(met['fees_paid'],100000-final,places=7)
        self.assertEqual(len(tr),4)

    def test_cash_stays_cash(self):
        d,p,o=small_prices();met,nav,tr,_=m.replay(p,o,[order(d[0],d[-1],[])],d)
        np.testing.assert_array_equal(nav,np.full(4,100000.));self.assertEqual(tr,[]);self.assertEqual(met['total_return'],0.)

    def test_reorder_does_not_create_trade(self):
        d,p,o=small_prices();p[:]=10.;o[:]=10.
        _,nav,tr,_=m.replay(p,o,[order(d[0],d[2],['A','B','C']),order(d[2],d[-1],['C','B','A'])],d)
        self.assertEqual(len(tr),3);np.testing.assert_array_equal(nav,np.full(4,100000.))

    def test_no_terminal_forced_sale(self):
        d,p,o=small_prices();_,_,tr,_=m.replay(p,o,[order(d[0],d[-1],['A','B','C'])],d)
        self.assertEqual(len(tr),3);self.assertTrue(all(t['side']=='buy' for t in tr))

    def test_consumed_missing_price_rejected(self):
        d,p,o=small_prices();p.at[d[2],'A']=np.nan
        with self.assertRaises(AssertionError):m.replay(p,o,[order(d[0],d[-1],['A','B'])],d)

    def test_unregistered_cost_rejected(self):
        d,p,o=small_prices()
        with self.assertRaises(ValueError):m.replay(p,o,[],d,'close',.01)

    def test_alias_normalization_does_not_mutate(self):
        rows=[dict(symbol='research_equity_FB'),dict(symbols=['BRK.B','A'])];before=deepcopy(rows)
        self.assertEqual(m.normalize(rows),[dict(symbol='META'),dict(symbols=['BRK-B','A'])]);self.assertEqual(rows,before)


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=synthetic_outputs()
    def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
    def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
    def candidates(self):
        return [dict(name=n,training=dict(cagr=.5,sharpe=1.,max_drawdown=-.2),
                     early=dict(cagr=.3),recent=dict(cagr=.4)) for n in m.NAMES]

    def test_nine_paths(self):
        out=self.check();self.assertTrue(out['passed']);self.assertEqual(out['execution_paths_verified'],9)
        self.assertEqual(out['total_monthly_orders_verified'],366);self.assertEqual(len(out['replay_details']),9)

    def test_exactly_one_first88_reconstruction_no_fold_read(self):
        reads=[]
        def read(name):reads.append(name);return self.data[name]
        with patch.object(m,'select',wraps=m.select) as pick:
            m.check_outcomes(self.px,self.op,self.orders,self.meta,read)
            self.assertEqual(pick.call_count,1)
        self.assertNotIn('WALKFORWARD.json',reads)

    def test_nav_tamper(self):
        self.data[m.NAMES[0]+'_same_close_DAILY.json'][4]['equity']+=10
        with self.assertRaises(AssertionError):self.check()

    def test_trade_fee_tamper(self):
        self.data[m.NAMES[0]+'_next_open_cost_TRADES.json'][0]['cost']+=1
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_trade_execution_date(self):
        self.data[m.NAMES[0]+'_next_open_zero_TRADES.json'][0]['date']=self.meta['dates'][0]
        with self.assertRaises(AssertionError):self.check()

    def test_holdings_tamper(self):
        self.data[m.NAMES[0]+'_same_close_HOLDINGS.json'][0]['symbols']=['A','D']
        with self.assertRaises(AssertionError):self.check()

    def test_metrics_and_slices_tamper(self):
        for section in ('metrics','slices'):
            with self.subTest(section=section):
                self.data=deepcopy(self.fixture[4]);result=self.data['RESULTS.json'][m.NAMES[1]]['same_close']
                target=result['metrics'] if section=='metrics' else result['slices']['later34']
                target['cagr']+=.001
                with self.assertRaises(ValueError):self.check()

    def test_annual_partial_flag_tamper(self):
        self.data['RESULTS.json'][m.NAMES[0]]['same_close']['metrics']['annual_returns'][0]['partial']=False
        with self.assertRaises(ValueError):self.check()

    def test_frozen_selection_and_count_tamper(self):
        self.data['FROZEN.json']['selection_count']=2
        with self.assertRaises(ValueError):self.check()
        self.data=deepcopy(self.fixture[4]);self.data['FROZEN.json']['selection']['primary']='bad'
        with self.assertRaises(AssertionError):self.check()

    def test_validation_count_and_no_fit_claims(self):
        for key,value in [('selection_count',True),('new_predictive_model_fits',1),('nuisance_estimates',1)]:
            with self.subTest(key=key):
                self.data=deepcopy(self.fixture[4]);self.data['VALIDATION.json'][key]=value
                with self.assertRaises(ValueError):self.check()

    def test_no_incremental_disclosure_cannot_be_forged(self):
        self.data['VALIDATION.json']['primary_no_incremental_evidence']=True
        with self.assertRaises(ValueError):self.check()

    def test_no_incremental_when_no_holdings_change(self):
        self.orders[m.NAMES[2]]=deepcopy(self.orders[m.NAMES[0]])
        got=m.incremental_evidence(self.orders,self.data['RANK_AUDIT.json']['months'])
        self.assertTrue(got['primary_no_incremental_evidence'])
        self.assertEqual(got['ordered_holdings_changed_months'][m.NAMES[2]],0)

    def test_tie_or_better_control_does_not_promote_control(self):
        rows=self.candidates();self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[0])
        rows[1]['training']['cagr']=100.;self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[0])

    def test_strict_challenger_and_epsilon(self):
        rows=self.candidates();rows[2]['training']['cagr']+=.5e-12;self.assertFalse(m.select(rows,rows)['admitted'])
        rows[2]['training']['cagr']+=.1;self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[2])

    def test_each_risk_guard_and_boundary(self):
        for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
            with self.subTest(key=key):
                rows=self.candidates();rows[2]['training']['cagr']+=.1
                rows[2][section][key]-=tol;self.assertTrue(m.select(rows,rows)['admitted'])
                rows[2][section][key]-=.001;self.assertFalse(m.select(rows,rows)['admitted'])

    def test_open_incumbent_tie_allowed_control_tie_rejected(self):
        rows=self.candidates();rows[1]['training']['cagr']=.4;rows[2]['training']['cagr']=.6
        opened=deepcopy(rows);opened[2]['training']['cagr']=.5;self.assertTrue(m.select(rows,opened)['admitted'])
        opened[1]['training']['cagr']=.5;self.assertFalse(m.select(rows,opened)['admitted'])

    def test_later34_never_selects(self):
        rows=self.candidates();rows[2]['later34']={'cagr':999};rows[0]['later34']={'cagr':-999}
        self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[0])

    def test_bad_metric_retains_incumbent(self):
        for value in (float('nan'),float('inf'),None,True,'0.9'):
            rows=self.candidates();rows[2]['training']['cagr']+=.1;rows[2]['training']['sharpe']=value
            self.assertFalse(m.select(rows,rows)['admitted'])

    def test_single_holding_and_unknown_identifier(self):
        for names in (['A'],['KO','A']):
            self.orders=deepcopy(self.fixture[2]);self.orders[m.NAMES[0]][0]['symbols']=names
            self.orders[m.NAMES[0]][0]['scores']=[.2]*len(names)
            with self.assertRaises(ValueError):m.validate_orders(self.px,self.op,self.orders,self.meta)

    def test_monthly_and_attribution_tamper(self):
        self.data['MONTH_COMPARISON.json'][0]['returns'][m.NAMES[1]]+=.001
        with self.assertRaises(ValueError):self.check()
        self.data=deepcopy(self.fixture[4]);self.data['ATTRIBUTION.json']['comparisons']['change_vs_current']['changed_months']+=1
        with self.assertRaises(ValueError):self.check()

    def test_three_comparisons_and_set_changes(self):
        self.assertEqual(set(self.data['ATTRIBUTION.json']['comparisons']),
            {'change_vs_incumbent','current_vs_incumbent','change_vs_current'})
        rows=deepcopy(self.data['MONTH_COMPARISON.json']);rows[0]['holdings'][m.NAMES[2]]=list(reversed(rows[0]['holdings'][m.NAMES[0]]))
        metrics={n:self.data['RESULTS.json'][n]['same_close']['metrics'] for n in m.NAMES}
        out=m.attribution(rows,metrics)
        self.assertNotIn(rows[0]['date'],[r['date'] for r in out['comparisons']['change_vs_incumbent']['changes']])

    def test_singleton_and_initial_are_zero_corrections(self):
        months=deepcopy(self.data['RANK_AUDIT.json']['months']);months[0].update(shared_query_population=1,informative_pair=False)
        with self.assertRaises(ValueError):m.validate_months(months,self.meta)
        meta=deepcopy(self.meta);meta['dates'][0]='2016-08-17';months[0].update(date='2016-08-17',initial_partial_fallback=True)
        with self.assertRaises(ValueError):m.validate_months(months,meta)

    def test_zero_correction_requires_exact_orders(self):
        months=deepcopy(self.data['RANK_AUDIT.json']['months']);months[0]['unchanged_change_correction']=True
        with self.assertRaises(AssertionError):m.monthly(self.orders,self.px,self.meta,months)

    def test_future_prices_do_not_affect_first88_nav(self):
        dates=m.validate_orders(self.px,self.op,self.orders,self.meta);end=pd.Timestamp(self.meta['ends'][87]);prefix=dates[dates<=end]
        for execution,cost in [('close',0.),('open',.0005)]:
            expected=m.replay(self.px.loc[:end],self.op.loc[:end],self.orders[m.NAMES[2]][:88],prefix,execution,cost)[1]
            px=self.px.copy();op=self.op.copy();px.loc[px.index>end]*=2;op.loc[op.index>end]*=3
            actual=m.replay(px,op,self.orders[m.NAMES[2]],dates,execution,cost)[1]
            np.testing.assert_array_equal(expected,actual[:len(expected)])


class FeatureBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_annual_lag158_independent_prepare import fixture
        cls.data=fixture()
        cls.prep=m.preparation()
        cls.expected=cls.prep.features_oracle(**cls.data)

    def setUp(self):
        self.arrays=deepcopy(self.expected['arrays']);self.arrays['eligible']=self.data['eligible'].copy()
        self.rank=self.prep.rank_audit_oracle(self.data['meta'],self.expected)
        self.digests={k:'a'*64 for k in ('source_overlay','source_review','source_prior')}
        self.feature=dict(rows=5978,months=122,dose=.01,max_absolute_score_correction=.005,
            fixed_issuers=dict(self.prep.ISSUERS),usable_rows=230,lag_pairs_evaluated=230,
            negative_current_lag_definition='-(filingDate_current-reportDate_current).days',
            negative_lag_change_definition='previous_lag_days-current_lag_days',
            rank_definition='(average_rank1-1)/(n-1)-0.5; n<=1 gives0; rank only common original-eligible mask',
            source_conditionally_admitted_rows=230,new_predictive_model_fits=0,nuisance_estimates=0,
            source_admission_changed=False,exact_acceptance_certified=False,historical_price_identity_certified=False,
            strategy_returns_or_nav_computed=False,numerical_features_computed=True,source_contradictions_hard_failed=True,
            feature_prefixes_checked=123,cells=deepcopy(self.expected['cells']),monthly=deepcopy(self.expected['monthly']),
            coverage=self.prep.coverage_oracle(self.data['rows'],self.expected),
            reason_counts=dict(Counter(r for c in self.expected['cells'] for r in c['reasons'])),
            informative_rank_months=sum(x['current_rank_nonzero']>0 or x['change_rank_nonzero']>0 for x in self.expected['monthly']))
        self.feature.update({k+'_sha256':v for k,v in self.digests.items()})

    def check(self):
        d=self.data
        return m.verify_features(self.arrays,self.feature,self.rank,d['meta'],d['rows'],
                                 d['event_metadata'],d['source_mask'],d['calendar'],self.digests)

    def test_independent_actual_interface_on_synthetic_grid(self):self.assertEqual(self.check(),230)

    def test_raw_rank_or_common_tamper(self):
        for key in m.EXTRA:
            with self.subTest(key=key):
                self.arrays=deepcopy(self.expected['arrays']);self.arrays['eligible']=self.data['eligible'].copy()
                self.arrays[key][0,0]=False if key=='common_available' else 999.
                with self.assertRaises(AssertionError):self.check()

    def test_feature_audit_source_and_cell_tamper(self):
        self.feature['source_overlay_sha256']='b'*64
        with self.assertRaises(ValueError):self.check()
        self.feature['source_overlay_sha256']='a'*64;self.feature['cells'][0]['current_lag_days']+=1
        with self.assertRaises(ValueError):self.check()

    def test_rank_audit_and_eligibility_tamper(self):
        self.rank['months'][0]['shared_query_population']=1
        with self.assertRaises(AssertionError):self.check()
        self.rank=self.prep.rank_audit_oracle(self.data['meta'],self.expected);self.arrays['eligible'][0,0]=False
        with self.assertRaises(ValueError):self.check()


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.out=self.root/m.STUDY;self.out.mkdir(parents=True)
        self.sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.write=lambda f,v:(self.out/f).write_text(json.dumps(v))
        source=self.root/'source.txt';source.write_text('synthetic independent source')
        self.sources={str(source):self.sha(source)}
        self.plan=dict(registered_utc='2020-01-01T00:00:00+00:00',independent_prepare_source_hashes=self.sources)
        self.write('PLAN.json',self.plan);self.plan_sha=self.sha(self.out/'PLAN.json')
        for name in m.CACHE_FILES:self.write(name,{})
        self.write('CACHE.json',{f:self.sha(self.out/f) for f in m.CACHE_FILES})
        self.artifacts={str(self.out/f):self.sha(self.out/f) for f in (*m.CACHE_FILES,'PLAN.json','CACHE.json')}
        self.report=dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
            orders_checked=366,candidate_cells_checked=5978,months_checked=122,prefixes_checked=123,
            lag_feature_matrices_verified=2,new_predictive_model_fits=0,nuisance_estimates=0,
            source_hashes=self.sources,artifact_hashes=self.artifacts,plan_sha256=self.plan_sha,
            cache_sha256=self.sha(self.out/'CACHE.json'),orders_sha256=self.sha(self.out/'PRETRAIN_ORDERS.json'),
            **dict.fromkeys(m.preparation().CHECKS,True))
        self.write('INDEPENDENT_PREPARE.json',self.report)
        for f in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json'):self.write(f,{})
        self.freeze()

    def freeze(self):
        self.write('FROZEN.json',dict(selection_count=1,pristine_oos=False,all_diagnostic_orders_frozen=True,
            hashes={f:self.sha(self.out/f) for f in m.FROZEN_FILES}))
        self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,
            frozen_sha256=self.sha(self.out/'FROZEN.json'),started_utc='2021-01-01T00:00:00+00:00'))

    def check_prepare(self):
        return m.verify_prepare_receipt(m.Reader(self.root),self.out,self.plan,self.plan_sha,self.sources)

    def check_frozen(self):
        return m.verify_frozen(m.Reader(self.root),self.out,self.plan,self.plan_sha,
                              datetime(2022,1,1,tzinfo=timezone.utc))

    def test_complete_prepare_and_phase_bindings(self):self.check_prepare();self.check_frozen()

    def test_every_prepare_flag_required(self):
        for key in m.preparation().CHECKS:
            with self.subTest(key=key):
                bad=deepcopy(self.report);bad[key]=False;self.write('INDEPENDENT_PREPARE.json',bad)
                with self.assertRaises(ValueError):self.check_prepare()

    def test_prepare_count_types_and_isolation(self):
        for key,value in [('lag_feature_matrices_verified',True),('orders_checked',488),
            ('nuisance_estimates',1),('producer_modules_imported',True),('returns_or_nav_computed',True)]:
            with self.subTest(key=key):
                bad=deepcopy(self.report);bad[key]=value;self.write('INDEPENDENT_PREPARE.json',bad)
                with self.assertRaises(ValueError):self.check_prepare()

    def test_cache_drift(self):
        self.write('SCORES.npz',{'changed':True})
        with self.assertRaises(ValueError):self.check_prepare()

    def test_unreviewed_cache_member(self):
        del self.report['artifact_hashes'][str(self.out/'RANK_AUDIT.json')]
        self.write('INDEPENDENT_PREPARE.json',self.report)
        with self.assertRaises(ValueError):self.check_prepare()

    def test_unbound_extra_prepare_artifact(self):
        self.write('extra.json',{});self.report['artifact_hashes'][str(self.out/'extra.json')]=self.sha(self.out/'extra.json')
        self.write('INDEPENDENT_PREPARE.json',self.report)
        with self.assertRaises(ValueError):self.check_prepare()

    def test_no_fold_artifact_allowed(self):
        self.write('WALKFORWARD.json',{'folds':[]})
        with self.assertRaisesRegex(ValueError,'fold-selection'):self.check_frozen()

    def test_frozen_prepare_binding_required(self):
        frozen=json.loads((self.out/'FROZEN.json').read_text());del frozen['hashes']['INDEPENDENT_PREPARE.json']
        self.write('FROZEN.json',frozen)
        with self.assertRaises(ValueError):self.check_frozen()

    def test_diagnostic_chronology_and_digest(self):
        for key,value in [('started_utc','2999-01-01T00:00:00+00:00'),('started_utc','2019-01-01T00:00:00+00:00'),
                          ('started_utc','2021-01-01T00:00:00'),('plan_sha256','a'*64),('frozen_sha256','b'*64)]:
            with self.subTest(key=key):
                self.freeze();marker=json.loads((self.out/'DIAGNOSE_STARTED.json').read_text());marker[key]=value
                self.write('DIAGNOSE_STARTED.json',marker)
                with self.assertRaises(ValueError):self.check_frozen()

    def test_source_files_close_over_independent_dependencies(self):
        paths=m.source_files();self.assertIn(m.ROOT/'work/annual_lag158_independent_outcome.py',paths)
        self.assertIn(m.ROOT/'work/monthly_ridge_support70_result_review.py',paths)
        self.assertIn(m.ROOT/'work/annual_lag158_independent_prepare.py',paths)
        self.assertTrue(all(p.is_file() for p in paths));m.independent_imports()


if __name__ == '__main__':
    unittest.main(verbosity=2)
