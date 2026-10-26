"""Synthetic-only independent159 checks; no new producer or market data reads."""
from copy import deepcopy
from datetime import date, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

import basic_shares159_independent_prepare as p
import basic_shares159_independent_outcome as o
import basic_shares159_independent as facade


def calendar():
    start = date(2015, 1, 1)
    sessions = [(start+timedelta(days=i)).isoformat() for i in range(1700)
                if (start+timedelta(days=i)).weekday() < 5]
    return dict(sessions=sessions, closes={s: p.ns(s+'T20:00:00Z') for s in sessions})


def event(symbol='AAPL'):
    ident = dict(security_id=symbol, issuer_id=symbol+':issuer',
                 identity_valid_from='2015-01-01', identity_valid_through='2019-01-01')
    periods = dict(current=dict(period_start='2016-01-01', period_end='2016-12-31', days_inclusive=366),
                   comparison=dict(period_start='2015-01-01', period_end='2015-12-31', days_inclusive=365))
    original = dict(event_id=symbol+':2016', symbol=symbol, identity=ident,
        inherited_holds=dict(inherited_rejection_codes=[]), release_date='2017-01-04', periods=periods,
        cells=dict(current=dict(value_in_shares='90'), comparison=dict(value_in_shares='100')))
    return dict(event_id=original['event_id'], symbol=symbol, identity=deepcopy(ident),
        original_event=original, inherited_holds=deepcopy(original['inherited_holds']),
        known_at='2017-01-06T20:00:00Z', usable_at='2017-01-11T20:00:00Z',
        selected_gaap_contexts=dict(selected_filed='2017-01-09'), period_end='2016-12-31',
        prior_period_end='2015-12-31', conditional_source_ready=True, unresolved_reasons=[],
        same_report_annual_pair=True, same_share_basis=True, strict_source_admitted=False,
        old_admission_flags_changed=False, current_shares='90', prior_shares='100')


def score_fixture():
    shape = (2, 49)
    baseline = np.tile(np.linspace(.20, .10, 49), (2, 1))
    arrays = dict(baseline=baseline, core=.5*baseline, core_weight=np.full(2, .5),
        eligible=np.ones(shape, bool), state_weak_narrow=np.zeros(2, bool), state_calm10=np.ones(2, bool),
        pre_score=baseline.copy(), effects=np.zeros(shape), forecast_available=np.zeros(shape, bool),
        rank_log_excess=np.zeros(shape), old_ml4=np.zeros(shape), extension_available=np.zeros(shape, bool),
        model_available=np.zeros(shape, bool), raw_local_constant=np.zeros(shape),
        raw_local_linear=np.zeros(shape), raw_source_log=np.zeros(shape))
    names=list(p.ISSUERS)+['S'+str(i) for i in range(28)]+['ETF'+str(i) for i in range(17)]
    meta=dict(symbols=[names[:], names[:]], etfs=names[-17:], dates=['2017-04-03', '2017-05-01'],
              ends=['2017-05-01', '2017-06-01'])
    hands={k:np.zeros(shape) for k in p.HANDS}
    hands[p.HANDS[0]][:, :4]=[-.5, -.2, .2, .5]
    hands[p.HANDS[1]][:, :4]=[-1., -1/3, 1/3, 1.]
    return arrays,meta,hands


def grid_fixture():
    cal=calendar();start=cal['sessions'].index('2017-03-13')
    signals=cal['sessions'][start:start+122]
    names=list(p.ISSUERS)+['S'+str(i)for i in range(28)]+['ETF'+str(i)for i in range(17)]
    meta=dict(dates=signals,symbols=[names[:]for _ in signals],etfs=names[-17:])
    events=[]
    for symbol,count,current in zip(p.ISSUERS,(10,10,12,12),('90','95','105','110')):
        e=event(symbol);e['current_shares']=current;e['original_event']['cells']['current']['value_in_shares']=current
        events.append(e)
        for k in range(count-1):
            events.append(dict(symbol=symbol,event_id=symbol+':future'+str(k),known_at='2020-01-01T20:00:00Z'))
    source,original=[],[]
    for i,signal in enumerate(signals):
        cut=cal['sessions'][start+i-42]
        for j,symbol in enumerate(names):
            row=dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=signal,
                     decision_cutoff=signal+'T20:00:00Z',decision_cutoff_ns=cal['closes'][signal])
            original.append(row)
            inherited=dict(original_row=deepcopy(row),cutoff=cut+'T20:00:00Z',cutoff_date=cut,cutoff_ns=cal['closes'][cut])
            source.append(dict(signal_index=i,candidate_index=j,symbol=symbol,cutoff=cut+'T20:00:00Z',
                original_availability_row=inherited,selected_event_id=symbol+':2016'if j<4 else None,
                conditional_source_ready=j<4,strict_source_admitted=False,old_admission_flags_changed=False))
    eligible=np.ones((122,49),bool)
    return dict(meta=meta,events=events,source_rows=source,original_rows=original,eligible=eligible,calendar=cal)


class ExactTests(unittest.TestCase):
    def test_negative_change_exact_and_unit_scale_invariant(self):
        self.assertEqual(p.negative_change('90', '100'), Fraction(1, 10))
        self.assertEqual(p.negative_change('9e9', '1e10'), Fraction(1, 10))
        self.assertEqual(p.negative_change('4', '3'), Fraction(-1, 3))
    def test_nonpositive_and_nonfinite_fail(self):
        for value in ('0', '-1', 'NaN', 'Infinity'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.negative_change(value, '100')
    def test_three_exact_values_same_float_have_distinct_ranks(self):
        a=Fraction(1, 3);epsilon=Fraction(1, 10**40)
        self.assertEqual(float(a), float(a+epsilon))
        _, rank=p.hands_oracle([a, a+epsilon, a+2*epsilon])
        np.testing.assert_array_equal(rank, [-1., 0., 1.])
    def test_ties_missing_and_scale(self):
        mag,rank=p.hands_oracle([Fraction(-1,20), Fraction(1,20), None])
        np.testing.assert_array_equal(mag, [-.25,.25,0.])
        np.testing.assert_array_equal(rank, [-1.,1.,0.])
        for values in ([None], [Fraction(2)], [Fraction(2),Fraction(2)]):
            self.assertTrue(all(not x.any() for x in p.hands_oracle(values)))
    def test_partial_ties_center_and_bound(self):
        mag,rank=p.hands_oracle([Fraction(0),Fraction(0),Fraction(1),Fraction(3)])
        np.testing.assert_allclose(rank, [-2/3,-2/3,1/3,1.], atol=1e-16)
        self.assertAlmostEqual(float(mag.sum()),0.)
        self.assertLessEqual(float(abs(mag).max()), .75)


class SourceTests(unittest.TestCase):
    def test_two_clocks_and_inclusive_usable_boundary(self):
        e=event();cal=calendar()
        self.assertEqual(p.source_status(e,'AAPL','2017-01-10',p.ns('2017-01-10T20:00:00Z'),cal), 'latest_not_yet_usable')
        self.assertIsNone(p.source_status(e,'AAPL','2017-01-11',p.ns('2017-01-11T20:00:00Z'),cal))
    def test_bad_latest_no_fallback_and_future_quality_not_read(self):
        old=event();bad=event();bad.update(event_id='bad',period_end='2017-01-01',conditional_source_ready=False)
        future=dict(symbol='AAPL',known_at='2020-01-01T20:00:00Z')
        self.assertIs(p.latest_event([old,bad,future],'AAPL',p.ns('2017-02-01T20:00:00Z')),bad)
    def test_ambiguous_latest_fails(self):
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            p.latest_event([event(),event()],'AAPL',p.ns('2017-02-01T20:00:00Z'))
    def test_wrong_filing_clock_fails(self):
        e=event();e['usable_at']=e['known_at']
        with self.assertRaisesRegex(ValueError,'Corroboration'):
            p.source_status(e,'AAPL','2017-02-01',p.ns('2017-02-01T20:00:00Z'),calendar())
    def test_550_day_boundary_and_identity(self):
        e=event();cal=calendar();end=date(2016,12,31)
        for age,expected in ((550,None),(551,'latest_period_outside_age_window')):
            cut=(end+timedelta(days=age)).isoformat()
            self.assertEqual(p.source_status(e,'AAPL',cut,p.ns(cut+'T20:00:00Z'),cal), expected)
        e['identity']['security_id']='AMD';e['original_event']['identity']['security_id']='AMD'
        self.assertEqual(p.source_status(e,'AAPL','2017-02-01',p.ns('2017-02-01T20:00:00Z'),cal),'identity_not_valid_at_cutoff')
    def test_full_grid_eligible_before_centering(self):
        fixture=grid_fixture();fixture['eligible'][0,0]=False
        result=p.features_oracle(**fixture)
        self.assertEqual(len(result['cells']),5978)
        self.assertFalse(result['arrays']['common_available'][0,0])
        np.testing.assert_array_equal(result['arrays'][p.HANDS[1]][0,:4],[0.,1.,0.,-1.])
        for n in (0,1,88):
            prefix=p.features_oracle(**fixture,n=n)
            for key in p.EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],result['arrays'][key][:n])
    def test_full_grid_rejects_wrong_cutoff_and_duplicate_issuer(self):
        fixture=grid_fixture();fixture['source_rows'][0]['cutoff']='2017-01-01T20:00:00Z'
        with self.assertRaisesRegex(ValueError,'t-42'):p.features_oracle(**fixture,n=1)
        fixture=grid_fixture()
        e=next(e for e in fixture['events']if e['event_id']=='AMD:2016')
        e['identity']['issuer_id']='AAPL:issuer';e['original_event']['identity']['issuer_id']='AAPL:issuer'
        with self.assertRaisesRegex(ValueError,'Duplicate issuer'):p.features_oracle(**fixture,n=1)


class ScoreTests(unittest.TestCase):
    def test_delta_once_and_modified_pre_top2_lock(self):
        arrays,meta,hands=score_fixture();result=p.score_oracle(arrays,meta,hands)
        for name,key in zip(p.NAMES[1:],p.HANDS):
            np.testing.assert_array_equal(result['deltas'][name], .01*hands[key])
            modified=arrays['pre_score'][0]+.01*hands[key][0]
            top=sorted(range(49),key=lambda j:(-modified[j],meta['symbols'][0][j]))[:2]
            self.assertEqual(result['episodes'][name][0]['top2'],[meta['symbols'][0][j]for j in top])
            self.assertTrue(set(top).issubset(result['choices'][name][0]))
    def test_neutral_exact_and_inputs_unchanged(self):
        arrays,meta,hands=score_fixture();before={k:v.tobytes()for k,v in arrays.items()}
        for value in hands.values():value[:]=0
        result=p.score_oracle(arrays,meta,hands)
        for name in p.NAMES[1:]:
            self.assertEqual(result['scores'][name].tobytes(),result['scores'][p.NAMES[0]].tobytes())
            self.assertEqual(result['choices'][name],result['choices'][p.NAMES[0]])
        self.assertEqual(before,{k:v.tobytes()for k,v in arrays.items()})
    def test_ineligible_and_etf_correction_fail(self):
        arrays,meta,hands=score_fixture();arrays['eligible'][0,0]=False
        with self.assertRaisesRegex(ValueError,'Ineligible'):p.score_oracle(arrays,meta,hands)
    def test_future_hand_poison_ignored(self):
        arrays,meta,hands=score_fixture();expected=p.score_oracle(arrays,meta,hands,n=1)
        for key in hands:
            hands[key]=hands[key].astype(object);hands[key][1]='future'
        # Prefix input itself is converted, so future rows cannot contaminate dtype.
        clean={key:np.asarray(value[:1],float)for key,value in hands.items()}
        actual=p.score_oracle(arrays,meta,clean,n=1)
        for name in p.NAMES:self.assertEqual(actual['scores'][name].tobytes(),expected['scores'][name].tobytes())


class SelectionTests(unittest.TestCase):
    def rows(self,gains=(0.,.006,.006)):
        return [dict(name=n,training=dict(cagr=.4+g,sharpe=1.5,max_drawdown=-.2),
                     early=dict(cagr=.4),recent=dict(cagr=.4))for n,g in zip(p.NAMES,gains)]
    def test_fixed_arm_tie_and_failed_source_guard(self):
        rows=self.rows();changed={n:dict(passed=True)for n in p.NAMES[1:]}
        chosen=o.select_first88_oracle(rows,rows,dict(passed=True),changed)
        self.assertEqual(chosen['primary'],p.NAMES[1])
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=False),changed)['primary'],p.NAMES[0])
    def test_fifty_basis_points_required(self):
        rows=self.rows((0.,.0049,.004));changed={n:dict(passed=True)for n in p.NAMES[1:]}
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[0])
    def test_order_permutation_is_not_holdings_change(self):
        orders={name:[dict(symbols=['AAPL','AMD','S'])for _ in range(122)]for name in p.NAMES}
        orders[p.NAMES[1]][0]['symbols'].reverse()
        self.assertEqual(o.changed_support_oracle(orders,[],p.NAMES[1])['months'],[])


class HostAndAccountingTests(unittest.TestCase):
    def test_invalid_plan_fails_before_market_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/facade.OUT;directory.mkdir(parents=True)
            plan=dict(round=158,status='REGISTERED',arms=list(p.NAMES))
            path=directory/'PLAN.json';path.write_text(json.dumps(plan))
            reader=facade.Reader(tmp)
            with self.assertRaisesRegex(ValueError,'Fixed registered'):
                facade.bind(reader,hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
            self.assertEqual(set(reader.seen),{str(path.resolve())})
    def test_prepare_refuses_existing_outcomes_before_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/facade.OUT;directory.mkdir(parents=True)
            plan=dict(round=159,status='REGISTERED',arms=list(p.NAMES),output_directory=str(directory),
                runtime=dict(python=facade.sys.version,numpy=np.__version__,pandas=pd.__version__),
                measurement=facade.measurement_contract(),selection_contract=facade.selection_contract(),
                actual_features_before_registration=0,actual_orders_before_registration=0,
                actual_returns_before_registration=0,no_observer_writes=True)
            path=directory/'PLAN.json';path.write_text(json.dumps(plan))
            (directory/'TRAIN_TRIALS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'precede all outcomes'):
                facade.bind(facade.Reader(tmp),hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
    def test_nine_paths_synthetic_fixed_unit_accounting(self):
        dates=pd.bdate_range('2020-01-01',periods=245)
        names=['AAPL','AMD','MSFT']
        close=pd.DataFrame(100.,index=dates,columns=names);opening=close.copy()
        signals=[str(d.date())for d in dates[::2][:-1]]
        ends=[str(d.date())for d in dates[::2][1:]]
        meta=dict(dates=signals,ends=ends,symbols=[names[:]for _ in signals])
        orders={name:[dict(date=d,signal_date=d,end=e,symbols=names[:],scores=[.3,.2,.1])
                      for d,e in zip(signals,ends)]for name in p.NAMES}
        result=o.nine_path_oracle(close,opening,orders,meta)
        self.assertEqual((result['paths'],result['orders'],len(result['details'])),(9,366,9))
        for name in p.NAMES:
            for key in ('same_close','next_open_zero'):
                self.assertEqual(result['results'][name][key]['metrics']['total_return'],0.)
            self.assertLess(result['results'][name]['next_open_cost']['metrics']['total_return'],0.)


if __name__ == '__main__':
    p.independent_imports()
    unittest.main(verbosity=2)
