"""Synthetic122-month accounting and binding tests; no real data are loaded."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_financial_coverage154_outcome_review as m


def order(day, end, names):
    return dict(date=str(pd.Timestamp(day).date()), signal_date=str(pd.Timestamp(day).date()),
                end=str(pd.Timestamp(end).date()), symbols=list(names), scores=[.2]*len(names))


def synthetic_outputs():
    days = pd.bdate_range("2016-01-04", periods=245)
    x = np.arange(len(days))
    specs = [("AAPL", 10, .001, 4), ("PLTR", 20, .0007, 7), ("PYPL", 30, .0013, 8)]
    close = pd.DataFrame({s: base*np.exp(rate*x+.007*np.sin(x/freq))
                          for s, base, rate, freq in specs}, index=days)
    opening = close*np.array([.999, 1.001, 1.0002])
    names = ["AAPL", "PLTR", "PYPL", "AMZN", "INTC", "KO", "MU"]+["S"+str(i) for i in range(25)]+["ETF"+str(i) for i in range(17)]
    meta = dict(dates=[str(d.date()) for d in days[:-1:2]], ends=[str(d.date()) for d in days[2::2]],
                symbols=[names.copy() for _ in range(122)], etfs=names[-17:])
    orders = {name: [order(day, end, ["AAPL", "PYPL"] if k and i % (k+2) == 0 else ["AAPL", "PLTR"])
                     for i, (day, end) in enumerate(zip(meta["dates"], meta["ends"]))]
              for k, name in enumerate(m.NAMES)}
    data, results, navs, open_navs = {}, {}, {}, {}
    bounds = dict(early2016_2020=(meta["dates"][0], meta["ends"][52]),
                  original2021_2026=(meta["dates"][53], meta["ends"][-1]),
                  training2016_2023=(meta["dates"][0], meta["ends"][87]),
                  later34=(meta["dates"][88], meta["ends"][-1]))
    for name in m.NAMES:
        results[name] = {}
        for execution, cost, key in m.PATHS:
            met, nav, trades, holdings = m.replay(close, opening, orders[name], days, execution, cost)
            prefix = name+"_"+key
            results[name][key] = dict(metrics=met, slices={
                k: m.h.slice_stat(days, nav, *span) for k, span in bounds.items()})
            data[prefix+"_DAILY.json"] = [dict(date=str(d.date()), equity=float(v)) for d, v in zip(days, nav)]
            data[prefix+"_TRADES.json"], data[prefix+"_HOLDINGS.json"] = trades, holdings
            if key == "same_close":
                navs[name] = nav
            if key == "next_open_cost":
                open_navs[name] = nav
    rows = [dict(name=name, **m.h.measures(days, navs[name], meta)) for name in m.NAMES]
    opened = [dict(name=name, **m.h.measures(days, open_navs[name], meta)) for name in m.NAMES]
    selection = m.select(rows, opened)
    arrays = {}
    hand = np.full((122, 49), np.nan)
    for name, index, value in zip(m.HANDS, (0, 3, 4, 5, 6, 1), (.2, .1, .15, -.2, .25, -.3)):
        hand = hand.copy()
        hand[:, index] = value
        arrays[name] = hand
    for key, name in zip(('original_available', 'batch1_available', 'batch2_available',
                          'batch3_available', 'batch4_available', 'expanded_available'), m.HANDS):
        arrays[key] = np.isfinite(arrays[name])
    cells = [dict(signal_index=i, pool_index=j, signal_date=day, symbol=symbol,
                  available=j == 1, reason="available" if j == 1 else "not_added",
                  earnings_bounded=-.3 if j == 1 else None)
             for i, (day, pool) in enumerate(zip(meta["dates"], meta["symbols"]))
             for j, symbol in enumerate(pool)]
    feature = dict(cells=cells, original_cells=122, batch1_cells=244, batch2_cells=366, batch3_cells=488, batch4_cells=610, expanded_cells=732, added_cells=122,
                   original_input_unchanged=True, batch1_input_unchanged=True, batch2_input_unchanged=True, batch3_input_unchanged=True, batch4_input_unchanged=True, new_model_fits=0, returns_computed=False,
                   reasons=dict(Counter(c["reason"] for c in cells)))
    data.update({
        "RESULTS.json": results, "TRAIN_TRIALS.json": rows, "TRAIN_OPEN_TRIALS.json": opened,
        "FROZEN.json": dict(selection=selection),
        "WALKFORWARD.json": m.walkforward(days, navs, open_navs, meta), "FEATURE_AUDIT.json": feature,
        "VALIDATION.json": dict(selection=selection, new_model_fits=0, no_observer_writes=True,
                               no_intramonth_trades=True, independent_outcome_review_pending=True,
                               pristine_oos=False)})
    return close, opening, orders, meta, data, arrays


class SelectionTests(unittest.TestCase):
    def groups(self):
        rows = [dict(name=name, training=dict(cagr=.5, sharpe=2., max_drawdown=-.2),
                     early=dict(cagr=.4), recent=dict(cagr=.6)) for name in m.NAMES]
        return rows, deepcopy(rows)

    def winner(self):
        rows, opened = self.groups()
        rows[6]["training"]["cagr"] = opened[6]["training"]["cagr"] = .6
        return rows, opened

    def test_ties_retain_incumbent(self):
        self.assertEqual(m.select(*self.groups())["primary"], m.NAMES[0])

    def test_original_control_cannot_be_promoted(self):
        rows, opened = self.groups()
        rows[1]["training"]["cagr"] = opened[1]["training"]["cagr"] = 9.
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_primary_wins_six_controls_both_paths(self):
        self.assertTrue(m.select(*self.winner())["admitted"])

    def test_open_incumbent_noninferiority(self):
        rows, opened = self.winner()
        opened[0]["training"]["cagr"] = .6
        self.assertTrue(m.select(rows, opened)["admitted"])

    def test_open_original_tie_rejected(self):
        rows, opened = self.winner()
        opened[1]["training"]["cagr"] = .6
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_open_batch1_tie_rejected(self):
        rows, opened = self.winner()
        opened[2]["training"]["cagr"] = .6
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_all_six_closed_ties_rejected(self):
        for i in range(6):
            rows, opened = self.winner()
            rows[i]["training"]["cagr"] = .6
            self.assertFalse(m.select(rows, opened)["admitted"])

    def test_each_risk_guard_each_path_each_control(self):
        for mode in (0, 1):
            for i in range(6):
                for section, key, tol in (("training", "sharpe", .05),
                                          ("training", "max_drawdown", .03),
                                          ("early", "cagr", .025), ("recent", "cagr", .025)):
                    pair = self.winner()
                    pair[mode][i][section][key] = pair[mode][6][section][key]+tol+.001
                    self.assertFalse(m.select(*pair)["admitted"])

    def test_exact_risk_boundaries_inclusive(self):
        rows, opened = self.winner()
        for group in (rows, opened):
            for section, key, tol in (("training", "sharpe", .05),
                                     ("training", "max_drawdown", .03),
                                     ("early", "cagr", .025), ("recent", "cagr", .025)):
                group[6][section][key] = group[0][section][key]-tol
        self.assertTrue(m.select(rows, opened)["admitted"])

    def test_tiny_strict_advantage_rejected(self):
        rows, opened = self.winner()
        rows[6]["training"]["cagr"] = .5+5e-13
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_invalid_metrics_fail(self):
        for value in (None, True, "2", float("nan"), float("inf")):
            rows, opened = self.winner()
            opened[6]["training"]["sharpe"] = value
            self.assertFalse(m.select(rows, opened)["admitted"])
        rows, opened = self.winner()
        del rows[6]["early"]["cagr"]
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_later34_does_not_select(self):
        rows, opened = self.groups()
        rows[6]["later34"] = {"cagr": 999}
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_exact_arm_order(self):
        rows, opened = self.winner()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(ValueError):
            m.select(rows, opened)


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic_outputs()

    def setUp(self):
        self.px, self.op, self.orders, self.meta, self.data, self.arrays = deepcopy(self.fixture)

    def check(self):
        return m.check_outcomes(self.px, self.op, self.orders, self.meta, self.data.__getitem__)

    def matrix(self):
        return m.verify_feature_matrices(self.arrays, self.data["FEATURE_AUDIT.json"], self.meta)

    def test_twenty_one_paths_and_854_orders(self):
        out = self.check()
        self.assertTrue(out["passed"])
        self.assertEqual(out["execution_paths_verified"], 21)
        self.assertEqual(out["total_monthly_orders_verified"], 854)
        self.assertEqual(len(out["independent_monthly_comparison"]), 122)

    def test_nav_tamper(self):
        self.data[m.NAMES[5]+"_next_open_cost_DAILY.json"][4]["equity"] += 10
        with self.assertRaises(AssertionError):
            self.check()

    def test_trade_cost_tamper(self):
        self.data[m.NAMES[0]+"_next_open_cost_TRADES.json"][0]["cost"] += 1
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_holdings_tamper(self):
        self.data[m.NAMES[1]+"_same_close_HOLDINGS.json"][0]["symbols"] = []
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_wrong_open_execution_date(self):
        self.data[m.NAMES[0]+"_next_open_zero_TRADES.json"][0]["date"] = self.meta["dates"][0]
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_single_name_order_rejected(self):
        self.orders[m.NAMES[0]][0].update(symbols=["AAPL"], scores=[.2])
        with self.assertRaises(ValueError):
            m.validate_orders(self.px, self.op, self.orders, self.meta)

    def test_frozen_selection_tamper(self):
        self.data["FROZEN.json"]["selection"]["primary"] = "forged"
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_training_open_metric_tamper(self):
        self.data["TRAIN_OPEN_TRIALS.json"][2]["training"]["sharpe"] += .01
        with self.assertRaises(ValueError):
            self.check()

    def test_walkforward_selection_tamper(self):
        entry = self.data["WALKFORWARD.json"]["folds"][0]["selection"]
        entry["admitted"] = not entry["admitted"]
        with self.assertRaises(ValueError):
            self.check()

    def test_new_model_or_noncausal_diagnostic_scope(self):
        for key, value in (("new_model_fits", 1), ("new_model_fits", False),
                           ("pristine_oos", True), ("no_intramonth_trades", False)):
            original = deepcopy(self.data["VALIDATION.json"])
            self.data["VALIDATION.json"][key] = value
            with self.assertRaises(ValueError):
                self.check()
            self.data["VALIDATION.json"] = original

    def test_distinct_masks_valid_and_twelve_arrays_checked(self):
        self.assertEqual(self.matrix(), 12)
        self.assertFalse(np.array_equal(self.arrays["original_available"],
                                       self.arrays["expanded_available"]))

    def test_old_earnings_cannot_change(self):
        self.arrays["earnings_expanded"][0, 0] += .01
        with self.assertRaises(ValueError):
            self.matrix()

    def test_batch1_earnings_cannot_change(self):
        self.arrays["earnings_expanded"][0, 3] += .01
        with self.assertRaises(ValueError):
            self.matrix()

    def test_batch1_coverage_cannot_disappear(self):
        self.arrays["expanded_available"][0, 3] = False
        self.arrays["earnings_expanded"][0, 3] = np.nan
        with self.assertRaises(ValueError):
            self.matrix()

    def test_old_coverage_cannot_disappear(self):
        self.arrays["expanded_available"][0, 0] = False
        self.arrays["earnings_expanded"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.matrix()

    def test_no_unrelated_new_coverage(self):
        self.arrays["expanded_available"][0, 6] = True
        self.arrays["earnings_expanded"][0, 6] = .1
        with self.assertRaises(ValueError):
            self.matrix()

    def test_new_value_matches_cell(self):
        self.arrays["earnings_expanded"][0, 1] += .01
        with self.assertRaises(ValueError):
            self.matrix()

    def test_new_value_must_be_bounded(self):
        self.arrays["earnings_expanded"][0, 1] = 1.1
        self.data["FEATURE_AUDIT.json"]["cells"][1]["earnings_bounded"] = 1.1
        with self.assertRaises(ValueError):
            self.matrix()

    def test_new_issuer_not_preexisting(self):
        self.arrays["earnings_original"][0, 1] = -.3
        self.arrays["original_available"][0, 1] = True
        with self.assertRaises(ValueError):
            self.matrix()

    def test_unavailable_pseudo_value_rejected(self):
        self.data["FEATURE_AUDIT.json"]["cells"][2]["earnings_bounded"] = .1
        with self.assertRaises(ValueError):
            self.matrix()

    def test_etf_cannot_have_either_financial_hand(self):
        self.arrays["earnings_original"][0, 32] = self.arrays["earnings_expanded"][0, 32] = .1
        self.arrays["original_available"][0, 32] = self.arrays["expanded_available"][0, 32] = True
        with self.assertRaises(ValueError):
            self.matrix()

    def test_exact_cell_axes_and_population(self):
        for mutate in ("drop", "reorder"):
            feature = deepcopy(self.data["FEATURE_AUDIT.json"])
            if mutate == "drop":
                feature["cells"].pop()
            else:
                feature["cells"][0], feature["cells"][1] = feature["cells"][1], feature["cells"][0]
            with self.assertRaises(ValueError):
                m.verify_feature_matrices(self.arrays, feature, self.meta)

    def test_coverage_counts_and_reasons(self):
        for key in ("original_cells", "batch1_cells", "batch2_cells", "batch3_cells", "batch4_cells", "expanded_cells", "added_cells"):
            feature = deepcopy(self.data["FEATURE_AUDIT.json"])
            feature[key] += 1
            with self.assertRaises(ValueError):
                m.verify_feature_matrices(self.arrays, feature, self.meta)
        self.data["FEATURE_AUDIT.json"]["reasons"] = {}
        with self.assertRaises(ValueError):
            self.matrix()

    def test_no_producer_modules_imported(self):
        for name in ("monthly_financial_coverage154_runner", "monthly_financial_coverage154_factor",
                     "monthly_financial_coverage154_selection"):
            self.assertNotIn(name, sys.modules)



class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.folder=self.root/m.STUDY;self.folder.mkdir(parents=True)
        self.audit_time=datetime(2026,10,20,tzinfo=timezone.utc)
    def put(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def npz(self,path,values):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('wb') as f:np.savez(f,**values)
        return p
    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def refs(self,directory):
        result={}
        for k,n in [('plan','PLAN.json'),('cache','CACHE.json'),('completion','COMPLETION.json'),('outcome','INDEPENDENT_OUTCOME.json')]:
            p=self.root/directory/n
            if not p.exists():self.put(directory+'/'+n,{})
            result[k]=dict(path=str(p),sha256=self.sha(p))
        return result
    def chain(self):
        base={str(i):np.full((122,49),float(i)) for i in range(27)}
        hand=np.full((122,49),np.nan);mask=np.isfinite(hand)
        self.npz(m.prep.LOCAL+'/DATA.npz',base)
        old=self.root/'legacy';previous=self.root/'prior'
        old_arrays=dict(base,earnings_bounded=hand,cash_flow_bounded=hand,comp_adjusted_cash_flow_bounded=hand,common_available=mask)
        prior_arrays=dict(base,earnings_original=hand,earnings_expanded=hand,original_available=mask,expanded_available=mask)
        for folder,arrays,arms in [('legacy',old_arrays,('incumbent_local67','earnings_control')),('prior',prior_arrays,('incumbent_local67','original_earnings','expanded_earnings'))]:
            self.npz(folder+'/DATA.npz',arrays)
            self.npz(folder+'/SCORES.npz',{k:np.zeros((122,49)) for k in arms})
            self.put(folder+'/PRETRAIN_ORDERS.json',{k:[] for k in arms})
            self.put(folder+'/CACHE.json',{n:self.sha(self.root/folder/n) for n in ('DATA.npz','SCORES.npz','PRETRAIN_ORDERS.json')})
        prior151_arrays=dict(prior_arrays,earnings_batch1=hand,batch1_available=mask)
        self.npz('prior151/DATA.npz',prior151_arrays)
        self.npz('prior151/SCORES.npz',{k:np.zeros((122,49)) for k in ('incumbent_local67','original_earnings','batch1_earnings','expanded_earnings')})
        self.put('prior151/PRETRAIN_ORDERS.json',{k:[] for k in ('incumbent_local67','original_earnings','batch1_earnings','expanded_earnings')})
        self.put('prior151/CACHE.json',{n:self.sha(self.root/'prior151'/n) for n in ('DATA.npz','SCORES.npz','PRETRAIN_ORDERS.json')})
        prior152_arrays=dict(prior151_arrays,earnings_batch2=hand,batch2_available=mask)
        names152=('incumbent_local67','original_earnings','batch1_earnings','batch2_earnings','expanded_earnings')
        self.npz('prior152/DATA.npz',prior152_arrays)
        self.npz('prior152/SCORES.npz',{k:np.zeros((122,49)) for k in names152})
        self.put('prior152/PRETRAIN_ORDERS.json',{k:[] for k in names152})
        self.put('prior152/CACHE.json',{n:self.sha(self.root/'prior152'/n) for n in ('DATA.npz','SCORES.npz','PRETRAIN_ORDERS.json')})
        prior153_arrays=dict(prior152_arrays,earnings_batch3=hand,batch3_available=mask)
        names153=(*names152[:-1],'batch3_earnings','expanded_earnings')
        self.npz('prior153/DATA.npz',prior153_arrays)
        self.npz('prior153/SCORES.npz',{k:np.zeros((122,49)) for k in names153})
        self.put('prior153/PRETRAIN_ORDERS.json',{k:[] for k in names153})
        self.put('prior153/CACHE.json',{n:self.sha(self.root/'prior153'/n) for n in ('DATA.npz','SCORES.npz','PRETRAIN_ORDERS.json')})
        source=self.put('synthetic_independent.py',{'synthetic':True})
        sources={str(source):self.sha(source)}
        inputs={k:self.put('input/'+k+'.json',[]) for k in m.INPUTS}
        self.put('input/original_orders.json',[]);self.put('input/sessions.json',{'sessions':['2020-01-01','2020-01-02']})
        legacy_refs,prior_refs=self.refs('legacy'),self.refs('prior')
        self.put('prior151/PLAN.json',{'prior150':prior_refs})
        prior151_refs=self.refs('prior151')
        self.put('prior152/PLAN.json',{'prior151':prior151_refs})
        prior152_refs=self.refs('prior152')
        self.put('prior153/PLAN.json',{'prior152':prior152_refs})
        prior153_refs=self.refs('prior153')
        raw={str(p):self.sha(p) for p in self.root.rglob('*') if p.is_file()}
        plan=dict(round=154,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[6],dose=.01,output_directory=str(self.folder),
            runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),hashes=raw,
            inputs={k:dict(path=str(p),sha256=self.sha(p)) for k,p in inputs.items()},
            raw_source_hashes=sources,independent_prepare_source_hashes=sources,
            legacy149=legacy_refs,prior153=prior153_refs,coverage_sources={'batch1':{},'batch2':{},'batch3':{},'batch4':{},'batch5':{}},
            actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
        self.put(m.STUDY+'/PLAN.json',plan);self.plan_sha=self.sha(self.folder/'PLAN.json')
        arrays=dict(base,earnings_batch4=hand,batch4_available=mask,earnings_batch3=hand,batch3_available=mask,earnings_batch2=hand,batch2_available=mask,earnings_original=hand,earnings_batch1=hand,earnings_expanded=hand,original_available=mask,batch1_available=mask,expanded_available=mask)
        self.npz(m.STUDY+'/DATA.npz',arrays);self.npz(m.STUDY+'/SCORES.npz',{k:np.zeros((122,49)) for k in m.NAMES})
        self.put(m.STUDY+'/PRETRAIN_ORDERS.json',{k:[] for k in m.NAMES})
        for n in ('META.json','PRICE_PARITY.json','FEATURE_AUDIT.json'):self.put(m.STUDY+'/'+n,{})
        self.put(m.STUDY+'/CACHE.json',{n:self.sha(self.folder/n) for n in m.CACHE_FILES})
        self.receipt=dict(stage='prepare',passed=True,**dict.fromkeys(m.PREPARE_CHECKS,True),orders_checked=854,source_cells_checked=5978,stock_cells_checked=3904,monthly_label_models_fit=0,producer_modules_imported=False,returns_or_nav_computed=False,
            plan_sha256=self.plan_sha,cache_sha256=self.sha(self.folder/'CACHE.json'),orders_sha256=self.sha(self.folder/'PRETRAIN_ORDERS.json'),source_hashes=sources,
            artifact_hashes={str(self.folder/n):self.sha(self.folder/n) for n in (*m.CACHE_FILES,'PLAN.json','CACHE.json')})
        self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.receipt)
        for n in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json'):self.put(m.STUDY+'/'+n,{})
        self.put(m.STUDY+'/FROZEN.json',dict(hashes={n:self.sha(self.folder/n) for n in m.FROZEN_FILES},pristine_oos=False,all_diagnostic_orders_frozen=True))
        self.put(m.STUDY+'/DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,started_utc='2026-10-19T00:00:00Z'))
        self.calls=[]
        # Source-package internals have dedicated oracle tests. These stubs
        # exercise the actual bindings call path and full artifact/control
        # checks, detecting the prior module/local-variable collision.
        for name,result in [('verify_proposal',{}),('verify_legacy',{}),('verify_prior153',{'prior152':prior152_refs}),('verify_coverage',({},{},{},{},{}))]:
            def call(*args,_name=name,_result=result,**kwargs):self.calls.append(_name);return _result
            patcher=patch.object(m.prep,name,side_effect=call);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(m,'source_files',return_value=[source]);patcher.start();self.addCleanup(patcher.stop)
    def check(self):return m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time)
    def test_full_bindings_source_gate_call_path(self):
        self.chain();self.assertEqual(self.check(),self.folder)
        self.assertEqual(self.calls,['verify_proposal','verify_legacy','verify_prior153','verify_coverage'])
    def test_all_four_prepare_flags(self):
        self.chain();self.receipt.pop('prior150_batch1_earnings_cache_and_mask_verified');self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'Independent prepare'):self.check()
    def test_854_count_required(self):
        self.chain();self.receipt['orders_checked']=488;self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'Exact prepare count'):self.check()
    def test_complete_prepare_artifacts_required(self):
        self.chain();self.receipt['artifact_hashes'].pop(str(self.folder/'PRICE_PARITY.json'));self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'Complete prepare artifacts'):self.check()
    def test_cache_tamper_rejected(self):
        self.chain();self.put(m.STUDY+'/PRICE_PARITY.json',{'changed':True})
        with self.assertRaises(ValueError):self.check()
    def test_prior_cache_tamper_rejected(self):
        self.chain();self.put('prior/PRETRAIN_ORDERS.json',{'changed':True})
        with self.assertRaises(ValueError):self.check()
    def test_source_gate_failure_propagates(self):
        self.chain()
        with patch.object(m.prep,'verify_coverage',side_effect=ValueError('bad source')):
            with self.assertRaisesRegex(ValueError,'bad source'):self.check()
    def test_sha_required_before_io(self):
        for value in (None,'', 'x'*64, 'A'*64):
            with self.assertRaises(ValueError):m.bindings(m.Reader(self.root),value)
    def test_marker_requires_prior_utc(self):
        self.chain();self.put(m.STUDY+'/DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,started_utc='2999-01-01T00:00:00Z'))
        with self.assertRaises(ValueError):self.check()
    def test_marker_binding(self):
        self.chain();self.put(m.STUDY+'/DIAGNOSE_STARTED.json',dict(plan_sha256='b'*64,started_utc='2026-10-19T00:00:00Z'))
        with self.assertRaises(ValueError):self.check()


class FifthArmTests(unittest.TestCase):
    def test_batch2_control_tie_blocks_promotion(self):
        t=SelectionTests();a,b=t.winner();a[3]['training']['cagr']=a[6]['training']['cagr']
        self.assertFalse(m.select(a,b)['admitted'])
    def test_open_batch2_control_tie_blocks_promotion(self):
        t=SelectionTests();a,b=t.winner();b[3]['training']['cagr']=b[6]['training']['cagr']
        self.assertFalse(m.select(a,b)['admitted'])
    def test_batch2_cache_cannot_change(self):
        t=BindingTests();t.setUp()
        try:
            t.chain();p=t.root/'prior151/DATA.npz'
            with np.load(p) as data:arrays={k:data[k] for k in data.files}
            arrays['earnings_expanded'][0,0]=.2;t.npz('prior151/DATA.npz',arrays)
            with self.assertRaises(ValueError):t.check()
        finally:t.doCleanups()
    def test_missing_new_prepare_proof_rejected(self):
        t=BindingTests();t.setUp()
        try:
            t.chain();t.receipt.pop('prior151_batch2_earnings_cache_and_mask_verified');t.put(m.STUDY+'/INDEPENDENT_PREPARE.json',t.receipt)
            with self.assertRaises(ValueError):t.check()
        finally:t.doCleanups()
    def test_batch2_feature_disappearance_rejected(self):
        _,_,_,meta,data,arrays=synthetic_outputs()
        arrays['earnings_batch2'][0,3]=np.nan;arrays['batch2_available'][0,3]=False
        with self.assertRaises(ValueError):m.verify_feature_matrices(arrays,data['FEATURE_AUDIT.json'],meta)


class SixthArmTests(unittest.TestCase):
    def test_batch3_control_tie_blocks_each_path(self):
        for mode in (0, 1):
            with self.subTest(mode=mode):
                pair = SelectionTests().winner()
                pair[mode][4]['training']['cagr'] = pair[mode][6]['training']['cagr']
                self.assertFalse(m.select(*pair)['admitted'])

    def test_missing_batch3_prepare_proofs_rejected(self):
        for key in ('prior152_batch3_earnings_cache_and_mask_verified',
                    'batch3_source_records_and_holds_preserved'):
            with self.subTest(key=key):
                t = BindingTests()
                t.setUp()
                try:
                    t.chain()
                    t.receipt.pop(key)
                    t.put(m.STUDY+'/INDEPENDENT_PREPARE.json', t.receipt)
                    with self.assertRaisesRegex(ValueError, 'Independent prepare'):
                        t.check()
                finally:
                    t.doCleanups()

    def test_prior152_cache_score_and_orders_cannot_change(self):
        for filename in ('DATA.npz', 'SCORES.npz', 'PRETRAIN_ORDERS.json'):
            with self.subTest(filename=filename):
                t = BindingTests()
                t.setUp()
                try:
                    t.chain()
                    p = t.root/'prior152'/filename
                    p.write_bytes(p.read_bytes()+b'changed')
                    with self.assertRaises(ValueError):
                        t.check()
                finally:
                    t.doCleanups()

    def test_batch3_feature_must_remain_bit_exact(self):
        _, _, _, meta, data, arrays = synthetic_outputs()
        arrays['earnings_expanded'][0, 5] += 1e-14
        with self.assertRaises(ValueError):
            m.verify_feature_matrices(arrays, data['FEATURE_AUDIT.json'], meta)

    def test_batch3_finite_coverage_cannot_disappear(self):
        _, _, _, meta, data, arrays = synthetic_outputs()
        arrays['earnings_expanded'][0, 5] = np.nan
        arrays['expanded_available'][0, 5] = False
        with self.assertRaises(ValueError):
            m.verify_feature_matrices(arrays, data['FEATURE_AUDIT.json'], meta)

    def test_prior152_cannot_add_other_issuer(self):
        _, _, _, meta, data, arrays = synthetic_outputs()
        for name in ('earnings_batch3', 'earnings_batch4', 'earnings_expanded'):
            arrays[name][0, 7] = .1
        for name in ('batch3_available', 'batch4_available', 'expanded_available'):
            arrays[name][0, 7] = True
        with self.assertRaisesRegex(ValueError, 'Prior152 unrelated'):
            m.verify_feature_matrices(arrays, data['FEATURE_AUDIT.json'], meta)

    def test_zero_is_available_and_not_missing(self):
        _, _, _, meta, data, arrays = synthetic_outputs()
        arrays['earnings_expanded'][0, 1] = 0.
        data['FEATURE_AUDIT.json']['cells'][1]['earnings_bounded'] = 0.
        self.assertEqual(m.verify_feature_matrices(arrays, data['FEATURE_AUDIT.json'], meta), 12)

    def test_signed_zero_old_control_is_preserved(self):
        _, _, _, meta, data, arrays = synthetic_outputs()
        for name in m.HANDS:
            arrays[name][0, 0] = -0.
        arrays['earnings_expanded'][0, 0] = 0.
        with self.assertRaisesRegex(ValueError, 'preserved finite'):
            m.verify_feature_matrices(arrays, data['FEATURE_AUDIT.json'], meta)

    def test_non_boolean_mask_is_rejected(self):
        _, _, _, meta, data, arrays = synthetic_outputs()
        arrays['batch3_available'] = arrays['batch3_available'].astype(int)
        with self.assertRaisesRegex(ValueError, 'Exact masks'):
            m.verify_feature_matrices(arrays, data['FEATURE_AUDIT.json'], meta)

    def test_missing_seventh_result_arm_rejected(self):
        px, op, orders, meta, data, _ = synthetic_outputs()
        data['RESULTS.json'].pop(m.NAMES[6])
        with self.assertRaisesRegex(ValueError, 'Seven result'):
            m.check_outcomes(px, op, orders, meta, data.__getitem__)

    def test_terminal_nav_and_date_are_checked(self):
        for key, value in (('date', '2099-01-01'), ('equity', 1.)):
            with self.subTest(key=key):
                px, op, orders, meta, data, _ = synthetic_outputs()
                data[m.NAMES[0]+'_same_close_DAILY.json'][-1][key] = value
                with self.assertRaises((ValueError, AssertionError)):
                    m.check_outcomes(px, op, orders, meta, data.__getitem__)


class SeventhArmTests(unittest.TestCase):
    def test_batch4_tie_blocks_each_selection_path(self):
        for mode in (0,1):
            pair=SelectionTests().winner()
            pair[mode][5]['training']['cagr']=pair[mode][6]['training']['cagr']
            self.assertFalse(m.select(*pair)['admitted'])

    def test_missing_prior153_and_batch4_prepare_flags_fail(self):
        for key in ('prior153_batch4_earnings_cache_and_mask_verified',
                    'batch4_source_records_and_holds_preserved'):
            t=BindingTests();t.setUp()
            try:
                t.chain();t.receipt.pop(key)
                t.put(m.STUDY+'/INDEPENDENT_PREPARE.json',t.receipt)
                with self.assertRaisesRegex(ValueError,'Independent prepare'):t.check()
            finally:t.doCleanups()

    def test_prior153_cache_scores_and_orders_are_bound(self):
        for name in ('DATA.npz','SCORES.npz','PRETRAIN_ORDERS.json'):
            t=BindingTests();t.setUp()
            try:
                t.chain();p=t.root/'prior153'/name;p.write_bytes(p.read_bytes()+b'changed')
                with self.assertRaises(ValueError):t.check()
            finally:t.doCleanups()

    def test_batch4_exact_value_and_mask_are_preserved(self):
        for missing in (False,True):
            _,_,_,meta,data,arrays=synthetic_outputs()
            arrays['earnings_expanded'][0,6]=np.nan if missing else .25000000000001
            arrays['expanded_available'][0,6]=not missing
            with self.assertRaises(ValueError):
                m.verify_feature_matrices(arrays,data['FEATURE_AUDIT.json'],meta)

    def test_prior153_cannot_add_an_unrelated_issuer(self):
        _,_,_,meta,data,arrays=synthetic_outputs()
        for name in ('earnings_batch4','earnings_expanded'):arrays[name][0,7]=.1
        arrays['batch4_available'][0,7]=arrays['expanded_available'][0,7]=True
        with self.assertRaisesRegex(ValueError,'Prior153 unrelated'):
            m.verify_feature_matrices(arrays,data['FEATURE_AUDIT.json'],meta)

    def test_nonboolean_batch4_mask_rejected(self):
        _,_,_,meta,data,arrays=synthetic_outputs()
        arrays['batch4_available']=arrays['batch4_available'].astype(int)
        with self.assertRaisesRegex(ValueError,'Exact masks'):
            m.verify_feature_matrices(arrays,data['FEATURE_AUDIT.json'],meta)


class SmallAccountingTests(unittest.TestCase):
    """Independent arithmetic expectations, not outputs created by replay."""

    def prices(self, values):
        days = pd.bdate_range('2026-01-05', periods=len(next(iter(values.values()))))
        return pd.DataFrame(values, index=days)

    def run_path(self, close, orders, execution='close', cost=0., opening=None):
        return m.replay(close, close if opening is None else opening, orders,
                        close.index, execution, cost)

    def test_equal_dollars_then_fixed_units_not_daily_rebalance(self):
        close = self.prices(dict(A=[10., 20., 10., 15.], B=[20., 20., 40., 20.]))
        rows = [order(close.index[0], close.index[-1], ['A', 'B'])]
        _, nav, trades, periods = self.run_path(close, rows)
        np.testing.assert_allclose(nav, [100000., 150000., 150000., 125000.], rtol=0., atol=1e-9)
        self.assertEqual([t['adjusted_units'] for t in trades], [5000., 2500.])
        self.assertEqual([t['dollars'] for t in trades], [50000., 50000.])
        self.assertEqual(len(periods), 1)
        self.assertTrue(all(t['side'] == 'buy' for t in trades))

    def test_three_equal_weight_positions(self):
        close = self.prices(dict(A=[10., 20., 20.], B=[20., 20., 20.], C=[40., 20., 20.]))
        rows = [order(close.index[0], close.index[-1], ['C', 'B', 'A'])]
        _, nav, trades, _ = self.run_path(close, rows)
        np.testing.assert_allclose(nav, [100000., 100000.*3.5/3., 100000.*3.5/3.], rtol=0., atol=1e-9)
        np.testing.assert_allclose([t['dollars'] for t in trades], [100000./3.]*3, rtol=0., atol=1e-9)

    def test_next_open_uses_next_session_prices_not_signal_close(self):
        close = self.prices(dict(A=[10., 20., 30., 40.], B=[20., 40., 40., 40.]))
        opening = close.copy()
        opening.iloc[1] = [25., 50.]
        rows = [order(close.index[0], close.index[-1], ['A', 'B'])]
        _, nav, trades, periods = self.run_path(close, rows, 'open', opening=opening)
        np.testing.assert_allclose(nav, [100000., 80000., 100000., 120000.], rtol=0., atol=1e-9)
        self.assertEqual({t['date'] for t in trades}, {str(close.index[1].date())})
        self.assertEqual([t['adjusted_price'] for t in trades], [25., 50.])
        self.assertEqual(periods[0]['signal_date'], str(close.index[0].date()))

    def test_entry_and_exit_cost_fixed_point_conserves_cash(self):
        close = self.prices(dict(A=[10.]*5, B=[20.]*5))
        rows = [order(close.index[0], close.index[2], ['A', 'B']),
                order(close.index[2], close.index[-1], [])]
        cost = .0005
        met, nav, trades, _ = self.run_path(close, rows, 'open', cost)
        post_entry = 100000./(1.+cost)
        post_exit = post_entry*(1.-cost)
        np.testing.assert_allclose(nav, [100000., post_entry, post_entry, post_exit, post_exit], rtol=0., atol=1e-8)
        self.assertAlmostEqual(met['fees_paid'], 100000.-post_exit, places=8)
        self.assertAlmostEqual(sum(t['cost'] for t in trades), met['fees_paid'], places=8)
        self.assertEqual([t['side'] for t in trades], ['buy', 'buy', 'sell', 'sell'])

    def test_cash_only_has_no_fabricated_fills(self):
        close = self.prices(dict(A=[10., 11., 12.], B=[20., 22., 24.]))
        rows = [order(close.index[0], close.index[-1], [])]
        met, nav, trades, periods = self.run_path(close, rows)
        np.testing.assert_array_equal(nav, [100000.]*3)
        self.assertEqual(trades, [])
        self.assertEqual(periods[0]['symbols'], [])
        self.assertEqual(met['fees_paid'], 0.)

    def test_end_date_is_valued_without_forced_sale(self):
        close = self.prices(dict(A=[10., 10., 20.], B=[20., 20., 40.]))
        rows = [order(close.index[0], close.index[-1], ['A', 'B'])]
        met, nav, trades, _ = self.run_path(close, rows)
        self.assertEqual(nav[-1], 200000.)
        self.assertEqual(len(trades), 2)
        self.assertEqual(met['total_return'], 1.)
        self.assertEqual(met['annual_returns'], [dict(year=2026, return_=1., partial=True)])

    def test_future_prices_and_plans_do_not_change_prefix(self):
        close = self.prices(dict(A=[10., 11., 12., 999., 1.], B=[20., 22., 24., 1., 999.]))
        rows = [order(close.index[0], close.index[3], ['A', 'B']),
                order(close.index[3], close.index[-1], [])]
        _, full, _, _ = self.run_path(close, rows)
        prefix = close.iloc[:3]
        _, short, _, _ = self.run_path(prefix, rows[:1])
        np.testing.assert_array_equal(full[:3], short)

    def test_invalid_held_price_rejected(self):
        for bad in (0., -1., float('nan'), float('inf')):
            with self.subTest(bad=bad):
                close = self.prices(dict(A=[10., bad, 10.], B=[20., 20., 20.]))
                rows = [order(close.index[0], close.index[-1], ['A', 'B'])]
                with self.assertRaises((ValueError, AssertionError)):
                    self.run_path(close, rows)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if args.receipt:
        if not result.wasSuccessful():raise SystemExit('Failed tests cannot issue prereview')
        if (m.ROOT/m.STUDY/'PLAN.json').exists():raise SystemExit('Already registered')
        receipt=m.synthetic_receipt(result)
        with args.receipt.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
