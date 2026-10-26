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
import monthly_financial_coverage152_outcome_review as m


def order(day, end, names):
    return dict(date=str(pd.Timestamp(day).date()), signal_date=str(pd.Timestamp(day).date()),
                end=str(pd.Timestamp(end).date()), symbols=list(names), scores=[.2]*len(names))


def synthetic_outputs():
    days = pd.bdate_range("2016-01-04", periods=245)
    x = np.arange(len(days))
    specs = [("AAPL", 10, .001, 4), ("KO", 20, .0007, 7), ("LRCX", 30, .0013, 8)]
    close = pd.DataFrame({s: base*np.exp(rate*x+.007*np.sin(x/freq))
                          for s, base, rate, freq in specs}, index=days)
    opening = close*np.array([.999, 1.001, 1.0002])
    names = ["AAPL", "KO", "LRCX", "AMZN"]+["S"+str(i) for i in range(28)]+["ETF"+str(i) for i in range(17)]
    meta = dict(dates=[str(d.date()) for d in days[:-1:2]], ends=[str(d.date()) for d in days[2::2]],
                symbols=[names.copy() for _ in range(122)], etfs=names[-17:])
    orders = {name: [order(day, end, ["AAPL", "LRCX"] if k and i % (k+2) == 0 else ["AAPL", "KO"])
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
    arrays = dict(earnings_batch2=np.full((122,49),np.nan),batch2_available=np.zeros((122,49),bool),earnings_batch1=np.full((122,49),np.nan),batch1_available=np.zeros((122,49),bool),earnings_original=np.full((122, 49), np.nan),
                  earnings_expanded=np.full((122, 49), np.nan),
                  original_available=np.zeros((122, 49), bool),
                  expanded_available=np.zeros((122, 49), bool))
    arrays["earnings_original"][:, 0] = .2
    arrays["earnings_expanded"][:, :2] = [.2, -.3]
    arrays["original_available"][:, 0] = True
    arrays["expanded_available"][:, :2] = True
    arrays["earnings_batch1"][:,0] = .2
    arrays["earnings_batch1"][:,3] = arrays["earnings_expanded"][:,3] = .1
    arrays["batch1_available"][:,[0,3]] = True
    arrays["expanded_available"][:,3] = True
    arrays["earnings_batch2"] = arrays["earnings_batch1"].copy()
    arrays["batch2_available"] = arrays["batch1_available"].copy()
    cells = [dict(signal_index=i, pool_index=j, signal_date=day, symbol=symbol,
                  available=j == 1, reason="available" if j == 1 else "not_added",
                  earnings_bounded=-.3 if j == 1 else None)
             for i, (day, pool) in enumerate(zip(meta["dates"], meta["symbols"]))
             for j, symbol in enumerate(pool)]
    feature = dict(cells=cells, original_cells=122, batch1_cells=244, batch2_cells=244, expanded_cells=366, added_cells=122,
                   original_input_unchanged=True, batch1_input_unchanged=True, batch2_input_unchanged=True, new_model_fits=0, returns_computed=False,
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
        rows[4]["training"]["cagr"] = opened[4]["training"]["cagr"] = .6
        return rows, opened

    def test_ties_retain_incumbent(self):
        self.assertEqual(m.select(*self.groups())["primary"], m.NAMES[0])

    def test_original_control_cannot_be_promoted(self):
        rows, opened = self.groups()
        rows[1]["training"]["cagr"] = opened[1]["training"]["cagr"] = 9.
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_primary_wins_three_controls_both_paths(self):
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

    def test_all_three_closed_ties_rejected(self):
        for i in range(3):
            rows, opened = self.winner()
            rows[i]["training"]["cagr"] = .6
            self.assertFalse(m.select(rows, opened)["admitted"])

    def test_each_risk_guard_each_path_each_control(self):
        for mode in (0, 1):
            for i in (0, 1, 2, 3):
                for section, key, tol in (("training", "sharpe", .05),
                                          ("training", "max_drawdown", .03),
                                          ("early", "cagr", .025), ("recent", "cagr", .025)):
                    pair = self.winner()
                    pair[mode][i][section][key] = pair[mode][4][section][key]+tol+.001
                    self.assertFalse(m.select(*pair)["admitted"])

    def test_exact_risk_boundaries_inclusive(self):
        rows, opened = self.winner()
        for group in (rows, opened):
            for section, key, tol in (("training", "sharpe", .05),
                                     ("training", "max_drawdown", .03),
                                     ("early", "cagr", .025), ("recent", "cagr", .025)):
                group[4][section][key] = group[0][section][key]-tol
        self.assertTrue(m.select(rows, opened)["admitted"])

    def test_tiny_strict_advantage_rejected(self):
        rows, opened = self.winner()
        rows[4]["training"]["cagr"] = .5+5e-13
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_invalid_metrics_fail(self):
        for value in (None, True, "2", float("nan"), float("inf")):
            rows, opened = self.winner()
            opened[4]["training"]["sharpe"] = value
            self.assertFalse(m.select(rows, opened)["admitted"])
        rows, opened = self.winner()
        del rows[4]["early"]["cagr"]
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_later34_does_not_select(self):
        rows, opened = self.groups()
        rows[4]["later34"] = {"cagr": 999}
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

    def test_fifteen_paths_and_610_orders(self):
        out = self.check()
        self.assertTrue(out["passed"])
        self.assertEqual(out["execution_paths_verified"], 15)
        self.assertEqual(out["total_monthly_orders_verified"], 610)
        self.assertEqual(len(out["independent_monthly_comparison"]), 122)

    def test_nav_tamper(self):
        self.data[m.NAMES[4]+"_next_open_cost_DAILY.json"][4]["equity"] += 10
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

    def test_distinct_masks_valid_and_eight_arrays_checked(self):
        self.assertEqual(self.matrix(), 8)
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
        self.arrays["expanded_available"][0, 4] = True
        self.arrays["earnings_expanded"][0, 4] = .1
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
        for key in ("original_cells", "batch1_cells", "expanded_cells", "added_cells"):
            feature = deepcopy(self.data["FEATURE_AUDIT.json"])
            feature[key] += 1
            with self.assertRaises(ValueError):
                m.verify_feature_matrices(self.arrays, feature, self.meta)
        self.data["FEATURE_AUDIT.json"]["reasons"] = {}
        with self.assertRaises(ValueError):
            self.matrix()

    def test_no_producer_modules_imported(self):
        for name in ("monthly_financial_coverage152_runner", "monthly_financial_coverage152_factor",
                     "monthly_financial_coverage152_selection"):
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
        self.npz('prior151/SCORES.npz',{k:np.zeros((122,49)) for k in m.prep.previous.NAMES})
        self.put('prior151/PRETRAIN_ORDERS.json',{k:[] for k in m.prep.previous.NAMES})
        self.put('prior151/CACHE.json',{n:self.sha(self.root/'prior151'/n) for n in ('DATA.npz','SCORES.npz','PRETRAIN_ORDERS.json')})
        source=self.put('synthetic_independent.py',{'synthetic':True})
        sources={str(source):self.sha(source)}
        inputs={k:self.put('input/'+k+'.json',[]) for k in m.INPUTS}
        self.put('input/original_orders.json',[]);self.put('input/sessions.json',{'sessions':['2020-01-01','2020-01-02']})
        legacy_refs,prior_refs=self.refs('legacy'),self.refs('prior')
        prior151_refs=self.refs('prior151')
        raw={str(p):self.sha(p) for p in self.root.rglob('*') if p.is_file()}
        plan=dict(round=152,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[4],dose=.01,output_directory=str(self.folder),
            runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),hashes=raw,
            inputs={k:dict(path=str(p),sha256=self.sha(p)) for k,p in inputs.items()},
            raw_source_hashes=sources,independent_prepare_source_hashes=sources,
            legacy149=legacy_refs,prior151=prior151_refs,coverage_sources={'batch1':{},'batch2':{},'batch3':{}},
            actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
        self.put(m.STUDY+'/PLAN.json',plan);self.plan_sha=self.sha(self.folder/'PLAN.json')
        arrays=dict(base,earnings_batch2=hand,batch2_available=mask,earnings_original=hand,earnings_batch1=hand,earnings_expanded=hand,original_available=mask,batch1_available=mask,expanded_available=mask)
        self.npz(m.STUDY+'/DATA.npz',arrays);self.npz(m.STUDY+'/SCORES.npz',{k:np.zeros((122,49)) for k in m.NAMES})
        self.put(m.STUDY+'/PRETRAIN_ORDERS.json',{k:[] for k in m.NAMES})
        for n in ('META.json','PRICE_PARITY.json','FEATURE_AUDIT.json'):self.put(m.STUDY+'/'+n,{})
        self.put(m.STUDY+'/CACHE.json',{n:self.sha(self.folder/n) for n in m.CACHE_FILES})
        self.receipt=dict(stage='prepare',passed=True,**dict.fromkeys(m.PREPARE_CHECKS,True),orders_checked=610,source_cells_checked=5978,stock_cells_checked=3904,monthly_label_models_fit=0,producer_modules_imported=False,returns_or_nav_computed=False,
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
        for name,result in [('verify_proposal',{}),('verify_legacy',{}),('verify_prior151',{'prior150':prior_refs}),('verify_coverage',({},{}))]:
            def call(*args,_name=name,_result=result,**kwargs):self.calls.append(_name);return _result
            patcher=patch.object(m.prep,name,side_effect=call);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(m,'source_files',return_value=[source]);patcher.start();self.addCleanup(patcher.stop)
    def check(self):return m.bindings(m.Reader(self.root),self.plan_sha,self.audit_time)
    def test_full_bindings_source_gate_call_path(self):
        self.chain();self.assertEqual(self.check(),self.folder)
        self.assertEqual(self.calls,['verify_proposal','verify_legacy','verify_prior151','verify_coverage'])
    def test_all_four_prepare_flags(self):
        self.chain();self.receipt.pop('prior150_batch1_earnings_cache_and_mask_verified');self.put(m.STUDY+'/INDEPENDENT_PREPARE.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'Independent prepare'):self.check()
    def test_610_count_required(self):
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
        t=SelectionTests();a,b=t.winner();a[3]['training']['cagr']=a[4]['training']['cagr']
        self.assertFalse(m.select(a,b)['admitted'])
    def test_open_batch2_control_tie_blocks_promotion(self):
        t=SelectionTests();a,b=t.winner();b[3]['training']['cagr']=b[4]['training']['cagr']
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
