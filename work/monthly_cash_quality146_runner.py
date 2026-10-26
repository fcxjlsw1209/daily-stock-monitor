"""Four fixed cash-quality arms on the unchanged local67 monthly strategy.

Actual features, orders and returns require a separately frozen round146 PLAN.
Source collection/admission are separate; this runner is strictly offline.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_same_report145_runner as previous
import monthly_cash_quality146_factor as factor

require, sha, read, dump = base.require, base.sha, base.read, base.dump
NAMES = ("incumbent_local67", "earnings_only", "cash_flow_only", "low_accrual_quality")
HANDS = ("earnings_bounded", "cash_flow_bounded", "low_accrual_bounded")
TRAIN, MONTHS, WIDTH, DOSE = 88, 122, 49, .01
OUT = "outputs/monthly_cash_quality146"
CACHE_FILES = base.CACHE_FILES
FROZEN_FILES = ("PLAN.json", "CACHE.json", "PRETRAIN_ORDERS.json", "TRAIN_TRIALS.json",
                "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json", "INDEPENDENT_PREPARE.json")
CHECKS = base.CHECKS
INPUTS = ("proposal", "meta", "policy", "original_orders", "prices", "sessions",
          "protected_manifest", "admitted_dataset", "source_admission")
TOLERANCES = dict(base.TOLERANCES, scores_atol=1e-12, feature_atol=1e-12)
CONCEPT_SHA = "4fe507528756a3da14b88d24929fb638a88b785c8294d476418bc8768a8ba4b2"


def selection_contract():
    return dict(total_months=122, training_months=88, early_months=53,
        recent_months=35, later_months=34,
        walkforward=[[36, 60], [60, 76], [76, 88]],
        execution_paths=[
            dict(execution="close", cost=0., key="same_close"),
            dict(execution="open", cost=0., key="next_open_zero"),
            dict(execution="open", cost=.0005, key="next_open_cost")],
        same_close_cagr="strict_greater_than_all_three_controls",
        next_open_cost_cagr="noninferior_incumbent_strict_greater_than_two_controls",
        strict_cagr_epsilon=1e-12, max_sharpe_degradation=.05,
        max_drawdown_degradation=.03, max_early_cagr_degradation=.025,
        max_recent_cagr_degradation=.025, new_model_fits=0,
        intramonth_trades=False, later34_for_selection=False)


def validate_proposal(p):
    require(p.get("round") == 146 and p.get("arms") == list(NAMES)
        and p.get("primary_challenger") == NAMES[3] and p.get("dose") == DOSE,
        "Wrong prospective experiment")
    require(p.get("concept") == dict(path="work/after145_cashflow_experiment_concept.json",
        sha256=CONCEPT_SHA), "Prospective concept changed")
    require(p.get("selection_contract") == selection_contract(),
        "Prospective selection contract changed")


def select(rows, opened):
    for group in (rows, opened):
        require([x["name"] for x in group] == list(NAMES), "Four exact arms required")
    close = {NAMES[i]: previous.guard(rows[3], rows[i], strict=True) for i in range(3)}
    op = {NAMES[i]: previous.guard(opened[3], opened[i], strict=i != 0) for i in range(3)}
    passed = all(close.values()) and all(op.values())
    return dict(primary=NAMES[3] if passed else NAMES[0], admitted=passed,
                guards_against=close, next_open_guards_against=op,
                mechanism_confirmed=all(op.values()))


def corrected(pre, hand, arrays, meta, n, backend, policy):
    """Only the declared .01*h correction; do not reuse the GP clipping map."""
    shape = np.asarray(arrays["baseline"]).shape
    hand = np.asarray(hand)
    require(hand.shape == shape and hand.dtype.kind == "f" and not np.isinf(hand[:n]).any(),
            "Finite or missing float hand on original axes")
    prefix = hand[:n]
    require(np.all(np.abs(prefix[np.isfinite(prefix)]) <= 1.), "Bounded continuous h required")
    good = np.asarray(arrays["eligible"][:n])
    require(good.dtype == bool and good.shape == (n, WIDTH), "Original eligibility mask")
    for i, symbols in enumerate(meta["symbols"][:n]):
        require(all(not np.isfinite(hand[i, j]) for j, s in enumerate(symbols)
                    if s in meta["etfs"]), "No financial features for ETFs")
    view = np.asarray(pre).copy()
    require(view.shape == (n, WIDTH), "Original pre-score shape")
    active = good & np.isfinite(hand[:n]) & (hand[:n] != 0.)
    delta = np.zeros(view.shape)
    delta[active] = DOSE * hand[:n][active]
    require(np.isfinite(delta).all() and np.max(np.abs(delta), initial=0.) <= DOSE,
            "Correction exceeded declared cap")
    view[active] += delta[active]
    full = np.zeros(shape)
    full[:n] = view
    return backend.adapter.score_and_choose("incumbent66", 0., hand, arrays, meta, n,
        pre_score=full, effects=arrays["effects"], episode_policy=policy,
        boundary_selector=backend.selector)


def choose(name, arrays, meta, n, backend, policy):
    require(name in NAMES, "Undeclared arm")
    if name == NAMES[0]:
        return backend.baseline(arrays, meta, n, policy)
    return corrected(backend.pre(arrays, n), arrays[HANDS[NAMES.index(name)-1]],
                     arrays, meta, n, backend, policy)


def panels(arrays, meta, backend, policy):
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    orders, scores = {}, {}
    common = arrays["common_available"]
    require(common.dtype == bool and common.shape == (MONTHS, WIDTH), "Common mask")
    for key in HANDS:
        require(np.array_equal(np.isfinite(arrays[key]), common), "Identical feature mask")
    for name in NAMES:
        c, s = choose(name, arrays, meta, MONTHS, backend, policy)
        orders[name], scores[name] = backend.make_orders(c, s, meta), s
        for n in range(MONTHS+1):
            pc, ps = choose(name, arrays, meta, n, backend, policy)
            require(pc == c[:n] and ps.tobytes() == s[:n].tobytes(), "Future-prefix dependence")
    for value in (0., np.nan):
        null = dict(arrays)
        for key in HANDS:
            null[key] = np.full((MONTHS, WIDTH), np.nan)
            for i, symbols in enumerate(meta["symbols"]):
                for j, symbol in enumerate(symbols):
                    if symbol not in meta["etfs"]:
                        null[key][i, j] = value
        for name in NAMES[1:]:
            c, s = choose(name, null, meta, MONTHS, backend, policy)
            require(backend.make_orders(c, s, meta) == orders[NAMES[0]]
                    and s.tobytes() == scores[NAMES[0]].tobytes(), "Zero/missing fallback")
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()},
            "Input arrays changed")
    return orders, scores


class Backend(previous.Backend):
    def training(self, orders, meta, px, op):
        rp = self.local.rp
        blocks, _, bounds, valid = rp.paths(px, meta, TRAIN)
        returns, rows = [], []
        for name in NAMES:
            selected = [[meta["symbols"][i].index(s) for s in row["symbols"]]
                        for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(selected, blocks, valid)
            returns.append(rr)
            rows.append(dict(name=name, **self.local.old.session.measure(rr, meta, bounds)))
        opened = self.open_training(orders, meta, px, op, TRAIN, NAMES)
        folds, stitched, control = [], [], []
        for n, end in ((36, 60), (60, 76), (76, 88)):
            prefix = [dict(name=name, **self.local.old.session.measure(rr, meta, bounds, n))
                      for name, rr in zip(NAMES, returns)]
            oprefix = self.open_training(orders, meta, px, op, n, NAMES)
            pick = select(prefix, oprefix)
            folds.append(dict(n=n, end=end, selection=pick, open_training=oprefix))
            stitched.extend(returns[NAMES.index(pick["primary"])][bounds[n]:bounds[end]])
            control.extend(returns[0][bounds[n]:bounds[end]])
        walk = dict(folds=folds, pristine_oos=False,
            metrics=rp.stats_rr(np.asarray(stitched), meta["dates"][36], meta["ends"][87]),
            incumbent=rp.stats_rr(np.asarray(control), meta["dates"][36], meta["ends"][87]))
        return rows, opened, walk

    def diagnostics(self, orders, meta, px, op, writer):
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (("close", 0., "same_close"), ("open", 0., "next_open_zero"),
                                     ("open", .0005, "next_open_cost")):
                met, eq, trades, holdings = self.local.ex.e.engine.replay(
                    px, op, orders[name], cost=cost, execution=mode,
                    start=pd.Timestamp(meta["dates"][0]), end=pd.Timestamp(meta["ends"][-1]))
                for row in met["annual_returns"]:
                    if row["year"] in (2016, 2026):
                        row["partial"] = True
                results[name][key] = dict(metrics=met,
                    slices=self.local.arch.old.prior.old.previous.cuts(eq, meta))
                writer(name+"_"+key+"_DAILY.json",
                       [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
                writer(name+"_"+key+"_TRADES.json", trades)
                writer(name+"_"+key+"_HOLDINGS.json", holdings)
        return results


class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha, backend_factory=Backend):
        self.root, self.plan_path = Path(root).resolve(), Path(plan_path).resolve()
        self.plan_sha = plan_sha
        require(sha(self.plan_path) == plan_sha, "Frozen PLAN changed")
        self.plan = p = read(self.plan_path)
        require(p["round"] == 146 and p["status"] == "REGISTERED"
                and p["arms"] == list(NAMES) and p["primary_challenger"] == NAMES[3],
                "Round146 registration required")
        require(p["dose"] == DOSE and p["tolerances"] == TOLERANCES, "Declared numeric policy")
        require(p["runtime"] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                "Pinned runtime changed")
        self.pins = self.verify(p["hashes"])
        require(set(p["inputs"]) == set(INPUTS), "Exact named inputs")
        self.inputs = {k: self.bound_path(v["path"]) for k, v in p["inputs"].items()}
        require(all(sha(self.inputs[k]) == v["sha256"] for k, v in p["inputs"].items()),
                "Input binding changed")
        validate_proposal(read(self.inputs["proposal"]))
        require(self.pins.get(str(self.root/"work/after145_cashflow_experiment_concept.json"))
                == CONCEPT_SHA, "Prospective concept not bound")
        for path, digest in source_files(self.root).items():
            require(self.pins.get(path) == digest, "Current implementation not registered")
        self.out = self.root / OUT
        require(self.plan_path == self.out/"PLAN.json" and Path(p["output_directory"]) == self.out,
                "Dedicated output directory")
        self.backend_factory = backend_factory
        self.require_admission()

    def require_admission(self):
        receipt = read(self.inputs["source_admission"])
        dataset = read(self.inputs["admitted_dataset"])
        require(receipt.get("stage") == "conditional_cash_quality_source_admission"
                and receipt.get("passed") is True
                and receipt.get("data_sha256") == sha(self.inputs["admitted_dataset"]),
                "Verified conditional source admission")
        require(dataset["method"] == receipt["method"] == factor.METHOD
                and len(dataset["events"]) == 44, "Full source population")
        require(dataset["policy_sha256"] == receipt["policy_sha256"]
                and receipt["policy_sha256"] in self.pins.values(), "Bound source policy")
        policy_ref = self.plan["source_policy"]
        source_policy = self.bound_path(policy_ref["path"])
        require(sha(source_policy) == policy_ref["sha256"] == receipt["policy_sha256"],
                "Exact source policy mismatch")
        for path, digest in receipt["source_hashes"].items():
            require(self.pins.get(str((self.root/path).resolve())) == digest,
                    "Source-admission dependency not registered")
        require(receipt.get("original_raw_events_preserved") is True
                and receipt.get("original_rejections_preserved") is True, "Source preservation")
        for key in ("first_public_version_verified", "complete_correction_history_verified",
                    "certified_daily_security_master"):
            require(receipt.get(key) is False, "Source assumptions cannot be upgraded")
        return dataset

    def prepare(self):
        self.refresh()
        require(not any((self.out/x).exists() for x in (*CACHE_FILES, "CACHE.json")),
                "Preparation already started; preserve partial outputs")
        backend = self.backend_factory(self.root, self)
        meta, original, px, op = backend.load()
        base.validate_meta(meta)
        require(len(original) == 27 and meta == read(self.inputs["meta"]), "Original cache")
        parity = backend.price_parity(meta, px, op, self.plan["raw_source_hashes"])
        result = factor.build_features(meta, self.require_admission())
        require(set(result["arrays"]) == set(HANDS)|{"common_available"}, "Exact new features")
        arrays = dict(original, **result["arrays"])
        orders, scores = panels(arrays, meta, backend, read(self.inputs["policy"]))
        require(orders[NAMES[0]] == read(self.inputs["original_orders"]), "Incumbent changed")
        result["audit"].update(source_admission_sha256=sha(self.inputs["source_admission"]),
                               admitted_dataset_sha256=sha(self.inputs["admitted_dataset"]))
        for filename, values in (("DATA.npz", arrays), ("SCORES.npz", scores)):
            with (self.out/filename).open("xb") as f:
                np.savez_compressed(f, **values)
        for filename, value in (("META.json", meta), ("PRETRAIN_ORDERS.json", orders),
                                ("PRICE_PARITY.json", parity), ("FEATURE_AUDIT.json", result["audit"])):
            self.write(filename, value)
        self.write("CACHE.json", {name: sha(self.out/name) for name in CACHE_FILES})

    def loaded(self):
        self.cache()
        backend = self.backend_factory(self.root, self)
        meta, original, px, op = backend.load()
        require(meta == read(self.out/"META.json"), "Original metadata changed")
        with np.load(self.out/"DATA.npz", allow_pickle=False) as data:
            require(set(data.files) == set(original)|set(HANDS)|{"common_available"},
                    "Prepared array names changed")
            require(all(data[k].tobytes() == v.tobytes() and data[k].dtype == v.dtype
                        and data[k].shape == v.shape for k, v in original.items()), "Original27 changed")
        return backend, meta, px, op

    def require_prepare_review(self):
        self.cache()
        r = read(self.out/"INDEPENDENT_PREPARE.json")
        require(r.get("passed") is True and all(r.get(k) is True for k in CHECKS),
                "Independent preparation checks incomplete")
        require(r.get("producer_modules_imported") is False and
                r.get("returns_or_nav_computed") is False, "Independent preparation isolation")
        require(r["plan_sha256"] == self.plan_sha and r["cache_sha256"] == sha(self.out/"CACHE.json")
                and r["orders_sha256"] == sha(self.out/"PRETRAIN_ORDERS.json"), "Review binding changed")
        reviewed = self.verify(r["artifact_hashes"])
        for name in ("PLAN.json", "CACHE.json", *CACHE_FILES):
            require(reviewed.get(str(self.out/name)) == sha(self.out/name), "Unreviewed prepared output")
        require(self.verify(r["source_hashes"]) == self.verify(self.plan["independent_prepare_source_hashes"]),
                "Independent reviewer code changed")

    def select_first88(self):
        self.refresh()
        self.require_prepare_review()
        require(not any((self.out/x).exists() for x in
                        ("TRAIN_TRIALS.json", "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json", "FROZEN.json")),
                "Selection already started")
        backend, meta, px, op = self.loaded()
        rows, opened, walk = backend.training(read(self.out/"PRETRAIN_ORDERS.json"), meta, px, op)
        self.write("TRAIN_TRIALS.json", rows)
        self.write("TRAIN_OPEN_TRIALS.json", opened)
        self.write("WALKFORWARD.json", walk)
        self.write("FROZEN.json", dict(selection=select(rows, opened), pristine_oos=False,
            all_diagnostic_orders_frozen=True, hashes={name: sha(self.out/name) for name in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh()
        self.require_prepare_review()
        frozen = read(self.out/"FROZEN.json")
        require(set(frozen["hashes"]) == set(FROZEN_FILES), "Frozen artifact list")
        self.verify({str(self.out/k): v for k, v in frozen["hashes"].items()})
        require(frozen["selection"] == select(read(self.out/"TRAIN_TRIALS.json"),
                                              read(self.out/"TRAIN_OPEN_TRIALS.json")), "Selection changed")
        require(not (self.out/"DIAGNOSE_STARTED.json").exists(), "Diagnostic already started")
        self.write("DIAGNOSE_STARTED.json", dict(plan_sha256=self.plan_sha,
            started_utc=datetime.now(timezone.utc).isoformat()))
        backend, meta, px, op = self.loaded()
        results = backend.diagnostics(read(self.out/"PRETRAIN_ORDERS.json"), meta, px, op, self.write)
        for filename, mode in (("TRAIN_TRIALS.json", "same_close"), ("TRAIN_OPEN_TRIALS.json", "next_open_cost")):
            for row in read(self.out/filename):
                require(abs(results[row["name"]][mode]["slices"]["training2016_2023"]["cagr"] -
                            row["training"]["cagr"]) < TOLERANCES["training_cagr"], "Training replay parity")
        self.write("RESULTS.json", results)
        self.write("VALIDATION.json", dict(selection=frozen["selection"],
            no_observer_writes=True, no_intramonth_trades=True, new_model_fits=0,
            independent_outcome_review_pending=True, pristine_oos=False))


def source_files(root):
    root = Path(root)
    files = (Path(__file__), Path(factor.__file__),
             root/"work/test_monthly_cash_quality146_factor.py",
             root/"work/test_monthly_cash_quality146_runner.py")
    return dict(previous.source_files(root), **{str(p.resolve()): sha(p) for p in files})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "select_first88", "diagnose_later34"))
    p.add_argument("--plan", required=True)
    p.add_argument("--plan-sha", required=True)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    getattr(Runner(a.root, a.plan, a.plan_sha), a.phase)()
