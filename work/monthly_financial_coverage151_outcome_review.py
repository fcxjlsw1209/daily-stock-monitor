"""Independent accounting for the four-arm coverage experiment.

Uses the frozen independent accounting engine, never the experiment producer.
Financial source/score correctness is separately bound to independent prepare.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "work/monthly_same_report145_outcome_review.py"
ENGINE_SHA = "968038244802fa5ea18253adbc99786979bd5e40890c96d04f859721ed728d8e"
if hashlib.sha256(ENGINE.read_bytes()).hexdigest() != ENGINE_SHA:
    raise ValueError("Frozen independent accounting changed")
import monthly_same_report145_outcome_review as old
import monthly_financial_coverage151_prepare_review as prep

h, require = old.h, old.require
Reader, replay, compare_stats = old.Reader, old.replay, old.compare_stats
normalize, rr_statistics = old.normalize, old.rr_statistics
STUDY = "outputs/monthly_financial_coverage151"
PRICE = old.PRICE
NAMES = ("incumbent_local67", "original_earnings", "batch1_earnings", "expanded_earnings")
HANDS = ("earnings_original", "earnings_batch1", "earnings_expanded")
PATHS, CACHE_FILES = old.PATHS, old.CACHE_FILES
FROZEN_FILES = ("PLAN.json", "CACHE.json", "PRETRAIN_ORDERS.json", "TRAIN_TRIALS.json",
                "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json", "INDEPENDENT_PREPARE.json")
INPUTS = ("proposal", "meta", "policy", "original_orders", "prices", "sessions",
          "protected_manifest")
PREPARE_CHECKS = prep.CHECKS


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    return sorted(set(prep.source_files(root)+[root/'work'/name for name in (
        'monthly_financial_coverage151_outcome_review.py','test_monthly_financial_coverage151_outcome_review.py',
        'monthly_same_report145_outcome_review.py','monthly_ridge_support70_result_review.py')]))


def select(rows, opened):
    fields = (("training", "cagr"), ("training", "sharpe"),
              ("training", "max_drawdown"), ("early", "cagr"), ("recent", "cagr"))
    def checks(group, at_open):
        require(type(group) is list and [r.get("name") for r in group] == list(NAMES),
                "Four exact selection arms")
        values = [[r.get(a, {}).get(b) for a, b in fields] for r in group]
        ans = {}
        for i in (0, 1, 2):
            candidate, control = values[3], values[i]
            valid = all(type(x) in (int, float) and math.isfinite(x)
                        for x in candidate + control)
            growth = valid and (candidate[0] >= control[0] - 1e-12 if at_open and i == 0
                                else candidate[0] > control[0] + 1e-12)
            ans[NAMES[i]] = bool(growth and all(c >= b - t for c, b, t in
                                zip(candidate[1:], control[1:], (.05, .03, .025, .025))))
        return ans
    close, opening = checks(rows, False), checks(opened, True)
    admitted = all(close.values()) and all(opening.values())
    return dict(primary=NAMES[3] if admitted else NAMES[0], admitted=admitted,
                guards_against=close, next_open_guards_against=opening,
                mechanism_confirmed=all(opening.values()))


def validate_orders(close, opening, orders, meta):
    dates = old.daily_axis(close.index, "prices")
    require(opening.index.equals(dates) and opening.columns.equals(close.columns)
            and close.columns.is_unique, "Same price axes")
    days, ends = old.daily_axis(meta["dates"], "signals"), pd.DatetimeIndex(meta["ends"])
    require(len(days) == len(ends) == len(meta["symbols"]) == 122
            and days.isin(dates).all() and ends.isin(dates).all() and (ends > days).all(),
            "122 ordered holding periods")
    require(days[1:].equals(ends[:-1]), "Adjacent holding periods")
    require(set(orders) == set(NAMES), "Four exact order arms")
    for name in NAMES:
        require(len(orders[name]) == 122, "122 orders per arm")
        for i, row in enumerate(orders[name]):
            require(set(row) == {"date", "end", "signal_date", "symbols", "scores"},
                    "Exact order fields")
            require(row["date"] == row["signal_date"] == meta["dates"][i]
                    and row["end"] == meta["ends"][i], "Order period")
            symbols = row["symbols"]
            require(type(symbols) is list and len(symbols) == len(set(symbols))
                    and len(symbols) in (0, 2, 3), "Cash or two/three positions")
            require(all(type(s) is str and s in meta["symbols"][i] and s in close.columns
                        for s in symbols), "Original security identifiers")
            require(len(row["scores"]) == len(symbols) and all(
                type(x) in (int, float) and math.isfinite(x) for x in row["scores"]),
                "Aligned finite scores")
            require(dates.get_loc(days[i]) + 1 < len(dates), "Next session exists")
    return dates[(dates >= days[0]) & (dates <= ends[-1])]


def walkforward(dates, navs, open_navs, meta):
    folds, selected, control = [], [], []
    for n, end in ((36, 60), (60, 76), (76, 88)):
        opened = [dict(name=name, **h.measures(dates, open_navs[name], meta, n))
                  for name in NAMES]
        rows = [dict(name=name, **h.measures(dates, navs[name], meta, n)) for name in NAMES]
        pick = select(rows, opened)
        folds.append(dict(n=n, end=end, selection=pick, open_training=opened))
        lo = dates.get_loc(pd.Timestamp(meta["dates"][n]))
        hi = dates.get_loc(pd.Timestamp(meta["ends"][end - 1]))
        for dest, name in ((selected, pick["primary"]), (control, NAMES[0])):
            nav = navs[name]
            dest.extend((nav[1:] / nav[:-1] - 1.)[lo:hi])
    return dict(folds=folds, pristine_oos=False,
                metrics=rr_statistics(selected, meta["dates"][36], meta["ends"][87]),
                incumbent=rr_statistics(control, meta["dates"][36], meta["ends"][87]))


def verify_feature_matrices(arrays,feature,meta):
    hands=[arrays[k] for k in HANDS]
    masks=[arrays[k] for k in ('original_available','batch1_available','expanded_available')]
    require(all(a.shape==(122,49) and a.dtype==np.float64 and not np.isinf(a).any() for a in hands),'Feature axes/dtypes')
    require(all(a.shape==(122,49) and a.dtype==bool and np.array_equal(a,np.isfinite(v)) for a,v in zip(masks,hands)),'Exact masks')
    for i in (0,1):
        require(np.all(~masks[i]|masks[i+1]) and hands[i][masks[i]].tobytes()==hands[i+1][masks[i]].tobytes(),'Nested preserved finite controls')
    identities=[(i,j,d,s) for i,(d,pool) in enumerate(zip(meta['dates'],meta['symbols'])) for j,s in enumerate(pool)]
    require(len(feature['cells'])==len(identities)==5978,'Complete source-cell census')
    for c,identity in zip(feature['cells'],identities):
        require(tuple(c[k] for k in ('signal_index','pool_index','signal_date','symbol'))==identity,'Ordered source cell')
        i,j,d,s=identity
        require(type(c['available']) is bool and type(c['reason']) is str,'Explicit source availability')
        if s not in ('INTC','KLAC'):
            require(hands[1][i:i+1,j:j+1].tobytes()==hands[2][i:i+1,j:j+1].tobytes() and not c['available'],'No unrelated extension')
        else:
            require(not masks[0][i,j] and not masks[1][i,j] and masks[2][i,j]==c['available'],'New issuer mask')
            if c['available']:
                value=c['earnings_bounded']
                require(type(value) is float and math.isfinite(value) and abs(value)<=1 and hands[2][i,j]==value,'New bounded value')
            else: require(c.get('earnings_bounded') is None,'No missing pseudo-value')
        if s not in ('AMZN','CSCO'):
            require(hands[0][i:i+1,j:j+1].tobytes()==hands[1][i:i+1,j:j+1].tobytes(),'Prior150 unrelated cell drift')
        if s in meta['etfs']: require(all(not a[i,j] for a in masks),'No ETF financial feature')
    for key,wanted in (('original_cells',int(masks[0].sum())),('batch1_cells',int(masks[1].sum())),('expanded_cells',int(masks[2].sum())),('added_cells',int(masks[2].sum()-masks[1].sum()))):
        require(feature.get(key)==wanted,'Coverage count '+key)
    require(feature['reasons']==dict(Counter(c['reason'] for c in feature['cells'])),'Reason census')
    require(feature['original_input_unchanged'] is feature['batch1_input_unchanged'] is True and type(feature['new_model_fits']) is int and feature['new_model_fits']==0 and feature['returns_computed'] is False,'Source-only preparation')
    return 6


def check_outcomes(close, opening, orders, meta, read):
    dates = validate_orders(close, opening, orders, meta)
    results = read("RESULTS.json")
    require(set(results) == set(NAMES), "Four result arms")
    bounds = dict(early2016_2020=(meta["dates"][0], meta["ends"][52]),
                  original2021_2026=(meta["dates"][53], meta["ends"][-1]),
                  training2016_2023=(meta["dates"][0], meta["ends"][87]),
                  later34=(meta["dates"][88], meta["ends"][-1]))
    errors, navs, open_navs, metrics, details = [], {}, {}, {}, {}
    maxnav = 0.
    for name in NAMES:
        require(set(results[name]) == {x[2] for x in PATHS}, "Three paths per arm")
        for execution, cost, key in PATHS:
            met, nav, trades, hold = replay(close, opening, orders[name], dates, execution, cost)
            prefix = name + "_" + key
            compare_stats(results[name][key]["metrics"], met, errors, prefix + "/metrics")
            compare_stats(results[name][key]["slices"], {
                k: h.slice_stat(dates, nav, *span) for k, span in bounds.items()},
                errors, prefix + "/slices")
            daily = read(prefix + "_DAILY.json")
            require(all(set(x) == {"date", "equity"} for x in daily)
                    and [x["date"] for x in daily] == [str(d.date()) for d in dates],
                    "Exact daily NAV dates")
            actual = np.asarray([x["equity"] for x in daily], float)
            require(np.isfinite(actual).all() and (actual > 0).all(), "Finite positive NAV")
            np.testing.assert_allclose(actual, nav, atol=1e-5, rtol=2e-11)
            maxnav = max(maxnav, float(np.max(np.abs(actual - nav))))
            actual_trades, actual_hold = read(prefix + "_TRADES.json"), read(prefix + "_HOLDINGS.json")
            h.compare(normalize(actual_trades), normalize(trades), errors, prefix + "/trades")
            h.compare(normalize(actual_hold), normalize(hold), errors, prefix + "/holdings")
            allowed = {row["date"] if execution == "close" else str(
                close.index[close.index.get_loc(pd.Timestamp(row["date"])) + 1].date())
                for row in orders[name]}
            require(all(x["date"] in allowed for x in actual_trades), "No intramonth trades")
            require([x["year"] for x in met["annual_returns"]] ==
                    sorted({int(x.year) for x in dates}) and all(
                        x["partial"] is (x["year"] in (2016, 2026)) for x in met["annual_returns"]),
                    "Annual coverage and partial flags")
            details[prefix] = dict(trades=len(trades), holding_periods=len(hold),
                                   fees_paid=met["fees_paid"])
            if execution == "close":
                navs[name], metrics[name] = nav, met
            if key == "next_open_cost":
                open_navs[name] = nav
        td = dates[dates <= pd.Timestamp(meta["ends"][87])]
        for execution, cost, full in (("close", 0., navs[name]),
                                      ("open", .0005, open_navs[name])):
            _, tn, _, _ = replay(close.loc[:td[-1]], opening.loc[:td[-1]],
                                orders[name][:88], td, execution, cost)
            np.testing.assert_allclose(tn, full[:len(td)], atol=1e-5, rtol=2e-11)
    rows = [dict(name=name, **h.measures(dates, navs[name], meta)) for name in NAMES]
    opened = [dict(name=name, **h.measures(dates, open_navs[name], meta)) for name in NAMES]
    compare_stats(read("TRAIN_TRIALS.json"), rows, errors, "training")
    compare_stats(read("TRAIN_OPEN_TRIALS.json"), opened, errors, "open_training")
    pick = select(rows, opened)
    h.assert_exact_tree(read("FROZEN.json")["selection"], pick)
    compare_stats(read("WALKFORWARD.json"), walkforward(dates, navs, open_navs, meta),
                  errors, "walkforward")
    validation = read("VALIDATION.json")
    h.assert_exact_tree(validation["selection"], pick)
    require(validation.get("pristine_oos") is False
            and validation.get("independent_outcome_review_pending") is True
            and validation.get("no_observer_writes") is True
            and validation.get("no_intramonth_trades") is True
            and type(validation.get("new_model_fits")) is int
            and validation["new_model_fits"] == 0, "Diagnostic scope")
    months = []
    for i, date in enumerate(meta["dates"]):
        held = {name: orders[name][i]["symbols"] for name in NAMES}
        returns = {name: math.fsum(
            float(close.at[pd.Timestamp(meta["ends"][i]), s]) /
            float(close.at[pd.Timestamp(date), s]) - 1. for s in held[name]) /
            len(held[name]) if held[name] else 0. for name in NAMES}
        months.append(dict(date=date, end=meta["ends"][i],
                           split="early53" if i < 53 else "recent35" if i < 88 else "later34",
                           holdings=held, returns=returns))
    return dict(passed=True, stage="evaluate", execution_paths_verified=12,
                total_monthly_orders_verified=488, daily_nav_trades_holdings_metrics_slices=True,
                first88_and_three_fold_selection_verified=True,
                first88_dual_execution_selection_verified=True, annual_partial_flags_verified=True,
                first88_causal_replay_verified=True, selected_name=pick["primary"],
                primary_cagr=metrics[pick["primary"]]["cagr"],
                primary_target_reached=metrics[pick["primary"]]["cagr"] >= .7,
                max_compared_absolute_difference=max((x for _, x in errors), default=0.),
                max_nav_absolute_difference=maxnav, replay_details=details,
                independent_monthly_comparison=months, producer_modules_imported=False,
                models_fit=0, producer_model_fits=0, zero_model_fit_count_verified=True,
                protected_member_scan_repeated=False)


def verify_diagnostic_start(reader, directory, plan_sha, audit_started):
    marker = reader.json(directory / "DIAGNOSE_STARTED.json")
    require(set(marker) == {"plan_sha256", "started_utc"}
            and marker["plan_sha256"] == plan_sha, "Diagnostic marker binding")
    value = marker["started_utc"]
    require(type(value) is str and value.endswith(("+00:00", "Z")), "UTC diagnostic time")
    start = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(start.tzinfo is not None and start.utcoffset().total_seconds() == 0
            and start < audit_started, "Diagnostic precedes audit")


def bindings(reader, plan_sha, audit_started=None):
    require(type(plan_sha) is str and len(plan_sha) == 64
            and all(c in "0123456789abcdef" for c in plan_sha), "Explicit PLAN SHA")
    p = reader.root / STUDY
    require(reader.sha(p / "PLAN.json") == plan_sha, "Exact registered PLAN")
    plan = reader.json(p / "PLAN.json")
    require(plan.get("round") == 151 and plan.get("status") == "REGISTERED"
            and plan.get("arms") == list(NAMES) and plan.get("primary_challenger") == NAMES[3],
            "Registered four-arm protocol")
    require(reader.path(plan["output_directory"]) == p and plan.get("dose") == .01,
            "Output and dose")
    require(plan["runtime"] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
            "Pinned runtime")
    pins = reader.verify(plan["hashes"])
    for file in source_files(reader.root):
        require(pins.get(str(file)) == reader.sha(file), "Registered outcome reviewer")
    for key in ("raw_source_hashes", "independent_prepare_source_hashes"):
        require(all(pins.get(k) == v for k, v in reader.verify(plan[key]).items()),
                "Registered closure " + key)
    for key in ("actual_feature_calculations_before_registration",
                "actual_returns_before_registration", "actual_model_fits_before_registration"):
        require(type(plan.get(key)) is int and plan[key] == 0, "Prospective registration")
    require(set(plan["inputs"]) == set(INPUTS), "Exact named inputs")
    for name, ref in plan["inputs"].items():
        require(reader.sha(ref["path"]) == ref["sha256"] ==
                pins.get(str(reader.path(ref["path"]))), "Named input " + name)
    prep.verify_proposal(reader,plan,pins)
    previous=prep.verify_legacy(reader,plan,pins)
    prep.verify_prior150(reader,plan,pins)
    # Recheck source policy and preservation independently; no feature/ratio
    # computation is performed by these source gates.
    sessions=reader.json(plan['inputs']['sessions']['path'])['sessions']
    prep.verify_coverage(reader,plan,pins,previous,sessions)
    cache = reader.json(p / "CACHE.json")
    require(set(cache) == set(CACHE_FILES), "Exact prepared files")
    cached = reader.verify(cache, p)
    prepare_receipt = reader.json(p / "INDEPENDENT_PREPARE.json")
    require(prepare_receipt.get("passed") is True and prepare_receipt.get("stage") == "prepare"
            and all(prepare_receipt.get(k) is True for k in PREPARE_CHECKS), "Independent prepare passed")
    for key, value in (("orders_checked", 488), ("source_cells_checked", 5978),
                       ("stock_cells_checked", 3904), ("monthly_label_models_fit", 0)):
        require(type(prepare_receipt.get(key)) is int and prepare_receipt[key] == value, "Exact prepare count " + key)
    require(prepare_receipt.get("producer_modules_imported") is False
            and prepare_receipt.get("returns_or_nav_computed") is False, "Independent prepare scope")
    for key, name in (("plan_sha256", "PLAN.json"), ("cache_sha256", "CACHE.json"),
                       ("orders_sha256", "PRETRAIN_ORDERS.json")):
        require(prepare_receipt.get(key) == reader.sha(p / name), "Prepare output link")
    require(reader.verify(prepare_receipt["source_hashes"]) ==
            reader.verify(plan["independent_prepare_source_hashes"]), "Prepare source closure")
    artifacts = reader.verify(prepare_receipt["artifact_hashes"])
    required = dict(cached, **{str(p / n): reader.sha(p / n) for n in ("PLAN.json", "CACHE.json")})
    require(all(artifacts.get(k) == v for k, v in required.items()), "Complete prepare artifacts")
    allowed = dict(pins, **required)
    require(all(allowed.get(k) == v for k, v in artifacts.items()), "Bound prepare artifacts")
    frozen = reader.json(p / "FROZEN.json")
    require(set(frozen["hashes"]) == set(FROZEN_FILES), "Exact frozen inputs")
    reader.verify(frozen["hashes"], p)
    require(frozen.get("pristine_oos") is False and frozen.get("all_diagnostic_orders_frozen") is True,
            "Frozen diagnostic scope")
    verify_diagnostic_start(reader, p, plan_sha, audit_started or datetime.now(timezone.utc))
    orders = reader.json(p / "PRETRAIN_ORDERS.json")
    h.assert_exact_tree(orders[NAMES[0]], reader.json(plan["inputs"]["original_orders"]["path"]))
    legacy_cache_path = reader.path(plan["legacy149"]["cache"]["path"])
    legacy_cache = reader.json(legacy_cache_path)
    legacy_orders_path = legacy_cache_path.parent / "PRETRAIN_ORDERS.json"
    require(reader.sha(legacy_orders_path) == legacy_cache["PRETRAIN_ORDERS.json"],
            "Legacy orders cache")
    h.assert_exact_tree(orders[NAMES[1]], reader.json(legacy_orders_path)["earnings_control"])
    prior_dir=reader.path(plan['prior150']['cache']['path']).parent
    previous_orders=reader.json(prior_dir/'PRETRAIN_ORDERS.json')
    for name,oldname in zip(NAMES[:3],('incumbent_local67','original_earnings','expanded_earnings')):
        h.assert_exact_tree(orders[name],previous_orders[oldname])
    arrays=reader.npz(p/'DATA.npz')
    original=reader.npz(prep.LOCAL+'/DATA.npz')
    prep.verify_original_arrays(original,arrays)
    reference,mask=prep.verify_reference(original,arrays,reader.npz(legacy_cache_path.parent/'DATA.npz'))
    prep.verify_prior_reference(original,arrays,reader.npz(prior_dir/'DATA.npz'),reference,mask)
    scores=reader.npz(p/'SCORES.npz');previous_scores=reader.npz(prior_dir/'SCORES.npz')
    for name,oldname in zip(NAMES[:3],('incumbent_local67','original_earnings','expanded_earnings')):
        require(scores[name].tobytes()==previous_scores[oldname].tobytes(),'Prior control full-score bytes')
    return p


def audit(root, plan_sha):

    started = datetime.now(timezone.utc)
    reader = Reader(root)
    p = bindings(reader, plan_sha, started)
    plan, meta, orders = (reader.json(p / n) for n in ("PLAN.json", "META.json", "PRETRAIN_ORDERS.json"))
    require(meta == reader.json(plan["inputs"]["meta"]["path"])
            and meta["dates"][0] == "2016-08-17" and meta["ends"][-1] == "2026-09-24",
            "Original full horizon/meta")
    data = reader.npz(plan["inputs"]["prices"]["path"])
    require(set(data) == {"dates", "symbols", "close", "opening"}, "Price payload")
    dates = pd.to_datetime(data["dates"])
    close = pd.DataFrame(data["close"], index=dates, columns=data["symbols"])
    opening = pd.DataFrame(data["opening"], index=dates, columns=data["symbols"])
    report = check_outcomes(close, opening, orders, meta, lambda n: reader.json(p / n))
    feature = reader.json(p / "FEATURE_AUDIT.json")
    report["financial_coverage_feature_matrices_verified"] = verify_feature_matrices(
        reader.npz(p / "DATA.npz"), feature, meta)
    require(feature["legacy149"] == plan["legacy149"]
            and feature["coverage_sources"] == plan["coverage_sources"] and feature["prior150"] == plan["prior150"], "Feature source linkage")
    report.update(schema_version=1, plan_sha256=plan_sha,
                  cache_sha256=reader.sha(p / "CACHE.json"),
                  orders_sha256=reader.sha(p / "PRETRAIN_ORDERS.json"),
                  frozen_sha256=reader.sha(p / "FROZEN.json"),
                  proposal_sha256=plan["inputs"]["proposal"]["sha256"],
                  source_admission_bound=True, old44_source_records_and_holds_preserved=True,
                  coverage_source_admission_verified=True,
                  original149_earnings_cache_and_mask_verified=True,
                  latest_catalog_blocking_independently_verified=True,
                  original149_earnings_control_unchanged=True,
                  prior150_batch1_earnings_cache_and_mask_verified=True,
                  batch1_source_records_and_holds_preserved=True,
                  prior150_earnings_control_unchanged=True,
                  baseline_local67_full_orders_unchanged=True,
                  diagnostic_start_marker_verified=True,
                  diagnose_started_sha256=reader.sha(p / "DIAGNOSE_STARTED.json"),
                  audit_started_utc=started.isoformat(),
                  source_hashes={str(f): reader.sha(f) for f in source_files(root)},
                  first_public_version_verified=False, source_admission_is_conditional=True,
                  expanded_available_cells=feature["expanded_cells"],
                  added_available_cells=feature["added_cells"],
                  limitations=["Repeatedly researched history is not pristine out of sample.",
                               "Conditional annual source coverage, not certified first-public history.",
                               "Full protected inventory is audited separately."])
    for path, digest in list(reader.seen.items()):
        require(reader.sha(path) == digest, "Post-review binding")
    report["artifact_hashes"] = dict(reader.seen)
    return report


def synthetic_receipt(result):
    contracts = [ROOT / "work" / name for name in (
        "monthly_financial_coverage151_runner.py",
        "monthly_financial_coverage151_factor.py",
        "monthly_financial_coverage151_proposal.json")]
    return dict(schema_version=1, stage="synthetic_outcome_review_only",
                passed=result.wasSuccessful(), tests_run=result.testsRun,
                failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source_files()},
                reviewed_contract_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in contracts},
                actual_evaluation_checked=False, actual_factors_calculated=0,
                actual_returns_calculated=0, producer_modules_imported=False,
                arms=list(NAMES), expected_execution_paths=12, expected_monthly_orders=488,
                expected_source_cells=5978,
                runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--plan-sha", required=True)
    cli.add_argument("--evaluate-complete", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    require(args.evaluate_complete, "Completed evaluation required")
    result = audit(ROOT, args.plan_sha)
    dest = args.output or ROOT / STUDY / "INDEPENDENT_OUTCOME.json"
    with dest.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=True, output=str(dest),
                          sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                          execution_paths=12)))
