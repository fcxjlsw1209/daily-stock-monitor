"""Three fixed session-clustering arms on the unchanged local67 strategy.

Only a registered round148 PLAN permits real feature/portfolio computation.
All prices are read from the existing frozen local cache, strictly offline.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_same_report145_runner as previous
import monthly_session_cluster148_factor as factor

require, sha, read, dump = base.require, base.sha, base.read, base.dump
NAMES = ("incumbent_local67", "event_frequency_control", "low_event_clustering")
HANDS = ("frequency_rank", "cluster_rank")
EXTRA = ("frequency", "cluster", "frequency_rank", "cluster_rank", "available")
TRAIN, MONTHS, WIDTH, DOSE = 88, 122, 49, .01
OUT = "outputs/monthly_session_cluster148"
LOCAL = "outputs/monthly_local_linear70"
PRICE = "outputs/monthly_2016_extension/PRICES.npz"
PRICE_SHA = "c877cbe443964d53972114bc654c473608362a6bb31577fb1de5e00cbd3c1485"
SESSIONS = "outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json"
SESSIONS_SHA = "c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4"
CALENDAR = "outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json"
CALENDAR_SHA = "4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea"
CONCEPT = "work/monthly_session_cluster148_concept.json"
CONCEPT_SHA = "1998f70944d1679ca29427825f284d1a08e0f0b0767767af9babd76a2fc9d8ee"
CACHE_FILES = base.CACHE_FILES
FROZEN_FILES = ("PLAN.json", "CACHE.json", "PRETRAIN_ORDERS.json", "TRAIN_TRIALS.json",
                "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json", "INDEPENDENT_PREPARE.json")
CHECKS = ("original27_arrays_byte_identical", "all_features_independently_reconstructed",
          "all_three_score_matrices_and_ordered_choices_independently_verified",
          "one_episode_and_prefix_isolation_verified", "zero_and_missing_exact_fallback")
INPUTS = ("proposal", "meta", "policy", "original_orders", "prices", "sessions", "protected_manifest")
TOLERANCES = dict(base.TOLERANCES, scores_atol=1e-12, feature_atol=1e-12)


def selection_contract():
    return dict(total_months=122, training_months=88, early_months=53,
        recent_months=35, later_months=34, walkforward=[[36, 60], [60, 76], [76, 88]],
        execution_paths=[
            dict(execution="close", cost=0., key="same_close"),
            dict(execution="open", cost=0., key="next_open_zero"),
            dict(execution="open", cost=.0005, key="next_open_cost")],
        same_close_cagr="strict_greater_than_both_controls",
        next_open_cost_cagr="noninferior_incumbent_strict_greater_than_frequency_control",
        strict_cagr_epsilon=1e-12, max_sharpe_degradation=.05,
        max_drawdown_degradation=.03, max_early_cagr_degradation=.025,
        max_recent_cagr_degradation=.025, new_model_fits=0,
        intramonth_trades=False, later34_for_selection=False)


def feature_contract():
    return dict(method="adjusted_session_event_lag1_covariance_skip42",
        event_sessions=210, adjacent_pairs=209, skip_sessions=42,
        first_event_offset=-251, last_event_offset=-42, first_close_offset=-252,
        sign_zero_tolerance=1e-12,
        event="overnight>1e-12 and intraday< -1e-12",
        covariance="(N11*209-Nleft*Nright)/(209*209)",
        rank="0.5-average_zero_based_rank/(n_valid-1)",
        rank_population="common_observed_original49_including_ETFs_before_trading_eligibility",
        missing_rank=0., singleton_rank=0., all_equal_rank=0.,
        max_abs_rank=.5, dose=DOSE, max_abs_correction=.005,
        apply_to="all_available_before_one_original_episode_selector_and_its_original_eligibility",
        required_prices="211_adjusted_closes_and210_adjusted_opens",
        missing_window="invalidate_both_features_without_dropping_pairs_or_filling")


def validate_proposal(p):
    require(p.get("round") == 148 and p.get("arms") == list(NAMES)
            and p.get("primary_challenger") == NAMES[2] and p.get("dose") == DOSE,
            "Wrong prospective experiment")
    require(p.get("concept") == dict(path=CONCEPT, sha256=CONCEPT_SHA),
            "Prospective concept changed")
    require(p.get("selection_contract") == selection_contract(), "Selection contract changed")
    require(p.get("feature_contract") == feature_contract(), "Feature contract changed")


def select(rows, opened):
    for group in (rows, opened):
        require([x["name"] for x in group] == list(NAMES), "Three exact arms required")
    close = {NAMES[i]: previous.guard(rows[2], rows[i], strict=True) for i in range(2)}
    op = {NAMES[i]: previous.guard(opened[2], opened[i], strict=i != 0) for i in range(2)}
    passed = all(close.values()) and all(op.values())
    return dict(primary=NAMES[2] if passed else NAMES[0], admitted=passed,
                guards_against=close, next_open_guards_against=op,
                mechanism_confirmed=all(op.values()))


def session_axis(values, name):
    axis = pd.DatetimeIndex(values)
    require(axis.tz is None and not axis.hasnans and axis.is_unique
            and axis.is_monotonic_increasing and axis.equals(axis.normalize()),
            "Canonical unique ordered "+name)
    return axis


def price_frames(payload, sessions, meta):
    """Validate the cached axes and select only the pinned calendar's coverage."""
    require(set(payload) == {"dates", "symbols", "close", "opening"}, "Exact price cache schema")
    dates = session_axis(payload["dates"], "price dates")
    calendar = session_axis(sessions, "official sessions")
    signals = session_axis(meta["dates"], "signals")
    require(len(calendar) and len(dates) and len(signals), "Nonempty price/calendar/signal axes")
    positions = calendar.get_indexer(signals)
    require((positions >= 252).all(), "All signals require the full fixed warmup")
    symbols = np.asarray(payload["symbols"])
    require(symbols.ndim == 1 and len(set(symbols)) == len(symbols)
            and all(isinstance(s, str) and bool(s) for s in symbols), "Unique named price columns")
    shape = (len(dates), len(symbols))
    frames = []
    keep = (dates >= calendar[0]) & (dates <= calendar[-1])
    covered = dates[keep]
    require(len(covered) and covered.equals(calendar[(calendar >= covered[0]) &
            (calendar <= covered[-1])]), "Cached date rows must match the official session grid")
    require(covered[0] <= calendar[int(positions.min())-252]
            and covered[-1] >= signals[-1], "Price axis must cover every fixed feature window")
    require(all(set(names) <= set(symbols) for names in meta["symbols"]), "Original pool columns missing")
    for key in ("close", "opening"):
        values = np.asarray(payload[key])
        require(values.dtype == np.dtype("float64") and values.shape == shape, "Price dtype/shape")
        # Do not fill or discard invalid monetary observations. The factor marks
        # the entire affected pair of features unavailable on its common mask.
        frames.append(pd.DataFrame(values[keep], index=covered, columns=symbols))
    return frames


def validate_features(arrays, n=MONTHS):
    """Missing ranks are zero; missing raw values retain their distinct boolean mask."""
    require(type(n) is int and 0 <= n <= MONTHS, "Valid feature prefix")
    mask = np.asarray(arrays["available"])
    require(mask.shape == (MONTHS, WIDTH) and mask.dtype == bool, "Common mask schema")
    mask = mask[:n]
    for key in EXTRA[:-1]:
        values = np.asarray(arrays[key])
        require(values.shape == (MONTHS, WIDTH) and values.dtype == np.dtype("float64"),
                "Feature shape/dtype: "+key)
        values = values[:n]
        if key in HANDS:
            require(np.isfinite(values).all() and (np.abs(values) <= .5).all()
                    and (values[~mask] == 0.).all(), "Finite bounded ranks with neutral missingness")
        else:
            require(not np.isinf(values).any() and np.array_equal(np.isfinite(values), mask),
                    "Common raw feature availability")
            lo, hi = (0., 1.) if key == "frequency" else (-.25, .25)
            require(((values[mask] >= lo) & (values[mask] <= hi)).all(), "Raw feature range")


def corrected(pre, hand, arrays, meta, n, backend, policy):
    require(type(n) is int and 0 <= n <= MONTHS, "Valid score prefix")
    shape = np.asarray(arrays["baseline"]).shape
    require(shape == (MONTHS, WIDTH), "Original full candidate axes")
    hand = np.asarray(hand)
    require(hand.shape == shape and hand.dtype == np.dtype("float64"), "Rank shape/dtype")
    prefix = hand[:n]
    require(np.isfinite(prefix).all() and (np.abs(prefix) <= .5).all(), "Finite centered rank")
    good = np.asarray(arrays["eligible"][:n])
    require(good.dtype == bool and good.shape == (n, WIDTH), "Original eligibility mask")
    available = np.asarray(arrays["available"][:n])
    require(available.dtype == bool and available.shape == (n, WIDTH)
            and (prefix[~available] == 0.).all(), "Unavailable ranks must remain zero")
    view = np.asarray(pre).copy()
    require(view.shape == (n, WIDTH) and view.dtype == np.dtype("float64"), "Original pre-score shape/dtype")
    active = available & (prefix != 0.)
    delta = np.zeros(view.shape)
    delta[active] = DOSE*prefix[active]
    require(np.isfinite(delta).all() and np.max(np.abs(delta), initial=0.) <= .005,
            "Correction exceeded the fixed half-rank cap")
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
    validate_features(arrays)
    orders, scores = {}, {}
    for name in NAMES:
        c, s = choose(name, arrays, meta, MONTHS, backend, policy)
        orders[name], scores[name] = backend.make_orders(c, s, meta), s
        for n in range(MONTHS+1):
            pc, ps = choose(name, arrays, meta, n, backend, policy)
            require(pc == c[:n] and ps.tobytes() == s[:n].tobytes(), "Future-prefix dependence")
    # Both a valid zero signal and an unavailable signal must leave the original
    # score byte-identical. Neither condition is inferred from the raw value.
    for available in (True, False):
        null = dict(arrays, available=np.full((MONTHS, WIDTH), available, dtype=bool))
        for key in HANDS:
            null[key] = np.zeros((MONTHS, WIDTH))
        for key in ("frequency", "cluster"):
            null[key] = np.full((MONTHS, WIDTH), 0. if available else np.nan)
        validate_features(null)
        for name in NAMES[1:]:
            c, s = choose(name, null, meta, MONTHS, backend, policy)
            require(backend.make_orders(c, s, meta) == orders[NAMES[0]]
                    and s.tobytes() == scores[NAMES[0]].tobytes(), "Zero/missing fallback")
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()},
            "Input arrays changed")
    return orders, scores


class Backend(previous.Backend):
    def pre(self, arrays, n):
        require(type(n) is int and 0 <= n <= MONTHS, "Valid pre-score prefix length")
        full = np.asarray(super().pre(arrays, n))
        require(full.shape == (MONTHS, WIDTH) and full.dtype == np.dtype("float64"),
                "Original full pre-score shape/dtype")
        return full[:n].copy()

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
        require(p["round"] == 148 and p["status"] == "REGISTERED"
                and p["arms"] == list(NAMES) and p["primary_challenger"] == NAMES[2],
                "Round148 registration required")
        require(p["dose"] == DOSE and p["tolerances"] == TOLERANCES, "Declared numeric policy")
        require(p["runtime"] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                "Pinned runtime changed")
        self.pins = self.verify(p["hashes"])
        require(set(p["inputs"]) == set(INPUTS), "Exact seven named inputs")
        self.inputs = {k: self.bound_path(v["path"]) for k, v in p["inputs"].items()}
        require(all(sha(self.inputs[k]) == v["sha256"] for k, v in p["inputs"].items()),
                "Input binding changed")
        validate_proposal(read(self.inputs["proposal"]))
        require(self.pins.get(str(self.root/CONCEPT)) == CONCEPT_SHA, "Concept not bound")
        require(sha(self.inputs["protected_manifest"]) == base.PROTECTED_SHA, "Protection manifest changed")
        for path, digest in source_files(self.root).items():
            require(self.pins.get(path) == digest, "Current implementation not registered")
        for section in ("raw_source_hashes", "independent_prepare_source_hashes"):
            require(all(self.pins.get(k) == v for k, v in self.verify(p[section]).items()),
                    "Unregistered "+section)
        local_plan = read(self.bound_path(LOCAL+"/PLAN.json"))
        require(self.verify(p["raw_source_hashes"]) == self.verify(local_plan["raw_source_hashes"])
                and len(p["raw_source_hashes"]) == 64, "Original raw64 unchanged")
        for key, expected in (("meta", LOCAL+"/META.json"),
                              ("policy", LOCAL+"/INCUMBENT_POLICY.json"),
                              ("original_orders", LOCAL+"/local_linear_ORDERS.json"),
                              ("prices", PRICE), ("sessions", SESSIONS)):
            require(self.inputs[key] == self.root/expected, "Original input path: "+key)
        require(sha(self.inputs["prices"]) == PRICE_SHA
                and sha(self.inputs["sessions"]) == SESSIONS_SHA, "Frozen price/calendar input")
        self.out = self.root/OUT
        require(self.plan_path == self.out/"PLAN.json" and Path(p["output_directory"]) == self.out,
                "Dedicated output directory")
        self.backend_factory = backend_factory
        self.require_calendar()

    def require_calendar(self):
        calendar_path = self.bound_path(CALENDAR)
        require(sha(calendar_path) == CALENDAR_SHA, "Original calendar plan")
        calendar, saved = read(calendar_path), read(self.inputs["sessions"])
        require(calendar["start"] == "2015-01-01" and calendar["end"] == "2027-12-31"
                and saved["plan_sha256"] == CALENDAR_SHA, "Calendar provenance/coverage")
        require(all(self.pins.get(k) == v for k, v in self.verify(calendar["bindings"]).items()),
                "Calendar dependency not registered")
        dates = session_axis(saved["sessions"], "official sessions")
        require(len(dates) and dates[0] >= pd.Timestamp(calendar["start"])
                and dates[-1] <= pd.Timestamp(calendar["end"]), "Calendar bounds")
        return dates

    def prepare(self):
        self.refresh()
        require(not any((self.out/x).exists() for x in (*CACHE_FILES, "CACHE.json")),
                "Preparation already started; preserve partial outputs")
        backend = self.backend_factory(self.root, self)
        meta, original, px, op = backend.load()
        base.validate_meta(meta)
        require(len(original) == 27 and meta == read(self.inputs["meta"]), "Original cache")
        require(not set(original) & set(EXTRA), "New feature names collide with original arrays")
        parity = backend.price_parity(meta, px, op, self.plan["raw_source_hashes"])
        sessions = self.require_calendar()
        with np.load(self.inputs["prices"], allow_pickle=False) as data:
            close, opening = price_frames(dict(data), sessions, meta)
        result = factor.compute_features(sessions, close, opening, meta["dates"], meta["symbols"])
        require(set(result) == set(EXTRA)|{"audit"}, "Exact factor output")
        features = {key: result[key] for key in EXTRA}
        validate_features(features)
        require(result["audit"].get("method") == factor.METHOD
                and result["audit"].get("new_model_fits") == 0
                and result["audit"].get("portfolio_returns_or_nav_computed") is False,
                "Pure declared feature computation")
        arrays = dict(original, **features)
        orders, scores = panels(arrays, meta, backend, read(self.inputs["policy"]))
        require(orders[NAMES[0]] == read(self.inputs["original_orders"]), "Incumbent changed")
        result["audit"].update(price_cache_sha256=sha(self.inputs["prices"]),
            sessions_sha256=sha(self.inputs["sessions"]), calendar_plan_sha256=CALENDAR_SHA,
            feature_contract=feature_contract())
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
            require(set(data.files) == set(original)|set(EXTRA), "Prepared array names changed")
            require(all(data[k].tobytes() == v.tobytes() and data[k].dtype == v.dtype
                        and data[k].shape == v.shape for k, v in original.items()), "Original27 changed")
        return backend, meta, px, op

    def require_prepare_review(self):
        self.cache()
        r = read(self.out/"INDEPENDENT_PREPARE.json")
        require(r.get("passed") is True and r.get("stage") == "prepare"
                and all(r.get(k) is True for k in CHECKS), "Independent preparation checks incomplete")
        require(r.get("producer_modules_imported") is False
                and r.get("returns_or_nav_computed") is False, "Independent preparation isolation")
        require(r.get("monthly_label_models_fit") == 0 and r.get("new_model_fits") == 0,
                "No new model fitting")
        require(r["plan_sha256"] == self.plan_sha and r["cache_sha256"] == sha(self.out/"CACHE.json")
                and r["orders_sha256"] == sha(self.out/"PRETRAIN_ORDERS.json"), "Review binding changed")
        reviewed = self.verify(r["artifact_hashes"])
        required = {str(self.out/name): sha(self.out/name) for name in ("PLAN.json", "CACHE.json", *CACHE_FILES)}
        require(all(reviewed.get(path) == digest for path, digest in required.items()), "Unreviewed cache output")
        allowed = dict(self.pins, **required)
        require(all(allowed.get(path) == digest for path, digest in reviewed.items()), "Unbound reviewed artifact")
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
             root/"work/test_monthly_session_cluster148_factor.py",
             root/"work/test_monthly_session_cluster148_runner.py")
    return dict(previous.source_files(root), **{str(p.resolve()): sha(p) for p in files})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "select_first88", "diagnose_later34"))
    p.add_argument("--plan", required=True)
    p.add_argument("--plan-sha", required=True)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    getattr(Runner(a.root, a.plan, a.plan_sha), a.phase)()
