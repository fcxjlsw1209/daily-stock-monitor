"""Synthetic-only runner checks; no historical prices, factors or returns."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_session_cluster148_runner as r


def row(name, cagr=.4, sharpe=1.5, dd=-.25, early=.3, recent=.5):
    return dict(name=name, training=dict(cagr=cagr, sharpe=sharpe, max_drawdown=dd),
                early=dict(cagr=early), recent=dict(cagr=recent))


def picks(scores, eligible):
    result = []
    for values, mask in zip(scores, eligible):
        order = sorted(np.flatnonzero(mask), key=lambda j: (-values[j], j))
        result.append([int(j) for j in order[:3]] if len(order) >= 2 else [])
    return result


class Adapter:
    def __init__(self):
        self.calls = []

    def score_and_choose(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        assert args[0] == "incumbent66" and args[1] == 0.
        assert kwargs["boundary_selector"] == "original"
        n, arrays = args[5], args[3]
        scores = kwargs["pre_score"][:n].copy()
        return picks(scores, arrays["eligible"][:n]), scores


def fixture():
    shape = (122, 49)
    a = dict(baseline=np.zeros(shape), eligible=np.ones(shape, bool), effects=np.zeros(shape),
             frequency=np.full(shape, .25), cluster=np.zeros(shape),
             frequency_rank=np.zeros(shape), cluster_rank=np.zeros(shape),
             available=np.ones(shape, bool))
    names = ["S"+str(i) for i in range(32)] + ["ETF"+str(i) for i in range(17)]
    m = dict(symbols=[names]*122, etfs=names[32:])
    b = SimpleNamespace(adapter=Adapter(), selector="original")
    return a, m, b


def proposal():
    return dict(round=148, arms=list(r.NAMES), primary_challenger=r.NAMES[2], dose=.01,
                concept=dict(path=r.CONCEPT, sha256=r.CONCEPT_SHA),
                selection_contract=r.selection_contract(), feature_contract=r.feature_contract())


class RankingIntegrationTests(unittest.TestCase):
    def test_exact_half_rank_cap_and_one_original_selector(self):
        a, m, b = fixture()
        hand = a["cluster_rank"]
        hand[0, :3] = [-.5, 0., .5]
        policy = {"synthetic": "policy"}
        _, scores = r.corrected(np.zeros((122, 49)), hand, a, m, 122, b, policy)
        self.assertEqual(list(scores[0, :3]), [-.005, 0., .005])
        self.assertEqual(len(b.adapter.calls), 1)
        args, kwargs = b.adapter.calls[0]
        self.assertIs(args[2], hand)
        self.assertIs(kwargs["effects"], a["effects"])
        self.assertIs(kwargs["episode_policy"], policy)

    def test_noneligible_rank_is_not_removed_before_original_selector(self):
        a, m, b = fixture()
        a["eligible"][0, 0] = False
        a["cluster_rank"][0, 0] = .5
        choices, scores = r.corrected(np.zeros((122, 49)), a["cluster_rank"], a, m, 122, b, {})
        self.assertEqual(scores[0, 0], .005)
        self.assertNotIn(0, choices[0])
        self.assertFalse(a["eligible"][0, 0])

    def test_etf_correction_is_allowed(self):
        a, m, b = fixture()
        a["frequency_rank"][0, 32] = .5
        _, scores = r.corrected(np.zeros((122, 49)), a["frequency_rank"], a, m, 122, b, {})
        self.assertEqual(scores[0, 32], .005)

    def test_zero_and_unavailable_preserve_score_bytes(self):
        a, m, b = fixture()
        pre = np.arange(122*49, dtype=float).reshape(122, 49)/100
        pre[0, 0] = -0.
        for available in (True, False):
            a["available"][:] = available
            _, scores = r.corrected(pre, a["frequency_rank"], a, m, 122, b, {})
            self.assertEqual(scores.tobytes(), pre.tobytes())

    def test_unavailable_nonzero_rank_is_rejected(self):
        a, m, b = fixture()
        a["available"][0, 0] = False
        a["cluster_rank"][0, 0] = .1
        with self.assertRaises(ValueError):
            r.corrected(np.zeros((122, 49)), a["cluster_rank"], a, m, 122, b, {})

    def test_invalid_rank_values_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf, .500001, -.500001):
            a, m, b = fixture()
            a["cluster_rank"][0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.corrected(np.zeros((122, 49)), a["cluster_rank"], a, m, 122, b, {})

    def test_future_rank_poison_does_not_affect_prefix(self):
        a, m, b = fixture()
        a["cluster_rank"][1:] = np.inf
        _, scores = r.corrected(np.zeros((1, 49)), a["cluster_rank"], a, m, 1, b, {})
        self.assertEqual(scores.shape, (1, 49))
        self.assertFalse(scores.any())

    def test_empty_prefix(self):
        a, m, b = fixture()
        choices, scores = r.corrected(np.zeros((0, 49)), a["cluster_rank"], a, m, 0, b, {})
        self.assertEqual(choices, [])
        self.assertEqual(scores.shape, (0, 49))

    def test_bad_shapes_types_or_prefix_are_rejected(self):
        a, m, b = fixture()
        for n in (-1, 123, True, 1.):
            with self.subTest(n=n), self.assertRaises(ValueError):
                r.corrected(np.zeros((1, 49)), a["cluster_rank"], a, m, n, b, {})
        with self.assertRaises(ValueError):
            r.corrected(np.zeros((1, 49)), a["cluster_rank"].astype("float32"), a, m, 1, b, {})
        with self.assertRaises(ValueError):
            r.corrected(np.zeros((122, 49)), a["cluster_rank"], a, m, 1, b, {})

    def test_feature_common_mask_and_neutral_rank_contract(self):
        a, _, _ = fixture()
        a["available"][0, 0] = False
        for key in ("frequency", "cluster"):
            a[key][0, 0] = np.nan
        r.validate_features(a)
        for key, value in (("frequency", 0.), ("cluster", np.inf),
                           ("frequency_rank", .1), ("cluster_rank", np.nan)):
            bad = {k: v.copy() for k, v in a.items()}
            bad[key][0, 0] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_features(bad)

    def test_raw_feature_ranges(self):
        a, _, _ = fixture()
        for key, value in (("frequency", -.001), ("frequency", 1.001),
                           ("cluster", -.251), ("cluster", .251)):
            bad = {k: v.copy() for k, v in a.items()}
            bad[key][0, 0] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                r.validate_features(bad)

    def test_feature_prefix_ignores_future_poison(self):
        a, _, _ = fixture()
        a["cluster"][1:] = np.inf
        a["cluster_rank"][1:] = np.nan
        r.validate_features(a, 1)
        with self.assertRaises(ValueError):
            r.validate_features(a)

    def test_pre_slices_all_prefixes_without_modifying_full_array(self):
        full = np.arange(122*49, dtype=float).reshape(122, 49)/100
        before = full.tobytes()
        b = object.__new__(r.Backend)
        with patch.object(r.previous.Backend, "pre", autospec=True, return_value=full) as old:
            for n in range(123):
                prefix = b.pre({}, n)
                self.assertEqual(prefix.shape, (n, 49))
                self.assertEqual(prefix.tobytes(), full[:n].tobytes())
                self.assertFalse(np.shares_memory(prefix, full))
                old.assert_called_with(b, {}, n)
                if n:
                    prefix[0, 0] = -999.
                self.assertEqual(full.tobytes(), before)

    def test_pre_rejects_bad_parent_contract(self):
        b = object.__new__(r.Backend)
        for value in (np.zeros((121, 49)), np.zeros((122, 48)),
                      np.zeros((122, 49), dtype=int), np.zeros((122, 49), dtype="float32")):
            with patch.object(r.previous.Backend, "pre", return_value=value), self.assertRaises(ValueError):
                b.pre({}, 1)

    def test_pre_rejects_bad_length_before_parent_call(self):
        b = object.__new__(r.Backend)
        with patch.object(r.previous.Backend, "pre") as old:
            for n in (-1, 123, True, 1., None):
                with self.assertRaises(ValueError):
                    b.pre({}, n)
            old.assert_not_called()

    def test_baseline_implementation_is_inherited(self):
        self.assertNotIn("baseline", r.Backend.__dict__)
        self.assertIs(r.Backend.baseline, r.previous.Backend.baseline)
        b = object.__new__(r.Backend)
        with patch.object(r.previous.Backend, "baseline", return_value=("old", "scores")) as old:
            self.assertEqual(r.choose(r.NAMES[0], {}, {}, 122, b, {}), ("old", "scores"))
            old.assert_called_once_with({}, {}, 122, {})

    def test_three_arm_panels_all_prefixes_and_366_synthetic_orders(self):
        a, m, b = fixture()
        for j, key in enumerate(r.HANDS):
            a[key][:, j] = .5
        for j in range(24):
            a["original_synthetic_"+str(j)] = np.zeros((122, 49))
        pre = np.tile(np.linspace(-1., 1., 49), (122, 1))
        b.pre = lambda arrays, n: pre[:n].copy()
        b.baseline = lambda arrays, meta, n, policy: (picks(pre[:n], arrays["eligible"][:n]), pre[:n].copy())
        b.make_orders = lambda choices, scores, meta: [
            dict(indices=choice, scores=[float(scores[i, j]) for j in choice])
            for i, choice in enumerate(choices)]
        before = {key: value.tobytes() for key, value in a.items()}
        orders, scores = r.panels(a, m, b, {})
        self.assertEqual(list(orders), list(r.NAMES))
        self.assertEqual(sum(len(value) for value in orders.values()), 366)
        self.assertEqual(set(scores), set(r.NAMES))
        self.assertEqual(before, {key: value.tobytes() for key, value in a.items()})


class PriceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2013-01-01", periods=600)
        self.sessions = self.dates[100:]
        self.symbols = np.array(["S"+str(i) for i in range(49)])
        self.payload = dict(dates=self.dates.strftime("%Y-%m-%d").to_numpy(),
            symbols=self.symbols, close=np.ones((600, 49)), opening=np.full((600, 49), 2.))
        self.meta = dict(dates=[str(self.dates[400].date())], symbols=[list(self.symbols)])

    def test_old_rows_are_excluded_without_shifting_warmup(self):
        close, opening = r.price_frames(self.payload, self.sessions, self.meta)
        self.assertTrue(close.index.equals(self.sessions))
        self.assertTrue(opening.index.equals(self.sessions))
        self.assertEqual(close.index.get_loc(self.meta["dates"][0]), 300)
        self.assertEqual(opening.iloc[0, 0], 2.)
        self.assertEqual(self.payload["close"].shape, (600, 49))

    def test_invalid_prices_remain_invalid_for_factor_common_mask(self):
        self.payload["close"][350, 0] = np.nan
        self.payload["opening"][351, 1] = 0.
        close, opening = r.price_frames(self.payload, self.sessions, self.meta)
        self.assertTrue(np.isnan(close.loc[self.dates[350], "S0"]))
        self.assertEqual(opening.loc[self.dates[351], "S1"], 0.)

    def test_removed_session_row_is_rejected_not_compressed(self):
        payload = {key: value.copy() for key, value in self.payload.items()}
        for key in ("dates", "close", "opening"):
            payload[key] = np.delete(payload[key], 350, axis=0)
        with self.assertRaises(ValueError):
            r.price_frames(payload, self.sessions, self.meta)

    def test_insufficient_calendar_warmup_rejected(self):
        with self.assertRaises(ValueError):
            r.price_frames(self.payload, self.dates[200:], self.meta)

    def test_price_cache_must_reach_required_first_close(self):
        for key in ("dates", "close", "opening"):
            self.payload[key] = self.payload[key][200:]
        with self.assertRaises(ValueError):
            r.price_frames(self.payload, self.sessions, self.meta)

    def test_duplicate_or_noncanonical_axes_rejected(self):
        self.payload["dates"][301] = self.payload["dates"][300]
        with self.assertRaises(ValueError):
            r.price_frames(self.payload, self.sessions, self.meta)
        with self.assertRaises(ValueError):
            r.session_axis(["2020-01-01 12:00:00"], "synthetic")

    def test_original_pool_columns_and_price_dtype_required(self):
        self.payload["opening"] = self.payload["opening"].astype("float32")
        with self.assertRaises(ValueError):
            r.price_frames(self.payload, self.sessions, self.meta)
        self.payload["opening"] = self.payload["opening"].astype("float64")
        self.meta["symbols"][0][0] = "MISSING"
        with self.assertRaises(ValueError):
            r.price_frames(self.payload, self.sessions, self.meta)

    def test_unregistered_cache_field_rejected(self):
        self.payload["replacement_close"] = self.payload["close"]
        with self.assertRaises(ValueError):
            r.price_frames(self.payload, self.sessions, self.meta)


class SelectionTests(unittest.TestCase):
    def test_proposal_binds_exact_concept_features_and_selection(self):
        p = proposal()
        r.validate_proposal(p)
        for key, value in (("round", 147), ("dose", .02), ("primary_challenger", r.NAMES[1]),
                           ("feature_contract", {}), ("selection_contract", {}), ("concept", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_proposal(dict(p, **{key: value}))

    def test_three_arm_double_gate_pass(self):
        rows = [row(name, cagr=.4+i*.01) for i, name in enumerate(r.NAMES)]
        self.assertEqual(r.select(rows, rows)["primary"], r.NAMES[2])

    def test_same_close_must_beat_both_controls_strictly(self):
        opened = [row(name, cagr=.4+i*.01) for i, name in enumerate(r.NAMES)]
        for index in (0, 1):
            rows = deepcopy(opened)
            rows[index]["training"]["cagr"] = rows[2]["training"]["cagr"]
            self.assertFalse(r.select(rows, opened)["admitted"])

    def test_open_noninferior_incumbent_but_strict_frequency(self):
        rows = [row(name, cagr=.4+i*.01) for i, name in enumerate(r.NAMES)]
        opened = [row(r.NAMES[0], cagr=.5), row(r.NAMES[1], cagr=.4), row(r.NAMES[2], cagr=.5)]
        self.assertTrue(r.select(rows, opened)["admitted"])
        opened[1]["training"]["cagr"] = .5
        self.assertFalse(r.select(rows, opened)["admitted"])

    def test_open_worsening_cannot_be_rescued_by_close(self):
        rows = [row(name, cagr=.4+i*.1) for i, name in enumerate(r.NAMES)]
        opened = deepcopy(rows)
        opened[0]["training"]["cagr"] = .9
        self.assertFalse(r.select(rows, opened)["admitted"])

    def test_risk_segment_nan_and_identical_portfolios_rejected(self):
        rows = [row(name) for name in r.NAMES]
        self.assertFalse(r.select(rows, rows)["admitted"])
        for field, value in (("sharpe", 1.44), ("dd", -.281), ("early", .274), ("recent", .474),
                             ("cagr", np.nan)):
            changed = rows[:-1]+[row(r.NAMES[-1], **dict(cagr=.9, **{field: value}))
                                     if field != "cagr" else row(r.NAMES[-1], cagr=value)]
            self.assertFalse(r.select(changed, changed)["admitted"])

    def test_arm_order_and_missing_control_rejected(self):
        rows = [row(name) for name in r.NAMES]
        for bad in (rows[::-1], rows[:2]):
            with self.assertRaises(ValueError):
                r.select(bad, rows)

    def test_seven_nonfinancial_inputs_and_nine_paths(self):
        self.assertEqual(len(r.INPUTS), 7)
        self.assertNotIn("admitted_dataset", r.INPUTS)
        self.assertEqual(len(r.selection_contract()["execution_paths"])*len(r.NAMES), 9)
        self.assertEqual(r.feature_contract()["max_abs_correction"], .005)


class RegistrationTests(unittest.TestCase):
    """Mock source I/O; no real PLAN, prices or account paths are opened."""
    def setUp(self):
        self.root = Path("/synthetic148")
        self.plan_path = self.root/r.OUT/"PLAN.json"
        paths = dict(proposal="synthetic_proposal.json", meta=r.LOCAL+"/META.json",
                     policy=r.LOCAL+"/INCUMBENT_POLICY.json",
                     original_orders=r.LOCAL+"/local_linear_ORDERS.json",
                     prices=r.PRICE, sessions=r.SESSIONS, protected_manifest="synthetic_protection.json")
        self.digests = {str(self.root/path): "d"*64 for path in paths.values()}
        self.digests[str(self.plan_path)] = "p"*64
        self.digests[str(self.root/r.PRICE)] = r.PRICE_SHA
        self.digests[str(self.root/r.SESSIONS)] = r.SESSIONS_SHA
        self.digests[str(self.root/paths["protected_manifest"])] = r.base.PROTECTED_SHA
        raw = {str(self.root/("raw_"+str(i))): "r"*64 for i in range(64)}
        independent = {str(self.root/"independent.py"): "i"*64}
        code = {str(self.root/"runner.py"): "c"*64}
        self.plan = dict(round=148, status="REGISTERED", arms=list(r.NAMES),
            primary_challenger=r.NAMES[2], dose=.01, tolerances=r.TOLERANCES,
            runtime=dict(python=r.sys.version, numpy=np.__version__, pandas=pd.__version__),
            inputs={key: dict(path=str(self.root/path), sha256=self.digests[str(self.root/path)])
                    for key, path in paths.items()},
            raw_source_hashes=raw, independent_prepare_source_hashes=independent,
            hashes=dict(self.digests, **raw, **independent, **code),
            output_directory=str(self.root/r.OUT))
        self.plan["hashes"][str(self.root/r.CONCEPT)] = r.CONCEPT_SHA
        self.documents = {str(self.plan_path): self.plan,
                          str(self.root/paths["proposal"]): proposal(),
                          str(self.root/r.LOCAL/"PLAN.json"): dict(raw_source_hashes=raw)}
        mocks = (
            patch.object(r, "sha", side_effect=lambda path: self.digests[str(path)]),
            patch.object(r, "read", side_effect=lambda path: self.documents[str(path)]),
            patch.object(r, "source_files", return_value=code),
            patch.object(r.Runner, "verify", lambda self, values: values),
            patch.object(r.Runner, "bound_path", lambda self, path: self.path(path)),
            patch.object(r.Runner, "require_calendar", return_value=None))
        for mock in mocks:
            mock.start()
            self.addCleanup(mock.stop)

    def construct(self):
        return r.Runner(self.root, self.plan_path, "p"*64)

    def test_valid_synthetic_registration(self):
        result = self.construct()
        self.assertEqual(result.out, self.root/r.OUT)

    def test_bad_runtime_or_unregistered_state_rejected(self):
        self.plan["runtime"] = {}
        with self.assertRaises(ValueError):
            self.construct()

    def test_missing_registered_implementation_rejected(self):
        del self.plan["hashes"][str(self.root/"runner.py")]
        with self.assertRaises(ValueError):
            self.construct()

    def test_extra_financial_input_rejected(self):
        self.plan["inputs"]["source_admission"] = {}
        with self.assertRaises(ValueError):
            self.construct()

    def test_raw_source_closure_cannot_shrink(self):
        self.plan["raw_source_hashes"] = dict(list(self.plan["raw_source_hashes"].items())[:-1])
        with self.assertRaises(ValueError):
            self.construct()

    def test_changed_price_cache_rejected(self):
        path = str(self.root/r.PRICE)
        self.digests[path] = self.plan["inputs"]["prices"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            self.construct()

    def test_changed_protection_manifest_rejected(self):
        path = self.plan["inputs"]["protected_manifest"]["path"]
        self.digests[path] = self.plan["inputs"]["protected_manifest"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            self.construct()


class PhaseInterfaceTests(unittest.TestCase):
    def test_prepare_reads_only_bound_feature_price_cache_with_mock_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dates = pd.bdate_range("2010-01-01", periods=600)
            arrays, meta, _ = fixture()
            meta["dates"] = [str(day.date()) for day in dates[300:422]]
            names = meta["symbols"][0]
            price_path = root/"synthetic_prices.npz"
            np.savez(price_path, dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                     symbols=np.array(names), close=np.ones((600, 49)),
                     opening=np.full((600, 49), 2.))
            orders = {name: [] for name in r.NAMES}
            refs = {}
            for key, value in (("meta", meta), ("original_orders", []), ("policy", {}),
                               ("sessions", dict(sessions=[str(d.date()) for d in dates]))):
                refs[key] = root/(key+".json")
                refs[key].write_text(json.dumps(value))
            refs["prices"] = price_path
            original = {key: arrays[key] for key in ("baseline", "eligible", "effects")}
            original.update({"original"+str(i): np.zeros((122, 49)) for i in range(24)})
            backend = SimpleNamespace(load=lambda: (meta, original, "original-close", "original-open"),
                price_parity=lambda m, px, op, raw: dict(px=px, op=op))
            obj = object.__new__(r.Runner)
            obj.root, obj.out, obj.inputs = root, root/"out", refs
            obj.out.mkdir()
            obj.plan = dict(raw_source_hashes={})
            obj.refresh = lambda: None
            obj.require_calendar = lambda: dates
            obj.backend_factory = lambda root, runner: backend
            result = {key: arrays[key].copy() for key in r.EXTRA}
            result["audit"] = dict(method=r.factor.METHOD, new_model_fits=0,
                                    portfolio_returns_or_nav_computed=False)
            scores = {name: np.zeros((122, 49)) for name in r.NAMES}
            with patch.object(r.base, "validate_meta"), \
                 patch.object(r.factor, "compute_features", return_value=result) as compute, \
                 patch.object(r, "panels", return_value=(orders, scores)):
                obj.prepare()
            args = compute.call_args.args
            self.assertTrue(args[0].equals(dates))
            self.assertIsInstance(args[1], pd.DataFrame)
            self.assertTrue((args[1].to_numpy() == 1.).all())
            self.assertTrue((args[2].to_numpy() == 2.).all())
            self.assertEqual(args[3:], (meta["dates"], meta["symbols"]))
            self.assertEqual(set(json.loads((obj.out/"CACHE.json").read_text())), set(r.CACHE_FILES))
            with np.load(obj.out/"DATA.npz", allow_pickle=False) as data:
                self.assertEqual(set(data.files), set(original)|set(r.EXTRA))
                for key, value in original.items():
                    self.assertEqual(data[key].tobytes(), value.tobytes())
            audit = json.loads((obj.out/"FEATURE_AUDIT.json").read_text())
            self.assertEqual(audit["price_cache_sha256"], r.sha(price_path))
            self.assertEqual(audit["feature_contract"], r.feature_contract())
            with self.assertRaises(ValueError):
                obj.prepare()

    def test_diagnostics_calls_nine_original_paths_only(self):
        backend = object.__new__(r.Backend)
        calls, written = [], []
        def replay(px, op, orders, **kwargs):
            calls.append(kwargs)
            return (dict(annual_returns=[dict(year=2016), dict(year=2026)]),
                    pd.Series([1., 1.], index=pd.to_datetime(["2020-01-01", "2020-01-02"])),
                    [], [])
        backend.local = SimpleNamespace(
            ex=SimpleNamespace(e=SimpleNamespace(engine=SimpleNamespace(replay=replay))),
            arch=SimpleNamespace(old=SimpleNamespace(prior=SimpleNamespace(
                old=SimpleNamespace(previous=SimpleNamespace(cuts=lambda eq, meta: {}))))))
        meta = dict(dates=["2020-01-01"], ends=["2020-01-02"])
        result = backend.diagnostics({name: [] for name in r.NAMES}, meta, None, None,
                                     lambda name, value: written.append(name))
        self.assertEqual(len(calls), 9)
        self.assertEqual(len(written), 27)
        self.assertEqual({(x["execution"], x["cost"]) for x in calls},
                         {("close", 0.), ("open", 0.), ("open", .0005)})
        self.assertEqual(set(result), set(r.NAMES))
        self.assertTrue(all(row["partial"] for values in result.values()
                            for path in values.values() for row in path["metrics"]["annual_returns"]))


if __name__ == "__main__":
    unittest.main()
