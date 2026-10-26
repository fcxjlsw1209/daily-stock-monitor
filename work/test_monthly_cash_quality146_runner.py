import unittest
from types import SimpleNamespace
import numpy as np
import monthly_cash_quality146_runner as r


class Adapter:
    def __init__(self):
        self.calls = 0

    def score_and_choose(self, *args, **kwargs):
        self.calls += 1
        assert args[0] == "incumbent66" and args[1] == 0.
        assert kwargs["boundary_selector"] == "original"
        return [], kwargs["pre_score"][:args[5]].copy()


def fixture():
    adapter = Adapter()
    arrays = dict(baseline=np.zeros((2, 49)), eligible=np.ones((2, 49), bool),
                  effects=np.zeros((2, 49)))
    names = ["STOCK"+str(i) for i in range(32)] + ["ETF"+str(i) for i in range(17)]
    meta = dict(symbols=[names, names], etfs=names[32:])
    return arrays, meta, SimpleNamespace(adapter=adapter, selector="original")


def row(name, cagr=.4, sharpe=1.5, dd=-.25, early=.3, recent=.5):
    return dict(name=name, training=dict(cagr=cagr, sharpe=sharpe, max_drawdown=dd),
                early=dict(cagr=early), recent=dict(cagr=recent))


class RunnerTests(unittest.TestCase):
    def test_proposal_binds_controls_and_evaluation(self):
        p = dict(round=146, arms=list(r.NAMES), primary_challenger=r.NAMES[3],
                 dose=r.DOSE, concept=dict(path="work/after145_cashflow_experiment_concept.json",
                                          sha256=r.CONCEPT_SHA),
                 selection_contract=r.selection_contract())
        r.validate_proposal(p)
        for key, value in (("dose", .02), ("primary_challenger", r.NAMES[1]),
                           ("selection_contract", {}), ("concept", {})):
            bad = dict(p, **{key: value})
            with self.assertRaises(ValueError):
                r.validate_proposal(bad)

    def test_exact_continuous_adjustment_and_one_episode(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[0, :3] = [-.8, 0., .7]
        a["eligible"][0, 2] = False
        original = hand.copy()
        _, score = r.corrected(np.zeros((2, 49)), hand, a, m, 2, b, {})
        self.assertEqual(score[0, 0], -.008)
        self.assertEqual(score[0, 1], 0.)
        self.assertEqual(score[0, 2], 0.)
        self.assertEqual(b.adapter.calls, 1)
        np.testing.assert_array_equal(hand, original)

    def test_missing_exact_fallback(self):
        a, m, b = fixture()
        pre = np.arange(98, dtype=float).reshape(2, 49) / 100
        _, score = r.corrected(pre, np.full((2, 49), np.nan), a, m, 2, b, {})
        self.assertEqual(score.tobytes(), pre.tobytes())

    def test_zero_prefix(self):
        a, m, b = fixture()
        _, score = r.corrected(np.zeros((0, 49)), np.full((2, 49), np.nan), a, m, 0, b, {})
        self.assertEqual(score.shape, (0, 49))

    def test_infinity_or_unbounded_is_rejected(self):
        for value in (np.inf, -np.inf, 1.001, -1.001, 2.):
            a, m, b = fixture()
            h = np.full((2, 49), np.nan)
            h[0, 0] = value
            with self.assertRaises(ValueError):
                r.corrected(np.zeros((2, 49)), h, a, m, 2, b, {})

    def test_rounded_unit_endpoint_within_fixed_cap(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[0, :2] = [-1., 1.]
        _, score = r.corrected(np.zeros((2, 49)), hand, a, m, 2, b, {})
        self.assertEqual(list(score[0, :2]), [-.01, .01])

    def test_future_values_do_not_affect_prefix(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[1] = np.inf
        _, score = r.corrected(np.zeros((1, 49)), hand, a, m, 1, b, {})
        np.testing.assert_array_equal(score, np.zeros((1, 49)))

    def test_etf_feature_rejected(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[0, 32] = 0.
        with self.assertRaises(ValueError):
            r.corrected(np.zeros((2, 49)), hand, a, m, 2, b, {})

    def test_all_guards_required(self):
        close = [row(n, cagr=.4 + i*.01) for i, n in enumerate(r.NAMES)]
        opened = [row(n, cagr=.4 + i*.01) for i, n in enumerate(r.NAMES)]
        self.assertEqual(r.select(close, opened)["primary"], r.NAMES[3])
        opened[2] = row(r.NAMES[2], cagr=.5)
        self.assertEqual(r.select(close, opened)["primary"], r.NAMES[0])

    def test_no_portfolio_change_cannot_validate_mechanism(self):
        rows = [row(n) for n in r.NAMES]
        self.assertFalse(r.select(rows, rows)["admitted"])

    def test_risk_and_segment_regression_rejected(self):
        for field, value in (("sharpe", 1.44), ("dd", -.281),
                             ("early", .274), ("recent", .474)):
            rows = [row(n) for n in r.NAMES]
            rows[-1] = row(r.NAMES[-1], cagr=.9, **{field: value})
            self.assertFalse(r.select(rows, rows)["admitted"])

    def test_arm_order_guard(self):
        rows = [row(n) for n in reversed(r.NAMES)]
        with self.assertRaises(ValueError):
            r.select(rows, rows)


if __name__ == "__main__":
    unittest.main()
