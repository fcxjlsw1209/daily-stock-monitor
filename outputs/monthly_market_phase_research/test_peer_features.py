import copy
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import peer_features as f


class PeerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2018-01-01", periods=380)
        self.day = self.dates[300]
        rng = np.random.default_rng(45)
        shared = rng.normal(.0004, .009, len(self.dates))
        self.path = 100 * np.exp(np.cumsum(shared))
        names = ["TARGET", "SELF_OTHER_CLASS"] + ["P%02d" % i for i in range(12)] + ["NEGATIVE"]
        prices = {s: self.path.copy() for s in names}
        prices["NEGATIVE"] = 100 * np.cumprod(1 - (np.r_[0., self.path[1:] / self.path[:-1] - 1.]))
        adj = pd.DataFrame(prices, index=self.dates)
        issuers = {s: "issuer_" + s for s in names}
        issuers["SELF_OTHER_CLASS"] = issuers["TARGET"]
        self.peers = dict(adj=adj, close=adj.copy(), eligible=pd.DataFrame(True, index=self.dates, columns=names),
                          issuers=issuers, aliases=[], source_hashes={})
        self.panel = dict(adj=adj[["TARGET"]].copy(), close=adj[["TARGET"]].copy())

    def build(self, panel=None, peers=None, symbols=None, etfs=()):
        return f.build_features(self.panel if panel is None else panel,
                                [(self.day, ["TARGET"] if symbols is None else symbols)], etfs,
                                peer_panel=self.peers if peers is None else peers)

    def test_self_issuer_excluded_unique_issuers_and_lexical_ties(self):
        self.peers["issuers"]["P01"] = self.peers["issuers"]["P00"]
        raw, audit = self.build()
        self.assertTrue(raw["peer_valid"][0, 0])
        chosen = audit["targets_by_event"][0]["targets"][0]["selected_peers"]
        names = [r["security_id"] for r in chosen]
        self.assertEqual(names, ["P00"] + ["P%02d" % i for i in range(2, 11)])
        self.assertEqual(len(set(r["issuer_id"] for r in chosen)), 10)
        self.assertNotIn("TARGET", names)
        self.assertNotIn("SELF_OTHER_CLASS", names)
        self.assertNotIn("NEGATIVE", names)

    def test_future_prices_and_future_members_cannot_affect_feature(self):
        original, audit = self.build()
        altered = copy.deepcopy(self.peers)
        altered["adj"].loc[self.dates[301]:] *= 100
        altered["close"].loc[self.dates[301]:] *= 100
        altered["eligible"].loc[self.dates[301]:] = False
        panel = {k: v.copy() for k, v in self.panel.items()}
        for value in panel.values():
            value.loc[self.dates[301]:] *= .01
        changed, changed_audit = self.build(panel, altered)
        prefix = {**self.peers, **{k: self.peers[k].loc[:self.day].copy() for k in ["adj", "close", "eligible"]}}
        truncated, trunc_audit = self.build({k: v.loc[:self.day] for k, v in self.panel.items()}, prefix)
        for key in original:
            np.testing.assert_array_equal(original[key], changed[key])
            np.testing.assert_array_equal(original[key], truncated[key])
        self.assertEqual(audit["targets_by_event"], changed_audit["targets_by_event"])
        self.assertEqual(audit["targets_by_event"], trunc_audit["targets_by_event"])

    def test_contemporary_membership_not_future_or_past_membership(self):
        self.peers["eligible"].loc[:self.day, "P00"] = False
        raw, audit = self.build()
        names = [r["security_id"] for r in audit["targets_by_event"][0]["targets"][0]["selected_peers"]]
        self.assertTrue(raw["peer_valid"][0, 0])
        self.assertNotIn("P00", names)
        self.peers["eligible"].loc[:self.dates[299], "P01"] = False
        raw2, audit2 = self.build()
        names2 = [r["security_id"] for r in audit2["targets_by_event"][0]["targets"][0]["selected_peers"]]
        self.assertIn("P01", names2)
        self.assertTrue(raw2["peer_valid"][0, 0])

    def test_min_five_positive_unique_peers_no_negative_fill(self):
        self.peers["eligible"].loc[:, ["P%02d" % i for i in range(4, 12)]] = False
        raw, audit = self.build()
        self.assertFalse(raw["peer_valid"][0, 0])
        self.assertEqual(raw["peer_participation"][0, 0], 0.)
        self.assertEqual(raw["peer_broadening"][0, 0], 0.)
        self.assertEqual(audit["missing_or_neutral_reasons"], {"fewer_than_five_positive_correlation_issuers": 1})

    def test_broadening_arithmetic_uses_same_current_peer_list(self):
        rates = np.where(np.arange(len(self.dates)) < 240, -.003, .02)
        rates += .0001 * np.sin(np.arange(len(self.dates)))
        path = 100 * np.exp(np.cumsum(rates))
        for panel in (self.panel, self.peers):
            for key in ("adj", "close"):
                panel[key].loc[:, :] = path[:, None]
        raw, audit = self.build()
        self.assertEqual(raw["peer_participation"][0, 0], .5)
        self.assertEqual(raw["peer_broadening"][0, 0], .5)
        target = audit["targets_by_event"][0]["targets"][0]
        self.assertEqual(target["above_MA200_63_sessions_ago"], 0.)
        self.assertEqual(target["above_MA200_now"], 1.)
        # Earlier membership is irrelevant: historical breadth is explicitly
        # conditional on peers selected from the current historical signal pool.
        self.peers["eligible"].loc[:self.dates[299]] = False
        changed, _ = self.build()
        for key in raw:
            np.testing.assert_array_equal(raw[key], changed[key])

    def test_missing_old_MA_price_or_nonpositive_raw_price_disqualifies_peer(self):
        self.peers["eligible"].loc[:, ["P%02d" % i for i in range(6, 12)]] = False
        self.peers["adj"].at[self.dates[300-262], "P00"] = np.nan
        raw, audit = self.build()
        self.assertTrue(raw["peer_valid"][0, 0])
        names = [r["security_id"] for r in audit["targets_by_event"][0]["targets"][0]["selected_peers"]]
        self.assertNotIn("P00", names)
        self.peers["close"].at[self.day, "P01"] = 0.
        raw2, _ = self.build()
        self.assertFalse(raw2["peer_valid"][0, 0])

    def test_target_missing_unmapped_and_ETFs_are_neutral(self):
        panel = {k: pd.concat([v, v.rename(columns={"TARGET": "UNKNOWN"}),
                               v.rename(columns={"TARGET": "QQQ"})], axis=1) for k, v in self.panel.items()}
        raw, audit = self.build(panel=panel, symbols=["UNKNOWN", "QQQ", "TARGET"], etfs={"TARGET"})
        self.assertFalse(raw["peer_valid"].any())
        self.assertEqual(audit["missing_or_neutral_reasons"], {"unmapped_or_ambiguous_target_issuer": 1, "ETF_target_neutral": 2})
        self.panel["adj"].at[self.day, "TARGET"] = np.nan
        missing, _ = self.build()
        self.assertFalse(missing["peer_valid"].any())

    def test_verified_dated_alias_only_and_canonical_ID(self):
        self.peers["issuers"]["research_equity_TARGET"] = self.peers["issuers"]["TARGET"]
        self.peers["aliases"] = [dict(symbol="OLD", security_id="research_equity_TARGET", issuer_id="issuer_TARGET",
                                      effective_from="2010-01-01", effective_to="2020-01-01",
                                      retrospective_identity_link_verified=True)]
        panel = {k: v.rename(columns={"TARGET": "OLD"}) for k, v in self.panel.items()}
        raw, audit = self.build(panel=panel, symbols=["OLD"])
        self.assertTrue(raw["peer_valid"].any())
        chosen = audit["targets_by_event"][0]["targets"][0]["selected_peers"]
        self.assertTrue(all(r["issuer_id"] != "issuer_TARGET" for r in chosen))
        self.peers["aliases"][0]["effective_to"] = "2018-01-02"
        unmapped, _ = self.build(panel=panel, symbols=["OLD"])
        self.assertFalse(unmapped["peer_valid"].any())

    def test_input_invariance(self):
        before, before_panel = copy.deepcopy(self.peers), copy.deepcopy(self.panel)
        self.build()
        for key in ("adj", "close", "eligible"):
            assert_frame_equal(before[key], self.peers[key])
        for key in self.panel:
            assert_frame_equal(before_panel[key], self.panel[key])
        self.assertEqual(before["issuers"], self.peers["issuers"])

    def test_exact_dates_alignment_and_boolean_membership(self):
        with self.assertRaises(ValueError):
            f.build_features(self.panel, [(self.day + pd.Timedelta(hours=1), ["TARGET"])], (), peer_panel=self.peers)
        self.peers["eligible"] = self.peers["eligible"].astype(float)
        with self.assertRaises(ValueError):
            self.build()

    def test_constant_target_and_insufficient_warmup(self):
        self.panel["adj"].loc[:, :] = 100.
        raw, _ = self.build()
        self.assertFalse(raw["peer_valid"].any())
        early, _ = f.build_features(self.panel, [(self.dates[250], ["TARGET"])], (), peer_panel=self.peers)
        self.assertFalse(early["peer_valid"].any())

    def test_market_controls_include_negative_correlation_without_top10(self):
        rates = np.where(np.arange(len(self.dates)) < 240, -.003, .02)
        rates += .0001 * np.sin(np.arange(len(self.dates)))
        path = 100 * np.exp(np.cumsum(rates))
        negative = 100 * np.cumprod(1 - np.r_[0., path[1:] / path[:-1] - 1.])
        for panel in (self.panel, self.peers):
            for key in ("adj", "close"):
                panel[key].loc[:, :] = path[:, None]
        for key in ("adj", "close"):
            self.peers[key]["NEGATIVE"] = negative
        raw, audit = self.build()
        self.assertTrue(raw["peer_valid"][0, 0])
        self.assertEqual(raw["peer_participation"][0, 0], .5)
        self.assertEqual(raw["peer_broadening"][0, 0], .5)
        self.assertAlmostEqual(raw["market_participation"][0, 0], 12 / 13 - .5)
        self.assertAlmostEqual(raw["market_broadening"][0, 0], (12 / 13 - 1 / 13) / 2)
        rec = audit["targets_by_event"][0]["targets"][0]
        self.assertEqual(rec["market_distinct_other_issuers"], 13)
        market = rec["market_representatives"]
        self.assertIn("NEGATIVE", [r["security_id"] for r in market])
        self.assertTrue(all(r["issuer_id"] != "issuer_TARGET" for r in market))
        self.assertEqual(len({r["issuer_id"] for r in market}), len(market))

    def test_market_controls_exactly_share_peer_valid_mask_and_deduplicate(self):
        self.peers["issuers"]["P01"] = self.peers["issuers"]["P00"]
        raw, audit = self.build()
        rec = audit["targets_by_event"][0]["targets"][0]
        self.assertEqual(rec["market_distinct_other_issuers"], 12)
        names = [r["security_id"] for r in rec["market_representatives"]]
        self.assertIn("P00", names)
        self.assertNotIn("P01", names)
        # There is still valid broad-market evidence when only four positive
        # peers remain, but the matched controls must also become exactly zero.
        self.peers["eligible"].loc[:, ["P%02d" % i for i in range(5, 12)]] = False
        invalid, _ = self.build()
        self.assertFalse(invalid["peer_valid"][0, 0])
        for key in f.FEATURE_NAMES:
            self.assertEqual(invalid[key][0, 0], 0.)


if __name__ == "__main__":
    unittest.main()
