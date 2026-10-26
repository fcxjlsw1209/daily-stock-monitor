"""Synthetic-only endpoint scoring, contracts and phase guard tests."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
import monthly_endpoint155_runner as r

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = r.module('endpoint155_synthetic_adapter', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
POLICY = dict(budget=.05, ceiling=5, gap=.02)


def arrays(n=4, width=5):
    scores = np.tile(np.linspace(.2, .1, width), (n, 1))
    mean = np.tile(np.arange(width, dtype=float), (n, 1))
    loss = mean.copy()
    rank = np.tile(np.linspace(-.5, .5, width), (n, 1)) if width > 1 else np.zeros((n, width))
    a = dict(baseline=scores.copy(), core=.5*scores, core_weight=np.full(n, .5),
        eligible=np.ones((n,width), bool), effects=np.zeros((n,width)),
        state_calm10=np.ones(n, bool), state_weak_narrow=np.zeros(n, bool),
        mean_return_prediction=mean, loss_prediction=loss, mean_return_rank=rank,
        negative_loss_rank=-rank, common_available=np.ones((n,width), bool))
    m = dict(dates=[str(d.date()) for d in pd.date_range('2020-01-31', periods=n, freq='ME')],
             symbols=[['S'+str(j) for j in range(width)] for _ in range(n)])
    return a, m


def rerank(a):
    ranked = r.model.rank_predictions(a['mean_return_prediction'], a['loss_prediction'],
                                     a['common_available'], a['eligible'])
    for key in ('mean_return_rank', 'negative_loss_rank', 'common_available'):
        a[key] = ranked[key]


def missing(a, row, col=None):
    where = row if col is None else (row, col)
    a['common_available'][where] = False
    a['mean_return_prediction'][where] = np.nan
    a['loss_prediction'][where] = np.nan
    rerank(a)


class FakeBackend:
    adapter = ADAPTER

    def __init__(self):
        self.episodes = 0

    def pre(self, a, n):
        result = np.zeros(a['baseline'].shape)
        result[:n] = a['baseline'][:n]
        if result.shape[1] > 3:
            result[:n, 3] += .2  # Demonstrates full local pre, not an older scope.
        return result

    def selector(self, scores, good, symbols, effects, original, policy):
        self.episodes += 1
        choices = [sorted(np.flatnonzero(ok).tolist(), key=lambda j: (-sc[j], names[j]))[:3]
                   for sc, ok, names in zip(scores, good, symbols)]
        return choices, scores.copy()

    def baseline(self, a, m, n, policy):
        return ADAPTER.score_and_choose('incumbent66', 0., np.full(a['baseline'].shape, np.nan),
            a, m, n, pre_score=self.pre(a,n), effects=a['effects'],
            episode_policy=policy, boundary_selector=self.selector)

    @staticmethod
    def make_orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=[meta['symbols'][i][j] for j in row],
                     scores=[float(scores[i, j]) for j in row])
                for i, row in enumerate(choices)]


def trials():
    return [dict(name=name, training=dict(cagr=.2, sharpe=1., max_drawdown=-.2),
                 early=dict(cagr=.2), recent=dict(cagr=.2)) for name in r.NAMES]


def proposal():
    return dict(round=155, arms=list(r.NAMES), primary_challenger=r.NAMES[2],
        status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
        measurement=r.measurement_contract(), selection_contract=r.selection_contract(),
        design=dict(path=r.DESIGN_PATH, sha256=r.DESIGN_SHA),
        previous_completion=dict(path=r.PREVIOUS_COMPLETION, sha256=r.PREVIOUS_COMPLETION_SHA),
        actual_new_labels=0, actual_new_model_fits=0, actual_new_orders=0, actual_new_returns=0)


class Scoring(unittest.TestCase):
    def setUp(self):
        self.a, self.m = arrays()
        self.backend = FakeBackend()

    def test_centered_extremes_and_loss_sign(self):
        rank, _, _ = r.ranked_pair(self.a, 4)
        self.assertEqual(rank['return'][0,0], -.5)
        self.assertEqual(rank['return'][0,-1], .5)
        np.testing.assert_array_equal(rank['loss'], -rank['return'])

    def test_ties_zero(self):
        self.a['mean_return_prediction'][:] = 9.
        self.a['loss_prediction'][:] = -.1
        rerank(self.a)
        ranks, _, _ = r.ranked_pair(self.a, 4)
        self.assertTrue(all((x == 0.).all() for x in ranks.values()))

    def test_partial_ties_average(self):
        self.a['mean_return_prediction'][0] = [0,0,1,2,2]
        rerank(self.a)
        rank, _, _ = r.ranked_pair(self.a,1)
        np.testing.assert_array_equal(rank['return'][0], [-.375,-.375,0,.375,.375])

    def test_singleton_zero(self):
        a, _ = arrays(width=1)
        rank, ready, use = r.ranked_pair(a, 1)
        self.assertTrue(ready[0] and use[0,0])
        self.assertEqual(rank['return'][0,0], 0.)

    def test_missing_cell_zero_in_both_and_does_not_force_row_fallback(self):
        missing(self.a, 0, 0)
        rank, ready, use = r.ranked_pair(self.a,1)
        self.assertTrue(ready[0]); self.assertFalse(use[0,0])
        self.assertEqual(rank['return'][0,0], 0.)
        self.assertEqual(rank['loss'][0,0], 0.)
        self.assertEqual(rank['return'][0,1], -.5)

    def test_empty_month_exact_fallback(self):
        missing(self.a, 0)
        choices, scores, audit = r.choices_all(self.a,self.m,1,self.backend,POLICY)
        self.assertFalse(audit[0]['available'])
        for key in r.NAMES[1:]:
            self.assertEqual(choices[key], choices[r.NAMES[0]])
            self.assertEqual(scores[key].tobytes(),scores[r.NAMES[0]].tobytes())

    def test_first_partial_query_exact_fallback(self):
        self.m['dates'][0]='2016-08-17'; missing(self.a,0)
        _, scores, audit = r.choices_all(self.a,self.m,1,self.backend,POLICY)
        self.assertTrue(audit[0]['initial_partial_fallback'])
        self.assertEqual(scores[r.NAMES[0]].tobytes(),scores[r.NAMES[2]].tobytes())

    def test_first_partial_with_forecast_rejected(self):
        self.m['dates'][0]='2016-08-17'
        with self.assertRaises(ValueError):r.choices_all(self.a,self.m,1,self.backend,POLICY)

    def test_fixed_dose_full_local_pre(self):
        _, scores, _ = r.choices_all(self.a,self.m,1,self.backend,POLICY)
        for name, hand in zip(r.NAMES[1:],('mean_return_rank','negative_loss_rank')):
            np.testing.assert_allclose(scores[name],self.backend.pre(self.a,1)[:1]+.01*self.a[hand][:1],atol=1e-16)
            self.assertLessEqual(np.max(np.abs(scores[name]-scores[r.NAMES[0]])),.005+1e-16)

    def test_no_old_forecast_availability_gate(self):
        self.a['forecast_available']=np.zeros_like(self.a['eligible'])
        _, ready, use = r.ranked_pair(self.a,1)
        self.assertTrue(ready.all() and use.all())

    def test_rank_not_only_boundary_candidates(self):
        self.assertEqual(r.ranked_pair(self.a,1)[0]['return'][0,0],-.5)

    def test_same_populations_for_both_targets(self):
        self.a['loss_prediction'][0,0]=np.nan
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)

    def test_nonfinite_forecast_rejected(self):
        self.a['loss_prediction'][0,0]=np.inf
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)

    def test_forecast_outside_mask_rejected(self):
        self.a['common_available'][0,0]=False
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)

    def test_ineligible_forecast_rejected(self):
        self.a['eligible'][0,0]=False
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)

    def test_boolean_mask_required(self):
        self.a['common_available']=self.a['common_available'].astype(int)
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)

    def test_boolean_or_string_forecast_rejected(self):
        for bad in (True,'bad'):
            a=deepcopy(self.a); a['loss_prediction']=a['loss_prediction'].astype(object);a['loss_prediction'][0,0]=bad
            with self.assertRaises(ValueError):r.ranked_pair(a,1)

    def test_bad_cached_rank_rejected(self):
        self.a['mean_return_rank'][0,0]=-.49
        with self.assertRaises(ValueError):r.ranked_pair(self.a,1)

    def test_invalid_prefix_rejected(self):
        for n in (-1,5,True,1.5):
            with self.assertRaises(ValueError):r.ranked_pair(self.a,n)

    def test_future_poison_ignored(self):
        before=r.ranked_pair(self.a,1)
        for key in ('loss_prediction','negative_loss_rank'):
            self.a[key]=self.a[key].astype(object);self.a[key][1:]='future'
        after=r.ranked_pair(self.a,1)
        np.testing.assert_array_equal(before[0]['loss'],after[0]['loss'])

    def test_one_episode_call_per_arm(self):
        r.choices_all(self.a,self.m,2,self.backend,POLICY)
        self.assertEqual(self.backend.episodes,3)

    def test_inputs_unchanged_and_prefix(self):
        snap={k:v.tobytes() for k,v in self.a.items()}
        c,s,_=r.choices_all(self.a,self.m,4,self.backend,POLICY)
        for n in range(5):
            d,t,_=r.choices_all(self.a,self.m,n,self.backend,POLICY)
            for name in r.NAMES:
                self.assertEqual(c[name][:n],d[name]);self.assertEqual(s[name][:n].tobytes(),t[name].tobytes())
        self.assertEqual(snap,{k:v.tobytes() for k,v in self.a.items()})

    def test_all_123_synthetic_prefixes_and_fallback(self):
        a,m=arrays(122); before={k:v.tobytes() for k,v in a.items()}
        orders, scores, audits=r.panels(a,m,FakeBackend(),POLICY)
        self.assertEqual(sum(len(v) for v in orders.values()),366)
        self.assertEqual(len(audits),122)
        self.assertEqual(before,{k:v.tobytes() for k,v in a.items()})


class Selection(unittest.TestCase):
    def test_tie_rejected(self):self.assertFalse(r.select(trials(),trials())['admitted'])
    def test_both_controls_required(self):
        a,b=trials(),trials(); a[2]['training']['cagr']=.21;b[2]['training']['cagr']=.21
        self.assertTrue(r.select(a,b)['admitted'])
    def test_open_return_control_must_strictly_win(self):
        a=trials();a[2]['training']['cagr']=.3
        self.assertFalse(r.select(a,trials())['admitted'])
    def test_open_incumbent_noninferiority(self):
        a,b=trials(),trials();a[2]['training']['cagr']=.3;b[1]['training']['cagr']=.1
        self.assertTrue(r.select(a,b)['admitted'])
        b[2]['training']['cagr']=.199
        self.assertFalse(r.select(a,b)['admitted'])
    def test_risk_tolerances(self):
        for section,field,value in [('training','sharpe',.949),('training','max_drawdown',-.231),('early','cagr',.174),('recent','cagr',.174)]:
            a,b=trials(),trials();a[2]['training']['cagr']=.3;b[2]['training']['cagr']=.3;b[2][section][field]=value
            self.assertFalse(r.select(a,b)['admitted'])
    def test_arm_order_required(self):
        with self.assertRaises(ValueError):r.select(trials()[::-1],trials())
    def test_nonfinite_values_rejected(self):
        for val in (np.nan,np.inf,True,'0.3'):
            a=trials();a[2]['training']['cagr']=val
            self.assertFalse(r.select(a,trials())['admitted'])


class ForecastAndProposal(unittest.TestCase):
    def test_valid_forecasts(self):
        a,_=arrays();r.validate_forecasts({k:a[k] for k in r.EXTRA},a['baseline'].shape)
    def test_invalid_forecast_type_or_shape(self):
        for val in (np.zeros((1,1)),np.zeros((4,5),bool)):
            a,_=arrays();a['loss_prediction']=val
            with self.assertRaises(ValueError):r.validate_forecasts({k:a[k] for k in r.EXTRA},(4,5))
    def test_mask_rank_zero_required(self):
        a,_=arrays();missing(a,0);a['mean_return_rank'][0,0]=.1
        with self.assertRaises(ValueError):r.validate_forecasts({k:a[k] for k in r.EXTRA},(4,5))
    def test_valid_proposal(self):r.validate_proposal(proposal())
    def test_wrong_dose_rejected(self):
        p=proposal();p['measurement']['dose']=.02
        with self.assertRaises(ValueError):r.validate_proposal(p)
    def test_changed_design_rejected(self):
        p=proposal();p['design']['sha256']='0'*64
        with self.assertRaises(ValueError):r.validate_proposal(p)
    def test_changed_selection_rejected(self):
        p=proposal();p['selection_contract']['first_months']=122
        with self.assertRaises(ValueError):r.validate_proposal(p)
    def test_no_actual_calculation_before_proposal(self):
        for key in ('actual_new_labels','actual_new_model_fits','actual_new_orders','actual_new_returns'):
            for val in (True,1):
                p=proposal();p[key]=val
                with self.assertRaises(ValueError):r.validate_proposal(p)


class ReviewAndPhases(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'out';self.out.mkdir()
        self.runner=object.__new__(r.Runner);self.runner.root=self.root;self.runner.out=self.out
        code=self.root/'review.py';code.write_text('# synthetic reviewer\n')
        self.code=code;self.runner.pins={str(code):r.sha(code)}
        self.runner.plan={'independent_prepare_source_hashes':dict(self.runner.pins)}
        self.cache={}
        for name in (*r.CACHE_FILES,'PLAN.json','CACHE.json'):
            (self.out/name).write_text('{}');self.cache[str(self.out/name)]=r.sha(self.out/name)
        self.review=dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
            source_hashes=dict(self.runner.pins),artifact_hashes=dict(self.cache),
            plan_sha256=self.cache[str(self.out/'PLAN.json')],cache_sha256=self.cache[str(self.out/'CACHE.json')],
            orders_sha256=self.cache[str(self.out/'PRETRAIN_ORDERS.json')],**{k:True for k in r.CHECKS})
    def check(self):
        (self.out/'INDEPENDENT_PREPARE.json').write_text(json.dumps(self.review))
        with patch.object(self.runner,'cache',return_value={k:v for k,v in self.cache.items() if Path(k).name in r.CACHE_FILES}):
            self.runner.require_prepare_review()
    def test_complete_receipt(self):self.check()
    def test_missing_artifact_closure(self):
        self.review['artifact_hashes']={}
        with self.assertRaises(ValueError):self.check()
    def test_changed_reviewer(self):
        self.code.write_text('# changed')
        with self.assertRaises(ValueError):self.check()
    def test_each_check_required(self):
        for key in r.CHECKS:
            self.review[key]=False
            with self.assertRaises(ValueError):self.check()
            self.review[key]=True
    def test_wrong_plan_receipt(self):
        self.review['plan_sha256']='f'*64
        with self.assertRaises(ValueError):self.check()
    def test_producer_import_forbidden(self):
        self.review['producer_modules_imported']=True
        with self.assertRaises(ValueError):self.check()
    def test_extra_unbound_artifact(self):
        extra=self.root/'extra';extra.write_text('extra');self.review['artifact_hashes'][str(extra)]=r.sha(extra)
        with self.assertRaises(ValueError):self.check()
    def test_selection_guard_before_loading(self):
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_prepare_review',side_effect=ValueError('gate')),patch.object(self.runner,'loaded') as load:
            with self.assertRaises(ValueError):self.runner.select_first88()
            load.assert_not_called()
    def test_existing_selection_not_overwritten(self):
        (self.out/'FROZEN.json').write_text('{}')
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'require_prepare_review') as review:
            with self.assertRaises(ValueError):self.runner.select_first88()
            review.assert_not_called()
    def test_diagnostics_needs_frozen_before_loading(self):
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'loaded') as load:
            with self.assertRaises(FileNotFoundError):self.runner.diagnose_later34()
            load.assert_not_called()
    def test_existing_diagnostics_not_overwritten(self):
        (self.out/'RESULTS.json').write_text('{}')
        with patch.object(self.runner,'refresh'),patch.object(self.runner,'loaded') as load:
            with self.assertRaises(ValueError):self.runner.diagnose_later34()
            load.assert_not_called()
    def test_prepare_rejects_started_cache(self):
        with patch.object(self.runner,'refresh'):
            with self.assertRaises(ValueError):self.runner.prepare()
    def test_cache_changed_content_rejected(self):
        (self.out/'CACHE.json').write_text(json.dumps({name:r.sha(self.out/name) for name in r.CACHE_FILES}))
        (self.out/'DATA.npz').write_text('changed')
        with self.assertRaises(ValueError):self.runner.cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)
