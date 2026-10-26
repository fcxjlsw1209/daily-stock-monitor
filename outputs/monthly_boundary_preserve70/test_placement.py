"""Pure synthetic checks of the registered support-preserving boundary slot."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


P = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location('preserve70_test_placement', P/'placement.py')
placement = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(placement)
EPISODE_PATH = P.parent/'monthly_daily_episodes'/'episodes.py'
_NODE = next(node for node in ast.parse(EPISODE_PATH.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'select_boundary')
_NS = {'np': np}
exec(compile(ast.Module(body=[_NODE], type_ignores=[]), str(EPISODE_PATH), 'exec'), _NS)
real_selector = _NS['select_boundary']
POLICY = dict(id=7, family='level', model_id=2, budget=.05, ceiling=5, gap=.02)


def fixture():
    shape = (4, 7)
    baseline = np.tile([.24, .22, .20, .191, .181, .1805, .16], (4, 1))
    cw = np.array([.52, .50, .52, .50])[:, None]
    support = np.tile([.1, -.1, .05, 0., -.05, .1, -.2], (4, 1))
    calm = np.array([False, True, False, True])
    weak = np.array([False, False, True, True])
    residual = np.tile([.1, .2, -.1, 0., .3, -.2, .05], (4, 1))
    old_ml = np.tile([-.2, -.1, .1, .3, 0., -.3, .4], (4, 1))
    local_ml = np.tile([.1, -.2, .2, .1, -.3, 0., -.4], (4, 1))
    extension = np.ones(shape, bool); extension[1, 4] = False
    residual_weight = .025*calm[:, None]
    ml_weight = .05*weak[:, None]+.01*(calm & ~weak)[:, None]*extension
    pre = baseline+residual_weight*(residual-support)+ml_weight*(old_ml-support)
    pre += .25*ml_weight*(local_ml-old_ml)
    a = dict(baseline=baseline, core=baseline-(1-cw)*support, core_weight=cw,
             eligible=np.ones(shape, bool), state_calm10=calm, state_weak_narrow=weak,
             effects=np.tile([0., 0., -.2, .1, .05, .5, -.5], (4, 1)),
             extension_available=extension)
    factor = np.tile([-.5, -.3, -.1, .1, .3, .5, .2], (4, 1))
    return a, dict(symbols=[list('ABCDEFG') for _ in range(4)]), pre, factor, residual, old_ml, local_ml


def canonical(a, n):
    cw = np.asarray(a['core_weight'])
    if cw.ndim == 1: cw = cw[:, None]
    return dict(a, core_weight=np.broadcast_to(cw, a['baseline'].shape)[:n].copy())


def cohorts(pre, eligible, symbols):
    ranks, allowed = [], []
    for row, good, names in zip(pre, eligible, symbols):
        rank = sorted(np.flatnonzero(good).tolist(), key=lambda j: (-row[j], names[j]))
        ranks.append(rank)
        allowed.append([j for j in rank[2:5] if row[rank[2]]-row[j] <= .02] if len(rank) >= 3 else [])
    return ranks, allowed


def independent(pre, factor, a, meta, n):
    """Scalar min/max oracle; never calls either production placement helper."""
    original = canonical(a, n)
    support = (a['baseline'][:n]-a['core'][:n])/(1-original['core_weight'])
    score = pre[:n].copy()
    prior = .025*a['state_calm10'][:n]+.05*a['state_weak_narrow'][:n]
    episode = np.minimum(.05, np.maximum(0., .10-prior))
    ranks, groups = cohorts(pre[:n], a['eligible'][:n], meta['symbols'][:n])
    choices = []
    for i, (rank, group) in enumerate(zip(ranks, groups)):
        if len(rank) < 3:
            choices.append(rank if len(rank) == 2 else [])
            continue
        for j in group:
            s = support[i, j]
            e = min(.5, max(-.5, s+a['effects'][i, j]))
            score[i, j] += episode[i]*(e-s)
            if np.isfinite(factor[i, j]) and factor[i, j] != 0.:
                h = min(.5, max(-.5, s+factor[i, j]))
                delta = .025*(h-s)
                if delta != 0.: score[i, j] += delta
        winner = min(group, key=lambda j: (-score[i, j], meta['symbols'][i][j]))
        choices.append(sorted(rank[:2]+[winner], key=lambda j: (-score[i, j], meta['symbols'][i][j])))
    return choices, score


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.a, self.meta, self.pre, self.factor, self.residual, self.old_ml, self.local_ml = fixture()

    def choose(self, n=4, selector=real_selector, policy=POLICY):
        return placement.score_and_choose(self.pre, self.factor, self.a, self.meta, n,
                                         policy=policy, boundary_selector=selector)

    def original(self, n=4):
        return real_selector(self.pre[:n], self.a['eligible'][:n], self.meta['symbols'][:n],
                             self.a['effects'][:n], canonical(self.a, n), POLICY)

    def simple(self, values, support=0.):
        self.pre[:] = values
        self.a['baseline'][:] = values
        self.a['core'][:] = self.a['baseline']-(1-self.a['core_weight'])*support
        self.a['effects'][:] = 0.
        self.a['state_calm10'][:] = False
        self.a['state_weak_narrow'][:] = False

    def test_fixed_interface_no_mode_or_tunable_budget(self):
        self.assertEqual(placement.HAND_BUDGET, .025)
        self.assertEqual(placement.SOURCE_PLACEMENT_PATH.name, 'placement.py')
        for extra in (dict(mode='boundary'), dict(budget=.05)):
            with self.assertRaises(TypeError):
                placement.score_and_choose(self.pre, self.factor, self.a, self.meta, 4,
                    policy=POLICY, boundary_selector=real_selector, **extra)

    def test_scalar_oracle_actual_local_pre_and_original_support(self):
        expected = independent(self.pre, self.factor, self.a, self.meta, 4)
        got = self.choose()
        self.assertEqual(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
        self.assertFalse(np.array_equal(self.pre, self.a['baseline']))
        s = (self.a['baseline']-self.a['core'])/(1-self.a['core_weight'])
        wrong = (self.pre-self.a['core'])/(1-self.a['core_weight'])
        self.assertTrue(np.any(s != wrong))

    def test_zero_signed_zero_and_missing_are_byte_exact(self):
        before = self.original()
        for neutral in (0., -0., np.nan, np.inf, -np.inf):
            self.factor[:] = neutral
            got = self.choose()
            self.assertEqual(got[0], before[0])
            self.assertEqual(got[1].tobytes(), before[1].tobytes())

    def test_zero_bypasses_clip_at_tolerated_support_roundoff(self):
        support = np.tile([.5+5e-13, -.5-5e-13, .5+5e-13, -.5-5e-13, 0., .1, -.1], (4, 1))
        self.a['baseline'][:] = 0.
        self.a['core'][:] = -(1-self.a['core_weight'])*support
        self.factor[:] = 0.
        expected = self.original()
        got = self.choose()
        self.assertEqual(got[0], expected[0])
        self.assertEqual(got[1].tobytes(), expected[1].tobytes())
        self.factor[:] = -0.
        self.assertEqual(self.choose()[1].tobytes(), expected[1].tobytes())

    def test_partial_missing_zero_and_outside_cells_unchanged(self):
        before = self.original()
        self.factor[:, 2] = np.nan
        self.factor[:, 3] = 0.
        got = self.choose()
        _, groups = cohorts(self.pre, self.a['eligible'], self.meta['symbols'])
        support = (self.a['baseline']-self.a['core'])/(1-self.a['core_weight'])
        mask = np.zeros(self.pre.shape, bool)
        for i, group in enumerate(groups):
            for j in group:
                if np.isfinite(self.factor[i,j]) and self.factor[i,j] != 0.: mask[i,j] = True
        self.assertEqual(got[1][~mask].tobytes(), before[1][~mask].tobytes())
        delta = .025*(np.clip(support[mask]+self.factor[mask], -.5, .5)-support[mask])
        np.testing.assert_array_equal(got[1][mask], before[1][mask]+delta)
        for i, group in enumerate(groups): self.factor[i, group] = np.nan
        unchanged = self.choose()
        self.assertEqual(unchanged[0], before[0])
        self.assertEqual(unchanged[1].tobytes(), before[1].tobytes())

    def test_clipping_both_sides_and_nonbinding_additive_equivalence(self):
        self.simple([.4, .3, .2, .199, .198, .1, 0.], support=np.array([0.,0.,.49,-.49,0.,0.,0.]))
        self.factor[:] = [0.,0.,.5,-.5,.2,0.,0.]
        original = self.original()[1]
        got = self.choose()[1]
        support = (self.a['baseline']-self.a['core'])/(1-self.a['core_weight'])
        expected = np.zeros(self.pre.shape)
        expected[:,2] = .025*(.5-support[:,2])
        expected[:,3] = .025*(-.5-support[:,3])
        expected[:,4] = .025*self.factor[:,4]
        np.testing.assert_array_equal(got, original+expected)
        self.assertTrue((got[:,2]-original[:,2] > 0.).all())
        self.assertTrue((got[:,3]-original[:,3] < 0.).all())
        self.assertLessEqual(np.abs(got-original).max(), .0125+1e-15)

    def test_saturated_nonzero_factor_with_zero_delta_preserves_custom_original_order(self):
        self.simple([.4,.3,.2,.199,.198,.1,0.], support=.5)
        self.factor[:] = .5
        calls = []
        def selector(*args):
            ch, score = real_selector(*args)
            calls.append(1)
            return [list(reversed(row)) for row in ch], score
        expected = selector(self.pre, self.a['eligible'], self.meta['symbols'], self.a['effects'], canonical(self.a,4), POLICY)
        calls.clear()
        got = self.choose(selector=selector)
        self.assertEqual(calls, [1])
        self.assertEqual(got[0], expected[0])
        self.assertEqual(got[1].tobytes(), expected[1].tobytes())

    def test_full_budget_decomposition_sum_one_and_remaining355(self):
        support = (self.a['baseline']-self.a['core'])/(1-self.a['core_weight'])
        calm, weak = self.a['state_calm10'], self.a['state_weak_narrow']
        r = .025*calm[:,None]
        ml = .05*weak[:,None]+.01*(calm&~weak)[:,None]*self.a['extension_available']
        ranks, groups = cohorts(self.pre, self.a['eligible'], self.meta['symbols'])
        mask = np.zeros(self.pre.shape,bool)
        for i, group in enumerate(groups): mask[i,group] = True
        e = np.minimum(.05,np.maximum(0.,.10-.025*calm-.05*weak))[:,None]*mask
        h = .025*mask*np.isfinite(self.factor)
        remaining = 1-self.a['core_weight']-r-ml-e-h
        self.assertGreaterEqual(remaining.min(),.355-1e-15)
        self.assertLessEqual((r+ml+e).max(),.10+1e-15)
        np.testing.assert_allclose(self.a['core_weight']+r+ml+e+h+remaining,1.,rtol=0.,atol=1e-15)
        composite = np.clip(support+self.factor,-.5,.5)
        expected = (self.a['core']+remaining*support+r*self.residual
                    +.75*ml*self.old_ml+.25*ml*self.local_ml
                    +e*np.clip(support+self.a['effects'],-.5,.5)+h*composite)
        np.testing.assert_allclose(self.choose()[1],expected,rtol=0.,atol=1e-16)
        # Test the conservative lower bound at every allowed core/state pair.
        for cw in (.50,.52):
            for calm in (False,True):
                for weak in (False,True):
                    prior=.025*calm+.05*weak
                    episode=min(.05,max(0.,.10-prior))
                    scope=.01*(calm and not weak)
                    remainder=1-cw-prior-episode-scope-.025
                    self.assertGreaterEqual(remainder,.355-1e-15)
                    self.assertAlmostEqual(cw+prior+episode+scope+.025+remainder,1.)

    def test_one_episode_call_original_payload_and_no_effect_feedback(self):
        calls=[]
        def capture(score,eligible,symbols,effects,original,policy):
            calls.append(copy.deepcopy((score,eligible,symbols,effects,original,policy)))
            return real_selector(score,eligible,symbols,effects,original,policy)
        got=self.choose(selector=capture)
        self.assertEqual(len(calls),1)
        score,good,names,effects,original,policy=calls[0]
        np.testing.assert_array_equal(score,self.pre)
        for key in ('baseline','core','state_calm10','state_weak_narrow'):
            np.testing.assert_array_equal(original[key],self.a[key])
        np.testing.assert_array_equal(original['core_weight'],np.broadcast_to(self.a['core_weight'],self.pre.shape))
        np.testing.assert_array_equal(effects,self.a['effects'])
        np.testing.assert_array_equal(good,self.a['eligible'])
        self.assertEqual(names,self.meta['symbols']);self.assertEqual(policy,POLICY)
        np.testing.assert_array_equal(got[1],independent(self.pre,self.factor,self.a,self.meta,4)[1])

    def test_original_top_two_membership_locked_even_if_final_order_changes(self):
        self.simple([.21,.205,.20,.19,.18,.17,.16])
        self.factor[:] = [-.5,-.5,.5,.5,.5,.5,.5]
        got,score=self.choose()
        self.assertEqual(got,[[2,0,1]]*4)
        np.testing.assert_array_equal(score[:,:2],self.pre[:,:2])
        self.assertTrue(all({0,1}.issubset(ch) for ch in got))

    def test_no_global_rerank_or_expansion_beyond_original_cohort(self):
        self.simple([.30,.25,.20,.19,.181,.1805,.17])
        self.a['effects'][:,2:5]=-.5
        self.factor[:] = -.5
        got,score=self.choose()
        self.assertTrue((score[:,5]>score[:,2]).all())
        self.assertEqual(got,[[0,1,2]]*4)
        np.testing.assert_array_equal(score[:,5:],self.pre[:,5:])

    def test_exact_gap_no_epsilon_and_no_candidate_relocking(self):
        outside=-np.nextafter(.02,np.inf)
        self.simple([.1,.05,0.,-.02,outside,-.03,-.04])
        self.factor[:]=.5
        score=self.choose()[1]
        self.assertEqual(score[0,3],self.pre[0,3]+.0125)
        self.assertEqual(score[0,4],self.pre[0,4])
        self.assertEqual(score[0,5],self.pre[0,5])

    def test_symbol_ties_and_three_output_order(self):
        self.simple(np.full(7,.2))
        self.meta['symbols']=[list('ZAMBLCY') for _ in range(4)]
        self.factor[:]=.2
        got,score=self.choose()
        self.assertEqual(got,[[5,1,3]]*4)
        np.testing.assert_array_equal(score[:,[0,1,3,6]],self.pre[:,[0,1,3,6]])

    def test_small_eligible_sets_exact_fallback_and_no_new_eligibility(self):
        for count in range(4):
            self.a['eligible'][:]=False;self.a['eligible'][:,:count]=True
            old=self.original();got=self.choose()
            self.assertTrue(all(len(ch)==(count if count>=2 else 0) for ch in got[0]))
            if count<3:
                self.assertEqual(got[0],old[0]);self.assertEqual(got[1].tobytes(),old[1].tobytes())
            else:np.testing.assert_array_equal(got[1][:,:2],old[1][:,:2])

    def test_all_prefixes_empty_and_core_weight_shapes(self):
        column=self.a['core_weight'].copy();full=self.choose()
        for cw in (column,column[:,0],np.broadcast_to(column,self.pre.shape).copy()):
            self.a['core_weight']=cw
            for n in range(5):
                got=self.choose(n=n)
                self.assertEqual(got[0],full[0][:n]);self.assertEqual(got[1].shape,(n,7))
                self.assertEqual(got[1].tobytes(),full[1][:n].tobytes())

    def test_future_poison_not_converted_or_inspected(self):
        class Poison:
            def __float__(self):raise AssertionError('future numeric value inspected')
            def __str__(self):raise AssertionError('future ID inspected')
        expected=self.choose(n=1)
        for key in ('baseline','core','core_weight','effects'):
            value=self.a[key].astype(object);value[1:]=Poison();self.a[key]=value
        for key in ('state_calm10','state_weak_narrow','eligible'):
            self.a[key][1:]=~self.a[key][1:]
        self.pre=self.pre.astype(object);self.factor=self.factor.astype(object)
        self.pre[1:]=Poison();self.factor[1:]=Poison()
        self.meta['symbols'][1:]=[[Poison()]]*3
        got=self.choose(n=1)
        self.assertEqual(got[0],expected[0]);self.assertEqual(got[1].tobytes(),expected[1].tobytes())
        empty=self.choose(n=0)
        self.assertEqual(empty[0],[]);self.assertEqual(empty[1].shape,(0,7))

    def test_inputs_unchanged_when_selector_mutates_copies(self):
        def mutate(score,good,names,effects,original,policy):
            ch,result=real_selector(score,good,names,effects,original,policy)
            score[:]=-9.;good[:]=False;names[0][0]='MUTATED';effects[:]=-9.
            for value in original.values():value[:]=False if value.dtype.kind=='b' else -9.
            policy['budget']=1.
            return ch,result
        saved=copy.deepcopy((self.a,self.meta,self.pre,self.factor,POLICY));expected=self.choose()
        got=self.choose(selector=mutate)
        self.assertEqual(got[0],expected[0]);np.testing.assert_array_equal(got[1],expected[1])
        got[1][:]=-99.
        for key in self.a:np.testing.assert_array_equal(self.a[key],saved[0][key])
        self.assertEqual(self.meta,saved[1]);np.testing.assert_array_equal(self.pre,saved[2])
        np.testing.assert_array_equal(self.factor,saved[3]);self.assertEqual(POLICY,saved[4])

    def test_invalid_inputs_fail(self):
        for n in (-1,5,True,1.5):
            with self.assertRaises(ValueError):self.choose(n=n)
        for key,value in (('budget',.025),('ceiling',4),('gap',.021)):
            with self.assertRaises(ValueError):self.choose(policy=dict(POLICY,**{key:value}))
        with self.assertRaises(ValueError):self.choose(selector=None)
        changes=('factor_shape','factor_bound','weight_shape','weight_value','eligible_type',
                 'state_type','state_shape','pre_shape','pre_nonfinite','core_nonfinite',
                 'support_bound','effect_nonfinite','symbol_duplicate','symbol_type','symbol_width')
        for kind in changes:
            self.setUp()
            if kind=='factor_shape':self.factor=self.factor[:3]
            elif kind=='factor_bound':self.factor[0,0]=.500001
            elif kind=='weight_shape':self.a['core_weight']=np.ones(7)
            elif kind=='weight_value':self.a['core_weight'][0]=.51
            elif kind=='eligible_type':self.a['eligible']=self.a['eligible'].astype(float)
            elif kind=='state_type':self.a['state_calm10']=self.a['state_calm10'].astype(float)
            elif kind=='state_shape':self.a['state_weak_narrow']=self.a['state_weak_narrow'][:,None]
            elif kind=='pre_shape':self.pre=self.pre[:3]
            elif kind=='pre_nonfinite':self.pre[0,0]=np.nan
            elif kind=='core_nonfinite':self.a['core'][0,0]=np.inf
            elif kind=='support_bound':self.a['core'][0,0]=-1.
            elif kind=='effect_nonfinite':self.a['effects'][0,0]=np.nan
            elif kind=='symbol_duplicate':self.meta['symbols'][0][0]='B'
            elif kind=='symbol_type':self.meta['symbols'][0][0]=None
            else:self.meta['symbols'][0]=['A']
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.choose(n=1)


if __name__=='__main__':unittest.main()
