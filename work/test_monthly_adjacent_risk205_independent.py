"""Only invented arrays, prices and source calendars; no actual research data."""
import copy
import hashlib
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import monthly_adjacent_risk205_independent as independent


def fixture(count=3, width=4):
    dates = pd.bdate_range('2014-01-01', periods=1000)
    t = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({'A': 10*np.exp(.0005*t+.06*np.sin(t/17.)),
        'B': 20*np.exp(.0002*t+.025*np.cos(t/29.)),
        'FUND': 15*np.exp(.0001*t+.07*np.sin(t/43.)), 'D': np.full(len(t), 12.)}, index=dates)
    names = ['A', 'B', 'FUND', 'D'][:width]
    query = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'adj': frame}, dict(dates=query, symbols=[names[:] for _ in query], etfs=['FUND']), dates.strftime('%Y-%m-%d').tolist()


def exact_path(returns=None):
    if returns is None:
        returns = [-.5]*31 + [0.] + [1.]*30 + [3.]
    p = np.full(295, 10.)
    p[231] = 1.
    for k, r in enumerate(returns, 232):
        p[k] = p[k-1]*(1+r)
    p[0] = p[252]/2.
    return p


def original_fixture(rows=3, width=4, shape='column'):
    calm = np.array([i % 3 == 2 for i in range(rows)], bool)
    cw = np.where(calm, .5, .52)
    ranks = (np.arange(width)+1)/max(1,width)-.5
    core = np.tile(ranks, (rows,1))*cw[:,None]
    if shape == 'matrix':
        cw = np.broadcast_to(cw[:,None], (rows,width)).copy()
    elif shape == 'column':
        cw = cw[:,None]
    return dict(core=core, core_weight=cw, state_calm10=calm,
                eligible=np.zeros((rows,width),bool), unused=object())


def targets(values, original, full=None):
    n, width = values['available'].shape
    result = {}
    for arm, key in independent.RAW.items():
        target = np.full((n if full is None else full,width), np.nan)
        target[:n] = values[key]
        target[:n][original['state_calm10'][:n]] = np.nan
        if n:
            target[0] = np.nan
        result[arm] = target
    return result


class Tests(unittest.TestCase):
    def test_exact_handworked_uncentered_products_and_two_risks(self):
        a = independent.measure_oracle(exact_path())
        self.assertTrue(a['available'])
        self.assertEqual(a['numerator'], 1.)
        self.assertEqual(a['adjacent_sum'], 39.5)
        self.assertEqual(a['squared_sum'], 46.75)
        self.assertEqual(len(a['absolute_adjacent_products']), 62)
        self.assertEqual(len(a['squared_returns']), 63)
        dp = math.sqrt((math.pi/2.)*39.5/62.)
        dr = math.sqrt(46.75/63.)
        self.assertEqual(a['candidate_denominator'], dp)
        self.assertEqual(a['control_denominator'], dr)
        self.assertEqual(a['raw_candidate'], 1./dp)
        self.assertEqual(a['raw_control'], 1./dr)

    def test_isolated_moves_zero_pair_risk_is_shared_missing_no_wrap(self):
        for returns in ([1.]+[0.]*61+[1.], [0.]*31+[1.]+[0.]*31):
            a = independent.measure_oracle(exact_path(returns))
            self.assertTrue(a['source_available'])
            self.assertFalse(a['available'])
            self.assertEqual(a['reason'], 'zero_risk')
            self.assertEqual(a['candidate_denominator'], 0.)
            self.assertGreater(a['control_denominator'], 0.)
            self.assertIsNone(a['raw_candidate'])
            self.assertIsNone(a['raw_control'])
            self.assertEqual(a['risk_estimates'], 2)
        a = independent.measure_oracle(np.ones(295))
        self.assertEqual(a['control_denominator'], 0.)
        self.assertFalse(a['available'])

    def test_constant_nonzero_return_not_demeaned(self):
        for r in (1., -.5):
            a = independent.measure_oracle(exact_path([r]*63))
            self.assertTrue(a['available'])
            self.assertEqual(a['control_denominator'], abs(r))
            self.assertEqual(a['adjacent_sum'], 62*r*r)
            self.assertEqual(a['candidate_denominator'], math.sqrt((math.pi/2.)*(62*r*r)/62.))

    def test_same_multiset_rms_invariant_adjacent_changes_and_sign_symmetry(self):
        clustered = [1.]*31+[0.]*32
        separated = [1.,0.]*31+[0.]
        a, b = [independent.risk_oracle(r) for r in (clustered, separated)]
        self.assertEqual(a['control_denominator'], b['control_denominator'])
        self.assertGreater(a['candidate_denominator'], b['candidate_denominator'])
        plus = independent.risk_oracle([.25,.5,0.]*21)
        minus = independent.risk_oracle([-.25,-.5,0.]*21)
        for key in ('adjacent_sum','squared_sum','candidate_denominator','control_denominator'):
            self.assertEqual(plus[key],minus[key])

    def test_no_positive_floor_zero_numerator_and_current_clock(self):
        p = exact_path([2.**-40,0.,-2.**-40]*21)
        a = independent.measure_oracle(p)
        self.assertTrue(a['available'])
        self.assertGreater(a['candidate_denominator'], 0.)
        self.assertLess(a['candidate_denominator'], 1e-12)
        p = exact_path(); p[0] = p[252]
        a = independent.measure_oracle(p)
        self.assertTrue(a['available'])
        self.assertEqual(a['raw_candidate'], 0.)
        self.assertEqual(a['raw_control'], 0.)
        p[-1] *= 2
        b = independent.measure_oracle(p)
        self.assertEqual(a['numerator'],b['numerator'])
        self.assertNotEqual(a['candidate_denominator'],b['candidate_denominator'])
        tiny = independent.risk_oracle([1e-200]*63)
        self.assertEqual(tiny['candidate_denominator'],0.)
        self.assertEqual(tiny['control_denominator'],0.)

    def test_scalar_scale_invariance_no_mutation_and_signed_numerator(self):
        p = exact_path(); before = p.copy(); a = independent.measure_oracle(p)
        for scale in (2.**-500,2.**500):
            b = independent.measure_oracle(p*scale)
            price_keys = {'start_price','formation_end_price','risk_start_price','end_price'}
            independent.record_check({k:v for k,v in a.items() if k not in price_keys},
                                     {k:v for k,v in b.items() if k not in price_keys})
            for key in price_keys:self.assertEqual(b[key],a[key]*scale)
        np.testing.assert_array_equal(p,before)
        p[0] = p[252]*2
        b = independent.measure_oracle(p)
        self.assertEqual(b['numerator'],-.5)
        self.assertLess(b['raw_candidate'],0.)
        self.assertLess(b['raw_control'],0.)

    def test_all295_complete_not_endpoints_only(self):
        for index in (0,1,150,230,231,252,293,294):
            for value in (0.,-1.,np.nan,np.inf):
                p=exact_path();p[index]=value
                a=independent.measure_oracle(p)
                self.assertFalse(a['source_available'])
                self.assertEqual(a['reason'],'invalid_adjusted_price')
        for value in (True,None,'2',complex(2,0)):
            p=exact_path().astype(object);p[200]=value
            self.assertEqual(independent.measure_oracle(p)['reason'],'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError):independent.measure_oracle(np.ones(294))

    def test_valid_price_arithmetic_overflow_hard_failure(self):
        p=exact_path();p[0]=5e-324;p[252]=1e308
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=exact_path();p[293]=5e-324;p[294]=1e308
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=exact_path();p[293]=1.;p[294]=1e200
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=exact_path([2.**-40]*63);p[0]=p[252]/1e308
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        for returns in ([1e200]*63, [1e154]*63, [1.4e153]*63):
            with self.assertRaises(FloatingPointError):independent.risk_oracle(returns)

    def test_original_pct_ties_empty_singleton_and_no_new_eligibility(self):
        for raw in ([1.,1.,1.],[2.],[np.nan,np.nan],[1.,1.,3.,np.nan],[-1.,0.,1.]):
            x=np.asarray([raw]);mask=np.isfinite(x)
            rank=independent.original_pct_ranks(x,mask)
            wanted=pd.Series(raw).rank(pct=True).to_numpy()-.5
            np.testing.assert_array_equal(rank[0],wanted)
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        self.assertTrue(v['available'][:,2].all()) # ETF included.
        self.assertFalse(v['available'][:,3].any()) # Genuine zero risk.

    def test_full_audit_windows_counts(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        self.assertEqual(a['source_available_by_event'],[4]*3)
        self.assertEqual(a['available_by_event'],[3]*3)
        self.assertEqual(a['valid_price_observations'],295*12)
        self.assertEqual(a['simple_return_observations'],63*12)
        self.assertEqual(a['risk_estimates'],24)
        self.assertEqual(a['adjacent_product_observations'],62*12)
        self.assertEqual(a['squared_return_observations'],63*12)
        self.assertEqual(a['ratio_observations'],18)
        self.assertEqual(a['ranked_score_observations'],18)
        for event in a['event_history']:
            t=c.index(event['date'])
            self.assertEqual(event['price_start'],c[t-294])
            self.assertEqual(event['price_end'],c[t])
            self.assertEqual(event['numerator_end'],c[t-42])
        r=independent.verify_features(p,m,c,v,a)
        self.assertTrue(all(r[key] is True for key in independent.FEATURE_FLAGS))
        self.assertFalse(r['computes_returns'])

    def test_prefix_future_numeric_isolation(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for n in range(4):
            frame=p['adj'].astype(object).copy()
            cutoff=pd.Timestamp.min if n==0 else pd.Timestamp(m['dates'][n-1])
            frame.loc[frame.index>cutoff,:]='FUTURE'
            z,b=independent.independent_features({'adj':frame},m,c,n)
            for key in v:np.testing.assert_array_equal(z[key],v[key][:n])
            self.assertEqual(b['event_history'],a['event_history'][:n])

    def test_missing_official_sessions_symbols_history_monthend(self):
        p,m,c=fixture();missing=pd.Timestamp('2016-07-05')
        v,a=independent.independent_features({'adj':p['adj'].drop(missing)},m,c)
        self.assertFalse(v['available'].any())
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_official_session')
        v,a=independent.independent_features({'adj':p['adj'].drop(columns='A')},m,c)
        self.assertFalse(v['available'][:,0].any())
        young=dict(dates=['2014-01-31'],symbols=[['A']],etfs=[])
        v,a=independent.independent_features(p,young,c)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        bad=copy.deepcopy(m);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)

    def test_corrupt_feature_audit_and_raw_rank_rejected(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for key in ('raw_candidate','raw_control','candidate','control'):
            bad=copy.deepcopy(v);bad[key][1,0]=np.nextafter(bad[key][1,0],np.inf)
            with self.assertRaises(ValueError):independent.verify_features(p,m,c,bad,a)
        bad=copy.deepcopy(a);bad['event_history'][0]['cells'][0]['candidate_denominator']*=2
        with self.assertRaises(ValueError):independent.verify_features(p,m,c,v,bad)

    def test_integration_half_noncalm_core_and_exact_calm_initial(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c)
        original=original_fixture();target=targets(v,original)
        delta=independent.independent_deltas(v,original,target)
        receipt=independent.verify_deltas(v,original,target,delta)
        self.assertTrue(receipt['calm_rows_unchanged'])
        for arm,key in independent.RAW.items():
            np.testing.assert_array_equal(delta[arm][0],np.zeros(4))
            np.testing.assert_array_equal(delta[arm][2],np.zeros(4))
            self.assertEqual(delta[arm][1,3],0.)
            expected=.5*(.52*float(v[key][1,0])-float(original['core'][1,0]))
            self.assertEqual(delta[arm][1,0],expected)
        bad=copy.deepcopy(target);bad[independent.ARMS[1]][2,0]=0.
        with self.assertRaises(ValueError):independent.verify_deltas(v,original,bad,delta)

    def test_delta_prefix_weights_nan_suffix_and_array_immutability(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c)
        for shape in ('column','matrix','vector'):
            original=original_fixture(shape=shape)
            saved={k:x.copy() for k,x in original.items() if isinstance(x,np.ndarray)}
            for n in range(4):
                vals={k:x[:n].copy() for k,x in v.items()}
                tar=targets(vals,original,full=3)
                d=independent.independent_deltas(vals,original,tar)
                independent.verify_deltas(vals,original,tar,d)
            for key,x in saved.items():np.testing.assert_array_equal(original[key],x)
        vv={k:x[:1].copy() for k,x in v.items()}
        original=original_fixture();original['core']=original['core'].astype(object)
        original['core'][1:]='FUTURE';original['core_weight']=original['core_weight'].astype(object)
        original['core_weight'][1:]='FUTURE'
        tar=targets(vv,original);d=independent.independent_deltas(vv,original,tar)
        independent.verify_deltas(vv,original,tar,d)

    def test_bad_delta_and_weights_fail(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c)
        a=original_fixture();tar=targets(v,a);d=independent.independent_deltas(v,a,tar)
        for mutate in ('delta','target','weights','blend','boolean_core'):
            aa,tt,dd=copy.deepcopy(a),copy.deepcopy(tar),copy.deepcopy(d)
            if mutate=='delta':dd[independent.ARMS[1]][1,0]+=1e-12
            if mutate=='target':tt[independent.ARMS[1]][1,0]=np.inf
            if mutate=='weights':aa['core_weight'][1,0]=.5
            if mutate=='boolean_core':
                aa['core']=aa['core'].astype(object);aa['core'][1,0]=False
            with self.assertRaises(ValueError):
                independent.verify_deltas(v,aa,tt,dd,blend=.25 if mutate=='blend' else .5)

    def test_pinned_accounting_only_no_producer_import(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(),independent.FROZEN_HELPER_SHA)
        self.assertTrue(callable(independent.accounting().audit_path))
        self.assertNotIn('import monthly_adjacent_risk205_kernel',Path(independent.__file__).read_text())

    def test_both_producer_and_independent_fixture_fullaudit_allprefix(self):
        import monthly_adjacent_risk205_kernel as production
        from test_monthly_adjacent_risk205_kernel import fixture as producer_fixture
        for make in (fixture,producer_fixture):
            p,m,c=make()
            for n in range(len(m['dates'])+1):
                stop=pd.Timestamp.min if n==0 else pd.Timestamp(m['dates'][n-1])
                small={'adj':p['adj'].loc[:stop].copy()}
                v,a=production.build_features(small,m,c,n)
                independent.verify_features(small,m,c,v,a,n)


if __name__=='__main__':
    unittest.main()
