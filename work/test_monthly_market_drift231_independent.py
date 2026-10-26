"""Invented data only: independent drift, causal market states and full-E hook."""
import copy
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_market_drift231_independent as oracle


def fixture():
    idx = pd.bdate_range('2014-01-01', periods=1100)
    x = np.arange(len(idx), dtype=float)
    px = pd.DataFrame(dict(SPY=np.exp(.00015*x+.18*np.sin(x/47.)),
        A=np.exp(.0005*x+.05*np.sin(x/19.)), B=np.exp(-.0002*x+.04*np.cos(x/31.)),
        C=np.exp(.0003*x+.08*np.cos(x/39.)), D=np.full(len(x), 10.), E=np.exp(.0001*x)), index=idx)
    ds = ['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31']
    meta = dict(dates=ds, ends=['2016-08-31', '2016-09-30', '2016-10-31', '2016-11-30'],
        symbols=[['A', 'B', 'C', 'D', 'E'] for _ in ds], etfs=['E'])
    a = dict(eligible=np.tile([True, True, True, True, False], (4, 1)),
        state_calm10=np.array([False, True, False, True], bool))
    return {'adj': px}, meta, idx.strftime('%Y-%m-%d').tolist(), a


def physical_prefix(args, n):
    p, m, c, a = copy.deepcopy(args)
    for key in ('dates', 'ends', 'symbols'): m[key] = m[key][:n]
    p['adj'] = p['adj'].loc[:m['dates'][-1]].copy() if n else p['adj'].iloc[:0].copy()
    a = {k: v[:n].copy() for k, v in a.items()}
    return p, m, c, a


def single(matches=42, calm=False):
    p, m, c, a = fixture()
    for key in ('dates', 'ends', 'symbols'): m[key] = m[key][-1:]
    a = dict(eligible=a['eligible'][-1:].copy(), state_calm10=np.array([calm], bool))
    t = c.index(m['dates'][0]); u = t-42
    p['adj']['SPY'] = 1.
    p['adj'].iloc[u-matches:, p['adj'].columns.get_loc('SPY')] = 2.
    return p, m, c, a


def original(args):
    from test_monthly_joint_error226_independent import original as old_original
    _, m, _, a = args
    n, w = a['eligible'].shape
    pre = np.tile(np.linspace(.025, .005, w), (n, 1))
    o = old_original(pre, a['eligible'])
    calm = a['state_calm10'].reshape(n)
    o['state_calm10'] = calm.copy(); o['core_weight'] = np.where(calm, .5, .52)[:, None]
    return o


def e(result, i=-1):
    return result['audit']['event_history'][i]


class ArithmeticTests(unittest.TestCase):
    def test_handworked_daily_log_and_fixed_shrink(self):
        prices = np.exp(np.r_[0., np.cumsum(np.r_[np.full(42, .002), np.full(210, -.001)])])
        mask = np.r_[np.ones(42, bool), np.zeros(210, bool)]
        got = oracle.drift_oracle(prices, mask)
        rs = [math.log(float(b))-math.log(float(a)) for a, b in zip(prices, prices[1:])]
        mu = math.fsum(rs)/252.; cm = math.fsum(rs[:42])/42.
        self.assertEqual(got, dict(unconditional_mean=mu, conditional_mean=cm,
            candidate_raw=mu+(42/(42+63.))*(cm-mu), control_raw=mu))
        self.assertGreater(got['candidate_raw'], mu)

    def test_41_42_minimum_and_equal_copy(self):
        p = np.exp(np.arange(253, dtype=float)/1000.)
        for n in (0, 41):
            with self.assertRaises(ValueError): oracle.drift_oracle(p, np.arange(252)<n)
        self.assertTrue(math.isfinite(oracle.drift_oracle(p, np.arange(252)<42)['candidate_raw']))
        got = oracle.drift_oracle(p, np.ones(252, bool))
        self.assertEqual(np.float64(got['candidate_raw']).tobytes(), np.float64(got['control_raw']).tobytes())
        flat = oracle.drift_oracle(np.ones(211), np.arange(210)<42)
        self.assertEqual(flat['candidate_raw'], 0.); self.assertEqual(flat['control_raw'], 0.)

    def test_stable_log_difference_extreme_prices(self):
        p = np.ones(253); p[0], p[1], p[2] = np.nextafter(0., 1.), 1e308, np.nextafter(0., 1.)
        result = oracle.drift_oracle(p, np.ones(252, bool))
        self.assertTrue(math.isfinite(result['candidate_raw']))
        self.assertEqual(result['candidate_raw'], result['control_raw'])
        with self.assertRaises(FloatingPointError): oracle.market_state([1e308]*200)

    def test_strict_spy_mean_equality_and_near_tie(self):
        self.assertEqual(oracle.market_state([10.]*200), (False, 10.))
        x = [10.]*199+[np.nextafter(10., math.inf)]
        state, mean = oracle.market_state(x)
        self.assertEqual(mean, math.fsum(x)/200.)
        self.assertEqual(state, x[-1] > mean)
        self.assertTrue(state)

    def test_exact_tie_ranks_constant_and_two_names(self):
        self.assertEqual(oracle.effect_ranks([1., 1., 3.]), [-.25, -.25, .5])
        self.assertEqual(oracle.effect_ranks([3., 1.]), [.5, -.5])
        self.assertEqual(oracle.effect_ranks([7., 7.]), [0., 0.])
        self.assertEqual(oracle.effect_ranks([-0., 0., 0.]), [0., 0., 0.])
        self.assertEqual(oracle.effect_ranks([1., np.nextafter(1., 2.)]), [-.5, .5])


class SourceClockTests(unittest.TestCase):
    def test_210_252_exact_endpoints_and_counts(self):
        for calm, L in ((True, 210), (False, 252)):
            args = single(42, calm); out = oracle.independent_build(*args); z = e(out)
            self.assertTrue(z['ready']); t=args[2].index(z['date']); u=t-42
            self.assertEqual(z['formation_start'], args[2][u-L]); self.assertEqual(z['formation_end'], args[2][u])
            self.assertEqual(z['historical_state_first'], args[2][u-L]); self.assertEqual(z['historical_state_last'], args[2][u-1])
            self.assertEqual(z['matching_state_days'], 42)
            c=z['counts']; self.assertEqual(c['stock_log_returns'], L*4)
            self.assertEqual(c['market_state_evaluations'], L+1)
            self.assertEqual(c['market_price_observations'], 200*(L+1))
            self.assertEqual(c['stock_price_observations'], (L+1)*4)
            self.assertEqual(c['drift_estimates'], 8); self.assertEqual(c['ranked_observations'], 8)

    def test_41_shared_fallback_42_supported(self):
        for matches in (41, 42):
            out=oracle.independent_build(*single(matches))
            self.assertEqual(e(out)['matching_state_days'], matches)
            self.assertEqual(bool(out['arrays']['ready'][0]), matches == 42)
            self.assertEqual(e(out)['counts']['state_support_fallback_months'], int(matches == 41))
            if matches == 41:
                self.assertEqual(e(out)['counts']['stock_log_returns'], 0)
                self.assertTrue(np.isnan(out['arrays']['control_raw']).all())

    def test_historical_state_is_previous_day_not_outcome_day(self):
        args=single(0); out=oracle.independent_build(*args)
        self.assertTrue(e(out)['query_state_up']); self.assertEqual(e(out)['historical_up_days'], 0)
        self.assertEqual(e(out)['matching_state_days'], 0)

    def test_current_market_state_t_can_change_after_u_stock_learning_ends(self):
        args=single(42); before=oracle.independent_build(*args)
        day=args[1]['dates'][0]; args[0]['adj'].loc[day,'SPY']=.5
        after=oracle.independent_build(*args)
        self.assertTrue(e(before)['query_state_up']); self.assertFalse(e(after)['query_state_up'])
        self.assertEqual(e(before)['historical_up_days'], e(after)['historical_up_days'])
        self.assertEqual(e(before)['formation_end'], e(after)['formation_end'])
        self.assertEqual(e(after)['matching_state_days'],252-42)

    def test_recent42_stock_returns_and_future_prices_not_consumed(self):
        args=single(42); out=oracle.independent_build(*args)
        t=args[2].index(args[1]['dates'][0]); cut=args[2][t-42]
        args[0]['adj'].loc[args[0]['adj'].index>cut,['A','B','C','D']]=np.nan
        args[0]['adj'].loc[args[0]['adj'].index>args[1]['dates'][0], 'SPY']=np.inf
        got=oracle.independent_build(*args)
        oracle.record_check(got['audit'], out['audit'])
        for key in oracle.ARRAY_KEYS: oracle.exact_array(got['arrays'][key],out['arrays'][key],key)

    def test_one_bad_eligible_source_falls_back_all_but_checks_remaining(self):
        args=single(); args[0]['adj'].loc[args[2][args[2].index(args[1]['dates'][0])-80], 'A']=np.nan
        out=oracle.independent_build(*args); z=e(out)
        self.assertEqual(z['status'], 'invalid_stock_source')
        self.assertEqual(z['counts']['checked_stock_cells'], 4)
        self.assertEqual(z['counts']['source_valid_stock_cells'], 3)
        self.assertEqual(z['counts']['stock_log_returns'], 0)
        self.assertTrue(all(cell['source_checked'] for cell in z['cells']))
        self.assertFalse(np.any(out['arrays']['candidate_effect']))
        self.assertTrue(np.isnan(out['arrays']['candidate_raw']).all())

    def test_noneligible_poison_and_missing_identity_ignored(self):
        args=single(); ref=oracle.independent_build(*args)
        args[0]['adj']=args[0]['adj'].drop(columns='E')
        oracle.record_check(oracle.independent_build(*args)['audit'], ref['audit'])
        args[3]['eligible'][0,4]=True
        self.assertEqual(e(oracle.independent_build(*args))['status'],'invalid_stock_source')

    def test_market_holes_invalid_types_and_early_history(self):
        for change,status in (('missing','missing_market_symbol'),('bad','invalid_market_price'),('text','nonnumeric_market_price')):
            args=single(); day=args[1]['dates'][0]
            if change=='missing': args[0]['adj']=args[0]['adj'].drop(columns='SPY')
            elif change=='bad': args[0]['adj'].loc[day,'SPY']=0.
            else:
                args[0]['adj']['SPY']=args[0]['adj']['SPY'].astype(object); args[0]['adj'].loc[day,'SPY']='2'
            z=e(oracle.independent_build(*args))
            self.assertEqual(z['status'],status); self.assertEqual(z['counts']['market_state_evaluations'],0)
            self.assertEqual(z['counts']['checked_stock_cells'],0)
        args=list(single()); t=args[2].index(args[1]['dates'][0]); args[2]=args[2][t-400:]
        z=e(oracle.independent_build(*args)); self.assertEqual(z['status'],'insufficient_calendar')
        self.assertIsNone(z['formation_start'])

    def test_initial_and_small_domain_do_not_read_prices(self):
        args=fixture(); args[0]['adj']=args[0]['adj'].iloc[:0]
        out=oracle.independent_build(*args,limit=1)
        self.assertEqual(e(out)['status'],'initial_partial'); self.assertEqual(e(out)['counts']['checked_stock_cells'],0)
        args=single(); args[0]['adj']=args[0]['adj'].iloc[:0]; args[3]['eligible'][:]=False
        self.assertEqual(e(oracle.independent_build(*args))['status'],'fewer_than_two_eligible')
        args[3]['eligible'][0,1]=True
        self.assertEqual(e(oracle.independent_build(*args))['status'],'fewer_than_two_eligible')

    def test_constant_all_eligible_supported_and_zero_effect(self):
        args=single()
        for name in ['A','B','C','D']: args[0]['adj'][name]=7.
        out=oracle.independent_build(*args)
        self.assertTrue(e(out)['ready']); self.assertFalse(np.any(out['arrays']['candidate_effect']))
        self.assertEqual(e(out)['counts']['ranked_observations'],8)

    def test_missing_official_session_no_compression(self):
        args=single(); t=args[2].index(args[1]['dates'][0]); day=args[2][t-50]
        args[0]['adj']=args[0]['adj'].drop(index=day)
        self.assertEqual(e(oracle.independent_build(*args))['status'],'missing_market_session')

    def test_metadata_shape_boolean_and_calendar_contract(self):
        for mutation in ('intelig','intcalm','duplicate','nonnormalized','notmonthend','unsorted'):
            args=single()
            if mutation=='intelig':args[3]['eligible']=args[3]['eligible'].astype(int)
            elif mutation=='intcalm':args[3]['state_calm10']=np.array([1])
            elif mutation=='duplicate':args[1]['symbols'][0][1]='A'
            elif mutation=='nonnormalized':args[1]['dates'][0]+='T00:00:00'
            elif mutation=='notmonthend':args[1]['dates'][0]='2016-10-28'
            else:args[2][1],args[2][2]=args[2][2],args[2][1]
            with self.assertRaises(ValueError):oracle.independent_build(*args)
        args=single(); args[3]['state_calm10']=args[3]['state_calm10'][:,None]
        self.assertTrue(e(oracle.independent_build(*args))['ready'])

    def test_physical_and_logical_prefixes_and_future_metadata_poison(self):
        args=fixture(); full=oracle.independent_build(*args)
        for n in range(len(args[1]['dates'])+1):
            logical=oracle.independent_build(*args,limit=n)
            physical=oracle.independent_build(*physical_prefix(args,n))
            oracle.record_check(logical['audit'],physical['audit'])
            oracle.record_check(logical['audit']['event_history'],full['audit']['event_history'][:n])
            for key in oracle.ARRAY_KEYS:
                oracle.exact_array(logical['arrays'][key],full['arrays'][key][:n],key)
                oracle.exact_array(logical['arrays'][key],physical['arrays'][key],key)
        args[1]['dates'][-1]='not-a-date';args[1]['symbols'][-1]=['BAD']
        oracle.record_check(oracle.independent_build(*args,limit=3)['audit']['event_history'],full['audit']['event_history'][:3])

    def test_full_audit_and_raw_effect_mutations_rejected(self):
        args=single(); out=oracle.independent_build(*args)
        receipt=oracle.check_build(*args,out)
        self.assertTrue(all(receipt[k] is True for k in oracle.MODEL_FLAGS))
        self.assertEqual(receipt['numerical_tolerance'],0.)
        for mode in ('count','clock','raw','effect','ready'):
            z=copy.deepcopy(out)
            if mode=='count':z['audit']['counts']['stock_log_returns']+=1
            elif mode=='clock':z['audit']['event_history'][0]['historical_state_last']=z['audit']['event_history'][0]['formation_end']
            elif mode=='raw':z['arrays']['candidate_raw'][0,0]+=1e-10
            elif mode=='effect':z['arrays']['candidate_effect'][0,0]+=1e-10
            else:z['arrays']['ready'][0]=False
            with self.assertRaises(ValueError):oracle.check_build(*args,z)


class SelectionTests(unittest.TestCase):
    def test_full_eligible_can_displace_original_top2_without_second_episode(self):
        args=single(); out=oracle.independent_build(*args); orig=original(args)
        # Synthetic supported raw ranking prefers C/D over A/B.
        for key in ('candidate','control'):
            out['arrays'][key+'_raw'][0,:4]=[0.,1.,3.,2.]
            out['arrays'][key+'_effect'][0,:4]=oracle.effect_ranks([0.,1.,3.,2.])
        scores, choices=oracle.selection_oracle(orig,args[1],out)
        self.assertNotIn(0, choices[oracle.ARMS[1]][0])
        self.assertEqual(choices[oracle.ARMS[1]][0], sorted(choices[oracle.ARMS[1]][0], key=lambda j:(-scores[oracle.ARMS[1]][0,j],args[1]['symbols'][0][j])))
        receipt=oracle.verify_selection(orig,args[1],out,scores,choices)
        self.assertTrue(all(receipt[k] for k in oracle.SELECTION_FLAGS))

    def test_supported_zero_effect_is_not_original_fallback(self):
        args=single()
        for name in ['A','B','C','D']:args[0]['adj'][name]=7.
        out=oracle.independent_build(*args); orig=original(args)
        scores, choices=oracle.selection_oracle(orig,args[1],out)
        self.assertNotEqual(choices[oracle.ARMS[0]][0], choices[oracle.ARMS[1]][0])
        helper=oracle.frozen_source_oracle().old.original_helper(); pre=helper.original_prefix(orig,args[1],None)['local']
        oracle.exact_array(scores[oracle.ARMS[1]],pre,'zero effect removes old effect')

    def test_original_clamp_and_budget_all_states(self):
        for calm in (False,True):
            for weak in (False,True):
                args=single(calm=calm); out=oracle.independent_build(*args); orig=original(args)
                orig['state_weak_narrow'][:]=weak
                support=np.tile([.49,-.49,0.,0.,0.],(1,1))
                orig['baseline']=orig['core']+(1-orig['core_weight'])*support
                scores,choices=oracle.selection_oracle(orig,args[1],out)
                helper=oracle.frozen_source_oracle().old.original_helper(); x=helper.original_prefix(orig,args[1],None)
                budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))[0]
                for j in range(4):
                    delta=budget*(min(.5,max(-.5,x['support'][0,j]+out['arrays']['candidate_effect'][0,j]))-x['support'][0,j])
                    self.assertEqual(scores[oracle.ARMS[1]][0,j],x['local'][0,j]+delta)
                oracle.verify_selection(orig,args[1],out,scores,choices)

    def test_cold_and_initial_exact_incumbent(self):
        for args in (single(41),physical_prefix(fixture(),1)):
            out=oracle.independent_build(*args); orig=original(args)
            scores,choices=oracle.selection_oracle(orig,args[1],out)
            for arm in oracle.ARMS[1:]:
                oracle.exact_array(scores[arm],scores[oracle.ARMS[0]],arm)
                self.assertEqual(choices[arm],choices[oracle.ARMS[0]])

    def test_reject_effect_outside_eligible_or_changed_choice(self):
        args=single(); out=oracle.independent_build(*args); orig=original(args)
        scores,choices=oracle.selection_oracle(orig,args[1],out)
        bad=copy.deepcopy(choices);bad[oracle.ARMS[1]][0].reverse()
        with self.assertRaises(ValueError):oracle.verify_selection(orig,args[1],out,scores,bad)
        out['arrays']['candidate_effect'][0,4]=.1
        with self.assertRaises(ValueError):oracle.selection_oracle(orig,args[1],out)


class BilateralTests(unittest.TestCase):
    def test_no_producer_import_in_oracle(self):
        source=inspect.getsource(oracle)
        self.assertNotIn('import monthly_market_drift231_model',source)
        self.assertNotIn('monthly_market_drift231_model.py',source)

    def test_both_fixtures_all_prefixes_arrays_full_audit(self):
        import monthly_market_drift231_model as producer
        from test_monthly_market_drift231_model import fixture as producer_fixture
        for args in (fixture(),producer_fixture()):
            full=producer.build(*args)
            oracle.check_build(*args,full)
            for n in range(len(args[1]['dates'])+1):
                p=physical_prefix(args,n)
                got=producer.build(*p)
                oracle.check_build(*p,got)
                small=producer.build(*args,limit=n)
                oracle.record_check(got['audit'],small['audit'])
                for key in oracle.ARRAY_KEYS:oracle.exact_array(got['arrays'][key],full['arrays'][key][:n],key)


if __name__ == '__main__':
    unittest.main()
