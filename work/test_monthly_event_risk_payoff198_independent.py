"""Independent198 synthetic-only scalar and complete-artifact checks."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_event_risk_payoff198_independent as independent


def fixture(count=3, width=4):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    frame = pd.DataFrame({'A': np.exp(.0001 * x + .03 * np.sin(x / 11.)),
        'B': 3000. - x, 'FUND': 5. + np.sin(x / 19.), 'D': np.full(len(index), 8.)}, index=index)
    names = ['A', 'B', 'FUND', 'D'][:width]
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'adj': frame}, dict(dates=dates, symbols=[names[:] for _ in dates], etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


def example_prices():
    p = 100. + np.arange(253)
    p[64] = 100.
    p[85] = 140.
    return p


class ScalarTests(unittest.TestCase):
    def test_two_events_formula_covariance_shrink_and_direction(self):
        p = example_prices(); r = independent.measure_oracle(p)
        self.assertEqual(r['anchor_indices'], [64, 85])
        self.assertIsNone(r['identity_branch'])
        events = r['anchor_events']; y = [e['payoff'] for e in events]; w = [e['inverse_sigma'] for e in events]
        n = len(y); ym = sum(y) / n; wm = sum(w) / n
        covariance = sum((a - ym) * (b - wm) for a, b in zip(y, w)) / n
        self.assertAlmostEqual(r['candidate'] - r['control'], n / (n + 20) / math.sqrt(21) * covariance, places=13)
        self.assertAlmostEqual(r['candidate'], sum(a*b for a,b in zip(y,w)) / ((n+20)*math.sqrt(21)), places=13)
        self.assertAlmostEqual(r['control'], wm * sum(y) / ((n+20)*math.sqrt(21)), places=13)
        old = sum(y) / ((n+20)*math.sqrt(21)*sum(e['sigma'] for e in events)*.5)
        self.assertNotAlmostEqual(old, r['control'], places=8)

    def test_risk_exact63_and_excludes_loss_and_payoff(self):
        p = example_prices(); a = independent.measure_oracle(p)['anchor_events'][0]
        self.assertEqual((a['risk_price_start_index'], a['risk_price_end_index']), (0, 63))
        self.assertEqual((a['risk_return_first_index'], a['risk_return_last_index']), (1, 63))
        self.assertEqual(len(a['pre_event_simple_returns']), 63)
        expected = [p[j]/p[j-1]-1 for j in range(1,64)]
        self.assertEqual(a['pre_event_simple_returns'], expected)
        self.assertAlmostEqual(a['sigma'], np.std(expected, ddof=1), places=15)
        p[64] = 80.; p[85] = 145.
        b = independent.measure_oracle(p)['anchor_events'][0]
        self.assertEqual(a['sigma'], b['sigma'])
        self.assertNotEqual(a['payoff'], b['payoff'])

    def test_cooldown_selected_before_risk_and_maximum_eight(self):
        p = 300. - np.arange(253)
        r = independent.measure_oracle(p)
        self.assertEqual(r['anchor_indices'], list(range(64,232,21)))
        self.assertEqual(r['selected_anchors'], 8)
        self.assertEqual(r['valid_event_payoffs'], 8)
        self.assertLess(r['candidate'], 0.)
        self.assertLess(r['control'], 0.)
        self.assertTrue(all(e['maturity_index'] <= 252 for e in r['anchor_events']))

    def test_latest_anchor_matures_at252_and_singleton_canonical(self):
        p = 100. + np.arange(253); p[231] = 200.
        r = independent.measure_oracle(p); e = r['anchor_events'][0]
        self.assertEqual(r['anchor_indices'], [231])
        self.assertEqual((e['risk_price_start_index'], e['risk_price_end_index'], e['maturity_index']), (167,230,252))
        self.assertEqual(r['identity_branch'], 'single_event')
        self.assertEqual(r['candidate'], r['control'])
        self.assertEqual(np.float64(r['candidate']).tobytes(), np.float64(r['control']).tobytes())
        p = 100. + np.arange(253); p[232] = 200.
        self.assertEqual(independent.measure_oracle(p)['reason'], 'no_selected_anchors')
        p = 1. + np.arange(253) * 1e-14; p[231] = 1. - 1e-12
        tiny = independent.measure_oracle(p)
        self.assertTrue(tiny['available'])
        self.assertGreater(tiny['anchor_events'][0]['sigma'], 0.)
        self.assertLess(tiny['anchor_events'][0]['sigma'], 1e-12)

    def test_zero_sigma_shared_missing_still_evaluates_all_events(self):
        p = np.full(253,100.); p[64] = p[85] = 90.
        r = independent.measure_oracle(p)
        self.assertTrue(r['source_available']); self.assertFalse(r['available'])
        self.assertEqual(r['reason'], 'zero_pre_event_sigma')
        self.assertEqual((r['selected_anchors'],r['valid_event_payoffs']), (2,0))
        self.assertEqual(r['anchor_indices'], [64,85])
        self.assertEqual(r['anchor_events'][0]['sigma'], 0.)
        self.assertIsNone(r['anchor_events'][0]['inverse_sigma'])
        self.assertGreater(r['anchor_events'][1]['sigma'], 0.)
        self.assertEqual(r['anchor_events'][0]['payoff'], 0.)
        self.assertGreater(r['anchor_events'][1]['payoff'], 0.)
        self.assertIsNotNone(r['payoff_sum']); self.assertIsNotNone(r['denominator'])
        self.assertIsNone(r['inverse_risk_sum']); self.assertIsNone(r['candidate'])

    def test_no_events_missing_vs_actual_zero_payoff_and_identity_branches(self):
        for p in (np.ones(253), np.arange(1,254,dtype=float)):
            r=independent.measure_oracle(p)
            self.assertTrue(r['source_available']); self.assertFalse(r['available'])
            self.assertEqual(r['reason'], 'no_selected_anchors')
            self.assertEqual(r['selected_anchors'], 0)
        r=independent.measure_oracle(np.resize([1.,2.],253))
        self.assertEqual(r['identity_branch'], 'equal_risks'); self.assertEqual(r['candidate'],r['control'])
        p=100.+np.arange(253)/100.
        for i in range(64,253): p[i]=100.+(i-64)%21
        r=independent.measure_oracle(p)
        self.assertEqual(r['identity_branch'],'equal_payoffs')
        self.assertEqual((r['candidate'],r['control']), (0.,0.))
        self.assertTrue(r['available'])

    def test_full_source_validation_exact_negative_and_inputs_unchanged(self):
        p=example_prices(); old=p.copy()
        independent.measure_oracle(p); np.testing.assert_array_equal(p,old)
        for position in (0,63,64,200,231,252):
            for bad in (0.,-1.,np.nan,np.inf):
                q=p.copy(); q[position]=bad
                self.assertEqual(independent.measure_oracle(q)['reason'], 'invalid_adjusted_price')
        for bad in (True,None,'1',complex(1,0)):
            q=p.astype(object);q[100]=bad
            self.assertEqual(independent.measure_oracle(q)['reason'], 'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError): independent.measure_oracle(p[:-1])
        q=100.+np.arange(253); q[64]=np.nextafter(q[63],0.)
        self.assertEqual(independent.measure_oracle(q)['anchor_indices'],[64])

    def test_valid_price_arithmetic_overflow_hard_fails_and_underflow_finite(self):
        p=example_prices();p[0]=5e-324;p[1]=1e308
        with self.assertRaises(FloatingPointError): independent.measure_oracle(p)
        p=example_prices();p[0]=1e-100;p[1]=1e100
        with self.assertRaises(FloatingPointError): independent.measure_oracle(p)
        p=np.full(253,1e-200);p[0]=1e308;p[64:85]=5e-201
        r=independent.measure_oracle(p)
        self.assertTrue(r['available'])
        self.assertEqual(r['anchor_events'][0]['pre_event_simple_returns'][0],-1.)

    def test_constant_price_unit_change_approximates_raw_and_preserves_anchor_set(self):
        p=example_prices();a=independent.measure_oracle(p)
        for scale in (2.**-300,2.**300):
            b=independent.measure_oracle(p*scale)
            self.assertEqual(a['anchor_indices'],b['anchor_indices'])
            for key in ('candidate','control'): self.assertAlmostEqual(a[key],b[key],places=10)
            for x,y in zip(a['anchor_events'],b['anchor_events']): self.assertEqual(x['sigma'],y['sigma'])


class BuildTests(unittest.TestCase):
    def test_all_counts_dates_and_unchanged_full_candidate_support(self):
        p,m,c=fixture();before=p['adj'].copy();v,a=independent.independent_features(p,m,c)
        receipt=independent.verify_features(p,m,c,v,a)
        self.assertEqual(receipt['source_available_by_event'],[4,4,4])
        self.assertEqual(receipt['available_by_event'],[3,3,3])
        self.assertEqual(receipt['valid_price_observations'],12*253)
        self.assertEqual(receipt['pre_event_return_observations'],63*receipt['selected_anchors'])
        self.assertEqual(receipt['matured_payoffs'],receipt['selected_anchors'])
        self.assertEqual(sum(receipt['valid_event_payoffs_by_event']),receipt['valid_event_payoffs'])
        self.assertTrue(v['available'][:,2].all())
        for row in a['event_history']:
            t=c.index(row['date'])
            self.assertEqual((row['price_start'],row['price_end']),(c[t-294],c[t-42]))
            self.assertEqual((row['anchor_first_date'],row['anchor_last_date']),(c[t-230],c[t-63]))
        pd.testing.assert_frame_equal(p['adj'],before)

    def test_all_prefixes_and_non_numeric_future_tail_poison(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for n in range(4):
            q={'adj':p['adj'].astype(object).copy()}
            stop=pd.Timestamp.min if not n else pd.Timestamp(c[c.index(m['dates'][n-1])-42])
            q['adj'].loc[q['adj'].index>stop,:]='unread future'
            vv,aa=independent.independent_features(q,m,c,n)
            independent.verify_features(q,m,c,vv,aa,n)
            for key in v: np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
            q['adj']=q['adj'].loc[:stop]
            vvv,aaa=independent.independent_features(q,m,c,n)
            for key in vv: np.testing.assert_array_equal(vvv[key],vv[key])
            self.assertEqual(aaa,aa)
        self.assertEqual(independent.independent_features(p,m,c,0)[0]['candidate'].shape,(0,4))

    def test_missing_priority_monthends_and_schema(self):
        p,m,c=fixture(1);t=c.index(m['dates'][0])
        p['adj']=p['adj'].drop(columns='A').drop(pd.Timestamp(c[t-200]))
        _,a=independent.independent_features(p,m,c)
        self.assertEqual(a['reason_counts'],{'missing_symbol':1,'missing_official_session':3})
        _,a=independent.independent_features(p,m,[x for x in c if x>='2016-01-01'])
        self.assertEqual(a['reason_counts'],{'insufficient_history':4})
        p,m,c=fixture()
        for limit in (-1,4,True,1.5):
            with self.assertRaises(ValueError): independent.independent_features(p,m,c,limit)
        bad=copy.deepcopy(m);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError): independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError): independent.independent_features(p,m,c[:c.index(m['dates'][-1])+1])
        bad=dict(m);bad.pop('etfs')
        with self.assertRaises(ValueError): independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError): independent.independent_features({'volume':p['adj']},m,c)

    def test_singleton_valid_rankzero_and_zero_sigma_counts(self):
        p,m,c=fixture(1,1);v,a=independent.independent_features(p,m,c)
        self.assertTrue(v['available'][0,0]);self.assertEqual(v['candidate_rank'][0,0],0.)
        t=c.index(m['dates'][0]);q=np.ones(253)*100;q[64]=q[85]=90.
        p['adj'].loc[c[t-294:t-41],'A']=q
        v,a=independent.independent_features(p,m,c);r=independent.verify_features(p,m,c,v,a)
        self.assertEqual((r['source_available_cells'],r['available_cells'],r['selected_anchors'],r['valid_event_payoffs']),(1,0,2,0))
        self.assertEqual((r['matured_payoffs'],r['pre_event_return_observations']),(2,126))

    def test_corrupt_source_risk_maturity_raw_rank_and_counter_rejected(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for kind in ('raw','rank','mask','anchor','risk','maturity','inverse','identity','count','type','date'):
            vv,aa=copy.deepcopy(v),copy.deepcopy(a);cell=aa['event_history'][0]['cells'][0]
            event=cell['anchor_events'][0]
            if kind=='raw': vv['candidate'][0,0]+=.001
            elif kind=='rank': vv['control_rank'][0,0]+=.001
            elif kind=='mask': vv['available'][0,0]=False
            elif kind=='anchor': event['anchor_index']+=1
            elif kind=='risk': event['sigma']*=1.001
            elif kind=='maturity': event['maturity_index']-=1
            elif kind=='inverse': event['inverse_sigma']*=1.001
            elif kind=='identity': cell['identity_branch']='single_event'
            elif kind=='count': aa['selected_anchors_by_event'][0]+=1
            elif kind=='type': aa['checked_cells']=float(aa['checked_cells'])
            else: aa['event_history'][0]['anchor_last_date']=aa['event_history'][0]['price_end']
            with self.assertRaises(ValueError): independent.verify_features(p,m,c,vv,aa)

    def test_centered_rank_deltas_firstpartial_bound_no_eligibility_filter(self):
        p,m,c=fixture();m['eligibility']='not consumed';v,_=independent.independent_features(p,m,c)
        d=independent.independent_deltas(v);r=independent.verify_deltas(v,d)
        self.assertEqual(r['max_absolute_delta'],.005)
        for arm,key in independent.RAW.items():
            np.testing.assert_array_equal(d[arm][0],np.zeros(4))
            np.testing.assert_array_equal(d[arm][1:],.01*v[key+'_rank'][1:])
        q=copy.deepcopy(d);q['event_risk_payoff'][0,0]=.001
        with self.assertRaises(ValueError): independent.verify_deltas(v,q)
        with self.assertRaises(ValueError): independent.verify_deltas(v,d,.02)
        raw=np.array([[1.,0.,0.,-1.]])
        np.testing.assert_array_equal(independent.centered_ranks(raw,np.ones_like(raw,bool)),[[.5,0,0,-.5]])

    def test_raw_numeric_tolerance_cannot_hide_rank_change(self):
        p,m,c=fixture(1)
        for name in p['adj']: p['adj'][name]=p['adj']['A']
        v,a=independent.independent_features(p,m,c)
        v['candidate'][0,0]+=5e-15
        v['candidate_rank']=independent.centered_ranks(v['candidate'],v['available'])
        with self.assertRaises(ValueError): independent.verify_features(p,m,c,v,a)

    def test_accounting_pin_and_no_producer_import(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(),independent.FROZEN_HELPER_SHA)
        self.assertTrue(callable(independent.accounting().audit_path))
        self.assertNotIn('import monthly_event_risk_payoff198_kernel',inspect.getsource(independent))
        with patch.object(independent,'FROZEN_HELPER_SHA','0'*64):
            with self.assertRaises(ValueError): independent.accounting()

    def test_unique_full_producer_cross_both_fixtures_all_prefix_and_edge_support(self):
        import monthly_event_risk_payoff198_kernel as producer
        import test_monthly_event_risk_payoff198_kernel as producer_tests
        for p,m,c in (fixture(),producer_tests.fixture()):
            for n in range(len(m['dates'])+1):
                stop=pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n-1])
                q={'adj':p['adj'].loc[:stop]}
                v,a=producer.build_features(q,m,c,limit=n)
                independent.verify_features(q,m,c,v,a,limit=n)
                independent.verify_deltas(v,independent.independent_deltas(v))
        p,m,c=fixture(1);t=c.index(m['dates'][0]);dates=c[t-294:t-41]
        zero=np.full(253,100.);zero[64]=zero[85]=90.
        equal=np.resize([1.,2.],253)
        for path in (example_prices(),np.ones(253),zero,equal,300.-np.arange(253)):
            q=copy.deepcopy(p)
            for name in q['adj']:q['adj'].loc[dates,name]=path
            v,a=producer.build_features(q,m,c)
            independent.verify_features(q,m,c,v,a)
        for bad in (0.,np.nan,'bad'):
            q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[dates[100],'A']=bad
            v,a=producer.build_features(q,m,c)
            independent.verify_features(q,m,c,v,a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
