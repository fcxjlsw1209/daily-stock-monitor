"""Generated prices only; independent scalar oracle and fixed protocol tests."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import io
import json
import math
import hashlib
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_session_cluster148_prepare_independent as m
c=m.contract


def fixture(n=2):
    dates=pd.bdate_range('2015-01-01',periods=253+max(n,1))
    names=['S'+str(i) for i in range(32)]+['ETF'+str(i) for i in range(17)]
    close=pd.DataFrame(100.,index=dates,columns=names)
    opening=close.copy()
    for j,s in enumerate(names):
        for t in range(len(dates)):
            if t%(j+2)==0:opening.loc[dates[t],s]=101.
    meta=dict(dates=[str(d.date()) for d in dates[252:252+n]],symbols=[names[:] for _ in range(n)],etfs=names[32:])
    return meta,dates,close,opening


def score_fixture(n=2,width=49):
    meta,_,_,_=fixture(n);x=np.tile(np.linspace(.16,-.1,width),(n,1))
    arrays={k:np.zeros_like(x) for k in ('baseline','pre_score','effects','forecast_available',
        'rank_log_excess','old_ml4','extension_available','model_available','raw_local_constant','raw_local_linear','raw_source_log')}
    arrays.update(baseline=x,pre_score=x.copy(),core=x*.52,eligible=np.ones_like(x,dtype=bool),core_weight=np.full_like(x,.52),
        state_weak_narrow=np.zeros(n,bool),state_calm10=np.zeros(n,bool))
    for key in ('forecast_available','extension_available','model_available'):
        arrays[key]=np.zeros_like(x,dtype=bool)
    return arrays,meta


def audit_fixture(meta,result):
    a=dict(method=c.METHOD,window_events=210,adjacent_pairs=209,close_rows=211,
        opening_rows=210,skip_sessions=42,width=49,signal_count=len(meta['dates']),sign_zero_tolerance=1e-12,
        rank_definition='0.5-average_tied_zero_based_rank/(n_valid-1); n_valid<=1 gives0',
        rank_direction='lower_frequency_or_covariance_is_higher',raw_missing='NaN',rank_missing=0,
        common_mask=True,constant_event_sequence_valid=True,covariance_uses_two209_pair_marginals=True,
        covariance_arithmetic='integer_joint_count_times209_minus_left_times_right_over209_squared',
        caller_verified_complete_session_grid_required=True,caller_verified_common_adjusted_price_basis_required=True,
        etfs_allowed=True,no_trading_eligibility_changes=True,dose_not_applied=True,
        new_model_fits=0,portfolio_returns_or_nav_computed=False,reasons=dict(Counter(r['reason'] for r in result['cells'])),
        cells=result['cells'],months=result['months'],feature_contract=c.feature_contract(),
        price_cache_sha256=c.PRICE_SHA,sessions_sha256=c.SESSIONS_SHA,calendar_plan_sha256=c.CALENDAR_SHA)
    return a


class EventTests(unittest.TestCase):
    def stats(self,events):
        return c.event_oracle([100.]*211,[101. if x else 100. for x in events])
    def test_all_zero_valid(self):
        s=self.stats([0]*210);self.assertTrue(s['available']);self.assertEqual(s['frequency'],0);self.assertEqual(s['cluster'],0)
    def test_all_one_valid_zero_covariance(self):
        s=self.stats([1]*210);self.assertEqual(s['frequency'],1);self.assertEqual(s['cluster'],0)
    def test_alternation_negative_covariance(self):
        s=self.stats([0,1]*105);self.assertEqual(s['covariance_numerator'],-104*105)
    def test_clustered_positive_covariance(self):
        s=self.stats([0]*105+[1]*105);self.assertEqual(s['adjacent_one_pairs'],104);self.assertGreater(s['cluster'],0)
    def test_asymmetric_marginals(self):
        s=self.stats([1]+[0]*209);self.assertEqual((s['left_ones'],s['right_ones']),(1,0));self.assertEqual(s['cluster'],0)
    def test_sign_is_log_ratio_not_raw_units(self):
        opening=[100.]*210;opening[0]=100.*math.exp(2e-12)
        self.assertEqual(c.event_oracle([100.]*211,opening)['event_ones'],1)
    def test_near_zero_not_event(self):
        self.assertEqual(c.event_oracle([100.]*211,[100.*math.exp(.5e-12)]*210)['event_ones'],0)
    def test_negative_overnight_not_event(self):
        self.assertEqual(c.event_oracle([100.]*211,[99.]*210)['event_ones'],0)
    def test_bad_price_invalidates_entire_window(self):
        for v in (0.,-1.,np.nan,np.inf):
            for which in (0,1):
                data=[[100.]*211,[101.]*210];data[which][40]=v
                s=c.event_oracle(*data);self.assertFalse(s['available']);self.assertIsNone(s['frequency'])
    def test_underflow_and_overflow_ratio_invalid(self):
        for x,y in ((1e-300,1e300),(1e300,1e-300)):
            s=c.event_oracle([x]*211,[y]*210);self.assertEqual(s['reason'],'unrepresentable_log_ratio')
    def test_exact_window_lengths(self):
        with self.assertRaises(ValueError):c.event_oracle([100.]*210,[100.]*210)


class RankTests(unittest.TestCase):
    def test_full_ties_zero(self):np.testing.assert_array_equal(c.rank_oracle([1,1,1],[True]*3),[0,0,0])
    def test_pairwise_tied_rank(self):np.testing.assert_allclose(c.rank_oracle([1,2,2,4],[True]*4),[.5,0,0,-.5])
    def test_single_and_empty(self):
        for mask in ([True,False],[False,False]):np.testing.assert_array_equal(c.rank_oracle([1,np.nan],mask),[0,0])
    def test_missing_does_not_dilute(self):np.testing.assert_array_equal(c.rank_oracle([1,np.nan,2],[True,False,True]),[.5,0,-.5])
    def test_near_ties_are_not_rounded(self):np.testing.assert_array_equal(c.rank_oracle([1,1+1e-14],[True]*2),[.5,-.5])
    def test_invalid_rank_input(self):
        with self.assertRaises(ValueError):c.rank_oracle([np.nan],[True])


class FeatureTests(unittest.TestCase):
    def test_first_and_last_window_events(self):
        meta,d,cl,op=fixture(1);op[:]=100.
        op.iloc[1,0]=101.;op.iloc[210,0]=101.
        r=c.features_oracle(meta,d,cl,op)
        self.assertEqual(r['cells'][0]['event_ones'],2)
        self.assertEqual(r['months'][0]['close_window_start'],str(d[0].date()))
        self.assertEqual(r['months'][0]['window_end'],str(d[210].date()))
    def test_excludes_recent42_and_future(self):
        meta,d,cl,op=fixture(1);a=c.features_oracle(meta,d,cl,op)
        op.iloc[211:]=999.;cl.iloc[211:]=np.nan
        b=c.features_oracle(meta,d,cl,op);self.assertEqual(a['cells'],b['cells'])
    def test_one_missing_invalidates_both_but_etfs_included(self):
        meta,d,cl,op=fixture(1);cl.iloc[80,0]=np.nan;r=c.features_oracle(meta,d,cl,op)
        self.assertFalse(r['arrays']['available'][0,0])
        self.assertTrue(r['arrays']['available'][0,32:].all())
        for k in c.HANDS:self.assertEqual(r['arrays'][k][0,0],0.)
    def test_missing_column_and_date_preserved_as_missing(self):
        meta,d,cl,op=fixture(1);cl=cl.drop(columns='S0');op=op.drop(index=d[100])
        self.assertFalse(c.features_oracle(meta,d,cl,op)['arrays']['available'].any())
    def test_insufficient_history_explicit(self):
        meta,d,cl,op=fixture(1);meta['dates']=[str(d[251].date())]
        self.assertEqual(c.features_oracle(meta,d,cl,op)['cells'][0]['reason'],'insufficient_history')
    def test_signal_must_be_session(self):
        meta,d,cl,op=fixture(1);meta['dates']=['2016-01-02']
        with self.assertRaises(ValueError):c.features_oracle(meta,d,cl,op)
    def test_unsorted_or_duplicate_grid(self):
        meta,d,cl,op=fixture(1)
        with self.assertRaises(ValueError):c.features_oracle(meta,d[::-1],cl,op)
    def test_exact49(self):
        meta,d,cl,op=fixture(1);meta['symbols'][0].pop()
        with self.assertRaises(ValueError):c.features_oracle(meta,d,cl,op)
    def test_no_input_mutation(self):
        meta,d,cl,op=fixture(1);before=(cl.copy(),op.copy(),deepcopy(meta));c.features_oracle(meta,d,cl,op)
        pd.testing.assert_frame_equal(before[0],cl);pd.testing.assert_frame_equal(before[1],op);self.assertEqual(before[2],meta)
    def test_audit_tamper_counts(self):
        meta,d,cl,op=fixture(1);r=c.features_oracle(meta,d,cl,op);a=audit_fixture(meta,r);m.verify_audit(a,r,meta)
        a['cells'][0]['event_ones']+=1
        with self.assertRaises(ValueError):c.feature_checks(a,meta)
    def test_rank_population_cannot_exclude_etf(self):
        meta,d,cl,op=fixture(1);r=c.features_oracle(meta,d,cl,op);a=audit_fixture(meta,r)
        a['cells'][0]['frequency_rank']=.123
        with self.assertRaises(AssertionError):c.validate_cells(a['cells'],meta)
    def test_window_audit_tamper(self):
        meta,d,cl,op=fixture(1);r=c.features_oracle(meta,d,cl,op);a=deepcopy(audit_fixture(meta,r))
        a['months'][0]['window_end']=meta['dates'][0]
        with self.assertRaises(ValueError):m.verify_audit(a,r,meta)
    def test_no_new_producer_import(self):
        self.assertNotIn('monthly_session_cluster148_factor',sys.modules)
        self.assertNotIn('monthly_session_cluster148_runner',sys.modules)


class PriceTests(unittest.TestCase):
    def payload(self):
        meta,d,cl,op=fixture(1)
        return meta,d,dict(dates=d.to_numpy(),symbols=np.array(cl.columns),close=cl.to_numpy(),opening=op.to_numpy())
    def test_complete_axis_and_missing_value_allowed(self):
        meta,d,p=self.payload();p['close'][40,0]=np.nan;cl,op=c.price_frames(p,d,meta);self.assertTrue(np.isnan(cl.iloc[40,0]))
    def test_missing_session_row_rejected(self):
        meta,d,p=self.payload()
        for k in ('dates','close','opening'):p[k]=np.delete(p[k],40,axis=0)
        with self.assertRaises(ValueError):c.price_frames(p,d,meta)
    def test_wrong_dtype_rejected(self):
        meta,d,p=self.payload();p['opening']=p['opening'].astype('float32')
        with self.assertRaises(ValueError):c.price_frames(p,d,meta)
    def test_missing_warmup_rejected(self):
        meta,d,p=self.payload()
        for k in ('dates','close','opening'):p[k]=p[k][1:]
        with self.assertRaises(ValueError):c.price_frames(p,d,meta)


class ScoreTests(unittest.TestCase):
    def test_delta_applied_before_eligibility(self):
        a,meta=score_fixture();a['eligible'][0,0]=False;hand=np.full((2,49),.5)
        _,_,delta=m.score_oracle(a,meta,hand,-hand)
        self.assertEqual(delta[m.NAMES[1]][0,0],.005)
    def test_etf_receives_delta(self):
        a,meta=score_fixture();hand=np.full((2,49),.5)
        _,_,delta=m.score_oracle(a,meta,hand,hand)
        self.assertEqual(delta[m.NAMES[1]][0,32],.005)
    def test_zero_exact_fallback(self):
        a,meta=score_fixture();s,ch,_=m.score_oracle(a,meta,np.zeros((2,49)),np.zeros((2,49)))
        for name in m.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[m.NAMES[0]].tobytes());self.assertEqual(ch[name],ch[m.NAMES[0]])
    def test_bound_and_nonfinite_fail(self):
        for value in (.50001,np.nan,np.inf):
            a,meta=score_fixture()
            with self.assertRaises(ValueError):m.score_oracle(a,meta,np.full((2,49),value),np.zeros((2,49)))
    def test_all123_prefixes(self):
        a,meta=score_fixture(122);meta,d,cl,op=fixture(122);r=c.features_oracle(meta,d,cl,op)
        s,ch,_=m.score_oracle(a,meta,*(r['arrays'][k] for k in c.HANDS))
        m.verify_prefixes(a,meta,d,cl,op,r,s,ch)
    def test_original27_bytes_checked(self):
        orig={str(i):np.array([i],float) for i in range(27)};out=dict(orig,**{k:np.array([0.]) for k in c.EXTRA})
        m.verify_original_arrays(orig,out);out['0']=np.array([-0.])
        with self.assertRaises(ValueError):m.verify_original_arrays(orig,out)


if __name__=='__main__':
    write='--write-review' in sys.argv
    if write:sys.argv.remove('--write-review')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    log=stream.getvalue();print(log)
    if write:
        directory=m.ROOT/m.STUDY;directory.mkdir(exist_ok=True,parents=True)
        logpath=directory/'PREPARE_REVIEW_TESTS.log'
        with logpath.open('x') as f:f.write(log)
        receipt=m.synthetic_receipt(result);receipt['test_log']=dict(path=str(logpath),sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
        dest=m.ROOT/'work/monthly_session_cluster148_prepare_review_SYNTHETIC.json'
        with dest.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
