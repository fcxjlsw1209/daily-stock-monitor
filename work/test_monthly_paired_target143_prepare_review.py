"""Artificial-only tests for independent143 reviewer, no real cache reads."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from datetime import datetime, timezone
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_paired_target143_prepare_review as r
from test_monthly_price_delay140_prepare_review import score_fixture


def label_fixture():
    dates=pd.bdate_range('2018-01-01','2023-04-30');t=np.arange(len(dates))[:,None];c=np.arange(5)[None,:]
    prices=100*np.exp(t*(.0001+c*.00015)+.02*np.sin(t*(.09+c*.013)))
    names=['A','B','C','D','SPY'];close=pd.DataFrame(prices,index=dates,columns=names)
    opening=close*(1.+.003*np.cos(t*.37+c*.1))
    sessions=dates;signals=pd.DatetimeIndex([sessions[sessions.to_period('M')==month][-1]for month in pd.period_range('2018-01','2022-12',freq='M')])
    days=np.repeat(signals.to_numpy(dtype='datetime64[D]'),4);symbols=np.tile(np.array(names[:4]),len(signals));good=np.ones(len(days),bool)
    return close,opening,sessions,days,symbols,good


def model_fixture():
    close,opening,sessions,days,symbols,good=label_fixture();labels=r.labels_oracle(close,opening,sessions,days,symbols,good,good,good)['arrays']
    t=np.arange(len(days))[:,None];j=np.arange(10)[None,:];x=.4*np.sin(t*.11+j*.7)
    events=np.repeat(np.array(['2021-03-31','2021-06-30','2022-04-29'],dtype='datetime64[D]'),4);es=np.tile(np.array(['A','B','C','D']),3);ex=.3*np.cos(np.arange(12)[:,None]*.2+j*.7);eg=np.ones(12,bool)
    return labels,x,events,es,ex,eg


def score_data():
    a,m=score_fixture();m=dict(m,dates=['2020-01-31','2020-02-28']);shape=a['baseline'].shape
    a['forecast_available'][:]=True;a['model_available'][:]=True;a['extension_available'][:]=True
    a.update(paired_raw_close=np.tile(np.arange(shape[1]),(shape[0],1)).astype(float),paired_raw_open=np.tile(-np.arange(shape[1]),(shape[0],1)).astype(float),paired_available=np.ones(shape,bool))
    return a,m


class LabelTests(unittest.TestCase):
    def setUp(self):self.f=label_fixture()
    def build(self,f=None):
        close,op,cal,d,s,g=self.f if f is None else f
        return r.labels_oracle(close,op,cal,d,s,g,g,g)
    def test_exact_four_endpoints(self):
        close,op,cal,d,s,g=self.f;x=self.build();a=x['arrays'];i=0
        for k,frame,lo,hi in [('close',close,a['dates'][i],a['calendar_ends'][i]),('open',op,a['entry_sessions'][i],a['exit_sessions'][i])]:
            expected=np.log(frame.loc[str(hi),'A']/frame.loc[str(lo),'A'])-np.log(frame.loc[str(hi),'SPY']/frame.loc[str(lo),'SPY'])
            self.assertAlmostEqual(x['raw'][k][i],expected,14)
    def test_shared_maturity_later_than_calendar_end(self):
        a=self.build()['arrays'];self.assertTrue((a['available_sessions']>a['calendar_ends']).all());np.testing.assert_array_equal(a['available_sessions'],a['exit_sessions'])
    def test_clipping_before_center(self):
        f=list(self.f);f[0]=f[0].copy();f[0].loc['2018-02-28','A']*=1e10;a=self.build(f)['arrays'];v=a['target_close'][:4]
        self.assertAlmostEqual(v.sum(),0.);self.assertGreater(v[0],.7);self.assertLess(v[0],1.)
    def test_missing_open_drops_both_labels(self):
        f=list(self.f);f[1]=f[1].copy();f[1].loc['2018-03-01','A']=np.nan;a=self.build(f)['arrays'];self.assertFalse(a['good'][0]);self.assertTrue(np.isnan(a['target_close'][0]));self.assertTrue(np.isnan(a['target_open'][0]))
    def test_singleton_month_all_missing(self):
        f=list(self.f);f[-1]=f[-1].copy();f[-1][1:4]=False;a=self.build(f)['arrays'];self.assertFalse(a['good'][:4].any())
    def test_zero_benchmark_no_label(self):
        f=list(self.f);f[1]=f[1].copy();f[1].loc['2018-03-01','SPY']=0.;self.assertFalse(self.build(f)['arrays']['good'][:4].any())
    def test_price_scale_invariant(self):
        a=self.build()['arrays'];f=list(self.f);f[0]=f[0]*[.01,10,3,7,.5];f[1]=f[1]*[.01,10,3,7,.5];b=self.build(f)['arrays']
        for k in ('target_close','target_open'):np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-14,equal_nan=True)
    def test_zero_gap_labels_equal(self):
        f=list(self.f);cal=f[2];f[1]=f[0].shift(1);x=self.build(f)['arrays'];np.testing.assert_allclose(x['target_close'],x['target_open'],rtol=0,atol=1e-14,equal_nan=True)
    def test_non_endpoint_prices_unused(self):
        f=list(self.f);f[0]=f[0].copy();f[0].loc['2018-02-15','A']=np.nan;x=self.build(f);y=self.build();np.testing.assert_array_equal(x['arrays']['target_close'],y['arrays']['target_close'])
    def test_input_unchanged(self):
        f=self.f;before=(f[0].copy(),f[1].copy(),f[-1].copy());self.build();pd.testing.assert_frame_equal(f[0],before[0]);pd.testing.assert_frame_equal(f[1],before[1]);np.testing.assert_array_equal(f[-1],before[2])
    def test_nonmonthend_rejected(self):
        f=list(self.f);f[3]=f[3].copy();f[3][0]=np.datetime64('2018-01-30')
        with self.assertRaises(ValueError):self.build(f)
    def test_duplicate_identity_rejected(self):
        f=list(self.f);f[4]=f[4].copy();f[4][1]=f[4][0]
        with self.assertRaises(ValueError):self.build(f)
    def test_insufficient_calendar_rejected(self):
        with self.assertRaises(ValueError):r.schedule_oracle(pd.bdate_range('2020-01-01','2020-02-28'),['2020-01-31'])
    def test_nonboolean_good_rejected(self):
        f=list(self.f);f[-1]=f[-1].astype(int)
        with self.assertRaises(ValueError):self.build(f)
    def test_calendar_order_rejected(self):
        with self.assertRaises(ValueError):r.schedule_oracle(self.f[2][::-1],self.f[3])
    def test_missing_price_session_preserves_schedule(self):
        f=list(self.f);f[0]=f[0].drop('2018-03-01');f[1]=f[1].drop('2018-03-01');a=self.build(f)['arrays'];self.assertEqual(str(a['exit_sessions'][0]),'2018-03-01');self.assertFalse(a['good'][:4].any())


class ModelTests(unittest.TestCase):
    def setUp(self):self.f=model_fixture()
    def run_model(self,f=None,**kw):return r.model_oracle(*(self.f if f is None else f),**kw)
    def test_maturity_excludes_november_signal_at_dec31(self):
        x=self.run_model();a=x['audit']['annual'][0];self.assertEqual(a['latest_available_session'],'2020-12-01');self.assertEqual(a['latest_calendar_end'],'2020-11-30')
    def test_same_targets_same_predictions(self):
        f=list(self.f);f[0]=dict(f[0],target_open=f[0]['target_close'].copy());x=self.run_model(f);np.testing.assert_array_equal(x['raw']['close'],x['raw']['open'])
    def test_joint_cohort_invalid_other_target(self):
        f=list(self.f);f[0]=copy.deepcopy(f[0]);f[0]['target_open'][:4]=np.nan;x=self.run_model(f);y=self.run_model();self.assertEqual(x['audit']['annual'][0]['training_rows'],y['audit']['annual'][0]['training_rows']-4)
    def test_five_year_exclusive_lower_boundary(self):
        x=self.run_model();self.assertEqual(x['audit']['annual'][0]['lookback_exclusive_start'],'2015-12-31')
    def test_future_labels_poison_no_earlier_prediction_effect(self):
        f=list(self.f);f[0]=copy.deepcopy(f[0]);f[1]=f[1].copy();future=f[0]['available_sessions']>np.datetime64('2020-12-31');f[0]['target_close'][future]=999.;f[0]['target_open'][future]=-999.;f[1][future]=999.;f[2]=f[2][:8];f[3]=f[3][:8];f[4]=f[4][:8];f[5]=f[5][:8];x=self.run_model(f);y=self.run_model()
        for k in r.ARMS:np.testing.assert_array_equal(x['raw'][k][:8],y['raw'][k][:8])
    def test_future_query_poison_does_not_affect_prefix(self):
        f=list(self.f);f[4]=f[4].copy();f[4][8:]=np.inf;f[5]=f[5].copy();f[5][8:]=False;x=self.run_model(f);y=self.run_model()
        for k in r.ARMS:np.testing.assert_array_equal(x['raw'][k][:8],y['raw'][k][:8])
    def test_event_good_separate_from_label_mask(self):
        f=list(self.f);f[0]=copy.deepcopy(f[0]);f[0]['good'][f[0]['dates']>=np.datetime64('2021-01-01')]=False;x=self.run_model(f);self.assertTrue(x['available'][:8].all())
    def test_insufficient_own_history_exact_pooled_copy(self):
        f=list(self.f);f[3]=np.full(12,'UNSEEN');x=self.run_model(f);self.assertFalse(x['local_available'].any())
        for k in r.ARMS:np.testing.assert_array_equal(x['raw'][k],x['pooled'][k])
    def test_pooled_cold_start_nan(self):
        x=self.run_model(min_months=100);self.assertFalse(x['available'].any());self.assertTrue(np.isnan(x['raw']['close']).all())
    def test_local_available_does_not_change_pooled_mask(self):
        x=self.run_model();y=self.run_model(min_own_months=100);np.testing.assert_array_equal(x['available'],y['available']);self.assertFalse(y['local_available'].any())
    def test_zero_targets_zero_correction(self):
        f=list(self.f);f[0]=dict(f[0],target_close=np.zeros(len(f[0]['dates'])),target_open=np.zeros(len(f[0]['dates'])));x=self.run_model(f);np.testing.assert_array_equal(x['raw']['close'],np.zeros(12))
    def test_weight_month_totals_and_normalization(self):
        d=np.array(['2020-09-30']*2+['2020-10-30']*4,dtype='datetime64[D]');w=r.weights_oracle(d,np.datetime64('2020-12-31'));self.assertAlmostEqual(w.sum(),6.);self.assertAlmostEqual(w[2:].sum()/w[:2].sum(),2**(1/12))
    def test_lstsq_matches_independent_direct_normal_equation(self):
        rng=np.random.default_rng(103);x=rng.normal(size=(40,10));y=rng.normal(size=40);w=np.linspace(.2,1,40);fit=r.fit_oracle(x,y,w);z=(x-fit[0])/fit[1];beta=np.linalg.solve(np.einsum('ni,n,nj->ij',z,w,z)+10*np.eye(10),np.einsum('ni,n,n->i',z,w,y-fit[3]));np.testing.assert_allclose(beta,fit[2],atol=1e-14,rtol=0)
    def test_all_query_prefixes(self):
        full=self.run_model()
        for n in range(13):
            f=list(self.f);f[2]=f[2][:n];f[3]=f[3][:n];f[4]=f[4][:n];f[5]=f[5][:n];a=self.run_model(f)
            for k in r.ARMS:np.testing.assert_allclose(a['raw'][k],full['raw'][k][:n],atol=1e-13,rtol=0,equal_nan=True)
    def test_missing_good_rejects_nonboolean(self):
        f=list(self.f);f[5]=f[5].astype(int)
        with self.assertRaises(ValueError):self.run_model(f)
    def test_input_arrays_unchanged(self):
        before=copy.deepcopy(self.f);self.run_model()
        for a,b in zip(self.f,before):
            if isinstance(a,dict):
                for k in a:np.testing.assert_array_equal(a[k],b[k])
            else:np.testing.assert_array_equal(a,b)
    def test_invalid_month_availability_rejected(self):
        f=list(self.f);f[0]=dict(f[0],available_sessions=f[0]['calendar_ends'].copy())
        with self.assertRaises(ValueError):self.run_model(f)
    def test_bad_actual_rank_rejected(self):
        f=list(self.f);f[1]=f[1].copy();f[1][0,0]=.7
        with self.assertRaises(ValueError):self.run_model(f)


class ScoreTests(unittest.TestCase):
    def test_episodes_once_per_arm(self):
        a,m=score_data();h=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=h),patch.object(h,'episode',wraps=h.episode)as spy:r.scores_oracle(a,m);self.assertEqual(spy.call_count,3)
    def test_whole_month_missing_exact_local_incumbent(self):
        a,m=score_data();a['paired_available'][:]=False;s,c,_=r.scores_oracle(a,m)
        for k in r.NAMES[1:]:self.assertEqual(s[k].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[k],c[r.NAMES[0]])
    def test_one_missing_query_falls_back_both(self):
        a,m=score_data();use=a['eligible']&a['forecast_available'];i,j=np.argwhere(use)[0];a['paired_raw_open'][i,j]=np.nan;s,c,audit=r.scores_oracle(a,m);self.assertFalse(audit['months'][i]['paired_ready'])
        for k in r.NAMES[1:]:np.testing.assert_array_equal(s[k][i],s[r.NAMES[0]][i]);self.assertEqual(c[k][i],c[r.NAMES[0]][i])
    def test_unused_query_missing_does_not_filter(self):
        a,m=score_data();a['forecast_available'][0,-1]=False;a['paired_raw_open'][0,-1]=np.nan;_,_,audit=r.scores_oracle(a,m);self.assertTrue(audit['months'][0]['paired_ready'])
    def test_input_bytes_unchanged(self):
        a,m=score_data();before={k:v.tobytes()for k,v in a.items()};r.scores_oracle(a,m);self.assertEqual(before,{k:v.tobytes()for k,v in a.items()})
    def test_exact_two_equal_half_and_one_cash(self):
        a,m=score_data();a['eligible'][0]=False;a['eligible'][0,:2]=True;a['eligible'][1]=False;a['eligible'][1,0]=True;_,choices,_=r.scores_oracle(a,m)
        for v in choices.values():self.assertEqual(set(v[0]),{0,1});self.assertEqual(v[1],[])
    def test_all_123_score_prefixes(self):
        a,m=score_data();a={k:np.tile(v,(61,1))if v.ndim==2 else np.tile(v,61)for k,v in a.items()};m=dict(m,symbols=m['symbols']*61,dates=m['dates']*61);s,c,_=r.scores_oracle(a,m)
        for n in range(123):
            ss,cc,_=r.scores_oracle(a,m,n)
            for k in r.NAMES:self.assertEqual(ss[k].tobytes(),s[k][:n].tobytes());self.assertEqual(cc[k],c[k][:n])
    def test_later_raw_poison_prefix_ignored(self):
        a,m=score_data();s,c,_=r.scores_oracle(a,m,1);a['paired_raw_open'][1]=np.inf;ss,cc,_=r.scores_oracle(a,m,1)
        for k in r.NAMES:np.testing.assert_array_equal(s[k],ss[k]);self.assertEqual(c[k],cc[k])


class IndependentForecastDecisionTests(unittest.TestCase):
    def fixture(self):
        a,m=score_data();shape=a['baseline'].shape
        a.update(paired_pooled_close=a['paired_raw_close'].copy(),paired_pooled_open=a['paired_raw_open'].copy(),paired_local_available=np.ones(shape,bool))
        f=dict(raw={arm:a['paired_raw_'+arm].reshape(-1).copy()for arm in r.ARMS},pooled={arm:a['paired_pooled_'+arm].reshape(-1).copy()for arm in r.ARMS},available=a['paired_available'].reshape(-1).copy(),local_available=a['paired_local_available'].reshape(-1).copy())
        return a,m,f
    def test_identical_independent_predictions_accepted(self):
        a,m,f=self.fixture();_,s,c,_=r.independent_forecast_decisions(a,m,f);ss,cc,_=r.scores_oracle(a,m)
        for k in r.NAMES:np.testing.assert_array_equal(s[k],ss[k]);self.assertEqual(c[k],cc[k])
    def test_subtolerance_near_tie_rank_reversal_rejected(self):
        a,m,f=self.fixture();a['paired_raw_open'][0,:2]=[0.,1e-14];f['raw']['open'][:2]=[1e-14,0.]
        np.testing.assert_allclose(a['paired_raw_open'].reshape(-1),f['raw']['open'],rtol=0,atol=r.ATOL)
        with self.assertRaisesRegex(ValueError,'discrete rank mismatch'):r.independent_forecast_decisions(a,m,f)
    def test_rank_reversal_without_active_slot_still_rejected(self):
        a,m,f=self.fixture();a['state_weak_narrow'][:]=False;a['extension_available'][:]=False
        a['paired_raw_close'][0,:2]=[0.,1e-14];f['raw']['close'][:2]=[1e-14,0.]
        with self.assertRaisesRegex(ValueError,'discrete rank mismatch'):r.independent_forecast_decisions(a,m,f)
    def test_missing_month_uses_identical_independent_fallback(self):
        a,m,f=self.fixture();a['paired_available'][0]=False;f['available'][:5]=False
        b,s,c,_=r.independent_forecast_decisions(a,m,f)
        for k in r.NAMES[1:]:np.testing.assert_array_equal(s[k][0],s[r.NAMES[0]][0]);self.assertEqual(c[k][0],c[r.NAMES[0]][0])


class IsolationTests(unittest.TestCase):
    def test_producer_import_forbidden(self):
        tree=ast.parse(Path(r.__file__).read_text());bad={'monthly_paired_target143_model','monthly_paired_target143_runner','after142_paired_target_labels'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertFalse(any(alias.name in bad for alias in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,bad)
    def test_bad_plan_before_actual_access(self):
        with self.assertRaisesRegex(ValueError,'PLAN SHA'):r.audit('/not-present','PLAN.json','no')
    def test_unregistered_rejected(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/r.STUDY;p.mkdir(parents=True);f=p/'PLAN.json';f.write_text(json.dumps(dict(round=143,status='DRAFT',arms=list(r.NAMES))))
            with self.assertRaisesRegex(ValueError,'Round143'):r.audit(t,f,hashlib.sha256(f.read_bytes()).hexdigest())

def prereview_receipt(result):
    paths=[r.ROOT/'work'/name for name in ('monthly_paired_target143_model.py','test_monthly_paired_target143_model.py','after142_paired_target_labels.py','after142_paired_target_labels_tests.py','monthly_paired_target143_runner.py','test_monthly_paired_target143_runner.py')]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proposal=r.ROOT/'work/monthly_paired_target143_proposal.json'
    assert sha(proposal)==r.PROPOSAL_SHA
    value=dict(passed=result.wasSuccessful(),round=143,stage='preregistration_prepare_review',created_at=datetime.now(timezone.utc).isoformat(),proposal_sha256=r.PROPOSAL_SHA,synthetic_only=True,real_143_label_fit_or_return_calculations=0,synthetic_tests=result.testsRun,source_hashes={str(p):sha(p)for p in r.source_files(r.ROOT)},input_hashes={str(proposal):sha(proposal)},implementation_hashes={str(p):sha(p)for p in paths},scope=['New model and target adapter inspected as specifications, never imported by independent reviewer.','Independent four-endpoint labels, shared maturity, weighted augmented least-squares Ridge, penalized local linear model.','Independent numerical predictions must also reproduce exact discrete ranks and ordered holdings, including sub-tolerance near ties.',
        'All full-data audit work explicitly deferred until registered producer completion and root authorization.','Synthetic calendar, joint cohort, December maturity, positive endpoint masks, zero-gap identity, unit scaling, pooled/local histories, future poisoning, all query and score prefixes, exact fallback and isolation.'],limitations=['No real market labels, model fits, ranks, orders or returns computed by this preregistration review.','Actual source closure, independent numerical parity and historical endpoint coverage remain mandatory preparation gates.'])
    assert result.wasSuccessful()
    out=r.ROOT/'work/monthly_paired_target143_prepare_prereview.json'
    assert not (r.ROOT/r.STUDY/'PLAN.json').exists(), 'Do not refresh prereview after registration'
    with out.open('w')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,receipt=str(out),sha256=sha(out),tests=result.testsRun)))

if __name__=='__main__':
    if sys.argv[1:]==['--write-prereview']:
        tests=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]);result=unittest.TextTestRunner(verbosity=2).run(tests)
        if result.wasSuccessful():prereview_receipt(result)
        else:sys.exit(1)
    else:unittest.main(verbosity=2)

