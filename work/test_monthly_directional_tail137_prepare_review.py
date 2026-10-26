"""Independent synthetic state/score checks. No real arrays or strategy returns."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_directional_tail137_prepare_review as r


def block(returns):
    a=np.ones((22,32));a[-1]=1+np.asarray(returns);return a


def fixture():
    shape=(2,5)
    baseline=np.array([[.16,.12,.08,.07,-.10],[.15,.13,.10,.09,-.10]])
    core=np.array([[.10,.08,.05,.04,-.10],[.10,.08,.06,.05,-.10]])
    a=dict(baseline=baseline,core=core,eligible=np.ones(shape,bool),core_weight=np.array([.5,.52]),
       state_weak_narrow=np.array([False,True]),state_calm10=np.array([True,False]),
       pre_score=baseline.copy(),effects=np.zeros(shape),forecast_available=np.zeros(shape,bool),
       rank_log_excess=np.zeros(shape),old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),
       model_available=np.zeros(shape,bool),raw_local_constant=np.zeros(shape),
       raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
    return a,dict(symbols=[list('ABCDE'),list('ABCDE')])


def source_fixture():
    dates=pd.bdate_range('2024-01-01',periods=50)
    stocks=['S%02d'%i for i in range(32)];etfs=['E%02d'%i for i in range(17)]
    prices=pd.DataFrame({s:100*np.exp(np.arange(50)*(.001+i*.0001)) for i,s in enumerate(stocks+etfs)},index=dates)
    meta=dict(dates=[str(dates[i].date())for i in(21,49)],symbols=[stocks+etfs,stocks+etfs],etfs=etfs)
    return prices,meta,dates


class DecimalTests(unittest.TestCase):
    def test_symmetric_half(self):
        q,d=r.tail_oracle(block([-.125]*16+[.125]*16));self.assertEqual(q,.5);self.assertEqual(d['positive_count'],16)
    def test_one_upper_outlier_exact(self):
        q,d=r.tail_oracle(block([0.]*31+[.5]));self.assertEqual(q,31/32);self.assertEqual(d['negative_count'],31)
    def test_one_lower_outlier_exact(self):
        q,_=r.tail_oracle(block([0.]*31+[-.5]));self.assertEqual(q,1/32)
    def test_reflection(self):
        v=np.array([-.25]*10+[.125]*21+[.5]);a,_=r.tail_oracle(block(v));b,_=r.tail_oracle(block(-v));self.assertAlmostEqual(a+b,1.,14)
    def test_common_shift(self):
        v=np.array([-.25]*10+[.125]*21+[.5]);a,_=r.tail_oracle(block(v));b,_=r.tail_oracle(block(v+.25));self.assertEqual(a,b)
    def test_positive_scale(self):
        v=np.array([-.25]*10+[.125]*21+[.5]);a,_=r.tail_oracle(block(v));b,_=r.tail_oracle(block(v*.5));self.assertEqual(a,b)
    def test_zero_dispersion_not_half(self):
        q,d=r.tail_oracle(block([.125]*32));self.assertTrue(np.isnan(q));self.assertEqual(d['reason'],'zero_dispersion');self.assertEqual(d['zero_count'],32)
    def test_small_nonzero_is_not_snapped(self):
        a=np.ones((22,32));a[-1,-1]=np.nextafter(1.,2.);q,d=r.tail_oracle(a);self.assertEqual(q,31/32);self.assertTrue(d['available'])
    def test_centered_zero_counts(self):
        q,d=r.tail_oracle(block([-.25]+[0.]*30+[.25]));self.assertEqual(q,.5);self.assertEqual(d['zero_count'],30)
    def test_all_interior_values_required(self):
        for value in (np.nan,np.inf,0.,-1.):
            a=block(np.linspace(-.2,.3,32));a[10,7]=value;q,d=r.tail_oracle(a);self.assertTrue(np.isnan(q));self.assertEqual(d['reason'],'invalid_price')
    def test_missing_gates_before_extreme_ratio(self):
        a=np.ones((22,32));a[0,0]=np.nextafter(0.,1.);a[-1,0]=np.finfo(float).max;a[10,31]=np.nan
        q,d=r.tail_oracle(a);self.assertTrue(np.isnan(q));self.assertEqual(d['reason'],'invalid_price')
    def test_extreme_ratio_rejected(self):
        a=np.ones((22,32));a[0,0]=1e-300;a[-1,0]=1e300
        with self.assertRaises(ValueError):r.tail_oracle(a)
    def test_shapes_types(self):
        for a in (np.ones((21,32)),np.ones((22,31)),np.ones((22,32),dtype=bool),np.full((22,32),'1')):
            with self.assertRaises(ValueError):r.tail_oracle(a)
    def test_permutation(self):
        a=block(np.linspace(-.25,.5,32)**3);b=a[:,::-1];self.assertEqual(r.tail_oracle(a)[0],r.tail_oracle(b)[0])


class WindowTests(unittest.TestCase):
    def test_exact22_dates_and32_order(self):
        p,m,s=source_fixture();z=r.state_oracle(p,m,s);d=z['audit']['months'][0]
        self.assertEqual(d['window_dates'],[str(t.date())for t in s[:22]]);self.assertEqual(d['stock_symbols'],m['symbols'][0][:32]);self.assertEqual(z['audit']['new_model_fits'],0)
    def test_missing_session_not_compressed(self):
        p,m,s=source_fixture();z=r.state_oracle(p.drop(s[10]),m,s);self.assertTrue(np.isnan(z['q'][0]));self.assertTrue(np.isfinite(z['q'][1]))
    def test_missing_stock_column(self):
        p,m,s=source_fixture();z=r.state_oracle(p.drop(columns='S04'),m,s);self.assertTrue(np.isnan(z['q']).all());self.assertEqual(z['audit']['months'][0]['reason'],'missing_stock_columns')
    def test_etf_values_irrelevant(self):
        p,m,s=source_fixture();a=r.state_oracle(p,m,s);p.loc[:,m['etfs']]=np.nan;b=r.state_oracle(p,m,s);r.compare(a['audit'],b['audit']);np.testing.assert_array_equal(a['q'],b['q'])
    def test_no_eligibility_filter(self):
        p,m,s=source_fixture();m['eligible']=np.zeros((2,49),bool);a=r.state_oracle(p,m,s);self.assertTrue(np.isfinite(a['q']).all())
    def test_future_poison_prefix(self):
        p,m,s=source_fixture();before=r.state_oracle(p,m,s);p.loc[s[22]:,:]=np.nan;after=r.state_oracle(p,m,s)
        r.compare(before['audit']['months'][0],after['audit']['months'][0]);self.assertEqual(before['q'][0],after['q'][0]);self.assertTrue(np.isnan(after['q'][1]))
    def test_future_text_is_never_interpreted(self):
        p,m,s=source_fixture();m=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);original=r.state_oracle(p,m,s)
        p=p.astype(object);p.loc[s[22]:,:]='unobserved future';actual=r.state_oracle(p,m,s)
        r.compare(original['audit'],actual['audit']);np.testing.assert_array_equal(original['q'],actual['q'])
    def test_prefix_truncates_prices(self):
        p,m,s=source_fixture();full=r.state_oracle(p,m,s);sub=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);cut=r.state_oracle(p.loc[:m['dates'][0]],sub,s)
        r.compare(cut['audit']['months'],full['audit']['months'][:1]);np.testing.assert_array_equal(cut['q'],full['q'][:1])
    def test_short_history(self):
        p,m,s=source_fixture();m=dict(m,dates=[str(s[20].date())],symbols=m['symbols'][:1]);z=r.state_oracle(p,m,s);self.assertEqual(z['audit']['months'][0]['reason'],'insufficient_history');self.assertEqual(z['audit']['months'][0]['window_dates'],[])
    def test_duplicate_calendar_rejected(self):
        p,m,s=source_fixture()
        with self.assertRaises(ValueError):r.state_oracle(p,m,list(s)+[s[-1]])
    def test_nonofficial_historical_price_rejected(self):
        p,m,s=source_fixture();extra=pd.DataFrame([p.iloc[0]],index=[pd.Timestamp('2024-01-06')]);p=pd.concat([p,extra]).sort_index()
        with self.assertRaises(ValueError):r.state_oracle(p,m,s)
    def test_unknown_signal_and_duplicate_symbols(self):
        p,m,s=source_fixture();m['symbols'][0][-1]='S00'
        with self.assertRaises(ValueError):r.state_oracle(p,m,s)
    def test_empty_prefix(self):
        p,m,s=source_fixture();m=dict(m,dates=[],symbols=[]);z=r.state_oracle(p.iloc[:0],m,s);self.assertEqual(len(z['q']),0);self.assertEqual(z['audit']['months'],[])


class ScoreTests(unittest.TestCase):
    def test_missing_exact_baseline(self):
        a,m=fixture();rank=np.full((2,5),np.nan);ss,cc,dd=r.score_oracle(a,m,rank,rank,[np.nan,np.nan])
        for n in r.NAMES[1:]:self.assertEqual(ss[n].tobytes(),ss[r.NAMES[0]].tobytes());self.assertEqual(cc[n],cc[r.NAMES[0]]);self.assertEqual(np.count_nonzero(dd[n]),0)
    def test_half_dose_not_full_replacement(self):
        a,m=fixture();u=np.full((2,5),.3);v=np.full((2,5),.1);_,_,dd=r.score_oracle(a,m,u,v,[.75,.25]);cw=a['core_weight'][:,None]
        np.testing.assert_allclose(dd[r.NAMES[2]],.5*(cw*(np.array([.75,.25])[:,None]*u+np.array([.25,.75])[:,None]*v)-a['core']),rtol=0,atol=1e-16)
    def test_half_matches_constant(self):
        a,m=fixture();u=np.full((2,5),.3);v=np.full((2,5),-.1);ss,cc,_=r.score_oracle(a,m,u,v,[.5,.5]);self.assertEqual(ss[r.NAMES[1]].tobytes(),ss[r.NAMES[2]].tobytes());self.assertEqual(cc[r.NAMES[1]],cc[r.NAMES[2]])
    def test_zero_is_valid_risk_state(self):
        a,m=fixture();u=np.full((2,5),.3);v=np.full((2,5),.1);_,_,d=r.score_oracle(a,m,u,v,[0.,0.]);np.testing.assert_allclose(d[r.NAMES[2]],.5*(a['core_weight'][:,None]*v-a['core']),rtol=0,atol=1e-16)
    def test_one_is_valid_raw_state(self):
        a,m=fixture();u=np.full((2,5),.3);v=np.full((2,5),.1);_,_,d=r.score_oracle(a,m,u,v,[1.,1.]);np.testing.assert_allclose(d[r.NAMES[2]],.5*(a['core_weight'][:,None]*u-a['core']),rtol=0,atol=1e-16)
    def test_zero_correction_exact_baseline(self):
        a,m=fixture();rank=a['core']/a['core_weight'][:,None];ss,cc,d=r.score_oracle(a,m,rank,rank,[.5,.5]);self.assertEqual(np.count_nonzero(d[r.NAMES[2]]),0);self.assertEqual(ss[r.NAMES[0]].tobytes(),ss[r.NAMES[2]].tobytes())
    def test_prefix_ignores_poisoned_future(self):
        a,m=fixture();u=np.full((2,5),.1);v=np.full((2,5),-.1);expected=r.score_oracle(a,m,u,v,[.3,.2],1);u[1]=np.inf;actual=r.score_oracle(a,m,u,v,[.3,np.inf],1)
        for n in r.NAMES:self.assertEqual(expected[0][n].tobytes(),actual[0][n].tobytes());self.assertEqual(expected[1][n],actual[1][n])
    def test_exactly_one_episode_each_arm(self):
        a,m=fixture();h=r.old.original_helper();rank=np.zeros((2,5))
        with patch.object(r.old,'original_helper',return_value=h),patch.object(h,'episode',wraps=h.episode)as spy:
            r.score_oracle(a,m,rank,rank,[.25,.75]);self.assertEqual(spy.call_count,3)
    def test_reject_bad_state_or_rank(self):
        a,m=fixture();u=np.full((2,5),.1)
        for q in ([-.1,.5],[1.1,.5],[np.inf,.5],[True,False]):
            with self.assertRaises(ValueError):r.score_oracle(a,m,u,u,q)
        with self.assertRaises(ValueError):r.score_oracle(a,m,u*6,u,[.2,.5])
    def test_original_arrays_not_mutated(self):
        a,m=fixture();before={k:v.tobytes()for k,v in a.items()};u=np.full((2,5),.1);r.score_oracle(a,m,u,u,[.2,.8]);self.assertEqual(before,{k:v.tobytes()for k,v in a.items()})


class ProducerParityTests(unittest.TestCase):
    def test_random_and_near_tie_math_parity(self):
        import monthly_directional_tail137_factor as p
        rng=np.random.default_rng(137)
        samples=[block(rng.uniform(-.9,2,32))for _ in range(12)]
        a=np.ones((22,32));a[-1]=1.+np.arange(32)*np.spacing(1.);samples.append(a)
        samples.extend([block([0.]*31+[.5]),block([0.]*32)])
        for a in samples:
            q,d=r.tail_oracle(a);actual,detail=p.tail_balance(a);r.compare(detail,d)
            np.testing.assert_allclose(actual,q,atol=1e-12,rtol=0,equal_nan=True)
    def test_full_window_schema_parity(self):
        import monthly_directional_tail137_factor as producer
        p,m,s=source_fixture()
        for missing in (False,True):
            if missing:p.iloc[10,7]=np.nan
            a=r.state_oracle(p,m,s);b=producer.build_features(p,m,s);r.compare(b['audit'],a['audit']);np.testing.assert_allclose(a['q'],b['q'],atol=1e-12,rtol=0,equal_nan=True)
    def test_schema_and_discrete_changes_rejected(self):
        a=dict(available=True,count=32,q=.5)
        for b in (dict(a,extra=1),dict(a,count=31),dict(a,q=.50001),dict(a,available=1)):
            with self.assertRaises(ValueError):r.compare(b,a)
    def test_digest_mismatch(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'a';p.write_text('x');reader=r.old.Reader(d)
            with self.assertRaises(ValueError):reader.verify({str(p):'0'*64})


def full_score_fixture():
    a,_=fixture();a={k:np.tile(v,(61,10))[:,:49] if v.ndim==2 else np.tile(v,61) for k,v in a.items()}
    raw=np.tile(np.linspace(-.48,.48,49),(122,1));risk=raw[:,::-1].copy()
    cw=a['core_weight'][:,None];a['core']=cw*np.where(a['state_calm10'][:,None],raw,risk)
    a['baseline']=a['core']+np.tile(np.sin(np.arange(49))*.015,(122,1));a['pre_score']=a['baseline'].copy()
    a['effects']=np.tile(np.cos(np.arange(49))*.05,(122,1))
    m=dict(symbols=[['S%02d'%j for j in range(49)]for i in range(122)])
    q=np.tile([.2,.5,.8,np.nan,0.,1.],21)[:122]
    a.update(directional_q=q,directional_raw_rank=raw,directional_risk_rank=risk)
    return a,m,raw,risk,q


class CoreIntegrationTests(unittest.TestCase):
    def backend(self):
        h=r.old.original_helper()
        class Backend:
            root=r.ROOT
            @staticmethod
            def baseline(a,m,n,policy):
                x=h.original_prefix(a,m,n);c,s,_=h.episode(x['local'],x,policy=policy);return c,s
            @staticmethod
            def pre(a,n):
                m=dict(symbols=[['S%02d'%j for j in range(49)]for i in range(122)])
                x=h.original_prefix(a,m,n);full=np.zeros_like(a['baseline']);full[:n]=x['local'];return full
            @staticmethod
            def selector(score,eligible,names,effects,original,policy):
                x=dict(baseline=original['baseline'],core=original['core'],cw=original['core_weight'],
                       support=(original['baseline']-original['core'])/(1-original['core_weight']),
                       eligible=eligible,names=names,effects=effects,calm=original['state_calm10'],weak=original['state_weak_narrow'])
                c,s,_=h.episode(score,x,policy=policy);return c,s
        return Backend()
    def test_all3_scores_and_choices_match_producer(self):
        import monthly_directional_tail137_runner as producer
        a,m,u,v,q=full_score_fixture();ss,cc,_=r.score_oracle(a,m,u,v,q);backend=self.backend()
        for name in r.NAMES:
            c,s=producer.choose(name,a,m,122,backend,r.old.original_helper().POLICY)
            np.testing.assert_allclose(s,ss[name],atol=1e-12,rtol=0);self.assertEqual(c,cc[name])
    def test_producer_and_independent_prefixes(self):
        import monthly_directional_tail137_runner as producer
        a,m,u,v,q=full_score_fixture();backend=self.backend()
        for n in (0,1,36,60,76,88):
            ss,cc,_=r.score_oracle(a,m,u,v,q,n)
            for name in r.NAMES:
                c,s=producer.choose(name,a,m,n,backend,r.old.original_helper().POLICY)
                np.testing.assert_allclose(s,ss[name],atol=1e-12,rtol=0);self.assertEqual(c,cc[name])
    def test_available_state_changes_etf_core_too(self):
        a,m,u,v,q=full_score_fixture();_,_,d=r.score_oracle(a,m,u,v,q)
        self.assertGreater(np.count_nonzero(d[r.NAMES[2]][0,32:]),0)
    def test_independent_core_math_does_not_call_producer(self):
        import monthly_directional_tail137_runner as producer
        a,m,u,v,q=full_score_fixture()
        with patch.object(producer,'choose',side_effect=AssertionError('not an oracle')):
            ss,_,_=r.score_oracle(a,m,u,v,q);self.assertEqual(len(ss),3)
    def test_receipt_gate_keys_match_producer(self):
        import monthly_directional_tail137_runner as producer
        self.assertEqual(r.CHECKS,producer.CHECKS);self.assertEqual(r.INPUTS,producer.INPUTS);self.assertEqual(r.NAMES,producer.NAMES);self.assertEqual(r.CACHE_FILES,producer.CACHE_FILES)


if __name__=='__main__':unittest.main()
