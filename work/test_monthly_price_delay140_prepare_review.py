"""Synthetic independent price-delay/selection checks; no real cache execution."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_price_delay140_prepare_review as r


def price_vectors(lag=2):
    x=np.random.default_rng(140).normal(0,.015,56)
    y=.003+.7*x[4-lag:56-lag]+.001*np.cos(np.arange(52)*.9)
    return 100*np.r_[1.,np.cumprod(1.+y)],100*np.r_[1.,np.cumprod(1.+x)]


class RegressionTests(unittest.TestCase):
    def test_lag_response_exceeds_current_fit(self):
        y,x=price_vectors();a=r.qr_oracle(y,x)
        self.assertTrue(a['available']);self.assertGreater(a['full_r2'],.98);self.assertGreater(a['raw_delay'],.9)
    def test_contemporaneous_response_zero_delay(self):
        _,x=price_vectors(0);returns=x[1:]/x[:-1]-1.;y=100*np.r_[1.,np.cumprod(1.+.003+.7*returns[4:])];a=r.qr_oracle(y,x);self.assertLess(a['raw_delay'],1e-12)
    def test_price_scale_invariance(self):
        y,x=price_vectors();a=r.qr_oracle(y,x);b=r.qr_oracle(y*100.,x*.01)
        for key in ('raw_nonfit','raw_delay'):self.assertAlmostEqual(a[key],b[key],12)
    def test_constant_stock_rejected(self):
        _,x=price_vectors();a=r.qr_oracle(np.full(53,100.),x);self.assertEqual(a['reason'],'zero_response_variance')
    def test_constant_market_rejected(self):
        y,_=price_vectors();a=r.qr_oracle(y,np.full(57,100.));self.assertEqual(a['reason'],'zero_market_variance')
    def test_nonboolean_positive_prices(self):
        y,x=price_vectors()
        for bad in (np.full(53,True),np.full(53,'1'),np.full(53,np.nan),np.full(53,np.inf),np.zeros(53),y[:-1]):
            with self.subTest(bad=str(bad)[:20]),self.assertRaises(ValueError):r.qr_oracle(bad,x)
    def test_rankdeficient_market(self):
        x=np.array([.01,-.01]*28);mp=100*np.r_[1.,np.cumprod(1+x)];y,_=price_vectors();a=r.qr_oracle(y,mp)
        self.assertFalse(a['available']);self.assertEqual(a['reason'],'rank_deficient_full')
    def test_qr_matches_lstsq_without_calling_it(self):
        import monthly_price_delay140_factor as p
        y,x=price_vectors();wanted=p.regression_delay(y,x)
        with patch.object(np.linalg,'lstsq',side_effect=AssertionError('Producer solver forbidden')):a=r.qr_oracle(y,x)
        r.compare(a,wanted)
    def test_contemporaneous_control_not_identical_to_delay(self):
        y,x=price_vectors();a=r.qr_oracle(y,x);self.assertNotEqual(a['raw_nonfit'],a['raw_delay'])
    def test_independent_module_has_no_new_producer_imports(self):
        text=Path(r.__file__).read_text();tree=r.ast.parse(text)
        for node in r.ast.walk(tree):
            if isinstance(node,r.ast.Import):self.assertFalse(any('price_delay140_factor' in a.name for a in node.names))
            if isinstance(node,r.ast.ImportFrom):self.assertNotIn('price_delay140_factor',node.module or '')


class RankTests(unittest.TestCase):
    def test_full_range(self):np.testing.assert_array_equal(r.rank_oracle([1.,2.,3.]),[-.5,0.,.5])
    def test_exact_ties(self):np.testing.assert_array_equal(r.rank_oracle([1.,1.,3.]),[-.25,-.25,.5])
    def test_allties_exact_zero(self):np.testing.assert_array_equal(r.rank_oracle([0.]*32),np.zeros(32))
    def test_fewer_than_two_missing(self):self.assertTrue(np.isnan(r.rank_oracle([np.nan,2.])).all())
    def test_raw_zero_is_regular_rank(self):self.assertEqual(r.rank_oracle([0.,1.])[0],-.5)


def panel_fixture():
    days=pd.bdate_range('2018-01-01',periods=430);stocks=['S%02d'%i for i in range(32)];etfs=['E%02d'%i for i in range(17)]
    rng=np.random.default_rng(140);z=rng.normal(.0004,.01,len(days));prices={'SPY':100*np.cumprod(1+z)}
    for i,s in enumerate(stocks):prices[s]=100*np.cumprod(1+.6*np.roll(z,i%5)+rng.normal(0,.002,len(days)))
    for e in etfs:prices[e]=np.full(len(days),100.)
    meta=dict(dates=[str(days[k].date())for k in(359,429)],symbols=[stocks+etfs,stocks+etfs],etfs=etfs)
    return dict(adj=pd.DataFrame(prices,index=days)),meta,days


class EndpointTests(unittest.TestCase):
    def test_exact57_market53_stock_endpoints(self):
        p,m,s=panel_fixture();a=r.cells_oracle(p,m,s)[0]
        self.assertEqual(a['cutoff'],str(s[359-42].date()));self.assertEqual(len(a['actual_week_ends']),57)
        self.assertEqual(a['stock_window_start'],a['actual_week_ends'][-53]);self.assertLessEqual(a['nominal_week_ends'][-1],a['cutoff'])
    def test_friday_holiday_uses_thursday(self):
        s=pd.bdate_range('2018-01-01',periods=430);holiday=pd.Timestamp('2019-03-29');s=s[s!=holiday]
        target=s[s.get_loc(pd.Timestamp('2019-04-01'))+42];cut,n,w=r.endpoint_oracle(s,target)
        self.assertEqual(w[list(n).index(holiday)],pd.Timestamp('2019-03-28'))
    def test_midweek_cutoff_excludes_current_week(self):
        s=pd.bdate_range('2018-01-01',periods=430);cut=pd.Timestamp('2019-03-27');target=s[s.get_loc(cut)+42]
        cutoff,n,w=r.endpoint_oracle(s,target);self.assertEqual(n[-1],pd.Timestamp('2019-03-22'))
    def test_no_per_symbol_endpoint_fallback(self):
        p,m,s=panel_fixture();_,_,w=r.endpoint_oracle(s,m['dates'][0]);p['adj'].loc[w[-1],'S00']=np.nan
        self.assertEqual(r.cells_oracle(p,m,s)[0]['reason'],'invalid_stock_path')
    def test_nonendpoint_missing_does_not_invalidate(self):
        p,m,s=panel_fixture();_,_,w=r.endpoint_oracle(s,m['dates'][0]);d=s[(s>w[-53])&(s<w[-1])&(~s.isin(w))][0];p['adj'].loc[d,'S00']=np.nan
        self.assertTrue(r.cells_oracle(p,m,s)[0]['available'])
    def test_after_cutoff_poison_ignored(self):
        p,m,s=panel_fixture();expected=r.cells_oracle(p,m,s)[:49];sub=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1])
        p['adj']=p['adj'].astype(object);p['adj'].loc[pd.Timestamp(expected[0]['cutoff'])+pd.Timedelta(days=1):]='not used'
        r.compare(r.cells_oracle(p,sub,s),expected)
    def test_missing_market_disables_both(self):
        p,m,s=panel_fixture();_,_,w=r.endpoint_oracle(s,m['dates'][0]);p['adj'].loc[w[0],'SPY']=np.nan;a=r.features_oracle(p,m,s)
        self.assertFalse(a['arrays']['common_available'][0].any());self.assertTrue(np.isnan(a['arrays']['rank_delay'][0]).all())
    def test_etf_always_missing(self):
        p,m,s=panel_fixture();a=r.features_oracle(p,m,s);self.assertTrue(np.isnan(a['arrays']['rank_delay'][:,32:]).all())
    def test_missing_first4_stock_endpoints_not_required(self):
        p,m,s=panel_fixture();_,_,w=r.endpoint_oracle(s,m['dates'][0]);p['adj'].loc[w[:4],'S00']=np.nan;self.assertTrue(r.cells_oracle(p,m,s)[0]['available'])
    def test_prefix_is_exact(self):
        p,m,s=panel_fixture();full=r.features_oracle(p,m,s);sub=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);cut=r.features_oracle(dict(adj=p['adj'].loc[:m['dates'][0]]),sub,s)
        r.compare(cut['audit']['cells'],full['audit']['cells'][:49])
        for k in r.EXTRA:np.testing.assert_array_equal(cut['arrays'][k],full['arrays'][k][:1])
    def test_empty_prefix(self):
        p,m,s=panel_fixture();a=r.features_oracle(dict(adj=p['adj'].iloc[:0]),dict(m,dates=[],symbols=[]),s);self.assertEqual(a['audit']['cells'],[])
    def test_fewer_than_two_usable(self):
        p,m,s=panel_fixture();p['adj'].loc[:,['S%02d'%i for i in range(1,32)]]=np.nan;a=r.features_oracle(p,m,s)
        self.assertFalse(a['arrays']['common_available'].any());self.assertTrue(a['audit']['cells'][0]['regression_available']);self.assertEqual(a['audit']['cells'][0]['reason'],'insufficient_cross_section')
    def test_all_arrays_and_cells_match_producer(self):
        import monthly_price_delay140_factor as producer
        p,m,s=panel_fixture();a=producer.build_features(p,m,s);b=r.features_oracle(p,m,s,a['arrays']);r.compare(a['audit'],b['audit'])
        for key in r.EXTRA:
            if key in r.EXTRA[:2]:np.testing.assert_allclose(a['arrays'][key],b['arrays'][key],atol=r.ATOL,rtol=0,equal_nan=True)
            else:np.testing.assert_array_equal(a['arrays'][key],b['arrays'][key])
    def test_changed_raw_cannot_feed_rank(self):
        p,m,s=panel_fixture();a=r.features_oracle(p,m,s);a['arrays']['raw_delay'][0,0]+=.01
        with self.assertRaises(AssertionError):r.features_oracle(p,m,s,a['arrays'])
    def test_nonofficial_price_rows_rejected(self):
        p,m,s=panel_fixture();p['adj'].loc[pd.Timestamp('2018-01-06')]=100.;p['adj'].sort_index(inplace=True)
        with self.assertRaisesRegex(ValueError,'Official source'):r.features_oracle(p,m,s)
    def test_eligibility_does_not_change_peer_ranking(self):
        p,m,s=panel_fixture();a=r.features_oracle(p,m,s);m['eligible']=np.zeros((2,49),bool);b=r.features_oracle(p,m,s);r.compare(a['audit'],b['audit'])


def score_fixture():
    shape=(2,5);baseline=np.array([[.16,.12,.08,.07,-.10],[.15,.13,.10,.09,-.10]])
    core=np.array([[.10,.08,.05,.04,-.10],[.10,.08,.06,.05,-.10]])
    a=dict(baseline=baseline,core=core,eligible=np.ones(shape,bool),core_weight=np.array([.5,.52]),
       state_weak_narrow=np.array([False,True]),state_calm10=np.array([True,False]),
       pre_score=baseline.copy(),effects=np.zeros(shape),forecast_available=np.zeros(shape,bool),
       rank_log_excess=np.zeros(shape),old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),
       model_available=np.zeros(shape,bool),raw_local_constant=np.zeros(shape),
       raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
    return a,dict(symbols=[list('ABCDE'),list('ABCDE')],etfs=['E'])


class ScoreTests(unittest.TestCase):
    def test_common_missing_exact_baseline(self):
        a,m=score_fixture();hand=np.full((2,5),np.nan);s,c,d=r.score_oracle(a,m,hand,hand)
        for name in r.NAMES[1:]:self.assertEqual(s[name].tobytes(),s[r.NAMES[0]].tobytes());self.assertEqual(c[name],c[r.NAMES[0]])
    def test_zero_hand_exact_baseline(self):
        a,m=score_fixture();hand=np.zeros((2,5));hand[:,-1]=np.nan;s,c,d=r.score_oracle(a,m,hand,hand)
        self.assertEqual(np.count_nonzero(d[r.NAMES[2]]),0);self.assertEqual(s[r.NAMES[2]].tobytes(),s[r.NAMES[0]].tobytes())
    def test_bounded_corrected_support(self):
        a,m=score_fixture();hand=np.full((2,5),.5);hand[:,-1]=np.nan;_,_,d=r.score_oracle(a,m,hand,hand)
        support=(a['baseline']-a['core'])/(1-a['core_weight'][:,None]);expected=.025*(np.minimum(.5,support+.5)-support);expected[:,-1]=0.
        np.testing.assert_allclose(d[r.NAMES[2]],expected,rtol=0,atol=1e-16)
    def test_ineligible_keeps_delta_zero(self):
        a,m=score_fixture();a['eligible'][0,0]=False;hand=np.full((2,5),.4);hand[:,-1]=np.nan;_,_,d=r.score_oracle(a,m,hand,hand);self.assertEqual(d[r.NAMES[2]][0,0],0.)
    def test_etf_finite_rejected(self):
        a,m=score_fixture();hand=np.full((2,5),.1)
        with self.assertRaises(ValueError):r.score_oracle(a,m,hand,hand)
    def test_unmatched_masks_rejected(self):
        a,m=score_fixture();hand=np.full((2,5),np.nan);other=hand.copy();other[0,0]=.2
        with self.assertRaises(ValueError):r.score_oracle(a,m,hand,other)
    def test_future_not_interpreted(self):
        a,m=score_fixture();hand=np.zeros((2,5));hand[:,-1]=np.nan;expected=r.score_oracle(a,m,hand,hand,1);hand[1]=np.inf;actual=r.score_oracle(a,m,hand,hand,1)
        for name in r.NAMES:self.assertEqual(expected[0][name].tobytes(),actual[0][name].tobytes());self.assertEqual(expected[1][name],actual[1][name])
    def test_one_episode_per_arm(self):
        a,m=score_fixture();hand=np.full((2,5),np.nan);h=r.old.original_helper()
        with patch.object(r.old,'original_helper',return_value=h),patch.object(h,'episode',wraps=h.episode)as spy:
            r.score_oracle(a,m,hand,hand);self.assertEqual(spy.call_count,3)
    def test_source_arrays_unchanged(self):
        a,m=score_fixture();before={k:v.tobytes()for k,v in a.items()};hand=np.full((2,5),np.nan);r.score_oracle(a,m,hand,hand);self.assertEqual(before,{k:v.tobytes()for k,v in a.items()})


def full_score_fixture():
    a,_=score_fixture();a={k:np.tile(v,(61,10))[:,:49] if v.ndim==2 else np.tile(v,61) for k,v in a.items()}
    names=['S%02d'%j for j in range(32)]+['E%02d'%j for j in range(17)]
    m=dict(symbols=[names[:] for _ in range(122)],etfs=names[32:])
    raw=np.tile(np.linspace(-.48,.48,49),(122,1));cw=a['core_weight'][:,None];a['core']=cw*raw
    a['baseline']=a['core']+np.tile(np.sin(np.arange(49))*.015,(122,1));a['pre_score']=a['baseline'].copy()
    a['effects']=np.tile(np.cos(np.arange(49))*.05,(122,1))
    beta=np.tile(r.rank_oracle(np.linspace(-1.,1.,32)),(122,1));rho=beta[:,::-1].copy()
    beta=np.c_[beta,np.full((122,17),np.nan)];rho=np.c_[rho,np.full((122,17),np.nan)]
    beta[5,:]=np.nan;rho[5,:]=np.nan;a.update(rank_nonfit=beta,rank_delay=rho)
    return a,m,beta,rho


class FullScoreIntegrationTests(unittest.TestCase):
    def backend(self,m):
        import monthly_price_delay140_runner as producer
        h=r.old.original_helper()
        class Backend:
            adapter=producer.module('independent140_test_adapter',r.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
            @staticmethod
            def baseline(a,meta,n,policy):
                x=h.original_prefix(a,meta,n);c,s,_=h.episode(x['local'],x,policy=policy);return c,s
            @staticmethod
            def pre(a,n):
                x=h.original_prefix(a,m,n);full=np.zeros_like(a['baseline']);full[:n]=x['local'];return full
            @staticmethod
            def selector(score,eligible,names,effects,original,policy):
                x=dict(baseline=original['baseline'],core=original['core'],cw=original['core_weight'],
                    support=(original['baseline']-original['core'])/(1-original['core_weight']),
                    eligible=eligible,names=names,effects=effects,calm=original['state_calm10'],weak=original['state_weak_narrow'])
                c,s,_=h.episode(score,x,policy=policy);return c,s
        return Backend()
    def test_all366_synthetic_choices_and_three_scores(self):
        import monthly_price_delay140_runner as producer
        a,m,beta,rho=full_score_fixture();ss,cc,_=r.score_oracle(a,m,beta,rho);backend=self.backend(m)
        for name in r.NAMES:
            c,s=producer.choose(name,a,m,122,backend,r.old.original_helper().POLICY)
            np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_prefixes_with_original_episode(self):
        import monthly_price_delay140_runner as producer
        a,m,beta,rho=full_score_fixture();backend=self.backend(m)
        for n in (0,1,36,60,76,88):
            ss,cc,_=r.score_oracle(a,m,beta,rho,n)
            for name in r.NAMES:
                c,s=producer.choose(name,a,m,n,backend,r.old.original_helper().POLICY)
                np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_interface_exact(self):
        import monthly_price_delay140_runner as producer
        for key in ('INPUTS','CACHE_FILES','NAMES','CHECKS','EXTRA'):self.assertEqual(getattr(r,key),getattr(producer,key))
    def test_independent_scores_do_not_call_new_producer(self):
        import monthly_price_delay140_runner as producer
        a,m,beta,rho=full_score_fixture()
        with patch.object(producer,'choose',side_effect=AssertionError('not an oracle')):
            scores,choices,_=r.score_oracle(a,m,beta,rho);self.assertEqual(len(choices[r.NAMES[2]]),122)


if __name__=='__main__':unittest.main()
