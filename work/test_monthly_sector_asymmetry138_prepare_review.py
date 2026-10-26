"""Synthetic-only independent conditional-sector tests; no real inputs."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_sector_asymmetry138_prepare_review as r


def return_vectors():
    x=np.r_[np.linspace(.001,.02,126),-np.linspace(.001,.02,126)]
    y=np.r_[2*x[:126]+.003,x[126:]-.004]
    return y,x


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


class MomentTests(unittest.TestCase):
    def test_beta_gap_not_correlation_gap(self):
        y,x=return_vectors();a=r.moments_oracle(y,x);self.assertAlmostEqual(a['raw_beta_gap'],1.,13);self.assertAlmostEqual(a['raw_rho_gap'],0.,13)
    def test_separate_intercepts(self):
        y,x=return_vectors();y[:126]+=1.;y[126:]-=2.;a=r.moments_oracle(y,x);self.assertAlmostEqual(a['beta_up'],2.,11);self.assertAlmostEqual(a['beta_down'],1.,11)
    def test_stock_scale_changes_beta_only(self):
        y,x=return_vectors();a=r.moments_oracle(y,x);b=r.moments_oracle(3*y,x);self.assertAlmostEqual(b['raw_beta_gap'],3*a['raw_beta_gap'],13);self.assertAlmostEqual(b['raw_rho_gap'],a['raw_rho_gap'],13)
    def test_benchmark_scale_changes_beta_only(self):
        y,x=return_vectors();a=r.moments_oracle(y,x);b=r.moments_oracle(y,3*x);self.assertAlmostEqual(b['raw_beta_gap'],a['raw_beta_gap']/3,13);self.assertAlmostEqual(b['raw_rho_gap'],a['raw_rho_gap'],13)
    def test_group_stock_constant_blocks_beta_too(self):
        y,x=return_vectors();y[:126]=.01;self.assertEqual(r.moments_oracle(y,x)['reason'],'constant_group')
    def test_sector_constant_group(self):
        y,x=return_vectors();x[:126]=.01;self.assertFalse(r.moments_oracle(y,x)['available'])
    def test_exact40_groups_suffice(self):
        y,x=return_vectors();x[40:126]=0.;x[166:]=0.;a=r.moments_oracle(y,x);self.assertTrue(a['available']);self.assertEqual(a['zero_count'],172)
    def test_39_fails(self):
        y,x=return_vectors();x[39:126]=0.;self.assertEqual(r.moments_oracle(y,x)['reason'],'insufficient_group')
    def test_zero_sector_days_excluded(self):
        y,x=return_vectors();x[40:126]=0.;a=r.moments_oracle(y,x);y[40:126]=1e50;b=r.moments_oracle(y,x);self.assertEqual(a,b)
    def test_zero_covariance_is_defined(self):
        g=r.group_moments([1.,-1.,-1.,1.],[1.,2.,3.,4.]);self.assertEqual(g,dict(beta=0.,rho=0.))
    def test_stock_reflection(self):
        y,x=return_vectors();a=r.moments_oracle(y,x);b=r.moments_oracle(-y,x);self.assertAlmostEqual(a['raw_beta_gap'],-b['raw_beta_gap'],13)
    def test_very_small_nonconstant_not_missing(self):
        y,x=return_vectors();a=r.moments_oracle(y*1e-100,x*1e-100);self.assertTrue(a['available']);self.assertAlmostEqual(a['raw_beta_gap'],1.,13)
    def test_invalid_shape_and_nonfinite(self):
        y,x=return_vectors()
        for z in (x[:-1],np.full(252,np.inf),np.full(252,np.nan),np.full(252,True)):
            with self.assertRaises(ValueError):r.moments_oracle(y,z)


class RankMappingTests(unittest.TestCase):
    def test_average_percentile_ties(self):
        a=r.rank_oracle([1.,1.,3.,np.nan]);np.testing.assert_array_equal(a[:3],[0.,0.,.5]);self.assertTrue(np.isnan(a[-1]))
    def test_raw_zero_has_ordinary_rank(self):
        a=r.rank_oracle([0.,1.,2.]);self.assertNotEqual(a[0],0.);self.assertAlmostEqual(a[0],1/3-.5)
    def test_all_missing(self):self.assertTrue(np.isnan(r.rank_oracle([np.nan]*4)).all())
    def test_single_available_rank_is_half(self):self.assertEqual(r.rank_oracle([np.nan,2.])[1],.5)
    def test_original_mapping_etf_precedence_and_foreign(self):
        s=pd.DataFrame({'ASML':['Healthcare'],'IBM':['Technology'],'X':['Unmapped']},index=[pd.Timestamp('2020-01-31')])
        self.assertEqual(r.benchmark_map('2020-01-31',['ASML','IBM','X','UNKNOWN'],s,[]),dict(ASML='XLK',IBM='XLK',X=None,UNKNOWN=None))
        self.assertIsNone(r.benchmark_map('2020-01-31',['ASML'],s,['ASML'])['ASML'])
    def test_missing_mapping_date_does_not_backfill(self):
        s=pd.DataFrame({'IBM':['Technology']},index=[pd.Timestamp('2020-01-30')]);self.assertIsNone(r.benchmark_map('2020-01-31',['IBM'],s,[])['IBM'])
    def test_original_literal_map_bound(self):
        r.verify_mapping_source(r.old.Reader(r.ROOT))


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


def panel_fixture():
    days=pd.bdate_range('2021-01-01',periods=340);stocks=['S%02d'%i for i in range(32)];etfs=['E%02d'%i for i in range(17)]
    sector=.007*np.sin(np.arange(339)*1.3)+.0001
    prices={'XLK':100*np.exp(np.r_[0.,np.cumsum(sector)])}
    for j,name in enumerate(stocks):
        z=sector*(.5+j*.03)+.002*np.cos(np.arange(339)*(.3+j*.02))
        prices[name]=100*np.exp(np.r_[0.,np.cumsum(z)])
    for name in etfs:prices[name]=np.full(340,100.)
    meta=dict(dates=[str(days[k].date())for k in (294,339)],symbols=[stocks+etfs,stocks+etfs],etfs=etfs)
    sectors=pd.DataFrame('Technology',index=pd.to_datetime(meta['dates']),columns=stocks)
    return dict(adj=pd.DataFrame(prices,index=days),sectors=sectors),meta,days


class WindowTests(unittest.TestCase):
    def test_253price_endpoints_skip42(self):
        p,m,s=panel_fixture();a=r.conditional_cells(p,m,s);self.assertEqual(len(a),98)
        self.assertEqual((a[0]['window_start'],a[0]['window_end']),(str(s[0].date()),str(s[252].date())));self.assertEqual(a[0]['benchmark'],'XLK');self.assertTrue(a[0]['available'])
    def test_recent42_and_future_poison_ignored(self):
        p,m,s=panel_fixture();a=r.conditional_cells(p,m,s);p['adj']=p['adj'].astype(object);p['adj'].loc[s[253]:,:]='unused recent/future'
        one=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);b=r.conditional_cells(p,one,s)
        for actual,expected in zip(b,a[:49]):r.compare(actual,expected)
    def test_missing_interior_official_row_not_compressed(self):
        p,m,s=panel_fixture();p['adj']=p['adj'].drop(s[100]);a=r.conditional_cells(p,m,s)
        self.assertEqual(a[0]['reason'],'invalid_sector_path');self.assertFalse(a[49]['available'])
    def test_no_sector_mapping_no_market_fallback(self):
        p,m,s=panel_fixture();p['sectors'].loc[:,'S00']='Unknown';a=r.conditional_cells(p,m,s);self.assertEqual(a[0]['reason'],'missing_sector_mapping')
    def test_all_etfs_unavailable(self):
        p,m,s=panel_fixture();a=r.conditional_cells(p,m,s)
        for i in list(range(32,49))+list(range(81,98)):self.assertEqual(a[i]['reason'],'etf_not_applicable')
    def test_missing_sector_price_column(self):
        p,m,s=panel_fixture();p['adj']=p['adj'].drop(columns='XLK');a=r.conditional_cells(p,m,s);self.assertEqual(a[0]['reason'],'invalid_sector_path')
    def test_short_history(self):
        p,m,s=panel_fixture();m=dict(m,dates=[str(s[293].date())],symbols=m['symbols'][:1]);p['sectors'].loc[s[293]]='Technology';a=r.conditional_cells(p,m,s);self.assertEqual(a[0]['reason'],'insufficient_history')
    def test_eligibility_not_used(self):
        p,m,s=panel_fixture();a=r.conditional_cells(p,m,s);m['eligible']=np.zeros((2,49),bool);self.assertEqual(a,r.conditional_cells(p,m,s))
    def test_prefix_exact(self):
        p,m,s=panel_fixture();a=r.conditional_cells(p,m,s);sub=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1]);p['adj']=p['adj'].loc[:m['dates'][0]];b=r.conditional_cells(p,sub,s);self.assertEqual(a[:49],b)
    def test_log_vector_matches_declared_frexp(self):
        p=np.exp(np.linspace(0,.5,253))*1e200;m,e=np.frexp(p);expected=np.log(m[1:]/m[:-1])+np.diff(e)*np.log(2.)
        np.testing.assert_array_equal(r.log_returns(p),expected)
    def test_all_equal_raw_has_positive_average_rank(self):
        a=r.rank_oracle([0.]*32);np.testing.assert_array_equal(a,np.full(32,1/64))





class FinalSchemaTests(unittest.TestCase):
    def test_group_full_diagnostics_against_producer(self):
        import monthly_sector_asymmetry138_factor as producer
        y,x=return_vectors()
        for yy,xx in ((y,x),(y*1e-100,x*1e-100),(y+1.,x),(-y,x)):
            r.compare(producer.conditional_gaps(yy,xx),r.detailed_moments(yy,xx))
    def test_unavailable_group_diagnostics_and_precedence(self):
        import monthly_sector_asymmetry138_factor as producer
        y,x=return_vectors()
        cases=[]
        yy=y.copy();yy[:126]=.003;cases.append((yy,x.copy()))
        xx=x.copy();xx[:126]=.003;cases.append((y.copy(),xx))
        xx=x.copy();xx[39:126]=0.;cases.append((y.copy(),xx))
        for yy,xx in cases:r.compare(producer.conditional_gaps(yy,xx),r.detailed_moments(yy,xx))
    def test_all_cells_arrays_schema_and_independent_ranks(self):
        import monthly_sector_asymmetry138_factor as producer
        p,m,s=panel_fixture();actual=producer.build_features(p,m,s,r.benchmark_map)
        expected=r.features_oracle(p,m,s,actual['arrays']);r.compare(actual['audit'],expected['audit'])
        for key in r.EXTRA:
            if key in r.EXTRA[:2]:np.testing.assert_allclose(actual['arrays'][key],expected['arrays'][key],rtol=0,atol=r.ATOL,equal_nan=True)
            else:np.testing.assert_array_equal(actual['arrays'][key],expected['arrays'][key])
    def test_missing_and_unknown_cells_match(self):
        import monthly_sector_asymmetry138_factor as producer
        p,m,s=panel_fixture();p['adj'].loc[s[100],'S00']=np.nan;p['sectors'].loc[:,'S01']='Unknown'
        actual=producer.build_features(p,m,s,r.benchmark_map);expected=r.features_oracle(p,m,s,actual['arrays'])
        r.compare(actual['audit'],expected['audit']);self.assertEqual(expected['audit']['cells'][0]['reason'],'invalid_stock_path')
    def test_changed_raw_not_accepted_for_ranking(self):
        p,m,s=panel_fixture();actual=r.features_oracle(p,m,s);actual['arrays']['raw_beta_gap'][0,0]+=.01
        with self.assertRaises(AssertionError):r.features_oracle(p,m,s,actual['arrays'])
    def test_empty_prefix(self):
        p,m,s=panel_fixture();sub=dict(m,dates=[],symbols=[]);result=r.features_oracle(dict(p,adj=p['adj'].iloc[:0]),sub,s)
        self.assertEqual(result['audit']['cells'],[]);self.assertEqual(result['arrays']['common_available'].shape,(0,49))
    def test_nonofficial_price_rejected(self):
        p,m,s=panel_fixture();extra=p['adj'].iloc[:1].copy();extra.index=[pd.Timestamp('2021-01-02')];p['adj']=pd.concat([p['adj'],extra]).sort_index()
        with self.assertRaises(ValueError):r.features_oracle(p,m,s)
    def test_input_bool_string_and_infinite_rejected(self):
        y,x=return_vectors()
        for bad in (np.full(252,True),np.full(252,'1'),np.full(252,np.inf)):
            with self.assertRaises(ValueError):r.detailed_moments(y,bad)
    def test_full_checker_has_no_producer_imports(self):
        tree=r.ast.parse(Path(r.__file__).read_text())
        names=[node.module for node in r.ast.walk(tree) if isinstance(node,r.ast.ImportFrom)]
        names+=[alias.name for node in r.ast.walk(tree) if isinstance(node,r.ast.Import) for alias in node.names]
        self.assertFalse(any(name and 'sector_asymmetry138' in name for name in names))
    def test_changed_discrete_schema_and_hash_rejected(self):
        for changed in (dict(available=1,q=.5),dict(available=True,q=.6),dict(available=True,q=.5,other=1)):
            with self.assertRaises(ValueError):r.compare(changed,dict(available=True,q=.5))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'input';p.write_text('one')
            with self.assertRaises(ValueError):r.old.Reader(directory).verify({str(p):'0'*64})


def full_score_fixture():
    a,_=score_fixture();a={k:np.tile(v,(61,10))[:,:49] if v.ndim==2 else np.tile(v,61) for k,v in a.items()}
    names=['S%02d'%j for j in range(32)]+['E%02d'%j for j in range(17)]
    m=dict(symbols=[names[:] for _ in range(122)],etfs=names[32:])
    raw=np.tile(np.linspace(-.48,.48,49),(122,1));cw=a['core_weight'][:,None];a['core']=cw*raw
    a['baseline']=a['core']+np.tile(np.sin(np.arange(49))*.015,(122,1));a['pre_score']=a['baseline'].copy()
    a['effects']=np.tile(np.cos(np.arange(49))*.05,(122,1))
    beta=np.tile(r.rank_oracle(np.linspace(-1.,1.,32)),(122,1));rho=beta[:,::-1].copy()
    beta=np.c_[beta,np.full((122,17),np.nan)];rho=np.c_[rho,np.full((122,17),np.nan)]
    beta[5,:]=np.nan;rho[5,:]=np.nan;a.update(rank_beta_gap=beta,rank_correlation_gap=rho)
    return a,m,beta,rho


class FullScoreIntegrationTests(unittest.TestCase):
    def backend(self,m):
        import monthly_sector_asymmetry138_runner as producer
        h=r.old.original_helper()
        class Backend:
            adapter=producer.module('independent138_test_adapter',r.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
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
        import monthly_sector_asymmetry138_runner as producer
        a,m,beta,rho=full_score_fixture();ss,cc,_=r.score_oracle(a,m,beta,rho);backend=self.backend(m)
        for name in r.NAMES:
            c,s=producer.choose(name,a,m,122,backend,r.old.original_helper().POLICY)
            np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_prefixes_with_original_episode(self):
        import monthly_sector_asymmetry138_runner as producer
        a,m,beta,rho=full_score_fixture();backend=self.backend(m)
        for n in (0,1,36,60,76,88):
            ss,cc,_=r.score_oracle(a,m,beta,rho,n)
            for name in r.NAMES:
                c,s=producer.choose(name,a,m,n,backend,r.old.original_helper().POLICY)
                np.testing.assert_allclose(s,ss[name],rtol=0,atol=r.ATOL);self.assertEqual(c,cc[name])
    def test_interface_exact(self):
        import monthly_sector_asymmetry138_runner as producer
        for key in ('INPUTS','CACHE_FILES','NAMES','CHECKS','EXTRA'):self.assertEqual(getattr(r,key),getattr(producer,key))
    def test_independent_scores_do_not_call_new_producer(self):
        import monthly_sector_asymmetry138_runner as producer
        a,m,beta,rho=full_score_fixture()
        with patch.object(producer,'choose',side_effect=AssertionError('not an oracle')):
            scores,choices,_=r.score_oracle(a,m,beta,rho);self.assertEqual(len(choices[r.NAMES[2]]),122)


if __name__=='__main__':unittest.main()
